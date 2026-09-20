"""Hard process deadlines for learned chat; numerical work stays in workers."""

from __future__ import annotations

import math
import sys
import threading
from copy import deepcopy
from typing import Any

from ..conversation.persistence import encode_json
from ..conversation.schema import ConversationLimits
from ..conversation.supervisor import run_json_worker
from .persistence import envelope

WORKER = "text_factors.learning.worker"
MAX_WIRE_BYTES = 16_000_000


class SupervisedLearnedConversation:
    def __init__(
        self,
        model: dict[str, Any],
        *,
        state: dict[str, Any] | None = None,
        seconds: float = 15.0,
        limits: ConversationLimits | None = None,
    ) -> None:
        if (
            type(seconds) not in (float, int)
            or not math.isfinite(seconds)
            or not 0 < seconds <= 300
        ):
            raise ValueError("worker seconds must be finite and in (0, 300]")
        header = envelope(model, kind="model")
        self.model = deepcopy(model)
        self.model_fingerprint = header["sha256"]
        self.seconds = float(seconds)
        self.state = deepcopy(state)
        self.limits = limits or ConversationLimits(max_state_bytes=3_000_000)
        if state is not None:
            envelope(state, kind="session")
            self.limits = ConversationLimits.from_dict(state["limits"])
            if state["model_fingerprint"] != self.model_fingerprint:
                raise ValueError("state belongs to a different model")
        if self.limits.max_state_bytes > 3_000_000:
            raise ValueError("supervised learned state budget is at most 3000000 bytes")
        self.last_status = "ready"
        self._lock = threading.Lock()

    def _failure(self, status: str, elapsed: float = 0.0) -> dict[str, Any]:
        return {
            "turn_id": 0 if self.state is None else self.state["turn_count"],
            "text": (
                "Не удалось завершить обработку в заданных пределах. "
                "Память не изменена."
            ),
            "action": "limit" if status in {"timeout", "output_limit"} else "error",
            "complete": False,
            "reason": "worker_" + status,
            "meaning": None,
            "assertions": [],
            "evidence": [],
            "diagnostics": {"emergency_response": True},
            "elapsed_seconds": elapsed,
        }

    def respond(self, text: str, *, request_id: str | None = None) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return self._failure("busy")
        try:
            payload = {
                "operation": "turn",
                "model": self.model,
                "state": self.state,
                "limits": self.limits.to_dict(),
                "text": text,
                "request_id": request_id,
            }
            # Check the actual combined model/state/request wire, including its
            # overhead, before launching. Nothing is persisted on rejection.
            encode_json(payload, max_bytes=MAX_WIRE_BYTES)
            result = run_json_worker(
                [sys.executable, "-m", WORKER],
                payload,
                seconds=self.seconds,
                max_output_bytes=MAX_WIRE_BYTES,
            )
            self.last_status = result["status"]
            if result["status"] == "completed":
                try:
                    output = result["result"]
                    if type(output) is not dict or set(output) != {"state", "response"}:
                        raise ValueError("invalid worker output fields")
                    state, response = output["state"], output["response"]
                    envelope(state, kind="session")
                    if (
                        state["model_fingerprint"] != self.model_fingerprint
                        or state["limits"] != self.limits.to_dict()
                    ):
                        raise ValueError("worker changed session identity or limits")
                    if (
                        type(response) is not dict
                        or set(response)
                        != {
                            "turn_id",
                            "text",
                            "action",
                            "complete",
                            "reason",
                            "meaning",
                            "assertions",
                            "evidence",
                            "diagnostics",
                            "elapsed_seconds",
                        }
                        or type(response["action"]) is not str
                        or response["action"]
                        not in {
                            "answer",
                            "ack",
                            "clarify",
                            "unknown",
                            "explain",
                            "greet",
                            "thanks",
                            "help",
                            "retracted",
                            "corrected",
                            "nonactual",
                            "error",
                            "limit",
                        }
                        or type(response["text"]) is not str
                        or len(response["text"]) > self.limits.max_chars
                        or type(response["complete"]) is not bool
                        or type(response["turn_id"]) is not int
                        or response["turn_id"] != state["turn_count"]
                    ):
                        raise ValueError("invalid worker response")
                    if response["action"] not in {"error", "limit"}:
                        self.state = deepcopy(state)
                    else:
                        self.last_status = response["action"]
                    return deepcopy(response)
                except (KeyError, TypeError, ValueError):
                    self.last_status = "error"
            return self._failure(self.last_status, result["elapsed_seconds"])
        finally:
            self._lock.release()

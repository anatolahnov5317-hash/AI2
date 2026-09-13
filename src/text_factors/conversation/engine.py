"""Transactional dialogue orchestration around AI2's learned predicate bridge.

No external language model or implicit online learning is used. This API has
cooperative deadlines; use the supervised CLI/worker for hard process deadlines.
Only explicit /teach commands can change the learned lexical associations.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any

from .bridge import LABELS, MAX_PAIRS, FactorSemanticBridge
from .language import DEFAULT_TEACHING_PAIRS, RussianParser
from .realization import realize
from .schema import (
    Budget,
    BudgetExceeded,
    ConversationLimits,
    SemanticFrame,
    StateOutcome,
    TurnResponse,
)
from .state import WorldState

_SCHEMA = "ai2-grounded-conversation-v1"
_MAX_ID = 2**53 - 1


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class ConversationSession:
    """A bounded, single-writer session with topic-local factual memory.

    Failed turns do not change facts or teaching. Recent request IDs are
    idempotent while their receipt remains in the bounded, byte-limited history.
    Public state is for inspection; modifying it directly is unsupported.
    A duck-typed bridge can be injected for diagnostics, but cannot be loaded
    through production persistence unless it uses the validated factor schema.
    """

    def __init__(
        self,
        seed: int = 42,
        limits: ConversationLimits | None = None,
        bridge: Any = None,
        train_defaults: bool = True,
    ) -> None:
        if limits is not None and not isinstance(limits, ConversationLimits):
            raise ValueError("limits must be ConversationLimits")
        if type(train_defaults) is not bool:
            raise ValueError("train_defaults must be boolean")
        self.limits = limits or ConversationLimits()
        self.parser = RussianParser(self.limits)
        self.state = WorldState(self.limits)
        self.bridge = bridge if bridge is not None else FactorSemanticBridge(seed=seed)
        self.turn_count = 0
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if train_defaults:
            if len(DEFAULT_TEACHING_PAIRS) > self.limits.max_training_examples:
                raise ValueError("default teaching exceeds configured capacity")
            self.bridge.fit(list(DEFAULT_TEACHING_PAIRS), seconds=15.0)
        if isinstance(self.bridge, FactorSemanticBridge) and (
            len(self.bridge.to_dict()["examples"]) > self.limits.max_training_examples
        ):
            raise ValueError("existing teaching exceeds configured capacity")
        if len(_canonical(self._snapshot())) > self.limits.max_state_bytes:
            raise ValueError("initial session exceeds configured byte capacity")

    def _snapshot(
        self,
        *,
        state: WorldState | None = None,
        history: list[dict[str, Any]] | None = None,
        turn_count: int | None = None,
        bridge: Any = None,
    ) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "limits": self.limits.to_dict(),
            "bridge": (self.bridge if bridge is None else bridge).to_dict(),
            "world": (self.state if state is None else state).to_dict(),
            "turn_count": self.turn_count if turn_count is None else turn_count,
            "history": deepcopy(self._history if history is None else history),
        }

    def to_dict(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ValueError("session is busy")
        try:
            return self._snapshot()
        finally:
            self._lock.release()

    @classmethod
    def from_dict(cls, value: Any) -> ConversationSession:
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "limits",
                "bridge",
                "world",
                "turn_count",
                "history",
            }
            or value["schema"] != _SCHEMA
        ):
            raise ValueError("invalid conversation snapshot schema")
        limits = ConversationLimits.from_dict(value["limits"])
        count = value["turn_count"]
        history = value["history"]
        if type(count) is not int or not 0 <= count < _MAX_ID:
            raise ValueError("invalid conversation turn count")
        if type(history) is not list or len(history) > limits.max_history:
            raise ValueError("invalid conversation history")
        previous = 0
        request_ids: set[str] = set()
        for entry in history:
            if type(entry) is not dict or set(entry) != {
                "input",
                "request_id",
                "response",
            }:
                raise ValueError("invalid history entry")
            raw = entry["input"]
            if type(raw) is not str or len(raw) > limits.max_chars or "\x00" in raw:
                raise ValueError("invalid history input")
            request_id = entry["request_id"]
            cls._validate_request_id(request_id)
            if request_id is not None:
                if request_id in request_ids:
                    raise ValueError("duplicate saved request ID")
                request_ids.add(request_id)
            response = TurnResponse.from_dict(entry["response"])
            if not previous < response.turn_id <= count:
                raise ValueError("nonmonotonic saved responses")
            if len(response.frames) > limits.max_clauses:
                raise ValueError("too many saved response clauses")
            previous = response.turn_id
        try:
            if len(_canonical(value)) > limits.max_state_bytes:
                raise ValueError("conversation snapshot exceeds byte capacity")
        except (TypeError, RecursionError) as error:
            raise ValueError("invalid conversation JSON") from error
        # Validate inexpensive event structure before bounded numerical replay.
        world = WorldState.from_dict(value["world"], limits=limits)
        if world.events and world.events[-1]["turn_id"] > count:
            raise ValueError("world contains a future turn")
        # A cached answer is not an independent authority. Validate its facts
        # against the event history AS OF that answer, and its text against the
        # same disclosed renderer used at runtime. Later corrections may have
        # legitimately changed today's facts without invalidating old receipts.
        historical = WorldState(limits)
        cursor = 0
        for entry in history:
            response = TurnResponse.from_dict(entry["response"])
            while (
                cursor < len(world.events)
                and world.events[cursor]["turn_id"] <= response.turn_id
            ):
                event = world.events[cursor]
                historical.apply(
                    SemanticFrame.from_dict(event["input_frame"]),
                    turn_id=event["turn_id"],
                    source=event["source"],
                )
                cursor += 1
            if response.assertions and response.action not in {"ack", "answer"}:
                raise ValueError("unexpected cached assertions")
            known_facts = set(historical.facts())
            if any(assertion not in known_facts for assertion in response.assertions):
                raise ValueError("cached assertion has no matching historical evidence")
            reconstructed = StateOutcome(
                response.action,
                assertions=response.assertions,
                reason=response.reason,
                alternatives=response.alternatives,
                resolved_frame=response.frames[-1] if response.frames else None,
            )
            if response.text != realize(reconstructed, max_facts=limits.max_candidates):
                raise ValueError(
                    "cached text does not match grounded response rendering"
                )
        bridge = FactorSemanticBridge.from_dict(value["bridge"])
        if len(bridge.to_dict()["examples"]) > limits.max_training_examples:
            raise ValueError("saved teaching exceeds configured capacity")
        session = cls(limits=limits, bridge=bridge, train_defaults=False)
        session.state, session.turn_count = world, count
        session._history = deepcopy(history)
        return session

    @staticmethod
    def _validate_request_id(request_id: Any) -> None:
        if request_id is not None and (
            type(request_id) is not str
            or not 1 <= len(request_id) <= 128
            or any(ord(c) < 33 or ord(c) == 127 for c in request_id)
        ):
            raise ValueError("request_id must be a bounded nonempty printable string")
        if request_id is not None:
            try:
                request_id.encode("utf-8")
            except UnicodeError as error:
                raise ValueError("request_id must be valid UTF-8") from error

    def _failure(
        self,
        reason: str,
        *,
        turn_id: int,
        started: float,
        frames: tuple[SemanticFrame, ...] = (),
        evidence: tuple[dict[str, Any], ...] = (),
        outcome: StateOutcome | None = None,
    ) -> TurnResponse:
        return TurnResponse(
            turn_id,
            realize(outcome or StateOutcome("clarify", reason=reason)),
            "clarify",
            frames=frames,
            evidence=evidence,
            complete=False,
            reason=reason,
            elapsed_seconds=perf_counter() - started,
            alternatives=outcome.alternatives if outcome else (),
        )

    def _teaching(self, text: str, budget: Budget) -> tuple[Any, StateOutcome]:
        parts = text.strip().split()
        if len(parts) != 3 or parts[0] != "/teach" or parts[2] not in LABELS:
            raise ValueError("invalid_teaching")
        if not isinstance(self.bridge, FactorSemanticBridge):
            raise ValueError("invalid_teaching")
        recipe = self.bridge.to_dict()
        pairs = [tuple(pair) for pair in recipe["examples"]]
        pair = (parts[1].casefold().replace("ё", "е"), parts[2])
        if len(set(pairs + [pair])) > min(MAX_PAIRS, self.limits.max_training_examples):
            raise ValueError("teaching_capacity")
        staged = FactorSemanticBridge(seed=self.bridge.seed, mode=self.bridge.mode)
        budget.check()
        staged.fit(
            sorted(set(pairs + [pair])),
            epochs=max(8, recipe["epochs"]),
            seconds=max(
                0.000001,
                min(15.0, budget.deadline - perf_counter()),
            ),
        )
        budget.check()
        return staged, StateOutcome("help", reason="taught")

    def respond(self, text: str, *, request_id: str | None = None) -> TurnResponse:
        started = perf_counter()
        self._validate_request_id(request_id)
        if type(text) is not str:
            raise ValueError("text must be a string")
        if not self._lock.acquire(blocking=False):
            return self._failure("busy", turn_id=self.turn_count, started=started)
        try:
            for entry in self._history:
                if request_id is not None and entry["request_id"] == request_id:
                    if entry["input"] == text:
                        return TurnResponse.from_dict(entry["response"])
                    return self._failure(
                        "request_id_conflict",
                        turn_id=self.turn_count,
                        started=started,
                    )
            if self.turn_count >= _MAX_ID - 1:
                return self._failure(
                    "session_capacity", turn_id=self.turn_count, started=started
                )
            turn_id = self.turn_count + 1
            # Oversized/invalid inputs must never enter persisted history.
            if len(text) > self.limits.max_chars or "\x00" in text:
                return self._failure("input_capacity", turn_id=turn_id, started=started)
            try:
                text.encode("utf-8")
            except UnicodeError:
                return self._failure("invalid_input", turn_id=turn_id, started=started)
            budget = Budget(self.limits.turn_seconds)
            staged = self.state.clone()
            proposed_bridge = self.bridge
            frames: list[SemanticFrame] = []
            evidence: list[dict[str, Any]] = []
            outcome: StateOutcome | None = None
            try:
                budget.check()
                if text.lstrip().startswith("/teach"):
                    proposed_bridge, outcome = self._teaching(text, budget)
                    response = TurnResponse(
                        turn_id,
                        realize(StateOutcome("taught")),
                        "taught",
                        reason="explicit_teaching",
                        elapsed_seconds=perf_counter() - started,
                    )
                else:
                    parsed = self.parser.parse(text)
                    budget.check()
                    if not parsed.complete or not parsed.frames:
                        outcome = StateOutcome(
                            "clarify",
                            reason=parsed.reason or "unsupported_input",
                            alternatives=parsed.alternatives,
                        )
                        response = self._failure(
                            outcome.reason,
                            turn_id=turn_id,
                            started=started,
                            outcome=outcome,
                        )
                    else:
                        for frame in parsed.frames:
                            budget.check()
                            if frame.cue:
                                resolution = self.bridge.classify(frame.cue)
                                budget.check()
                                evidence.append(resolution.to_dict())
                                if resolution.label not in LABELS:
                                    outcome = StateOutcome(
                                        "clarify",
                                        reason=resolution.evidence.get(
                                            "reason", "unrecognized_predicate"
                                        ),
                                        alternatives=resolution.candidates[
                                            : self.limits.max_candidates
                                        ],
                                    )
                                    break
                                frame = replace(frame, predicate=resolution.label)
                            frames.append(frame)
                            outcome = staged.apply(frame, turn_id=turn_id)
                            budget.check()
                            if outcome.action == "clarify":
                                break
                        if outcome is None or outcome.action == "clarify":
                            response = self._failure(
                                outcome.reason if outcome else "unsupported_input",
                                turn_id=turn_id,
                                started=started,
                                frames=tuple(frames),
                                evidence=tuple(evidence),
                                outcome=outcome,
                            )
                        else:
                            response = TurnResponse(
                                turn_id,
                                realize(outcome, max_facts=self.limits.max_candidates),
                                outcome.action,
                                frames=tuple(frames),
                                assertions=outcome.assertions,
                                evidence=tuple(evidence),
                                reason=outcome.reason,
                                elapsed_seconds=perf_counter() - started,
                            )
                budget.check()
            except (BudgetExceeded, TimeoutError):
                response = self._failure(
                    "turn_time_budget", turn_id=turn_id, started=started
                )
            except ValueError as error:
                reason = str(error)
                response = self._failure(
                    reason
                    if reason in {"invalid_teaching", "teaching_capacity"}
                    else "invalid_input",
                    turn_id=turn_id,
                    started=started,
                )
            except Exception:
                # Numerical/adapter faults are contained, not silently asserted.
                # The public trace exposes the failure, never sensitive internals.
                response = self._failure(
                    "internal_error", turn_id=turn_id, started=started
                )
            if not response.complete:
                staged, proposed_bridge = self.state, self.bridge
            history = (
                self._history
                + [
                    {
                        "input": text,
                        "request_id": request_id,
                        "response": response.to_dict(),
                    }
                ]
            )[-self.limits.max_history :]
            while (
                history
                and len(
                    _canonical(
                        self._snapshot(
                            state=staged,
                            bridge=proposed_bridge,
                            history=history,
                            turn_count=turn_id,
                        )
                    )
                )
                > self.limits.max_state_bytes
            ):
                history.pop(0)
            if (
                len(
                    _canonical(
                        self._snapshot(
                            state=staged,
                            bridge=proposed_bridge,
                            history=history,
                            turn_count=turn_id,
                        )
                    )
                )
                > self.limits.max_state_bytes
            ):
                return self._failure(
                    "session_capacity", turn_id=turn_id, started=started
                )
            # Serialization belongs to the turn budget too. A completed answer
            # may not commit merely because it timed out after generating text.
            if response.complete and perf_counter() >= budget.deadline:
                return self._failure(
                    "turn_time_budget", turn_id=turn_id, started=started
                )
            self.state, self.bridge = staged, proposed_bridge
            self._history, self.turn_count = history, turn_id
            return response
        finally:
            self._lock.release()

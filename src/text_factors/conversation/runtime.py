"""Hard-deadline runtime and single-writer session-file guard for the CLI."""

from __future__ import annotations

import math
import os
import stat
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .persistence import session_envelope, state_from_envelope
from .realization import realize
from .schema import ConversationLimits, StateOutcome, TurnResponse
from .supervisor import run_json_worker


class SupervisedConversation:
    """Keep the last successful snapshot outside the disposable worker.

    Numeric model reconstruction, parsing, inference, state replay and response
    generation all run inside the same hard wall-clock limit. A failed worker
    never replaces the parent's previous snapshot.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        limits: ConversationLimits | None = None,
        seconds: float = 10.0,
        state: dict[str, Any] | None = None,
    ) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 0 < seconds <= 300
        ):
            raise ValueError("worker timeout must be finite and in (0, 300]")
        self.seed, self.seconds = seed, float(seconds)
        self.limits = limits or ConversationLimits()
        self.state = deepcopy(state)
        if state is not None:
            self.limits = ConversationLimits.from_dict(state["limits"])
            # Header/checksum/byte validation only: no numerical work in parent.
            self.state = state_from_envelope(
                session_envelope(
                    state,
                    max_bytes=self.limits.max_state_bytes,
                )
            )
        self.last_status = "ready"
        if self.limits.max_state_bytes > 12_000_000:
            raise ValueError("supervised sessions require max_state_bytes <= 12000000")

    def respond(self, text: str, *, request_id: str | None = None) -> TurnResponse:
        result = run_json_worker(
            [sys.executable, "-m", "text_factors.conversation.worker"],
            {
                "operation": "turn",
                "state": self.state,
                "seed": self.seed,
                "limits": self.limits.to_dict(),
                "text": text,
                "request_id": request_id,
            },
            seconds=self.seconds,
            max_output_bytes=16_000_000,
        )
        self.last_status = result["status"]
        if result["status"] == "completed":
            try:
                output = result["result"]
                if type(output) is not dict or set(output) != {"state", "response"}:
                    raise ValueError("invalid turn worker result")
                response = TurnResponse.from_dict(output["response"])
                next_state = state_from_envelope(
                    session_envelope(
                        output["state"],
                        max_bytes=self.limits.max_state_bytes,
                    )
                )
                if next_state["limits"] != self.limits.to_dict():
                    raise ValueError("worker changed session limits")
                self.state = next_state
                return response
            except (KeyError, TypeError, ValueError):
                self.last_status = "error"
        reason = (
            "turn_time_budget" if result["status"] == "timeout" else "internal_error"
        )
        return TurnResponse(
            turn_id=0 if self.state is None else self.state["turn_count"],
            text=realize(StateOutcome("clarify", reason=reason)),
            action="clarify",
            complete=False,
            reason=reason,
            elapsed_seconds=result["elapsed_seconds"],
        )


class SessionFileLock:
    """Nonblocking OS lock; a killed process releases it automatically.

    The small companion .lock file is deliberately retained: unlinking a locked
    inode would let another process bypass the lock by creating a new inode.
    Network-filesystem locking semantics are outside this local CLI contract.
    """

    def __init__(self, state_path: Path) -> None:
        self.path = state_path.with_name(state_path.name + ".lock")
        self._fd: int | None = None

    def __enter__(self) -> SessionFileLock:
        try:
            before = self.path.lstat()
        except FileNotFoundError:
            before = None
        if before is not None and not stat.S_ISREG(before.st_mode):
            raise ValueError("session lock must be a regular file, not a symlink")
        flags = (
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(self.path, flags, 0o600)
        try:
            current = os.fstat(fd)
            if not stat.S_ISREG(current.st_mode):
                raise ValueError("invalid session lock file")
            if os.name == "posix":
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":
                import msvcrt

                if current.st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                raise ValueError("session file locking is unsupported on this platform")
        except OSError as error:
            os.close(fd)
            raise ValueError("session file is in use; no state was modified") from error
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *_: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

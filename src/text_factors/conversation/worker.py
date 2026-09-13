"""One JSON request per process; state is returned, never saved in place.

The parent supervisor owns the wall-clock deadline and state-file commit.
Evaluation checkpoints are the only optional worker writes and target a fresh
private temporary directory supplied by that parent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .engine import ConversationSession
from .persistence import atomic_write_json, decode_json, encode_json
from .schema import ConversationLimits

MAX_WIRE_BYTES = 16_000_000


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    operation = payload.get("operation")
    if operation == "turn":
        if set(payload) != {
            "operation",
            "state",
            "limits",
            "seed",
            "text",
            "request_id",
        }:
            raise ValueError("invalid turn worker request")
        if payload["state"] is None:
            session = ConversationSession(
                seed=payload["seed"],
                limits=ConversationLimits.from_dict(payload["limits"]),
            )
        else:
            session = ConversationSession.from_dict(payload["state"])
        response = session.respond(payload["text"], request_id=payload["request_id"])
        return {"state": session.to_dict(), "response": response.to_dict()}
    if operation == "evaluate":
        if set(payload) != {
            "operation",
            "seeds",
            "modes",
            "seconds",
            "split",
            "checkpoint",
        }:
            raise ValueError("invalid evaluation worker request")
        from ..evaluation.grounded_dialogue import evaluate_grounded_dialogue

        checkpoint = payload["checkpoint"]
        if checkpoint is not None and (
            type(checkpoint) is not str or not Path(checkpoint).is_absolute()
        ):
            raise ValueError("checkpoint must be an absolute temporary path")

        def progress(report: dict[str, Any]) -> None:
            if checkpoint is not None:
                path = Path(checkpoint)
                atomic_write_json(
                    path,
                    report,
                    overwrite=path.exists(),
                    max_bytes=MAX_WIRE_BYTES,
                )

        return evaluate_grounded_dialogue(
            seeds=tuple(payload["seeds"]),
            modes=tuple(payload["modes"]),
            seconds=payload["seconds"],
            split=payload["split"],
            progress=progress,
        )
    raise ValueError("unknown worker operation")


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_WIRE_BYTES + 1)
        payload = decode_json(raw, max_bytes=MAX_WIRE_BYTES)
        result = execute(payload)
        sys.stdout.buffer.write(encode_json(result, max_bytes=MAX_WIRE_BYTES))
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        # Do not echo arbitrary input, state, paths or environment variables.
        print(f"conversation worker failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

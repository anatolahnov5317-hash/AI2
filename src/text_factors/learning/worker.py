"""Single-request learned worker; only the parent may save a model or session."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..conversation.persistence import atomic_write_json, decode_json, encode_json
from ..conversation.schema import ConversationLimits
from .model import ModelBundle
from .runtime import MAX_WIRE_BYTES
from .schema import exact_fields


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    operation = payload.get("operation")
    if operation == "train":
        exact_fields(
            payload, {"operation", "seed", "seconds", "dataset"}, "training request"
        )
        bundle = ModelBundle.fit(
            seed=payload["seed"], seconds=payload["seconds"], dataset=payload["dataset"]
        )
        return {"model": bundle.to_dict(), "fingerprint": bundle.fingerprint}
    if operation == "turn":
        from .session import LearnedSession

        exact_fields(
            payload,
            {"operation", "model", "state", "limits", "text", "request_id"},
            "learned turn request",
        )
        bundle = ModelBundle.from_dict(payload["model"])
        limits = ConversationLimits.from_dict(payload["limits"])
        session = (
            LearnedSession(bundle, limits)
            if payload["state"] is None
            else LearnedSession.from_dict(payload["state"], bundle)
        )
        if session.limits != limits:
            raise ValueError("worker request limits do not match saved session")
        response = session.respond(payload["text"], request_id=payload["request_id"])
        return {"state": session.to_dict(), "response": response}
    if operation == "evaluate":
        from .evaluation import evaluate_learned_dialogue

        exact_fields(
            payload,
            {"operation", "model", "split", "seconds", "source_freeze", "checkpoint"},
            "evaluation request",
        )
        bundle = ModelBundle.from_dict(payload["model"])
        checkpoint = payload["checkpoint"]
        if checkpoint is not None and (
            type(checkpoint) is not str or not Path(checkpoint).is_absolute()
        ):
            raise ValueError("evaluation checkpoint must be an absolute private path")

        def progress(report: dict[str, Any]) -> None:
            if checkpoint is not None:
                path = Path(checkpoint)
                atomic_write_json(
                    path, report, overwrite=path.exists(), max_bytes=MAX_WIRE_BYTES
                )

        return evaluate_learned_dialogue(
            bundle,
            split=payload["split"],
            seconds=payload["seconds"],
            source_freeze=payload["source_freeze"],
            progress=progress,
        )
    if operation == "freeze":
        from .evaluation import capture_source_freeze

        exact_fields(payload, {"operation", "model"}, "freeze request")
        return capture_source_freeze(ModelBundle.from_dict(payload["model"]))
    raise ValueError("unsupported learned worker operation")


def main() -> int:
    try:
        payload = decode_json(
            sys.stdin.buffer.read(MAX_WIRE_BYTES + 1), max_bytes=MAX_WIRE_BYTES
        )
        result = execute(payload)
        sys.stdout.buffer.write(encode_json(result, max_bytes=MAX_WIRE_BYTES))
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:
        print(f"learned worker failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

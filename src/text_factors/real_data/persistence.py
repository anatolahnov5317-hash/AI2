"""Atomic JSON persistence for experimental real-data state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contexts import ContextRegistry
from .engine import RealDataEngine

STATE_SCHEMA = "ai2-real-data-test-state-v1"
DEFAULT_MAX_STATE_BYTES = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("state is not canonical JSON") from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_limit(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("max_state_bytes must be a positive integer")
    return value


def _validate_destination(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("state path cannot be a symlink")
    if path.exists() and not path.is_file():
        raise ValueError("state path must be a regular file")


def state_payload(
    engine: RealDataEngine,
    *,
    contexts: ContextRegistry | None = None,
    model_version: str,
) -> dict[str, Any]:
    if type(model_version) is not str or not model_version:
        raise ValueError("model_version must be a nonempty string")
    core = {
        "model_version": model_version,
        "engine": engine.to_dict(),
        "contexts": contexts.to_dict() if contexts is not None else None,
    }
    return {
        "schema": STATE_SCHEMA,
        "fingerprint": _fingerprint(core),
        "payload": core,
    }


def save_state(
    path: str | Path,
    engine: RealDataEngine,
    *,
    contexts: ContextRegistry | None = None,
    model_version: str,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> Path:
    limit = _positive_limit(max_state_bytes)
    destination = Path(path)
    _validate_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(
        state_payload(
            engine,
            contexts=contexts,
            model_version=model_version,
        )
    )
    if len(payload) > limit:
        raise ValueError("real-data state exceeds max_state_bytes")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_state(
    path: str | Path,
    *,
    max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
) -> tuple[RealDataEngine, ContextRegistry | None, str]:
    limit = _positive_limit(max_state_bytes)
    source = Path(path)
    _validate_destination(source)
    if not source.exists():
        raise ValueError("real-data state does not exist")
    size = source.stat().st_size
    if size <= 0 or size > limit:
        raise ValueError("real-data state file size is invalid")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid real-data state JSON") from exc
    if (
        type(value) is not dict
        or set(value) != {"schema", "fingerprint", "payload"}
        or value.get("schema") != STATE_SCHEMA
        or type(value.get("payload")) is not dict
        or type(value.get("fingerprint")) is not str
    ):
        raise ValueError("unexpected real-data state schema")
    payload = value["payload"]
    if _fingerprint(payload) != value["fingerprint"]:
        raise ValueError("real-data state fingerprint mismatch")
    if set(payload) != {"model_version", "engine", "contexts"}:
        raise ValueError("invalid real-data payload fields")
    model_version = payload["model_version"]
    if type(model_version) is not str or not model_version:
        raise ValueError("invalid persisted model_version")
    if type(payload["engine"]) is not dict:
        raise ValueError("invalid persisted engine")
    engine = RealDataEngine.from_dict(payload["engine"])
    raw_contexts = payload["contexts"]
    if raw_contexts is not None and type(raw_contexts) is not dict:
        raise ValueError("invalid persisted contexts")
    contexts = (
        ContextRegistry.from_dict(raw_contexts) if raw_contexts is not None else None
    )
    return engine, contexts, model_version

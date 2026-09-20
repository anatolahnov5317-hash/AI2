"""Checksummed learned artifacts, using v0.4's tested atomic file primitive."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any

from ..conversation.persistence import _atomic_write, encode_json, read_json
from .model import MAX_MODEL_BYTES
from .model import SCHEMA as MODEL_SCHEMA
from .schema import exact_fields

KINDS = {
    "model": (MODEL_SCHEMA, MAX_MODEL_BYTES),
    "session": ("ai2-learned-dialogue-session-v1", 3_000_000),
}


def envelope(payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if type(kind) is not str or kind not in KINDS:
        raise ValueError("invalid learned artifact kind")
    schema, limit = KINDS[kind]
    if type(payload) is not dict or payload.get("schema") != schema:
        raise ValueError("invalid learned artifact schema")
    raw = encode_json(payload, max_bytes=limit)
    return {
        "kind": "ai2.learned." + kind,
        "version": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }


def unpack(value: Any, *, kind: str) -> dict[str, Any]:
    value = exact_fields(
        value, {"kind", "version", "sha256", "payload"}, "learned artifact"
    )
    expected = envelope(value["payload"], kind=kind)
    if (
        value["kind"] != expected["kind"]
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise ValueError("unsupported learned artifact kind/version")
    digest = value["sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or not hmac.compare_digest(digest, expected["sha256"])
    ):
        raise ValueError("learned artifact checksum mismatch")
    return value["payload"]


def read_artifact(path: Path, *, kind: str) -> dict[str, Any]:
    if type(kind) is not str or kind not in KINDS:
        raise ValueError("invalid learned artifact kind")
    return unpack(read_json(path, max_bytes=KINDS[kind][1] + 1024), kind=kind)


def save_artifact(
    payload: dict[str, Any], path: Path, *, kind: str, overwrite: bool = False
) -> Path:
    value = envelope(payload, kind=kind)
    raw = encode_json(value, max_bytes=KINDS[kind][1] + 1024)
    return _atomic_write(
        path,
        raw,
        overwrite=overwrite,
        existing_validator=lambda old: unpack(old, kind=kind),
    )

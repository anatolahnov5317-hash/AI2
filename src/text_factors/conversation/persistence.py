"""Bounded JSON snapshots with checksums and recoverable atomic writes.

The checksum detects accidental corruption; it is not an authentication scheme.
No state is deserialized with pickle or evaluated as Python.  Existing unrelated
files are never accepted as an overwrite target for ``save_session``.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import ConversationSession

DEFAULT_MAX_BYTES = 4_000_000
MAX_STATE_BYTES = 16_000_000
_ENVELOPE_ALLOWANCE = 1024
MAX_JSON_BYTES = MAX_STATE_BYTES + _ENVELOPE_ALLOWANCE
_MAX_DEPTH = 64
_MAX_NODES = 1_000_000
_MAX_INTEGER = 10**128
_SESSION_KIND = "ai2.conversation"
_SESSION_VERSION = 1
_STATE_SCHEMA = "ai2-grounded-conversation-v1"


def _check_byte_limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_JSON_BYTES:
        raise ValueError(f"max_bytes must be an integer in [1, {MAX_JSON_BYTES}]")


def _validate_json_tree(value: Any, max_bytes: int) -> None:
    """Validate before serialization, including objects ``json`` coerces."""
    nodes = 0
    estimated_bytes = 0
    ancestors: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes, estimated_bytes
        nodes += 1
        estimated_bytes += 1
        if nodes > min(_MAX_NODES, max_bytes) or estimated_bytes > max_bytes:
            raise ValueError("JSON exceeds its size or node budget")
        item_type = type(item)
        if item is None or item_type is bool:
            return
        if item_type is int:
            if abs(item) >= _MAX_INTEGER:
                raise ValueError("JSON integer exceeds its digit budget")
            return
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            return
        if item_type is str:
            estimated_bytes += len(item)
            if estimated_bytes > max_bytes:
                raise ValueError("JSON exceeds its byte budget")
            return
        if item_type not in (dict, list):
            raise ValueError("state must contain only JSON primitives")
        if depth > _MAX_DEPTH:
            raise ValueError("JSON nesting exceeds 64 levels")
        identity = id(item)
        if identity in ancestors:
            raise ValueError("JSON contains a circular reference")
        ancestors.add(identity)
        try:
            if item_type is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            ancestors.remove(identity)

    if type(value) is not dict:
        raise ValueError("JSON root must be an object")
    visit(value, 1)


def _encode_json(value: dict[str, Any], *, max_bytes: int) -> bytes:
    _check_byte_limit(max_bytes)
    _validate_json_tree(value, max_bytes)
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    output = bytearray()
    try:
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if len(output) + len(encoded) > max_bytes:
                raise ValueError("JSON exceeds its byte budget")
            output.extend(encoded)
    except (UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError("state cannot be encoded as bounded UTF-8 JSON") from exc
    return bytes(output)


def _decode_json(raw: bytes, *, max_bytes: int) -> dict[str, Any]:
    _check_byte_limit(max_bytes)
    if type(raw) is not bytes:
        raise ValueError("JSON input must be UTF-8 bytes")
    if len(raw) > max_bytes:
        raise ValueError("JSON exceeds its byte budget")
    # Reject deep syntax before json.loads can recurse into it.  Brackets inside
    # quoted strings are not nesting; escaping is handled one byte at a time.
    depth = 0
    quoted = False
    escaped = False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > _MAX_DEPTH:
                raise ValueError("JSON nesting exceeds 64 levels")
        elif char in (93, 125):
            depth -= 1

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key[:80]!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        if len(value) > 256:
            raise ValueError("JSON number exceeds its digit budget")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number

    def bounded_integer(value: str) -> int:
        if len(value.removeprefix("-")) > 128:
            raise ValueError("JSON integer exceeds its digit budget")
        return int(value)

    def reject_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON number: {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_int=bounded_integer,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid bounded UTF-8 JSON") from exc
    _validate_json_tree(decoded, max_bytes)
    return decoded


def encode_json(value: dict[str, Any], *, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    """Encode strict, bounded, canonical UTF-8 JSON without touching disk."""
    return _encode_json(value, max_bytes=max_bytes)


def decode_json(raw: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    """Decode one bounded JSON object, rejecting duplicates and nonfinite numbers."""
    return _decode_json(raw, max_bytes=max_bytes)


def _file_snapshot(path: Path) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("destination must be a regular file, not a symlink")
    return info


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _read_json_with_snapshot(
    path: Path, *, max_bytes: int
) -> tuple[dict[str, Any], os.stat_result]:
    _check_byte_limit(max_bytes)
    before = _file_snapshot(path)
    if before is None:
        raise FileNotFoundError(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise ValueError("file changed while it was being opened")
        if opened.st_size > max_bytes:
            raise ValueError("JSON exceeds its byte budget")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        after = os.fstat(descriptor)
        if not _same_file(opened, after):
            raise ValueError("file changed while it was being read")
    finally:
        os.close(descriptor)
    return _decode_json(raw, max_bytes=max_bytes), opened


def read_json(
    path: str | os.PathLike[str], *, max_bytes: int = DEFAULT_MAX_BYTES
) -> dict[str, Any]:
    """Read one regular UTF-8 JSON object without following a leaf symlink."""
    value, _ = _read_json_with_snapshot(Path(path), max_bytes=max_bytes)
    return value


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def _atomic_write(
    path: Path,
    raw: bytes,
    *,
    overwrite: bool,
    existing_validator: Callable[[dict[str, Any]], Any] | None = None,
) -> Path:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a boolean")
    path = path.absolute()
    before = _file_snapshot(path)
    if before is not None:
        if not overwrite:
            raise FileExistsError(path)
        if existing_validator is not None:
            previous, opened = _read_json_with_snapshot(path, max_bytes=MAX_JSON_BYTES)
            if not _same_file(before, opened):
                raise ValueError("overwrite target changed during validation")
            existing_validator(previous)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name[:80]}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        current = _file_snapshot(path)
        if before is None:
            # A hard link installs the fully written file with atomic O_EXCL-like
            # semantics.  os.replace would silently clobber a racing creator.
            if current is not None:
                raise FileExistsError(path)
            os.link(temporary, path)
            temporary.unlink()
        else:
            if current is None or not _same_file(before, current):
                raise ValueError("overwrite target changed before commit")
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        # This exact random temporary path is the only cleanup target.
        with suppress(FileNotFoundError):
            temporary.unlink()
    return path


def atomic_write_json(
    path: str | os.PathLike[str],
    data: dict[str, Any],
    *,
    overwrite: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Atomically write a bounded JSON report; parent directory must exist."""
    raw = _encode_json(data, max_bytes=max_bytes)
    return _atomic_write(Path(path), raw, overwrite=overwrite)


def _unpack_session(value: dict[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"kind", "version", "sha256", "state"}:
        raise ValueError("invalid conversation snapshot fields")
    if (
        value["kind"] != _SESSION_KIND
        or type(value["version"]) is not int
        or value["version"] != _SESSION_VERSION
    ):
        raise ValueError("unsupported conversation snapshot kind or version")
    digest = value["sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("invalid conversation snapshot checksum")
    state = value["state"]
    canonical = _encode_json(state, max_bytes=MAX_STATE_BYTES)
    actual = hashlib.sha256(canonical).hexdigest()
    if not hmac.compare_digest(digest, actual):
        raise ValueError("conversation snapshot checksum mismatch")
    return state


def _state_limit(state: dict[str, Any]) -> int:
    """Check inexpensive metadata only; semantic/numerical replay is a worker job."""
    from .schema import ConversationLimits

    if (
        type(state) is not dict
        or set(state)
        != {"schema", "limits", "bridge", "world", "turn_count", "history"}
        or state["schema"] != _STATE_SCHEMA
    ):
        raise ValueError("invalid conversation state schema")
    limits = ConversationLimits.from_dict(state["limits"])
    if (
        type(state["bridge"]) is not dict
        or type(state["world"]) is not dict
        or type(state["history"]) is not list
        or type(state["turn_count"]) is not int
        or not 0 <= state["turn_count"] < 2**53 - 1
        or len(state["history"]) > limits.max_history
    ):
        raise ValueError("invalid conversation state metadata")
    return limits.max_state_bytes


def session_envelope(
    state: dict[str, Any], max_bytes: int = DEFAULT_MAX_BYTES
) -> dict[str, Any]:
    """Wrap worker-validated state without importing or rebuilding the engine.

    This checks shape and declared limits, not the contents of learned memory or
    event history.  Callers must only commit state returned by a trusted worker.
    ``max_bytes`` applies to the canonical state, excluding envelope overhead.
    """
    _check_byte_limit(max_bytes)
    if max_bytes > MAX_STATE_BYTES:
        raise ValueError(f"state byte budget cannot exceed {MAX_STATE_BYTES}")
    declared_limit = _state_limit(state)
    canonical = _encode_json(state, max_bytes=min(max_bytes, declared_limit))
    return {
        "kind": _SESSION_KIND,
        "version": _SESSION_VERSION,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "state": state,
    }


def state_from_envelope(value: dict[str, Any]) -> dict[str, Any]:
    """Validate envelope/checksum/header/declared budget without engine replay."""
    state = _unpack_session(value)
    declared_limit = _state_limit(state)
    _encode_json(state, max_bytes=declared_limit)
    return state


def save_session(
    session: ConversationSession,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Save a versioned snapshot, refusing unrelated overwrite targets."""
    state = session.to_dict()
    _encode_json(state, max_bytes=session.limits.max_state_bytes)
    return save_session_state(state, path, overwrite=overwrite)


def save_session_state(
    state: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Commit worker-validated state without numerical replay in the parent.

    The existing destination must itself be a valid versioned conversation
    envelope.  The state metadata and checksum are checked here; validation of
    learned memory and world events remains the supervised worker's job.
    """
    limit = _state_limit(state)
    envelope = session_envelope(state, max_bytes=limit)
    raw = _encode_json(envelope, max_bytes=limit + _ENVELOPE_ALLOWANCE)
    return _atomic_write(
        Path(path), raw, overwrite=overwrite, existing_validator=state_from_envelope
    )


def load_session(path: str | os.PathLike[str]) -> ConversationSession:
    """Validate bytes/envelope/checksum before loading a bounded session."""
    envelope = read_json(path, max_bytes=MAX_JSON_BYTES)
    state = state_from_envelope(envelope)
    from .engine import ConversationSession

    session = ConversationSession.from_dict(state)
    return session

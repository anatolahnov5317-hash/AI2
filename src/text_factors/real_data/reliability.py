"""Atomic publication of bounded, immutable operational result bundles.

Readers address a committed run directory only. Staging directories are never
considered runs. This covers process termination, not a simulated power failure
or a filesystem without atomic same-directory rename semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

RUN_SCHEMA = "ai2-operational-run-v1"
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _run_path(root: Path, run_id: str) -> Path:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid run ID")
    return root / run_id


def _artifact_name(name: str) -> str:
    if (
        type(name) is not str
        or _ARTIFACT.fullmatch(name) is None
        or name == "manifest.json"
        or name in {".", ".."}
    ):
        raise ValueError("invalid artifact name")
    return name


def _limit(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("max_total_bytes must be a positive integer")
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_run(
    root: str | Path,
    run_id: str,
    artifacts: Mapping[str, bytes],
    *,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> Path:
    """Publish complete artifacts at once; equal retries return the same run.

    Competing publishers of the same run ID with differing payloads are
    rejected. Distinct run IDs can be published concurrently. The result is
    immutable; callers must not edit files inside a published run.
    """
    limit = _limit(max_total_bytes)
    directory = Path(root)
    final = _run_path(directory, run_id)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("run root must be a real directory")
    if not isinstance(artifacts, Mapping) or not artifacts or len(artifacts) > 64:
        raise ValueError("artifacts must be a nonempty bounded mapping")
    entries: dict[str, bytes] = {}
    total = 0
    for name, content in artifacts.items():
        name = _artifact_name(name)
        if type(content) is not bytes:
            raise ValueError("artifact content must be bytes")
        total += len(content)
        if total > limit:
            raise ValueError("run exceeds max_total_bytes")
        entries[name] = content

    if final.exists() or final.is_symlink():
        if read_run(directory, run_id, max_total_bytes=limit) != entries:
            raise ValueError("run ID already published with different artifacts")
        return final
    directory.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=directory))
    try:
        items = {}
        for name, content in sorted(entries.items()):
            _write_fsync(stage / name, content)
            items[name] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        _write_fsync(
            stage / "manifest.json",
            _canonical({"schema": RUN_SCHEMA, "run_id": run_id, "artifacts": items}),
        )
        _fsync_directory(stage)
        try:
            os.rename(stage, final)
        except OSError:
            if (
                not final.is_dir()
                or read_run(directory, run_id, max_total_bytes=limit) != entries
            ):
                raise
        else:
            stage = None
        _fsync_directory(directory)
    finally:
        if stage is not None:
            for item in stage.iterdir():
                item.unlink()
            stage.rmdir()
    return final


def read_run(
    root: str | Path,
    run_id: str,
    *,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, bytes]:
    """Read only committed runs, verify every size and checksum."""
    limit = _limit(max_total_bytes)
    directory = _run_path(Path(root), run_id)
    manifest_path = directory / "manifest.json"
    if (
        Path(root).is_symlink()
        or directory.is_symlink()
        or not directory.is_dir()
        or manifest_path.is_symlink()
    ):
        raise ValueError("run is absent or not a real directory")
    if not manifest_path.is_file() or manifest_path.stat().st_size > 16_384:
        raise ValueError("run manifest is missing or oversized")
    try:
        with manifest_path.open("rb") as stream:
            manifest_raw = stream.read(16_385)
        if len(manifest_raw) > 16_384:
            raise ValueError("run manifest is oversized")
        manifest = json.loads(manifest_raw.decode("utf-8", "strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid run manifest") from exc
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "run_id", "artifacts"}
        or manifest["schema"] != RUN_SCHEMA
        or manifest["run_id"] != run_id
        or type(manifest["artifacts"]) is not dict
        or not 0 < len(manifest["artifacts"]) <= 64
    ):
        raise ValueError("invalid run manifest schema")
    result = {}
    total = 0
    for raw_name, meta in manifest["artifacts"].items():
        name = _artifact_name(raw_name)
        if (
            type(meta) is not dict
            or set(meta) != {"sha256", "bytes"}
            or type(meta["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]) is None
            or type(meta["bytes"]) is not int
            or meta["bytes"] < 0
        ):
            raise ValueError("invalid artifact metadata")
        total += meta["bytes"]
        if total > limit:
            raise ValueError("run exceeds max_total_bytes")
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError("run artifact is missing or has wrong size")
        with path.open("rb") as stream:
            payload = stream.read(meta["bytes"] + 1)
        if len(payload) != meta["bytes"]:
            raise ValueError("run artifact changed during read")
        if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
            raise ValueError("run artifact checksum mismatch")
        result[name] = payload
    if set(path.name for path in directory.iterdir()) != set(result) | {
        "manifest.json"
    }:
        raise ValueError("run contains unlisted artifacts")
    return result

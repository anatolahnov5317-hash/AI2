"""Version-pinned, integrity-checked checkpoints for the bounded dialogue path.

The SQLite operational store is the *live* authority. Checkpoints retain old
versions for recovery, but loading always selects its active state version.
Revocations registered after a checkpoint cannot be undone by restoring it.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, cast

from .engine import RealDataEngine
from .persistence import DEFAULT_MAX_STATE_BYTES, load_state, save_state, state_payload
from .question_language import QuestionLanguageModel
from .raw_language import RawSemanticModel
from .storage import OperationalStore

_SCHEMA = "ai2-research-dialogue-checkpoint-v1"


class RecoveryRequiresRevision(ValueError):
    """The last committed state needs an explicit recovery or source revision.

    ``engine`` already has the latest source tombstones overlaid, but its new
    version is *not* active. A caller must checkpoint and activate a revision
    before constructing a dialogue from it.
    """

    def __init__(
        self,
        reason: str,
        engine: RealDataEngine,
        raw_model: RawSemanticModel,
        question_model: QuestionLanguageModel,
    ) -> None:
        super().__init__(f"recovery_requires_revision: {reason}")
        self.engine = engine
        self.raw_model = raw_model
        self.question_model = question_model


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("checkpoint is not canonical JSON") from exc


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _version_paths(directory: Path, version: str) -> tuple[Path, Path]:
    if type(version) is not str or not version or len(version) > 512:
        raise ValueError("invalid checkpoint state version")
    name = _digest(version.encode("utf-8"))
    return directory / f"state-{name}.json", directory / f"checkpoint-{name}.json"


def _check_reviewed_model(engine: RealDataEngine, raw_model: RawSemanticModel) -> None:
    """A reviewed interpretation must name the parser stored in this bundle."""
    if any(
        claim.reviewer_id is not None
        and claim.model_version != raw_model.model_fingerprint
        for claim in engine.claims.values()
    ):
        raise ValueError("reviewed claims and raw language model are incompatible")


def _directory(path: str | Path, *, create: bool, store: OperationalStore) -> Path:
    directory = Path(path).absolute()
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("checkpoint directory must be a real directory")
    if create:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not directory.exists():
        raise ValueError("checkpoint directory does not exist")
    if stat.S_IMODE(directory.stat().st_mode) & 0o077:
        raise ValueError("checkpoint directory must be private to its owner")
    if store.path == directory or directory in store.path.parents:
        raise ValueError("live access store must reside outside checkpoints")
    return directory


def _read_file(path: Path, *, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint component must be a regular file")
    if (path.stat().st_mode & 0o077) or not 0 < path.stat().st_size <= limit:
        raise ValueError("checkpoint component has unsafe permissions or size")
    return path.read_bytes()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = _read_file(path, limit=DEFAULT_MAX_STATE_BYTES)
        manifest = json.loads(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid research checkpoint JSON") from exc
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "fingerprint", "payload"}
        or manifest["schema"] != _SCHEMA
        or type(manifest["fingerprint"]) is not str
        or type(manifest["payload"]) is not dict
        or _digest(_canonical(manifest["payload"])) != manifest["fingerprint"]
        or _canonical(manifest) != raw
    ):
        raise ValueError("invalid or noncanonical research checkpoint")
    payload = manifest["payload"]
    if set(payload) != {
        "state_version",
        "model_version",
        "state_file",
        "state_sha256",
        "raw_model",
        "question_model",
    }:
        raise ValueError("unexpected research checkpoint payload")
    return payload


def _atomic_write(path: Path, value: bytes) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("checkpoint path must be a regular non-symlink file")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_research_checkpoint(
    directory: str | Path,
    engine: RealDataEngine,
    raw_model: RawSemanticModel,
    question_model: QuestionLanguageModel,
    *,
    model_version: str,
    store: OperationalStore,
) -> Path:
    """Write a complete candidate version before activating it in the store.

    Reusing a state version with divergent bytes is rejected. Saving does not
    change the live access authority, so a crash before ``finish_revision``
    leaves the previously committed checkpoint available.
    """
    if type(model_version) is not str or not model_version or len(model_version) > 512:
        raise ValueError("invalid checkpoint model version")
    _check_reviewed_model(engine, raw_model)
    parent = _directory(directory, create=True, store=store)
    state_path, manifest_path = _version_paths(parent, engine.state_version)
    state_bytes = _canonical(state_payload(engine, model_version=model_version))
    if len(state_bytes) > DEFAULT_MAX_STATE_BYTES:
        raise ValueError("research state exceeds byte budget")
    payload = {
        "state_version": engine.state_version,
        "model_version": model_version,
        "state_file": state_path.name,
        "state_sha256": _digest(state_bytes),
        "raw_model": raw_model.to_dict(),
        "question_model": question_model.to_dict(),
    }
    manifest_bytes = _canonical(
        {
            "schema": _SCHEMA,
            "fingerprint": _digest(_canonical(payload)),
            "payload": payload,
        }
    )
    if len(manifest_bytes) > DEFAULT_MAX_STATE_BYTES:
        raise ValueError("research checkpoint exceeds byte budget")
    if manifest_path.exists() or manifest_path.is_symlink():
        if (
            _read_file(manifest_path, limit=DEFAULT_MAX_STATE_BYTES) != manifest_bytes
            or _read_file(state_path, limit=DEFAULT_MAX_STATE_BYTES) != state_bytes
        ):
            raise ValueError("state version is already checkpointed with other content")
        return manifest_path
    save_state(state_path, engine, model_version=model_version)
    if _read_file(state_path, limit=DEFAULT_MAX_STATE_BYTES) != state_bytes:
        raise ValueError("saved engine differs from checkpointed engine")
    _atomic_write(manifest_path, manifest_bytes)
    return manifest_path


def load_research_checkpoint(
    directory: str | Path,
    *,
    expected_model_version: str,
    store: OperationalStore,
) -> tuple[RealDataEngine, RawSemanticModel, QuestionLanguageModel]:
    """Load only the store's active version, overlaying live revocations.

    Pending revisions and newly overlaid tombstones require explicit recovery
    before any restored dialogue can issue a new answer.
    """
    if (
        type(expected_model_version) is not str
        or not expected_model_version
        or len(expected_model_version) > 512
    ):
        raise ValueError("invalid expected model version")
    parent = _directory(directory, create=False, store=store)
    epoch = store.revision_epoch()
    active = store.active_state_version()
    if active is None:
        raise ValueError("active_state_version_unbound: checkpoint not activated")
    state_path, manifest_path = _version_paths(parent, active)
    payload = _read_manifest(manifest_path)
    if (
        payload["state_version"] != active
        or payload["model_version"] != expected_model_version
        or payload["state_file"] != state_path.name
        or type(payload["state_sha256"]) is not str
        or len(payload["state_sha256"]) != 64
    ):
        raise ValueError("research checkpoint does not match active version or model")
    state_bytes = _read_file(state_path, limit=DEFAULT_MAX_STATE_BYTES)
    if _digest(state_bytes) != payload["state_sha256"]:
        raise ValueError("research state file fingerprint mismatch")
    engine, contexts, model_version = load_state(state_path)
    if (
        contexts is not None
        or model_version != expected_model_version
        or engine.state_version != active
    ):
        raise ValueError("research state does not match checkpoint manifest")
    raw_model = RawSemanticModel.from_dict(payload["raw_model"])
    question_model = QuestionLanguageModel.from_dict(payload["question_model"])
    _check_reviewed_model(engine, raw_model)
    revoked = set(store.revoked_sources())
    families = set(store.revoked_source_families())
    obsolete = {
        (entry["source_id"], entry["source_version"])
        for entry in cast(list[dict[str, Any]], engine.to_dict()["obsolete_sources"])
    }
    roots = engine.evidence.to_dict()["roots"]
    new_tombstones = sorted(
        {
            (root["source_id"], root["source_version"])
            for root in roots
            if (
                (root["source_id"], root["source_version"]) in revoked
                or root["source_id"] in families
            )
            and (root["source_id"], root["source_version"]) not in obsolete
        }
    )
    for source_id, version in new_tombstones:
        engine.invalidate_source(source_id, version)
    if store.revision_epoch() != epoch or store.active_state_version() != active:
        raise RecoveryRequiresRevision(
            "active authority changed during restore", engine, raw_model, question_model
        )
    if new_tombstones or store.pending_claim_ids():
        reason = (
            "revoked sources need a new active revision"
            if new_tombstones
            else "revision is pending"
        )
        raise RecoveryRequiresRevision(reason, engine, raw_model, question_model)
    return engine, raw_model, question_model

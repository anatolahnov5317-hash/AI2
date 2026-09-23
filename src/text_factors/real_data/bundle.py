"""Atomic, version-pinned experimental model bundle and explicit v1 migration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .composition import CompositionalEncoder
from .contexts import ContextRegistry
from .engine import RealDataEngine
from .open_semantics import OpenSemanticModel
from .persistence import _canonical, load_state, state_payload
from .storage import OperationalStore

BUNDLE_SCHEMA = "ai2-real-data-bundle-v2"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _name(value: str, field: str) -> str:
    if type(value) is not str or not value or len(value) > 512:
        raise ValueError(f"invalid {field}")
    return value


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    engine: RealDataEngine
    contexts: ContextRegistry
    semantics: OpenSemanticModel
    encoder: CompositionalEncoder
    model_version: str
    corpus_version: str
    policy_version: str


def _body(
    engine: RealDataEngine,
    contexts: ContextRegistry,
    semantics: OpenSemanticModel,
    encoder: CompositionalEncoder,
    *,
    model_version: str,
    corpus_version: str,
    policy_version: str,
) -> dict[str, Any]:
    _name(model_version, "model_version")
    _name(corpus_version, "corpus_version")
    _name(policy_version, "policy_version")
    if contexts.width != encoder.width:
        raise ValueError("encoder/context width mismatch")
    if any(
        claim.model_version is not None and claim.model_version != model_version
        for claim in engine.claims.values()
    ):
        raise ValueError("claims refer to another model version")
    state = state_payload(engine, contexts=contexts, model_version=model_version)
    semantic = semantics.to_dict()
    encoder_config = {
        "width": encoder.width,
        "active_bits_per_atom": encoder.active_bits_per_atom,
        "seed": encoder.seed,
    }
    return {
        "model_version": model_version,
        "corpus_version": corpus_version,
        "policy_version": policy_version,
        "state_version": engine.state_version,
        "state": state,
        "semantic": semantic,
        "encoder": encoder_config,
    }


def save_bundle(
    path: str | Path,
    engine: RealDataEngine,
    *,
    contexts: ContextRegistry,
    semantics: OpenSemanticModel,
    encoder: CompositionalEncoder,
    model_version: str,
    corpus_version: str,
    policy_version: str,
    max_bytes: int = MAX_BUNDLE_BYTES,
) -> Path:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = Path(path)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("bundle path must be a regular non-symlink file")
    body = _body(
        engine,
        contexts,
        semantics,
        encoder,
        model_version=model_version,
        corpus_version=corpus_version,
        policy_version=policy_version,
    )
    envelope = {
        "schema": BUNDLE_SCHEMA,
        "components": {
            key: _digest(body[key]) for key in ("state", "semantic", "encoder")
        },
        "fingerprint": _digest(body),
        "body": body,
    }
    payload = _canonical(envelope)
    if len(payload) > max_bytes:
        raise ValueError("bundle exceeds max_bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def load_bundle(
    path: str | Path,
    *,
    expected_model_version: str,
    expected_corpus_version: str,
    expected_policy_version: str,
    authority: OperationalStore,
    max_bytes: int = MAX_BUNDLE_BYTES,
) -> LoadedBundle:
    """Restore a pinned bundle and overlay irreversible access tombstones."""
    if type(authority) is not OperationalStore:
        raise ValueError("a separate live access authority is required")
    for name, value in (
        ("model", expected_model_version),
        ("corpus", expected_corpus_version),
        ("policy", expected_policy_version),
    ):
        _name(value, f"expected_{name}_version")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or not 0 < source.stat().st_size <= max_bytes
    ):
        raise ValueError("missing or invalid bundle file")
    try:
        value = json.loads(source.read_bytes().decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid bundle JSON") from exc
    if (
        type(value) is not dict
        or set(value) != {"schema", "components", "fingerprint", "body"}
        or value["schema"] != BUNDLE_SCHEMA
        or type(value["body"]) is not dict
        or type(value["components"]) is not dict
        or set(value["components"]) != {"state", "semantic", "encoder"}
        or type(value["fingerprint"]) is not str
    ):
        raise ValueError("unsupported bundle schema")
    body = value["body"]
    if set(body) != {
        "model_version",
        "corpus_version",
        "policy_version",
        "state_version",
        "state",
        "semantic",
        "encoder",
    }:
        raise ValueError("invalid bundle components")
    if _digest(body) != value["fingerprint"] or any(
        type(value["components"][name]) is not str
        or _digest(body[name]) != value["components"][name]
        for name in ("state", "semantic", "encoder")
    ):
        raise ValueError("bundle component digest mismatch")
    if (
        body["model_version"] != expected_model_version
        or body["corpus_version"] != expected_corpus_version
        or body["policy_version"] != expected_policy_version
    ):
        raise ValueError("incompatible model/corpus/policy versions")
    raw_state = body["state"]
    if type(raw_state) is not dict or set(raw_state) != {
        "schema",
        "fingerprint",
        "payload",
    }:
        raise ValueError("invalid inner state")
    from .persistence import STATE_SCHEMA

    if (
        raw_state["schema"] != STATE_SCHEMA
        or _digest(raw_state["payload"]) != raw_state["fingerprint"]
    ):
        raise ValueError("inner state digest mismatch")
    state = raw_state["payload"]
    if (
        type(state) is not dict
        or set(state) != {"model_version", "engine", "contexts"}
        or state["model_version"] != expected_model_version
    ):
        raise ValueError("inner model version mismatch")
    if type(state["engine"]) is not dict or type(state["contexts"]) is not dict:
        raise ValueError("bundle needs engine and contexts")
    engine = RealDataEngine.from_dict(state["engine"])
    contexts = ContextRegistry.from_dict(state["contexts"])
    semantics = OpenSemanticModel.from_dict(body["semantic"])
    config = body["encoder"]
    if type(config) is not dict or set(config) != {
        "width",
        "active_bits_per_atom",
        "seed",
    }:
        raise ValueError("invalid encoder configuration")
    encoder = CompositionalEncoder(**config)
    if engine.state_version != body["state_version"] or contexts.width != encoder.width:
        raise ValueError("incompatible engine/context/encoder bundle")
    if any(
        claim.model_version is not None
        and claim.model_version != expected_model_version
        for claim in engine.claims.values()
    ):
        raise ValueError("persisted claim model version mismatch")
    roots = (
        engine.evidence.root(item["root_id"])
        for item in engine.evidence.to_dict()["roots"]
    )
    present = {(root.source_id, root.source_version) for root in roots}
    family_revocations = set(authority.revoked_source_families())
    # A source family tombstone covers every current and future version, while
    # absent old revisions have nothing to invalidate in this bundle.
    invalid = (
        set(authority.revoked_sources())
        | {
            (source_id, version)
            for source_id, version in present
            if source_id in family_revocations
        }
    ) & present
    for source_id, version in sorted(invalid):
        engine.invalidate_source(source_id, version)
    return LoadedBundle(
        engine,
        contexts,
        semantics,
        encoder,
        expected_model_version,
        expected_corpus_version,
        expected_policy_version,
    )


def migrate_v1_state(
    legacy_path: str | Path,
    destination: str | Path,
    *,
    semantics: OpenSemanticModel,
    encoder: CompositionalEncoder,
    expected_model_version: str,
    corpus_version: str,
    policy_version: str,
) -> Path:
    """Explicit additive migration: v1 state needs matching learned artifacts."""
    engine, contexts, version = load_state(legacy_path)
    if version != expected_model_version or contexts is None:
        raise ValueError("legacy state/model/contexts mismatch")
    return save_bundle(
        destination,
        engine,
        contexts=contexts,
        semantics=semantics,
        encoder=encoder,
        model_version=version,
        corpus_version=corpus_version,
        policy_version=policy_version,
    )

"""Reproducible train/calibrate/evaluate stages and read-only proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..conversation.persistence import _atomic_write, encode_json, read_json
from .archive import ObservationArchive
from .assessment import calibrate_model, evaluate_model, propose, validate_policy
from .learning import CandidateModel, LearningConfig, train_model
from .learning_data import (
    MAX_CORPUS_BYTES,
    archive_learning_corpus,
    corpus_summary,
    fingerprint,
    load_learning_corpus,
    split_documents,
    validate_learning_corpus,
)

BUNDLE_SCHEMA = "ai2-open-mention-model-bundle-v1"
STAGE_SCHEMA = "ai2-open-mention-trained-stage-v1"
PROGRESS_SCHEMA = "ai2-open-mention-progress-v1"
REPORT_SCHEMA = "ai2-open-mention-evaluation-v1"


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in (
        "learning.py",
        "assessment.py",
        "learning_data.py",
        "learning_commands.py",
    ):
        digest.update(name.encode() + b"\x00")
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _print(value: Any) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )


def write_artifact(
    path: Path, value: dict[str, Any], *, overwrite: bool = False
) -> None:
    def existing_validator(previous: Any) -> None:
        if type(previous) is not dict or previous.get("schema") != value["schema"]:
            raise ValueError("refusing to replace an unrelated file")

    _atomic_write(
        path,
        encode_json(value, max_bytes=MAX_CORPUS_BYTES),
        overwrite=overwrite,
        existing_validator=existing_validator,
    )


def _envelope(schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema": schema, "fingerprint": fingerprint(payload), "payload": payload}


def _load_envelope(path: Path, schema: str) -> dict[str, Any]:
    data = read_json(path, max_bytes=MAX_CORPUS_BYTES)
    if (
        set(data) != {"schema", "fingerprint", "payload"}
        or data.get("schema") != schema
        or type(data.get("payload")) is not dict
    ):
        raise ValueError("unexpected checkpoint schema")
    if fingerprint(data["payload"]) != data.get("fingerprint"):
        raise ValueError("checkpoint fingerprint mismatch")
    return data["payload"]


def load_bundle(path: Path) -> tuple[CandidateModel, dict[str, Any]]:
    payload = _load_envelope(path, BUNDLE_SCHEMA)
    if set(payload) != {"model", "policy", "provenance"}:
        raise ValueError("invalid model bundle fields")
    model = CandidateModel.from_dict(payload["model"])
    if (
        type(payload.get("policy")) is not dict
        or type(payload.get("provenance")) is not dict
    ):
        raise ValueError("checkpoint requires frozen policy and provenance")
    validate_policy(model, payload["policy"])
    for key in ("languages", "calibration_languages"):
        languages = payload["provenance"].get(key)
        if (
            type(languages) is not list
            or not languages
            or any(type(language) is not str or not language for language in languages)
        ):
            raise ValueError("invalid model language provenance")
    if type(payload["provenance"].get("spec")) is not dict:
        raise ValueError("model bundle requires its experiment specification")
    return model, payload


def train_stages(
    corpus: dict[str, Any],
    output_dir: Path,
    config: LearningConfig,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Commit the trained stage before calibration; never fit on held-out test."""
    validate_learning_corpus(corpus)
    train = split_documents(corpus, "train")
    validation = split_documents(corpus, "validation")
    if not train or not validation:
        raise ValueError("training and separate validation documents are required")
    spec = {
        "implementation_fingerprint": implementation_fingerprint(),
        "corpus_fingerprint": fingerprint(corpus),
        "train_fingerprint": fingerprint(train),
        "validation_fingerprint": fingerprint(validation),
        "config": asdict(config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_path, bundle_path = output_dir / "trained.json", output_dir / "model.json"
    progress_path = output_dir / "progress.json"
    if not resume and any(
        path.exists() for path in (stage_path, bundle_path, progress_path)
    ):
        raise ValueError(
            "experiment outputs exist; use --resume for the same specification"
        )
    if resume and progress_path.exists():
        previous = read_json(progress_path, max_bytes=MAX_CORPUS_BYTES)
        if previous.get("schema") != PROGRESS_SCHEMA or previous.get("spec") != spec:
            raise ValueError("resume specification differs from prior experiment")

    def progress(event: dict[str, Any]) -> None:
        value = {"schema": PROGRESS_SCHEMA, "spec": spec, "time": time.time(), **event}
        write_artifact(progress_path, value, overwrite=progress_path.exists())
        _print(event)

    if resume and bundle_path.exists():
        _, payload = load_bundle(bundle_path)
        if payload["provenance"]["spec"] != spec:
            raise ValueError("completed model belongs to another experiment")
        progress(
            {"phase": "completed", "resumed": True, "model_path": str(bundle_path)}
        )
        return payload
    started = time.monotonic()
    if resume and stage_path.exists():
        trained = _load_envelope(stage_path, STAGE_SCHEMA)
        if trained["spec"] != spec:
            raise ValueError("trained checkpoint belongs to another experiment")
        model = CandidateModel.from_dict(trained["model"])
        progress({"phase": "trained", "resumed": True})
    else:
        progress({"phase": "training", "documents": len(train)})
        model = train_model(train, config, progress=progress)
        write_artifact(
            stage_path,
            _envelope(STAGE_SCHEMA, {"spec": spec, "model": model.to_dict()}),
        )
        progress({"phase": "trained", "training_summary": model.training_summary})
    progress({"phase": "calibrating", "documents": len(validation)})
    policy = calibrate_model(model, validation, progress=progress)
    payload = {
        "model": model.to_dict(),
        "policy": policy,
        "provenance": {
            "spec": spec,
            "languages": sorted({doc["language"] for doc in train}),
            "calibration_languages": sorted({doc["language"] for doc in validation}),
            "corpus_origin": corpus["provenance"],
            "method": "supervised hashed-feature candidate baseline",
            "test_used_for_fitting": False,
            "external_language_model_used": False,
            "seconds_this_invocation": time.monotonic() - started,
        },
    }
    write_artifact(bundle_path, _envelope(BUNDLE_SCHEMA, payload))
    progress({"phase": "completed", "model_path": str(bundle_path)})
    return payload


def evaluate_bundle(bundle_path: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    validate_learning_corpus(corpus)
    model, payload = load_bundle(bundle_path)
    if payload["provenance"]["spec"]["corpus_fingerprint"] != fingerprint(corpus):
        raise ValueError("evaluation corpus differs from the frozen experiment")
    test = split_documents(corpus, "test")
    if not test:
        raise ValueError("held-out test documents are required")
    started = time.monotonic()
    metrics = evaluate_model(
        model,
        test,
        payload["policy"],
        split_documents(corpus, "train"),
        progress=_print,
    )
    return {
        "schema": REPORT_SCHEMA,
        "model_fingerprint": fingerprint(payload["model"]),
        "policy_fingerprint": fingerprint(payload["policy"]),
        "corpus": corpus_summary(corpus),
        "training_summary": model.training_summary,
        "policy": payload["policy"],
        "metrics": metrics,
        "evaluation_seconds": time.monotonic() - started,
        "production_ready": False,
        "evaluated_languages": sorted({doc["language"] for doc in test}),
    }


def propose_bundle(bundle_path: Path, text: str, *, language: str) -> dict[str, Any]:
    model, payload = load_bundle(bundle_path)
    in_training = language in payload["provenance"]["languages"]
    in_calibration = language in payload["provenance"]["calibration_languages"]
    in_domain = in_training and in_calibration
    result = propose(model, text, payload["policy"], allow_selection=in_domain)
    return {
        **result,
        "schema": "ai2-open-mention-proposals-v1",
        "model_fingerprint": fingerprint(payload["model"]),
        "text_fingerprint": fingerprint(text),
        "language": language,
        "language_in_training": in_training,
        "language_in_calibration": in_calibration,
        "semantic_status": "unconfirmed_model_proposals",
        "archive_mutated": False,
    }


def run(args: argparse.Namespace) -> int:
    if args.learning_operation == "train":
        corpus = load_learning_corpus(args.corpus)
        config = LearningConfig(
            seed=args.seed,
            epochs=args.epochs,
            feature_dim=args.feature_dim,
            max_span_tokens=args.max_span_tokens,
            max_antecedents=args.max_antecedents,
            negative_ratio=args.negative_ratio,
        )
        train_stages(corpus, Path(args.output_dir), config, resume=args.resume)
    elif args.learning_operation == "evaluate":
        result = evaluate_bundle(Path(args.model), load_learning_corpus(args.corpus))
        write_artifact(Path(args.output), result, overwrite=args.overwrite)
        _print(
            {"status": "evaluated", "output": args.output, "metrics": result["metrics"]}
        )
    elif args.learning_operation == "propose":
        with Path(args.input).open("rb") as handle:
            raw = handle.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("input exceeds the 1 MiB proposal request budget")
        result = propose_bundle(
            Path(args.model),
            raw.decode("utf-8", errors="strict"),
            language=args.language,
        )
        if args.output:
            write_artifact(Path(args.output), result, overwrite=args.overwrite)
        else:
            _print(result)
    elif args.learning_operation == "import-corpus":
        corpus = load_learning_corpus(args.corpus)
        with ObservationArchive(args.archive, create=True) as archive:
            result = archive_learning_corpus(archive, corpus, namespace=args.namespace)
        write_artifact(Path(args.output), result, overwrite=args.overwrite)
        _print(
            {
                "status": "archived",
                "sources": len(result["payload"]["members"]),
                "output": args.output,
            }
        )
    return 0


def add_parsers(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "mention-learning", help="train and evaluate open mention/link proposals"
    )
    operations = parser.add_subparsers(dest="learning_operation", required=True)
    for name in ("train", "evaluate", "propose", "import-corpus"):
        command = operations.add_parser(name)
        command.set_defaults(handler=run)
        if name in {"train", "evaluate", "import-corpus"}:
            command.add_argument("--corpus", required=True)
        if name in {"evaluate", "propose"}:
            command.add_argument("--model", required=True)
        if name == "train":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--seed", type=int, default=17)
            command.add_argument("--epochs", type=int, default=6)
            command.add_argument("--feature-dim", type=int, default=8192)
            command.add_argument("--max-span-tokens", type=int, default=8)
            command.add_argument("--max-antecedents", type=int, default=64)
            command.add_argument("--negative-ratio", type=int, default=3)
            command.add_argument("--resume", action="store_true")
        else:
            command.add_argument("--output", required=name != "propose")
            command.add_argument("--overwrite", action="store_true")
        if name == "propose":
            command.add_argument("--input", required=True)
            command.add_argument("--language", required=True)
        if name == "import-corpus":
            command.add_argument("--archive", required=True)
            command.add_argument("--namespace", required=True)

"""Bounded, read-only P03 diagnostics on train/validation, never sealed test.

Each selected long document is timed in a separate worker. Its process peak RSS
includes Python/NumPy and the model; it is not an incremental model allocation.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from text_factors.conversation.persistence import read_json
from text_factors.observations.diagnostics import (
    NGramSpanControl,
    calibrate_ngram_control,
    diagnose_mentions,
    score_ngram_control,
)
from text_factors.observations.learning_commands import load_bundle
from text_factors.observations.learning_data import MAX_CORPUS_BYTES, fingerprint


def _worker(model_path: Path) -> int:
    # No labels are sent into the worker; only a single public input text.
    payload = json.load(sys.stdin)
    if set(payload) != {"text"} or type(payload["text"]) is not str:
        raise ValueError("worker expects one text without annotations")
    model, _ = load_bundle(model_path)
    started = perf_counter()
    spans = model.span_scores(payload["text"])
    seconds = perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss //= 1024
    print(
        json.dumps(
            {
                "candidate_count": len(spans),
                "scoring_seconds": seconds,
                "peak_worker_rss_kib": peak_rss,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _measure_long_documents(
    model_path: Path,
    documents: list[dict[str, Any]],
    *,
    maximum: int,
    timeout: float,
) -> dict[str, Any]:
    if type(maximum) is not int or maximum < 1 or maximum > 128:
        raise ValueError("max-documents must be in [1, 128]")
    if not 0 < timeout <= 300:
        raise ValueError("timeout must be in (0, 300]")
    chosen = sorted(documents, key=lambda doc: (-len(doc["text"]), doc["document_id"]))[
        :maximum
    ]
    rows = []
    for index, doc in enumerate(chosen, 1):
        print(
            f"P03 long document {index}/{len(chosen)}: {doc['document_id']}",
            file=sys.stderr,
            flush=True,
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            str(model_path),
        ]
        try:
            process = subprocess.run(
                cmd,
                input=json.dumps({"text": doc["text"]}, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "document_id": doc["document_id"],
                    "characters": len(doc["text"]),
                    "status": "timed_out",
                    "timeout_seconds": timeout,
                }
            )
            continue
        if process.returncode:
            rows.append(
                {
                    "document_id": doc["document_id"],
                    "characters": len(doc["text"]),
                    "status": "failed",
                    "reason": process.stderr[-800:],
                }
            )
            continue
        measured = json.loads(process.stdout)
        rows.append(
            {
                "document_id": doc["document_id"],
                "characters": len(doc["text"]),
                "status": "completed",
                **measured,
            }
        )
    return {
        "selection": "longest_public_validation_documents_by_character_count",
        "available_documents": len(documents),
        "requested_documents": len(chosen),
        "completed_documents": sum(row["status"] == "completed" for row in rows),
        "measurements": rows,
        "peak_rss_includes_process_and_model_loading": True,
        "scoring_seconds_excludes_model_loading": True,
    }


def diagnose(
    public_data_path: Path,
    model_path: Path,
    *,
    max_documents: int = 3,
    timeout: float = 60,
) -> dict[str, Any]:
    public = read_json(public_data_path, max_bytes=MAX_CORPUS_BYTES)
    if (
        type(public) is not dict
        or set(public) != {"schema", "train", "validation"}
        or public["schema"] != "ai2-p03-public-splits-v1"
    ):
        raise ValueError("expected only train and validation in a public-only file")
    train, validation = public["train"], public["validation"]
    if (
        type(train) is not list
        or type(validation) is not list
        or not train
        or not validation
    ):
        raise ValueError("train and validation must be nonempty lists")
    if any(doc.get("split") != "train" for doc in train) or any(
        doc.get("split") != "validation" for doc in validation
    ):
        raise ValueError("public file contains a document outside train/validation")
    model, bundle = load_bundle(model_path)
    spec = bundle["provenance"]["spec"]
    if spec.get("train_fingerprint") != fingerprint(train) or spec.get(
        "validation_fingerprint"
    ) != fingerprint(validation):
        raise ValueError("public splits differ from frozen model train/validation")
    learned = diagnose_mentions(
        model, validation, threshold=bundle["policy"]["mention_threshold"]
    )
    control = NGramSpanControl(model.config)
    control.fit(train)
    calibrated = calibrate_ngram_control(control, validation)
    comparison = score_ngram_control(control, validation, calibrated)
    return {
        "schema": "ai2-open-mention-p03-diagnostics-v1",
        "splits_read": ["train", "validation"],
        "test_split_accessed": False,
        "train_sha256": fingerprint(train),
        "validation_sha256": fingerprint(validation),
        "model_sha256": fingerprint(bundle["model"]),
        "learned_validation": learned,
        "ngram_span_adapter": {
            **calibrated,
            **comparison,
            "validation_reused_to_select_threshold": True,
            "independent_holdout_comparison": False,
        },
        "surface_baseline": (
            "original assessment.evaluate_model baseline: exact train surfaces, "
            "not n-gram detection"
        ),
        "long_documents": _measure_long_documents(
            model_path, validation, maximum=max_documents, timeout=timeout
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--public-data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-documents", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.model)
    if args.public_data is None or args.output is None:
        parser.error("--public-data and --output are required")
    report = diagnose(
        args.public_data,
        args.model,
        max_documents=args.max_documents,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing diagnostic report")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"P03 diagnostics saved: {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

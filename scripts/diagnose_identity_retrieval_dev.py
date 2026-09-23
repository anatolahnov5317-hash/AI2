"""Reproducible P04 retrieval ablation on public train/validation only.

Usage:
  python scripts/diagnose_identity_retrieval_dev.py --public-data public.json \
    --baseline baseline.json --output p04-validation.json

The public JSON must contain exactly schema/train/validation. Optional baseline
JSON can enforce the fingerprint of an earlier deterministic run. Otherwise this
script calibrates the recency baseline anew using *only* public validation.
All three strategies share the same trained model and mention threshold. The
90% / 10-decision gate is checked for each strategy independently; this tool
does not edit the policy or activate a candidate link in production.
"""

from __future__ import annotations

import argparse
import json
import resource
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

from text_factors.observations.antecedent_retrieval import BoundedSurfaceRetrieval
from text_factors.observations.assessment import (
    LINK_ENDPOINT_THRESHOLDS,
    LINK_MARGINS,
    LINK_THRESHOLDS,
    MIN_LINK_DECISIONS,
    MIN_LINK_PRECISION,
    _accepted_counts,
    _accepted_metrics,
    _apply_gate,
    _digest,
    _gold,
    _model_digest,
    _ranked,
    _spans,
    calibrate_model,
)
from text_factors.observations.identity_diagnostics import _counts
from text_factors.observations.learning import LearningConfig, train_model


def _public_documents(value: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    if (
        type(value) is not dict
        or set(value) != {"schema", "train", "validation"}
        or value["schema"] != "ai2-p03-public-splits-v1"
    ):
        raise ValueError("input must contain only train and validation")
    train, validation = value["train"], value["validation"]
    if (
        type(train) is not list
        or type(validation) is not list
        or not train
        or not validation
    ):
        raise ValueError("both public splits must be nonempty")
    if any(d.get("split") != "train" for d in train) or any(
        d.get("split") != "validation" for d in validation
    ):
        raise ValueError("sealed/test documents are forbidden in a P04 ablation")
    for field in ("document_id", "group_id"):
        if {d[field] for d in train} & {d[field] for d in validation}:
            raise ValueError(f"public splits overlap by {field}")
    return train, validation


def _strategies(model: Any, validation: list[dict], threshold: float) -> dict:
    result: dict[str, Any] = {}
    for strategy in ("recency", "exact", "prefix4"):
        print(f"P04 public validation: {strategy}", flush=True)
        strategy_started = monotonic()
        ranked: list[tuple[dict, dict, list, list]] = []
        document_timings = []
        counts = Counter()
        for document in validation:
            document_started = monotonic()
            text, gold = document["text"], _gold(document)
            span_rows = [s for s in _spans(model, text) if s["score"] >= threshold]
            gold_rows = [
                {"start": begin, "end": end, "score": 1.0}
                for begin, end in sorted(gold)
            ]
            pair_cache: dict[tuple[int, int, int, int], float] = {}
            if strategy == "recency":
                predicted = _ranked(model, text, span_rows, pair_cache)
                oracle = _ranked(model, text, gold_rows, pair_cache)
            else:
                prefix = strategy == "prefix4"
                predicted = _ranked(
                    model,
                    text,
                    span_rows,
                    pair_cache,
                    antecedent_selector=BoundedSurfaceRetrieval(
                        text, span_rows, prefix_forms=prefix
                    ),
                )
                oracle = _ranked(
                    model,
                    text,
                    gold_rows,
                    pair_cache,
                    antecedent_selector=BoundedSurfaceRetrieval(
                        text, gold_rows, prefix_forms=prefix
                    ),
                )
            document_timings.append(
                {
                    "document_id": document["document_id"],
                    "characters": len(text),
                    "predicted_mentions": len(span_rows),
                    "scoring_seconds": round(monotonic() - document_started, 3),
                    "process_peak_rss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                }
            )
            ranked.append((document, gold, predicted, oracle))
            counts.update(
                _counts(
                    text,
                    gold,
                    predicted,
                    oracle,
                    _apply_gate(predicted, {"enabled": False}),
                    model.config.max_antecedents,
                )
            )
        anaphoric = counts["anaphoric_known_mentions"]
        options = []
        endpoint_thresholds = tuple(
            value for value in LINK_ENDPOINT_THRESHOLDS if value >= threshold
        ) or (threshold,)
        for score_threshold in LINK_THRESHOLDS:
            for margin_threshold in LINK_MARGINS:
                for mention_score_threshold in endpoint_thresholds:
                    gate = {
                        "enabled": True,
                        "score_threshold": score_threshold,
                        "margin_threshold": margin_threshold,
                        "mention_score_threshold": mention_score_threshold,
                    }
                    accepted = Counter()
                    for _, gold, predicted, _ in ranked:
                        accepted.update(
                            _accepted_counts(_apply_gate(predicted, gate), gold)
                        )
                    options.append({**gate, **_accepted_metrics(accepted)})
        eligible = [
            row
            for row in options
            if row["evaluable_accepted_count"] >= MIN_LINK_DECISIONS
            and row["precision"] >= MIN_LINK_PRECISION
        ]
        supported = [
            row
            for row in options
            if row["evaluable_accepted_count"] >= MIN_LINK_DECISIONS
        ]
        eligible.sort(
            key=lambda row: (row["evaluable_accepted_count"], row["precision"]),
            reverse=True,
        )
        supported.sort(
            key=lambda row: (row["precision"], row["evaluable_accepted_count"]),
            reverse=True,
        )
        result[strategy] = {
            "counts": dict(counts),
            "oracle_top3_recall": counts["oracle_top3_retrieval_hits"] / anaphoric
            if anaphoric
            else None,
            "predicted_top3_recall": counts["predicted_top3_retrieval_hits"] / anaphoric
            if anaphoric
            else None,
            "gate_enabled_by_validation": bool(eligible),
            "best_eligible_gate": eligible[0] if eligible else None,
            "best_supported_precision_gate": supported[0] if supported else None,
            "elapsed_seconds": round(monotonic() - strategy_started, 2),
            "document_scoring": document_timings,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing ablation")
    public = json.loads(args.public_data.read_text(encoding="utf-8"))
    train, validation = _public_documents(public)
    previous = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline is not None
        else None
    )
    config = (
        LearningConfig(**previous["config"])
        if previous is not None
        else LearningConfig(
            seed=17,
            epochs=6,
            feature_dim=4096,
            max_span_tokens=4,
            max_antecedents=32,
        )
    )
    started = monotonic()
    model = train_model(train, config)
    training_elapsed = round(monotonic() - started, 2)
    policy = (
        previous["policy"]
        if previous is not None
        else calibrate_model(model, validation)
    )
    model_sha = _model_digest(model)
    if (
        model_sha != policy["model_sha256"]
        or _digest(validation) != policy["validation_data_sha256"]
    ):
        raise ValueError("public data/model differ from recorded baseline")
    report = {
        "schema": "ai2-p04-public-retrieval-ablation-v1",
        "purpose": "exploratory validation ablation, not independent held-out evidence",
        "splits_read": ["train", "validation"],
        "sealed_test_accessed": False,
        "train_document_count": len(train),
        "validation_document_count": len(validation),
        "train_sha256": model.training_summary["train_data_sha256"],
        "validation_sha256": policy["validation_data_sha256"],
        "model_sha256": model_sha,
        "policy_sha256": policy["policy_sha256"],
        "config": model.to_dict()["config"],
        "unchanged_requirements": {
            "minimum_empirical_precision": MIN_LINK_PRECISION,
            "minimum_evaluable_decisions": MIN_LINK_DECISIONS,
        },
        "mention_threshold": policy["mention_threshold"],
        "strategies": _strategies(model, validation, policy["mention_threshold"]),
        "training_elapsed_seconds": training_elapsed,
        "total_elapsed_seconds": round(monotonic() - started, 2),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "production_policy_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"P04 ablation saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen Block-1 quality gate for open mention and identity learning.

The gate is deliberately stricter than a regression smoke test. It freezes the
held-out document/group identities and thresholds before evaluation, then reports
where the current model fails. A failing gate is a research result, not an
exception to be hidden by changing the test split after seeing predictions.
"""

from __future__ import annotations

from typing import Any

from .assessment import evaluate_model, propose
from .learning_commands import load_bundle
from .learning_data import (
    fingerprint,
    split_documents,
    validate_learning_corpus,
)

GATE_SCHEMA = "ai2-block1-quality-gate-v1"
REPORT_SCHEMA = "ai2-block1-quality-gate-report-v1"

DEFAULT_REQUIREMENTS = {
    "max_unsupported_gold_rate": 0.10,
    "min_unseen_surface_recall": 0.25,
    "max_same_surface_false_merges": 0,
    "min_accepted_link_precision": 0.90,
    "min_accepted_link_decisions": 10,
    "min_mention_f1_delta_from_baseline": 0.0,
    "min_coreference_f1_delta_from_baseline": 0.0,
}


def _bounded_rate(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _requirements(value: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {**DEFAULT_REQUIREMENTS, **(value or {})}
    expected = set(DEFAULT_REQUIREMENTS)
    if set(merged) != expected:
        raise ValueError("quality-gate requirement fields changed")
    for name in (
        "max_unsupported_gold_rate",
        "min_unseen_surface_recall",
        "min_accepted_link_precision",
    ):
        merged[name] = _bounded_rate(merged[name], name)
    for name in (
        "min_mention_f1_delta_from_baseline",
        "min_coreference_f1_delta_from_baseline",
    ):
        raw = merged[name]
        if type(raw) not in (int, float) or not -1.0 <= float(raw) <= 1.0:
            raise ValueError(f"{name} must be in [-1, 1]")
        merged[name] = float(raw)
    for name in ("max_same_surface_false_merges", "min_accepted_link_decisions"):
        merged[name] = _nonnegative_int(merged[name], name)
    return merged


def freeze_quality_gate(
    corpus: dict[str, Any],
    *,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a test gate without opening model predictions."""

    validate_learning_corpus(corpus)
    train = split_documents(corpus, "train")
    validation = split_documents(corpus, "validation")
    test = split_documents(corpus, "test")
    if not train or not validation or not test:
        raise ValueError("quality gate requires train, validation and test splits")
    train_groups = {item["group_id"] for item in train}
    validation_groups = {item["group_id"] for item in validation}
    test_groups = {item["group_id"] for item in test}
    if train_groups & validation_groups or train_groups & test_groups:
        raise ValueError("quality gate has cross-split group leakage")
    if validation_groups & test_groups:
        raise ValueError("quality gate has validation/test group leakage")
    payload = {
        "schema": GATE_SCHEMA,
        "corpus_fingerprint": fingerprint(corpus),
        "train_fingerprint": fingerprint(train),
        "validation_fingerprint": fingerprint(validation),
        "test_fingerprint": fingerprint(test),
        "test_documents": [
            {
                "document_id": document["document_id"],
                "group_id": document["group_id"],
                "language": document["language"],
                "text_fingerprint": fingerprint(document["text"]),
                "mentions_fingerprint": fingerprint(document["mentions"]),
            }
            for document in test
        ],
        "requirements": _requirements(requirements),
        "test_predictions_seen_before_freeze": False,
    }
    return {**payload, "gate_fingerprint": fingerprint(payload)}


def validate_quality_gate(gate: dict[str, Any], corpus: dict[str, Any]) -> None:
    if type(gate) is not dict or gate.get("schema") != GATE_SCHEMA:
        raise ValueError("unsupported Block-1 quality gate")
    expected_fingerprint = fingerprint(
        {key: value for key, value in gate.items() if key != "gate_fingerprint"}
    )
    if gate.get("gate_fingerprint") != expected_fingerprint:
        raise ValueError("quality gate changed after freeze")
    validate_learning_corpus(corpus)
    if gate.get("corpus_fingerprint") != fingerprint(corpus):
        raise ValueError("quality gate corpus differs from frozen corpus")
    test = split_documents(corpus, "test")
    if gate.get("test_fingerprint") != fingerprint(test):
        raise ValueError("quality gate test split differs from frozen split")
    _requirements(gate.get("requirements"))


def _surface(document: dict[str, Any], mention: dict[str, Any]) -> str:
    return document["text"][mention["start"] : mention["end"]].casefold()


def _training_surfaces(train: list[dict[str, Any]]) -> set[str]:
    return {
        _surface(document, mention)
        for document in train
        for mention in document["mentions"]
    }


def _predicted_spans(model: Any, text: str, threshold: float) -> set[tuple[int, int]]:
    return {
        (item["start"], item["end"])
        for item in model.span_scores(text)
        if float(item["score"]) >= threshold
    }


def _unseen_surface_metrics(
    model: Any,
    policy: dict[str, Any],
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    known = _training_surfaces(train)
    gold_count = 0
    found = 0
    unsupported = 0
    for document in test:
        predicted = _predicted_spans(
            model,
            document["text"],
            float(policy["mention_threshold"]),
        )
        candidates = {
            (item["start"], item["end"]) for item in model.span_scores(document["text"])
        }
        for mention in document["mentions"]:
            if _surface(document, mention) in known:
                continue
            gold_count += 1
            span = (mention["start"], mention["end"])
            found += span in predicted
            unsupported += span not in candidates
    return {
        "gold_count": gold_count,
        "recovered_count": found,
        "unsupported_count": unsupported,
        "recall": found / gold_count if gold_count else None,
    }


def _same_surface_false_merges(
    model: Any,
    policy: dict[str, Any],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    false_merges = 0
    evaluable_links = 0
    risky_gold_mentions = 0
    risky_surfaces = 0
    for document in test:
        gold_by_span = {
            (mention["start"], mention["end"]): mention.get("entity_id")
            for mention in document["mentions"]
        }
        surface_entities: dict[str, set[str]] = {}
        for mention in document["mentions"]:
            entity = mention.get("entity_id")
            if entity is None:
                continue
            surface_entities.setdefault(_surface(document, mention), set()).add(entity)
        risky = {
            surface
            for surface, entities in surface_entities.items()
            if len(entities) > 1
        }
        risky_surfaces += len(risky)
        risky_gold_mentions += sum(
            _surface(document, mention) in risky for mention in document["mentions"]
        )
        rows = propose(model, document["text"], policy)["mentions"]
        by_id = {row["mention_id"]: row for row in rows}
        for row in rows:
            selected = row["selected"]
            if selected is None:
                continue
            left = by_id[selected]
            left_span = (left["start"], left["end"])
            right_span = (row["start"], row["end"])
            left_entity = gold_by_span.get(left_span)
            right_entity = gold_by_span.get(right_span)
            if left_entity is None or right_entity is None:
                continue
            left_surface = document["text"][slice(*left_span)].casefold()
            right_surface = document["text"][slice(*right_span)].casefold()
            if left_surface != right_surface or left_surface not in risky:
                continue
            evaluable_links += 1
            false_merges += left_entity != right_entity
    return {
        "risky_surface_count": risky_surfaces,
        "risky_gold_mention_count": risky_gold_mentions,
        "evaluable_selected_links": evaluable_links,
        "false_merge_count": false_merges,
    }


def _failure_reasons(
    metrics: dict[str, Any],
    unseen: dict[str, Any],
    same_surface: dict[str, Any],
    requirements: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    mentions = metrics["mentions"]
    unsupported_rate = (
        mentions["unsupported_gold_spans"] / mentions["gold_count"]
        if mentions["gold_count"]
        else 0.0
    )
    if unsupported_rate > requirements["max_unsupported_gold_rate"]:
        failures.append("unsupported_gold_rate")
    unseen_recall = unseen["recall"]
    if (
        unseen_recall is not None
        and unseen_recall < requirements["min_unseen_surface_recall"]
    ):
        failures.append("unseen_surface_recall")
    if (
        same_surface["false_merge_count"]
        > requirements["max_same_surface_false_merges"]
    ):
        failures.append("same_surface_false_merges")
    accepted = metrics["accepted_links"]
    evaluable = accepted["evaluable_accepted_count"]
    precision = accepted["precision"]
    if evaluable < requirements["min_accepted_link_decisions"]:
        failures.append("accepted_link_decision_count")
    elif precision is None or precision < requirements["min_accepted_link_precision"]:
        failures.append("accepted_link_precision")
    comparison = metrics["comparison"]
    if (
        comparison["mention_f1_delta_from_baseline"]
        < requirements["min_mention_f1_delta_from_baseline"]
    ):
        failures.append("mention_below_required_baseline_delta")
    if (
        comparison["coreference_f1_delta_from_baseline"]
        < requirements["min_coreference_f1_delta_from_baseline"]
    ):
        failures.append("coreference_below_required_baseline_delta")
    return failures


def evaluate_quality_gate(
    bundle_path: Any,
    corpus: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one frozen model/policy against a pre-frozen Block-1 gate."""

    validate_quality_gate(gate, corpus)
    model, payload = load_bundle(bundle_path)
    train = split_documents(corpus, "train")
    test = split_documents(corpus, "test")
    metrics = evaluate_model(model, test, payload["policy"], train)
    unseen = _unseen_surface_metrics(model, payload["policy"], train, test)
    same_surface = _same_surface_false_merges(model, payload["policy"], test)
    requirements = _requirements(gate["requirements"])
    failures = _failure_reasons(metrics, unseen, same_surface, requirements)
    mention_gold = metrics["mentions"]["gold_count"]
    unsupported_rate = (
        metrics["mentions"]["unsupported_gold_spans"] / mention_gold
        if mention_gold
        else 0.0
    )
    return {
        "schema": REPORT_SCHEMA,
        "gate_fingerprint": gate["gate_fingerprint"],
        "model_fingerprint": fingerprint(payload["model"]),
        "policy_fingerprint": fingerprint(payload["policy"]),
        "test_fingerprint": gate["test_fingerprint"],
        "requirements": requirements,
        "metrics": metrics,
        "slices": {
            "unseen_surfaces": unseen,
            "same_surface_multiple_entities": same_surface,
        },
        "derived": {"unsupported_gold_rate": unsupported_rate},
        "failure_reasons": failures,
        "passed": not failures,
        "thresholds_changed_on_test": False,
    }

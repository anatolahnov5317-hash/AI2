"""Held-out assessment for transient mention and antecedent proposals.

Scores are model responses, not calibrated probabilities. Validation chooses a
fixed-grid operating point. No function in this module writes to the archive.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

POLICY_SCHEMA = "ai2-open-candidate-policy-v1"
MENTION_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
LINK_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
LINK_MARGINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
LINK_ENDPOINT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MIN_LINK_DECISIONS = 10
MIN_LINK_PRECISION = 0.9


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_digest(model: Any) -> str:
    """Bind a policy to the complete serialized model, including its weights."""
    return _digest(model.to_dict())


def _documents(documents: Any, split: str) -> list[dict[str, Any]]:
    result = list(documents)
    if not result:
        raise ValueError(f"{split} documents must not be empty")
    identities = set()
    for document in result:
        if document.get("split") != split:
            raise ValueError(f"only {split} documents are permitted here")
        identity = document["document_id"]
        if identity in identities:
            raise ValueError("duplicate document_id")
        identities.add(identity)
    return result


def _disjoint(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    for key in ("document_id", "group_id"):
        if {item[key] for item in left} & {item[key] for item in right}:
            raise ValueError(f"overlap across splits: {key}")


def _training_disjoint(model: Any, documents: list[dict[str, Any]]) -> None:
    for key, summary_key in (
        ("document_id", "train_document_ids"),
        ("group_id", "train_group_ids"),
    ):
        if set(model.training_summary.get(summary_key, [])) & {
            document[key] for document in documents
        }:
            raise ValueError(f"training overlap: {key}")


def _score(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("candidate score must be finite and between zero and one")
    return result


def _spans(model: Any, text: str) -> list[dict[str, Any]]:
    spans = []
    seen = set()
    for item in model.span_scores(text):
        start, end = item["start"], item["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError("invalid proposed span")
        if (start, end) in seen:
            raise ValueError("duplicate proposed span")
        seen.add((start, end))
        spans.append({"start": start, "end": end, "score": _score(item["score"])})
    return sorted(spans, key=lambda span: (span["start"], span["end"]))


def _gold(document: dict[str, Any]) -> dict[tuple[int, int], str | None]:
    return {
        (mention["start"], mention["end"]): mention.get("entity_id")
        for mention in document["mentions"]
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_count": tp + fp,
        "gold_count": tp + fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _mention_counts(
    gold: Mapping[tuple[int, int], Any], spans: list[dict[str, Any]]
) -> tuple[int, int, int]:
    predicted = {(span["start"], span["end"]) for span in spans}
    expected = set(gold)
    return (
        len(predicted & expected),
        len(predicted - expected),
        len(expected - predicted),
    )


def _ranked(
    model: Any,
    text: str,
    spans: list[dict[str, Any]],
    pair_cache: dict[tuple[int, int, int, int], float] | None = None,
    *,
    antecedent_selector: Callable[[int, int], tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Rank pairs using learned link response and mention confidence.

    The learned pair response is cached independently from endpoint confidence.
    On oracle gold spans the endpoint score is exactly one, so oracle diagnostics
    continue to measure the pair model itself. No gold identity enters ranking.
    """
    budget = model.config.max_antecedents
    if type(budget) is not int or budget < 1:
        raise ValueError("max_antecedents must be a positive integer")
    if pair_cache is None:
        pair_cache = {}
    result = []
    for index, span in enumerate(spans):
        candidates = []
        antecedents = (
            antecedent_selector(index, budget)
            if antecedent_selector is not None
            else range(max(0, index - budget), index)
        )
        for previous in antecedents:
            antecedent = spans[previous]
            pair = (antecedent["start"], antecedent["end"], span["start"], span["end"])
            if pair not in pair_cache:
                pair_cache[pair] = _score(model.link_score(text, antecedent, span))
            endpoint_confidence = math.sqrt(
                _score(antecedent["score"]) * _score(span["score"])
            )
            candidates.append(
                {
                    "mention_id": f"m{previous:06d}",
                    "start": antecedent["start"],
                    "end": antecedent["end"],
                    "score": pair_cache[pair] * endpoint_confidence,
                    "order": previous,
                }
            )
        candidates.sort(key=lambda item: (-item["score"], -item["order"]))
        for candidate in candidates:
            del candidate["order"]
        top = candidates[0]["score"] if candidates else None
        second = candidates[1]["score"] if len(candidates) > 1 else 0.0
        result.append(
            {
                "mention_id": f"m{index:06d}",
                **span,
                "surface": text[span["start"] : span["end"]],
                "candidates": candidates[:3],
                "considered_antecedents": len(candidates),
                "margin": top - second if top is not None else None,
            }
        )
    return result


def _apply_gate(
    rows: list[dict[str, Any]], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    by_id = {row["mention_id"]: row for row in rows}
    endpoint_threshold = float(gate.get("mention_score_threshold") or 0.0)
    for row in rows:
        candidates = row["candidates"]
        selected = None
        top = candidates[0] if candidates else None
        antecedent = by_id.get(top["mention_id"]) if top is not None else None
        if (
            gate["enabled"]
            and top is not None
            and antecedent is not None
            and row["score"] >= endpoint_threshold
            and antecedent["score"] >= endpoint_threshold
            and top["score"] >= gate["score_threshold"]
            and row["margin"] >= gate["margin_threshold"]
            and row["margin"] > 0
        ):
            selected = top["mention_id"]
        result.append(
            {
                **row,
                "selected": selected,
                "status": (
                    "proposed_link"
                    if selected is not None
                    else "ambiguous"
                    if candidates
                    else "unresolved"
                ),
            }
        )
    return result


def _accepted_counts(
    rows: list[dict[str, Any]], gold: Mapping[tuple[int, int], str | None]
) -> dict[str, int]:
    counts = {
        "mention_count": len(rows),
        "candidate_bearing_mentions": sum(bool(row["candidates"]) for row in rows),
        "accepted_count": 0,
        "evaluable_accepted_count": 0,
        "correct_count": 0,
        "unknown_gold_excluded_count": 0,
        "wrong_boundary_count": 0,
    }
    by_id = {row["mention_id"]: row for row in rows}
    for row in rows:
        if row["selected"] is None:
            continue
        counts["accepted_count"] += 1
        left = by_id[row["selected"]]
        left_span = (left["start"], left["end"])
        right_span = (row["start"], row["end"])
        # An invented boundary is an error even if the other endpoint is unknown.
        if left_span not in gold or right_span not in gold:
            counts["wrong_boundary_count"] += 1
            counts["evaluable_accepted_count"] += 1
        elif gold[left_span] is None or gold[right_span] is None:
            counts["unknown_gold_excluded_count"] += 1
        else:
            counts["evaluable_accepted_count"] += 1
            counts["correct_count"] += gold[left_span] == gold[right_span]
    return counts


def _accepted_metrics(counts: Counter) -> dict[str, Any]:
    return {
        **dict(counts),
        "precision": (
            counts["correct_count"] / counts["evaluable_accepted_count"]
            if counts["evaluable_accepted_count"]
            else None
        ),
        "coverage": (
            counts["accepted_count"] / counts["candidate_bearing_mentions"]
            if counts["candidate_bearing_mentions"]
            else None
        ),
    }


def calibrate_model(
    model: Any, validation_documents: Any, *, progress: Any = None
) -> dict[str, Any]:
    """Choose fixed-grid thresholds only on validation; leave the model untouched."""
    documents = _documents(validation_documents, "validation")
    _training_disjoint(model, documents)
    model_sha = _model_digest(model)
    cache = []
    for index, document in enumerate(documents, 1):
        cache.append((document, _spans(model, document["text"])))
        if progress is not None:
            progress(
                {
                    "phase": "validation_spans",
                    "document_id": document["document_id"],
                    "completed": index,
                    "total": len(documents),
                }
            )
    mention_grid = []
    for threshold in MENTION_THRESHOLDS:
        counts = [0, 0, 0]
        for document, spans in cache:
            current = _mention_counts(
                _gold(document), [span for span in spans if span["score"] >= threshold]
            )
            counts = [left + right for left, right in zip(counts, current, strict=True)]
        mention_grid.append({"threshold": threshold, **_prf(*counts)})
    operating_point = max(
        mention_grid,
        key=lambda row: (row["f1"], row["precision"] or 0.0, row["threshold"]),
    )
    threshold = operating_point["threshold"]
    ranked_cache = []
    oracle_ranked_cache = []
    for index, (document, spans) in enumerate(cache, 1):
        gold = _gold(document)
        pair_cache: dict[tuple[int, int, int, int], float] = {}
        ranked_cache.append(
            (
                gold,
                _ranked(
                    model,
                    document["text"],
                    [span for span in spans if span["score"] >= threshold],
                    pair_cache,
                ),
            )
        )
        oracle_ranked_cache.append(
            (
                gold,
                _ranked(
                    model,
                    document["text"],
                    [
                        {"start": start, "end": end, "score": 1.0}
                        for start, end in sorted(gold)
                    ],
                    pair_cache,
                ),
            )
        )
        if progress is not None:
            progress(
                {
                    "phase": "validation_links",
                    "document_id": document["document_id"],
                    "completed": index,
                    "total": len(documents),
                }
            )
    pair_enabled = model.training_summary.get("pair_training_enabled") is True
    gate: dict[str, Any] = {
        "enabled": False,
        "score_threshold": None,
        "margin_threshold": None,
        "mention_score_threshold": None,
        "reason": "no_supported_validation_operating_point",
        "min_evaluable_decisions": MIN_LINK_DECISIONS,
        "min_empirical_precision": MIN_LINK_PRECISION,
        "statistical_guarantee": False,
    }
    link_grid = []
    oracle_link_grid = []
    if pair_enabled:
        endpoint_thresholds = tuple(
            value for value in LINK_ENDPOINT_THRESHOLDS if value >= threshold
        )
        if not endpoint_thresholds:
            endpoint_thresholds = (threshold,)
        for score_threshold in LINK_THRESHOLDS:
            for margin_threshold in LINK_MARGINS:
                oracle_trial = {
                    "enabled": True,
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                    "mention_score_threshold": 0.0,
                }
                oracle_counts = Counter()
                for gold, rows in oracle_ranked_cache:
                    oracle_counts.update(
                        _accepted_counts(_apply_gate(rows, oracle_trial), gold)
                    )
                oracle_link_grid.append(
                    {**oracle_trial, **_accepted_metrics(oracle_counts)}
                )
                for mention_score_threshold in endpoint_thresholds:
                    trial = {
                        "enabled": True,
                        "score_threshold": score_threshold,
                        "margin_threshold": margin_threshold,
                        "mention_score_threshold": mention_score_threshold,
                    }
                    counts = Counter()
                    for gold, rows in ranked_cache:
                        counts.update(_accepted_counts(_apply_gate(rows, trial), gold))
                    link_grid.append({**trial, **_accepted_metrics(counts)})
        eligible = [
            row
            for row in link_grid
            if row["evaluable_accepted_count"] >= MIN_LINK_DECISIONS
            and row["precision"] >= MIN_LINK_PRECISION
        ]
        if eligible:
            best = max(
                eligible,
                key=lambda row: (
                    row["evaluable_accepted_count"],
                    row["precision"],
                    row["mention_score_threshold"],
                    row["score_threshold"],
                    row["margin_threshold"],
                ),
            )
            gate.update(
                enabled=True,
                score_threshold=best["score_threshold"],
                margin_threshold=best["margin_threshold"],
                mention_score_threshold=best["mention_score_threshold"],
                reason="validation_empirical_operating_point",
                validation_evaluable_decisions=best["evaluable_accepted_count"],
                validation_correct_decisions=best["correct_count"],
                validation_empirical_precision=best["precision"],
            )
    else:
        gate["reason"] = "pair_training_disabled"
    policy = {
        "schema": POLICY_SCHEMA,
        "model_sha256": model_sha,
        "mention_threshold": threshold,
        "link_gate": gate,
        "validation_documents": [
            {key: document[key] for key in ("document_id", "group_id")}
            for document in documents
        ],
        "validation_data_sha256": _digest(documents),
        "mention_grid": mention_grid,
        "link_grid": link_grid,
        "oracle_link_grid": oracle_link_grid,
        "mention_tie_break": "highest_f1_then_precision_then_threshold",
        "link_tie_break": (
            "most_evaluable_decisions_then_precision_then_endpoint_and_pair_thresholds"
        ),
        "score_semantics": "pair_sigmoid_times_geometric_endpoint_confidence",
        "calibration_scope": "predicted_validation_spans",
        "archive_mutation": False,
    }
    if model_sha != _model_digest(model):
        raise ValueError("model changed during calibration")
    policy["policy_sha256"] = _digest(policy)
    return policy


def _check_policy(model: Any, policy: dict[str, Any]) -> None:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("unsupported policy schema")
    expected = _digest(
        {key: value for key, value in policy.items() if key != "policy_sha256"}
    )
    if expected != policy.get("policy_sha256"):
        raise ValueError("policy changed after calibration")
    if _model_digest(model) != policy["model_sha256"]:
        raise ValueError("model changed after calibration")
    if (
        policy["link_gate"]["enabled"]
        and model.training_summary.get("pair_training_enabled") is not True
    ):
        raise ValueError("link gate cannot be enabled without pair training")


def validate_policy(model: Any, policy: dict[str, Any]) -> None:
    """Check frozen policy and model identity without changing either artifact."""
    _check_policy(model, policy)


def _propose_rows(
    model: Any, text: str, policy: dict[str, Any], *, allow_selection: bool = True
) -> list[dict[str, Any]]:
    spans = [
        span
        for span in _spans(model, text)
        if span["score"] >= policy["mention_threshold"]
    ]
    gate = {
        **policy["link_gate"],
        "enabled": policy["link_gate"]["enabled"] and allow_selection,
    }
    return _apply_gate(_ranked(model, text, spans), gate)


def propose(
    model: Any, text: str, policy: dict[str, Any], *, allow_selection: bool = True
) -> dict[str, Any]:
    """Return transient alternatives from text alone; never read gold identity."""
    _check_policy(model, policy)
    mentions = _propose_rows(model, text, policy, allow_selection=allow_selection)
    _check_policy(model, policy)
    return {
        "schema": "ai2-open-candidate-proposals-v1",
        "mentions": mentions,
        "mention_id_scope": "this_proposal_only",
        "score_semantics": "pair_sigmoid_times_geometric_endpoint_confidence",
        "policy_sha256": policy["policy_sha256"],
        "archive_mutation": False,
        "entity_creation": False,
        "selection_allowed": allow_selection,
    }


def _choose_two(count: int) -> int:
    return count * (count - 1) // 2


def _cluster_counts(
    rows: list[dict[str, Any]], gold: Mapping[tuple[int, int], str | None]
) -> Counter:
    """Compare clusters on all known gold spans; missing mentions are singletons."""
    parents = {row["mention_id"]: row["mention_id"] for row in rows}

    def root(identity: str) -> str:
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    for row in rows:
        if row["selected"] is not None:
            parents[root(row["mention_id"])] = root(row["selected"])
    predicted = {(row["start"], row["end"]): root(row["mention_id"]) for row in rows}
    joint, expected, actual = Counter(), Counter(), Counter()
    known = 0
    for index, (span, entity) in enumerate(gold.items()):
        if entity is None:
            continue
        known += 1
        cluster = predicted.get(span, f"missing-{index}")
        joint[(entity, cluster)] += 1
        expected[entity] += 1
        actual[cluster] += 1
    true_positive = sum(_choose_two(count) for count in joint.values())
    predicted_positive = sum(_choose_two(count) for count in actual.values())
    gold_positive = sum(_choose_two(count) for count in expected.values())
    false_merge = predicted_positive - true_positive
    false_split = gold_positive - true_positive
    pairs = _choose_two(known)
    return Counter(
        true_positive=true_positive,
        false_positive=false_merge,
        false_negative=false_split,
        true_negative=pairs - true_positive - false_merge - false_split,
        false_merge_pairs=false_merge,
        false_split_pairs=false_split,
        pair_count=pairs,
        known_gold_mentions=known,
        unknown_gold_mentions_excluded=len(gold) - known,
        unknown_gold_pairs_excluded=_choose_two(len(gold)) - pairs,
        unmatched_predicted_mentions=len(set(predicted) - set(gold)),
    )


def _cluster_metrics(counts: Counter) -> dict[str, Any]:
    return {
        **dict(counts),
        **_prf(
            counts["true_positive"], counts["false_positive"], counts["false_negative"]
        ),
        "pair_universe": "all_known_gold_mentions_missing_predictions_are_singletons",
    }


def _baseline_surfaces(documents: list[dict[str, Any]]) -> set[str]:
    return {
        document["text"][mention["start"] : mention["end"]]
        for document in documents
        for mention in document["mentions"]
    }


def _surface_spans(text: str, surfaces: set[str]) -> list[dict[str, Any]]:
    found = set()
    for surface in sorted(surfaces):
        if not surface:
            continue
        start = text.find(surface)
        while start != -1:
            end = start + len(surface)
            left_ok = (
                not surface[0].isalnum() or start == 0 or not text[start - 1].isalnum()
            )
            right_ok = (
                not surface[-1].isalnum() or end == len(text) or not text[end].isalnum()
            )
            if left_ok and right_ok:
                found.add((start, end))
            start = text.find(surface, start + 1)
    return [{"start": start, "end": end, "score": 1.0} for start, end in sorted(found)]


def _baseline_rows(
    text: str, spans: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    rows = []
    for index, span in enumerate(spans):
        surface = text[span["start"] : span["end"]]
        previous = rows[max(0, index - budget) : index]
        matches = [row for row in previous if row["surface"] == surface]
        rows.append(
            {
                **span,
                "mention_id": f"m{index:06d}",
                "surface": surface,
                "candidates": [{"mention_id": row["mention_id"]} for row in previous],
                "selected": matches[-1]["mention_id"] if matches else None,
            }
        )
    return rows


def evaluate_model(
    model: Any,
    test_documents: Any,
    policy: dict[str, Any],
    train_documents: Any,
    *,
    progress: Any = None,
) -> dict[str, Any]:
    """Evaluate frozen model/policy; held-out gold is used only by metric functions."""
    _check_policy(model, policy)
    documents = _documents(test_documents, "test")
    training = _documents(train_documents, "train")
    _training_disjoint(model, documents)
    _disjoint(training, documents)
    _disjoint(policy["validation_documents"], documents)
    _disjoint(training, policy["validation_documents"])
    trained_ids = model.training_summary.get("train_document_ids")
    if trained_ids is not None and set(trained_ids) != {
        document["document_id"] for document in training
    }:
        raise ValueError("baseline training documents differ from model training set")
    surfaces = _baseline_surfaces(training)
    totals: dict[str, Counter] = {
        key: Counter()
        for key in (
            "mentions",
            "oracle",
            "end_to_end",
            "accepted",
            "oracle_accepted",
            "baseline_mentions",
            "baseline_oracle",
            "baseline_end_to_end",
            "baseline_accepted",
            "baseline_oracle_accepted",
        )
    }
    per_document = []
    for index, document in enumerate(documents, 1):
        text, gold = document["text"], _gold(document)
        candidates = _spans(model, text)
        predicted = [
            span for span in candidates if span["score"] >= policy["mention_threshold"]
        ]
        pair_cache = {}
        rows = _apply_gate(
            _ranked(model, text, predicted, pair_cache), policy["link_gate"]
        )
        oracle_spans = [
            {"start": start, "end": end, "score": 1.0} for start, end in sorted(gold)
        ]
        oracle = _apply_gate(
            _ranked(model, text, oracle_spans, pair_cache), policy["link_gate"]
        )
        baseline_spans = _surface_spans(text, surfaces)
        baseline = _baseline_rows(text, baseline_spans, model.config.max_antecedents)
        baseline_oracle = _baseline_rows(
            text, oracle_spans, model.config.max_antecedents
        )
        mention_counts = _mention_counts(gold, predicted)
        totals["mentions"].update(
            dict(zip(("tp", "fp", "fn"), mention_counts, strict=True))
        )
        totals["mentions"]["unsupported_gold_spans"] += len(
            set(gold) - {(span["start"], span["end"]) for span in candidates}
        )
        totals["baseline_mentions"].update(
            dict(
                zip(
                    ("tp", "fp", "fn"),
                    _mention_counts(gold, baseline_spans),
                    strict=True,
                )
            )
        )
        for name, current in (
            ("oracle", oracle),
            ("end_to_end", rows),
            ("baseline_oracle", baseline_oracle),
            ("baseline_end_to_end", baseline),
        ):
            totals[name].update(_cluster_counts(current, gold))
        for name, current in (
            ("accepted", rows),
            ("oracle_accepted", oracle),
            ("baseline_accepted", baseline),
            ("baseline_oracle_accepted", baseline_oracle),
        ):
            totals[name].update(_accepted_counts(current, gold))
        per_document.append(
            {"document_id": document["document_id"], "mentions": _prf(*mention_counts)}
        )
        if progress is not None:
            progress(
                {
                    "phase": "test_evaluation",
                    "document_id": document["document_id"],
                    "completed": index,
                    "total": len(documents),
                }
            )
    mention = _prf(*(totals["mentions"][key] for key in ("tp", "fp", "fn")))
    mention["unsupported_gold_spans"] = totals["mentions"]["unsupported_gold_spans"]
    baseline_mention = _prf(
        *(totals["baseline_mentions"][key] for key in ("tp", "fp", "fn"))
    )
    end_to_end = _cluster_metrics(totals["end_to_end"])
    baseline_end = _cluster_metrics(totals["baseline_end_to_end"])
    _check_policy(model, policy)
    return {
        "schema": "ai2-open-candidate-evaluation-v1",
        "split": "test",
        "document_count": len(documents),
        "test_data_sha256": _digest(documents),
        "model_sha256": policy["model_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "threshold_selection_on_test": False,
        "mentions": mention,
        "coreference_oracle_mentions": _cluster_metrics(totals["oracle"]),
        "coreference_end_to_end": end_to_end,
        "accepted_links": _accepted_metrics(totals["accepted"]),
        "accepted_links_oracle_mentions": _accepted_metrics(totals["oracle_accepted"]),
        "baseline": {
            "kind": "simple_control_not_target_architecture",
            "span_method": "exact_train_surface_with_unicode_alphanumeric_boundaries",
            "link_method": "nearest_identical_surface_with_same_antecedent_budget",
            "learned_surface_count": len(surfaces),
            "train_document_count": len(training),
            "mentions": baseline_mention,
            "coreference_oracle_mentions": _cluster_metrics(totals["baseline_oracle"]),
            "coreference_end_to_end": baseline_end,
            "accepted_links": _accepted_metrics(totals["baseline_accepted"]),
            "accepted_links_oracle_mentions": _accepted_metrics(
                totals["baseline_oracle_accepted"]
            ),
        },
        "comparison": {
            "mention_f1_delta_from_baseline": mention["f1"] - baseline_mention["f1"],
            "coreference_f1_delta_from_baseline": end_to_end["f1"] - baseline_end["f1"],
            "mention_below_baseline": mention["f1"] < baseline_mention["f1"],
            "coreference_below_baseline": end_to_end["f1"] < baseline_end["f1"],
        },
        "per_document": per_document,
    }

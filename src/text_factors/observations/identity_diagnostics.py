"""Gold-aware identity error analysis confined to train and frozen validation.

This module does not train, retune a policy, write to the archive or inspect a
test split. Gold entity IDs never enter model scoring or proposed links.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .assessment import (
    _apply_gate,
    _check_policy,
    _choose_two,
    _cluster_counts,
    _digest,
    _documents,
    _gold,
    _ranked,
    _spans,
    _training_disjoint,
)
from .schema import canonical_json

COUNT_NAMES = (
    "gold_mentions",
    "known_identity_mentions",
    "unknown_identity_mentions",
    "missing_predicted_boundaries",
    "anaphoric_known_mentions",
    "distant_only_mentions",
    "gold_antecedent_inside_window",
    "oracle_top3_retrieval_hits",
    "oracle_top3_ranking_misses",
    "predicted_top3_retrieval_hits",
    "same_surface_different_entity_pairs",
    "same_surface_false_merge_pairs",
    "different_surface_same_entity_pairs",
    "different_surface_false_split_pairs",
    "false_merge_pairs",
    "false_split_pairs",
)


def _gold_clusters(
    rows: list[dict[str, Any]], gold: Mapping[tuple[int, int], str | None]
) -> dict[tuple[int, int], str]:
    """Assign known gold spans to predicted clusters; missing spans stay apart."""
    parents = {row["mention_id"]: row["mention_id"] for row in rows}

    def root(mention_id: str) -> str:
        while parents[mention_id] != mention_id:
            parents[mention_id] = parents[parents[mention_id]]
            mention_id = parents[mention_id]
        return mention_id

    for row in rows:
        if row["selected"] is not None:
            parents[root(row["mention_id"])] = root(row["selected"])
    predicted = {(row["start"], row["end"]): root(row["mention_id"]) for row in rows}
    return {
        span: predicted.get(span, f"missing-{index}")
        for index, (span, entity) in enumerate(gold.items())
        if entity is not None
    }


def _counts(
    text: str,
    gold: Mapping[tuple[int, int], str | None],
    ranked: list[dict[str, Any]],
    oracle_ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    window: int,
) -> Counter:
    """Separate mention loss, retrieval loss, homonym merges and form splits."""
    counts = Counter({name: 0 for name in COUNT_NAMES})
    counts["gold_mentions"] = len(gold)
    ordered_gold = sorted(gold)
    gold_position = {span: index for index, span in enumerate(ordered_gold)}
    known = [(span, gold[span]) for span in ordered_gold if gold[span] is not None]
    counts["known_identity_mentions"] = len(known)
    counts["unknown_identity_mentions"] = len(gold) - len(known)
    predicted = {(row["start"], row["end"]): row for row in ranked}
    oracle = {(row["start"], row["end"]): row for row in oracle_ranked}
    counts["missing_predicted_boundaries"] = len(set(gold) - set(predicted))
    clusters = _gold_clusters(selected, gold)

    surface_count: Counter = Counter()
    surface_entity: Counter = Counter()
    surface_cluster: Counter = Counter()
    surface_cluster_entity: Counter = Counter()
    entity_count: Counter = Counter()
    entity_surface: Counter = Counter()
    entity_cluster: Counter = Counter()
    entity_cluster_surface: Counter = Counter()
    earlier_by_entity: dict[str, list[tuple[int, int]]] = {}
    for span, entity in known:
        # Null identities were removed above; no null is treated as a negative.
        assert entity is not None
        surface, cluster = text[slice(*span)], clusters[span]
        surface_count[surface] += 1
        surface_entity[(surface, entity)] += 1
        surface_cluster[(surface, cluster)] += 1
        surface_cluster_entity[(surface, cluster, entity)] += 1
        entity_count[entity] += 1
        entity_surface[(entity, surface)] += 1
        entity_cluster[(entity, cluster)] += 1
        entity_cluster_surface[(entity, cluster, surface)] += 1
        earlier = earlier_by_entity.setdefault(entity, [])
        if earlier:
            counts["anaphoric_known_mentions"] += 1
            # Index positions include unknown mentions, as the oracle window does.
            if gold_position[span] - gold_position[earlier[-1]] > window:
                counts["distant_only_mentions"] += 1
            else:
                counts["gold_antecedent_inside_window"] += 1
            oracle_candidates = {
                (item["start"], item["end"]) for item in oracle[span]["candidates"]
            }
            if any(prior in oracle_candidates for prior in earlier):
                counts["oracle_top3_retrieval_hits"] += 1
            elif gold_position[span] - gold_position[earlier[-1]] <= window:
                counts["oracle_top3_ranking_misses"] += 1
            predicted_candidates = {
                (item["start"], item["end"])
                for item in predicted.get(span, {}).get("candidates", [])
            }
            if any(prior in predicted_candidates for prior in earlier):
                counts["predicted_top3_retrieval_hits"] += 1
        earlier.append(span)

    counts["same_surface_different_entity_pairs"] = sum(
        _choose_two(n) for n in surface_count.values()
    ) - sum(_choose_two(n) for n in surface_entity.values())
    counts["same_surface_false_merge_pairs"] = sum(
        _choose_two(n) for n in surface_cluster.values()
    ) - sum(_choose_two(n) for n in surface_cluster_entity.values())
    counts["different_surface_same_entity_pairs"] = sum(
        _choose_two(n) for n in entity_count.values()
    ) - sum(_choose_two(n) for n in entity_surface.values())
    correctly_joined = sum(_choose_two(n) for n in entity_cluster.values()) - sum(
        _choose_two(n) for n in entity_cluster_surface.values()
    )
    counts["different_surface_false_split_pairs"] = (
        counts["different_surface_same_entity_pairs"] - correctly_joined
    )
    clusters_metrics = _cluster_counts(selected, gold)
    counts["false_merge_pairs"] = clusters_metrics["false_merge_pairs"]
    counts["false_split_pairs"] = clusters_metrics["false_split_pairs"]
    return counts


def _training_digest(items: list[dict[str, Any]]) -> str:
    """Recreate the immutable training-summary digest from normalized records."""
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda row: row["document_id"]):
        mentions = sorted(
            (
                {
                    "start": mention["start"],
                    "end": mention["end"],
                    "entity_id": mention["entity_id"],
                }
                for mention in item["mentions"]
            ),
            key=lambda mention: (mention["start"], mention["end"]),
        )
        digest.update(
            canonical_json(
                {
                    "document_id": item["document_id"],
                    "group_id": item["group_id"],
                    "split": "train",
                    "text": item["text"],
                    "mentions": mentions,
                }
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def diagnose_identity(
    model: Any,
    documents: Any,
    policy: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Report aggregate identity failures using registered training or validation.

    The validation corpus must exactly match the one used to freeze the policy.
    Training records must reproduce the full model training digest. New
    development or sealed documents cannot enter by renaming their split.
    """
    if split not in ("train", "validation"):
        raise ValueError("identity diagnostics permit only train or validation")
    _check_policy(model, policy)
    items = _documents(documents, split)
    if split == "validation":
        _training_disjoint(model, items)
        if _digest(items) != policy["validation_data_sha256"]:
            raise ValueError("validation documents differ from frozen calibration")
    else:
        for field, recorded in (
            ("document_id", "train_document_ids"),
            ("group_id", "train_group_ids"),
        ):
            if {item[field] for item in items} != set(
                model.training_summary.get(recorded, [])
            ):
                raise ValueError(f"training {field} differs from recorded full corpus")
        if _training_digest(items) != model.training_summary.get("train_data_sha256"):
            raise ValueError("training data differs from model training fingerprint")
    totals = Counter({name: 0 for name in COUNT_NAMES})
    per_document = []
    for document in items:
        text, gold = document["text"], _gold(document)
        spans = [
            span
            for span in _spans(model, text)
            if span["score"] >= policy["mention_threshold"]
        ]
        pair_cache: dict[tuple[int, int, int, int], float] = {}
        ranked = _ranked(model, text, spans, pair_cache)
        oracle = _ranked(
            model,
            text,
            [{"start": start, "end": end, "score": 1.0} for start, end in sorted(gold)],
            pair_cache,
        )
        selected = _apply_gate(ranked, policy["link_gate"])
        counts = _counts(
            text, gold, ranked, oracle, selected, model.config.max_antecedents
        )
        totals.update(counts)
        per_document.append({"document_id": document["document_id"], **dict(counts)})
    _check_policy(model, policy)
    anaphoric = totals["anaphoric_known_mentions"]
    return {
        "schema": "ai2-identity-diagnostics-v1",
        "split": split,
        "document_count": len(items),
        "data_sha256": _digest(items),
        "policy_sha256": policy["policy_sha256"],
        "max_antecedents": model.config.max_antecedents,
        "link_gate_enabled": policy["link_gate"]["enabled"],
        "link_gate_reason": policy["link_gate"]["reason"],
        "counts": dict(totals),
        "oracle_top3_recall": (
            totals["oracle_top3_retrieval_hits"] / anaphoric if anaphoric else None
        ),
        "predicted_top3_recall": (
            totals["predicted_top3_retrieval_hits"] / anaphoric if anaphoric else None
        ),
        "per_document": per_document,
    }

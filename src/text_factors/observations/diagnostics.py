"""Read-only diagnostics for open mention candidates on public splits.

These measurements do not choose thresholds, modify model weights, or read a
closed test split. The character n-gram control adapts the older string scorer
to candidate spans; it is a new comparison protocol, not an old mention model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from ..evaluation.baselines import NGramMemory
from .learning import LearningConfig, _span_indices, _tokens


def _public_documents(documents: Sequence[dict[str, Any]], split: str) -> None:
    if split not in {"train", "validation"} or not documents:
        raise ValueError("diagnostics require nonempty train or validation documents")
    if any(document.get("split") != split for document in documents):
        raise ValueError(f"diagnostics accept only {split} documents")


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _coverage(gold: set[tuple[int, int]], predicted: set[tuple[int, int]]) -> dict:
    represented = len(gold & predicted)
    return {
        "gold_count": len(gold),
        "represented_count": represented,
        "unsupported_count": len(gold) - represented,
        "recall": represented / len(gold) if gold else None,
    }


def _nested(span: tuple[int, int], other: tuple[int, int]) -> bool:
    return span != other and (
        (span[0] <= other[0] and other[1] <= span[1])
        or (other[0] <= span[0] and span[1] <= other[1])
    )


def _bucket(span: tuple[int, int], starts: dict[int, int], ends: dict[int, int]) -> str:
    if span[0] not in starts or span[1] not in ends:
        return "unaligned"
    width = ends[span[1]] - starts[span[0]]
    if width <= 0:
        return "unaligned"
    if width == 1:
        return "1"
    if width <= 4:
        return "2-4"
    if width <= 8:
        return "5-8"
    if width <= 16:
        return "9-16"
    return "17+"


_BUCKETS = ("1", "2-4", "5-8", "9-16", "17+", "unaligned")


def diagnose_mentions(
    model: Any,
    documents: Sequence[dict[str, Any]],
    *,
    split: str = "validation",
    threshold: float,
) -> dict[str, Any]:
    """Report candidate ceiling and selected exact spans, including nested gold.

    The candidate count is before thresholding; selected counts use the frozen
    supplied threshold. All gold spans, including unaligned and overwide spans,
    stay in the denominators. Nested means strict containment with another gold
    mention, regardless of whether that other mention was predicted.
    """
    _public_documents(documents, split)
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("mention threshold must be in [0, 1]")
    totals: Counter[str] = Counter()
    bucket_totals: dict[str, Counter[str]] = defaultdict(Counter)
    nested_totals: dict[str, Counter[str]] = defaultdict(Counter)
    per_document: list[dict[str, Any]] = []
    for document in sorted(documents, key=lambda item: item["document_id"]):
        text = document["text"]
        tokens = _tokens(text, model.config)
        starts = {token.start: index for index, token in enumerate(tokens)}
        ends = {token.end: index + 1 for index, token in enumerate(tokens)}
        gold = {(item["start"], item["end"]) for item in document["mentions"]}
        scored = model.span_scores(text)
        candidates = {(item["start"], item["end"]) for item in scored}
        selected = {
            (item["start"], item["end"])
            for item in scored
            if item["score"] >= threshold
        }
        if len(candidates) != len(scored):
            raise ValueError("model returned duplicate candidate spans")
        represented = gold & candidates
        matched = gold & selected
        totals.update(
            gold=len(gold),
            represented=len(represented),
            tp=len(matched),
            fp=len(selected - gold),
            fn=len(gold - selected),
        )
        for span in gold:
            bucket = _bucket(span, starts, ends)
            group = bucket_totals[bucket]
            group["gold"] += 1
            group["represented"] += span in candidates
            group["selected"] += span in selected
            nesting = (
                "nested" if any(_nested(span, other) for other in gold) else "flat"
            )
            nested_totals[nesting]["gold"] += 1
            nested_totals[nesting]["represented"] += span in candidates
            nested_totals[nesting]["selected"] += span in selected
        # The length slice has a well-defined false-positive denominator. For
        # nesting, only gold recall is meaningful without an arbitrary policy
        # for classifying invented mention boundaries.
        for span in selected - gold:
            bucket_totals[_bucket(span, starts, ends)]["false_positive"] += 1
        per_document.append(
            {
                "document_id": document["document_id"],
                "characters": len(text),
                "tokens": len(tokens),
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "candidate_coverage": _coverage(gold, candidates),
                "selected_spans": _prf(
                    len(matched), len(selected - gold), len(gold - selected)
                ),
            }
        )
    slices: dict[str, dict[str, Any]] = {}
    for name in _BUCKETS:
        counts = bucket_totals[name]
        tp = counts["selected"]
        slices[name] = {
            "candidate_coverage": {
                "gold_count": counts["gold"],
                "represented_count": counts["represented"],
                "unsupported_count": counts["gold"] - counts["represented"],
                "recall": counts["represented"] / counts["gold"]
                if counts["gold"]
                else None,
            },
            "selected_spans": _prf(tp, counts["false_positive"], counts["gold"] - tp),
        }
    nesting = {}
    for name in ("nested", "flat"):
        counts = nested_totals[name]
        nesting[name] = {
            "gold_count": counts["gold"],
            "represented_count": counts["represented"],
            "selected_count": counts["selected"],
            "candidate_recall": counts["represented"] / counts["gold"]
            if counts["gold"]
            else None,
            "selected_recall": counts["selected"] / counts["gold"]
            if counts["gold"]
            else None,
        }
    return {
        "split": split,
        "mention_threshold": float(threshold),
        "document_count": len(documents),
        "candidate_coverage": {
            "gold_count": totals["gold"],
            "represented_count": totals["represented"],
            "unsupported_count": totals["gold"] - totals["represented"],
            "recall": totals["represented"] / totals["gold"]
            if totals["gold"]
            else None,
        },
        "selected_spans": _prf(totals["tp"], totals["fp"], totals["fn"]),
        "span_length_in_unicode_tokens": slices,
        "gold_nesting": nesting,
        "per_document": per_document,
    }


class NGramSpanControl:
    """Explicit span adapter around the existing NGramMemory string scorer."""

    def __init__(self, config: LearningConfig, *, n: int = 2) -> None:
        self.config = config
        self.scorer = NGramMemory(n=n)
        self.trained = False

    def fit(self, documents: Sequence[dict[str, Any]]) -> None:
        _public_documents(documents, "train")
        if any(doc.get("coverage") != "complete" for doc in documents):
            raise ValueError("n-gram control requires complete train annotations")
        surfaces = [
            doc["text"][mention["start"] : mention["end"]]
            for doc in documents
            for mention in doc["mentions"]
        ]
        if not surfaces:
            raise ValueError("n-gram control needs training mention surfaces")
        self.scorer.fit(surfaces)
        self.trained = True

    def span_scores(self, text: str) -> list[dict[str, Any]]:
        if not self.trained:
            raise ValueError("n-gram control must fit train surfaces first")
        tokens = _tokens(text, self.config)
        return [
            {
                "start": tokens[begin].start,
                "end": tokens[stop - 1].end,
                "score": self.scorer.score(
                    text[tokens[begin].start : tokens[stop - 1].end]
                ),
            }
            for begin, stop in _span_indices(tokens, self.config)
        ]


def calibrate_ngram_control(
    control: NGramSpanControl, documents: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Choose the threshold from validation labels, with no test access.

    Ties prefer precision then the higher threshold. A missing threshold means
    that validation selected no mentions. The old NGramMemory had no span API.
    """
    _public_documents(documents, "validation")
    scored = []
    gold_total = 0
    for doc in documents:
        gold = {(m["start"], m["end"]) for m in doc["mentions"]}
        gold_total += len(gold)
        scored.extend(
            (row["score"], (row["start"], row["end"]) in gold)
            for row in control.span_scores(doc["text"])
        )
    scored.sort(key=lambda pair: -pair[0])
    tp, fp = 0, 0
    best: tuple[float, float, float] = (0.0, 0.0, float("inf"))
    best_counts = (0, 0, gold_total)
    index = 0
    while index < len(scored):
        value = scored[index][0]
        while index < len(scored) and scored[index][0] == value:
            if scored[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        f1 = 2 * tp / (tp + fp + gold_total) if tp + fp + gold_total else 0.0
        precision = tp / (tp + fp)
        current = (f1, precision, float(value))
        if current > best:
            best, best_counts = current, (tp, fp, gold_total - tp)
    return {
        "control_kind": "new_span_adapter_to_existing_ngram_string_memory",
        "n": control.scorer.n,
        "candidate_policy": "same_unicode_contiguous_spans_as_learned_model",
        "calibration_split": "validation",
        "threshold": best[2] if best[2] != float("inf") else None,
        "validation_spans": _prf(*best_counts),
    }


def score_ngram_control(
    control: NGramSpanControl,
    documents: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Public validation diagnostic; a test run needs a separate frozen gate."""
    _public_documents(documents, "validation")
    if calibration.get("calibration_split") != "validation":
        raise ValueError("ngram threshold must be calibrated on validation")
    threshold = calibration["threshold"]
    tp = fp = fn = 0
    for doc in documents:
        gold = {(m["start"], m["end"]) for m in doc["mentions"]}
        proposed = {
            (row["start"], row["end"])
            for row in control.span_scores(doc["text"])
            if threshold is not None and row["score"] >= threshold
        }
        tp += len(proposed & gold)
        fp += len(proposed - gold)
        fn += len(gold - proposed)
    return {"split": "validation", "selected_spans": _prf(tp, fp, fn)}

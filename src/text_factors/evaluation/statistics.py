"""Finite, auditable classification metrics and dev-only threshold fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from statistics import NormalDist

import numpy as np


def _scores(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a non-empty one-dimensional finite sequence")
    return scores


@dataclass(frozen=True, slots=True)
class NoveltyGate:
    """An empirical negative-dev quantile, NOT a population FPR guarantee.

    Scores must exceed the threshold strictly; ties are rejected. Never fit this
    on test scores. This gate turns a familiarity score into a testable decision;
    a nonempty SDR by itself is not a recognition decision.
    """

    threshold: float
    target_dev_fpr: float
    negative_count: int

    def __post_init__(self) -> None:
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if (
            type(self.target_dev_fpr) not in (int, float)
            or not isfinite(self.target_dev_fpr)
            or not 0 <= self.target_dev_fpr < 1
        ):
            raise ValueError("target_dev_fpr must be in [0, 1)")
        if type(self.negative_count) is not int or self.negative_count < 1:
            raise ValueError("negative_count must be a positive integer")

    @classmethod
    def fit(
        cls, negative_scores: Sequence[float], *, target_fpr: float = 0.05
    ) -> NoveltyGate:
        if (
            type(target_fpr) not in (int, float)
            or not isfinite(target_fpr)
            or not 0 <= target_fpr < 1
        ):
            raise ValueError("target_fpr must be in [0, 1)")
        scores = np.sort(_scores(negative_scores))
        index = max(0, ceil((1 - target_fpr) * len(scores)) - 1)
        return cls(float(scores[index]), float(target_fpr), len(scores))

    def accepts(self, score: float) -> bool:
        if not isfinite(score):
            raise ValueError("score must be finite")
        return score > self.threshold


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Mann-Whitney AUROC with half credit for ties; requires both classes."""
    values = _scores(scores)
    truth = np.asarray(labels)
    if truth.dtype.kind != "b" or truth.shape != values.shape:
        raise ValueError("labels must be booleans with the same shape as scores")
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative examples")
    order = np.argsort(values, kind="stable")
    wins = 0.0
    negatives_before = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        group_positives = int(truth[order[start:end]].sum())
        group_negatives = end - start - group_positives
        wins += group_positives * (negatives_before + 0.5 * group_negatives)
        negatives_before += group_negatives
        start = end
    return wins / (positives * negatives)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Nominal 95% Wilson interval for a binomial rate, conditional on this suite."""
    if type(trials) is not int or type(successes) is not int:
        raise ValueError("successes and trials must be integers")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    z = NormalDist().inv_cdf(0.975)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def classification_metrics(
    labels: Sequence[bool], scores: Sequence[float], gate: NoveltyGate
) -> dict[str, object]:
    values = _scores(scores)
    truth = np.asarray(labels)
    if truth.dtype.kind != "b" or truth.shape != values.shape:
        raise ValueError("labels must be booleans with the same shape as scores")
    predicted = values > gate.threshold
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    if not tp + fn or not fp + tn:
        raise ValueError("classification report requires both classes")
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(truth),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn),
        "false_positive_rate_ci95": list(wilson_interval(fp, fp + tn)),
        "recall_ci95": list(wilson_interval(tp, tp + fn)),
        "acceptance_rate": (tp + fp) / len(truth),
        "roc_auc": roc_auc(labels, scores),
    }


def paired_accuracy_interval(
    labels: Sequence[bool],
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    seed: int,
    resamples: int = 1000,
) -> dict[str, object]:
    """Paired bootstrap over cases; never treats model seeds as extra cases."""
    arrays = [np.asarray(values) for values in (labels, first, second)]
    if any(a.dtype.kind != "b" or a.ndim != 1 for a in arrays):
        raise ValueError("inputs must be one-dimensional boolean sequences")
    if not len(arrays[0]) or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("inputs must have the same non-empty shape")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    differences = (arrays[1] == arrays[0]).astype(float) - (arrays[2] == arrays[0])
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(np.mean(rng.choice(differences, size=len(differences))))
            for _ in range(resamples)
        ]
    )
    return {
        "accuracy_difference": float(np.mean(differences)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "resamples": resamples,
        "unit": "held-out case within this fixed rule family",
        "scope": "diagnostic interval, not evidence of general intelligence",
    }

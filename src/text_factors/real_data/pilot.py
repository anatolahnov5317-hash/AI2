"""Pilot-gate metrics for groundedness and useful answer coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PilotGateConfig:
    confidence: float = 0.95
    max_ungrounded_upper: float = 0.01
    min_resolvable_coverage: float = 0.70

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence", self.confidence),
            ("max_ungrounded_upper", self.max_ungrounded_upper),
            ("min_resolvable_coverage", self.min_resolvable_coverage),
        ):
            if type(value) not in (int, float) or not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class PilotSample:
    resolvable: bool
    answered: bool
    grounded: bool
    critical_error: bool = False


def _exact_binomial_upper(errors: int, total: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper bound, without a SciPy dependency.

    For 0 < errors < total, invert P(X <= errors | p) = 1 - confidence.
    The recurrence sums backwards from the observed count. In the searched
    interval p >= errors / total, each preceding probability is no larger.
    """

    if type(errors) is not int or type(total) is not int or not 0 <= errors <= total:
        raise ValueError("invalid binomial counts")
    if type(confidence) not in (float, int) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if total <= 0:
        return 1.0
    if errors == 0:
        alpha = 1.0 - confidence
        return 1.0 - alpha ** (1.0 / total)
    if errors == total:
        return 1.0

    def left_tail(probability: float) -> float:
        log_at_observed = (
            math.lgamma(total + 1)
            - math.lgamma(errors + 1)
            - math.lgamma(total - errors + 1)
            + errors * math.log(probability)
            + (total - errors) * math.log1p(-probability)
        )
        term = math.exp(log_at_observed)
        result = term
        for count in range(errors, 0, -1):
            term *= count * (1.0 - probability) / ((total - count + 1) * probability)
            result += term
            if term < result * 1e-16:
                break
        return result

    alpha = 1.0 - confidence
    lower, upper = errors / total, 1.0
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        if midpoint in (lower, upper):
            break
        if left_tail(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def evaluate_pilot(
    samples: tuple[PilotSample, ...], config: PilotGateConfig | None = None
) -> dict[str, float | int | bool]:
    config = config or PilotGateConfig()
    resolvable = sum(sample.resolvable for sample in samples)
    answered_resolvable = sum(
        sample.resolvable and sample.answered for sample in samples
    )
    answered = [sample for sample in samples if sample.answered]
    ungrounded = sum(not sample.grounded for sample in answered)
    critical = sum(sample.critical_error for sample in samples)
    coverage = 0.0 if resolvable == 0 else answered_resolvable / resolvable
    upper = _exact_binomial_upper(ungrounded, len(answered), config.confidence)
    passed = (
        critical == 0
        and coverage >= config.min_resolvable_coverage
        and upper <= config.max_ungrounded_upper
    )
    return {
        "samples": len(samples),
        "resolvable": resolvable,
        "answered": len(answered),
        "answered_resolvable": answered_resolvable,
        "resolvable_coverage": coverage,
        "ungrounded_answers": ungrounded,
        "ungrounded_upper_bound": upper,
        "critical_errors": critical,
        "passed": passed,
    }

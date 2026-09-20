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


def _wilson_upper(errors: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    if errors == 0:
        alpha = 1.0 - confidence
        return 1.0 - alpha ** (1.0 / total)
    # 95% is the declared default gate. For custom confidence use a conservative
    # normal approximation without adding a scipy dependency.
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else 2.5758293035489004
    p = errors / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    )
    return min(1.0, (centre + spread) / denominator)


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
    upper = _wilson_upper(ungrounded, len(answered), config.confidence)
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

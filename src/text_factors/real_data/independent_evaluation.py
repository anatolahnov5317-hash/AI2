"""P22 open, history-grouped evaluation; sealed data are unavailable here.

This scorer accepts frozen predictions, never a model or a sealed example. It
cannot establish that a supplied source was accessible or that a gold label is
correct: those are independent corpus and P18/P20 custody responsibilities.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _name(value: str, field: str) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError(f"invalid {field}")
    return value


def _labels(value: tuple[str, ...], field: str) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or len(value) > 128
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError(f"{field} needs sorted unique labels")
    for label in value:
        _name(label, field)
    return value


@dataclass(frozen=True, slots=True)
class EvidencePointer:
    claim: str
    source_id: str
    source_version: int
    source_sha256: str

    def __post_init__(self) -> None:
        _name(self.claim, "referenced claim")
        _name(self.source_id, "source ID")
        if type(self.source_version) is not int or self.source_version < 1:
            raise ValueError("invalid source version")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("invalid source SHA-256")


@dataclass(frozen=True, slots=True)
class FrozenPrediction:
    """Output produced before the evaluator sees gold; empty claims abstain."""

    claims: tuple[str, ...]
    evidence: tuple[EvidencePointer, ...] = ()

    def __post_init__(self) -> None:
        _labels(self.claims, "prediction")
        if type(self.evidence) is not tuple or any(
            not isinstance(pointer, EvidencePointer) for pointer in self.evidence
        ):
            raise ValueError("invalid evidence pointers")
        if len({pointer.claim for pointer in self.evidence}) != len(self.evidence):
            raise ValueError("at most one pointer per predicted claim")
        if any(pointer.claim not in self.claims for pointer in self.evidence):
            raise ValueError("evidence cannot justify an unpredicted claim")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    history_group_id: str
    relation_id: str
    split: str
    gold_claims: tuple[str, ...]
    gold_complete: bool
    predictions: Mapping[str, FrozenPrediction]

    def __post_init__(self) -> None:
        for field in ("case_id", "history_group_id", "relation_id"):
            _name(getattr(self, field), field)
        _labels(self.gold_claims, "gold")
        if type(self.gold_complete) is not bool:
            raise ValueError("gold completeness must be explicit")
        if not isinstance(self.predictions, Mapping) or any(
            type(name) is not str or not isinstance(prediction, FrozenPrediction)
            for name, prediction in self.predictions.items()
        ):
            raise ValueError("invalid predictions")
        object.__setattr__(
            self, "predictions", MappingProxyType(dict(self.predictions))
        )


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Named, fully paired methods and disclosed training relation budget."""

    candidate: str
    baseline: str
    ablation: str
    trained_history_groups: frozenset[str]
    baseline_relation_ids: frozenset[str]
    newly_trained_relation_ids: frozenset[str]
    bootstrap_seed: int = 2209
    bootstrap_repeats: int = 2000

    def __post_init__(self) -> None:
        methods = (self.candidate, self.baseline, self.ablation)
        if len(set(methods)) != len(methods):
            raise ValueError("candidate, baseline and ablation must be distinct")
        for method in methods:
            _name(method, "method")
        for field in (
            "trained_history_groups",
            "baseline_relation_ids",
            "newly_trained_relation_ids",
        ):
            groups = getattr(self, field)
            if type(groups) is not frozenset or not groups:
                raise ValueError(f"{field} needs pinned, nonempty IDs")
            for group in groups:
                _name(group, field)
        if self.baseline_relation_ids & self.newly_trained_relation_ids:
            raise ValueError("new relations already exist in the baseline")
        if (
            type(self.bootstrap_seed) is not int
            or type(self.bootstrap_repeats) is not int
        ):
            raise ValueError("invalid bootstrap configuration")
        if not 1 <= self.bootstrap_repeats <= 10000:
            raise ValueError("bootstrap must be bounded")

    @property
    def methods(self) -> tuple[str, str, str]:
        return self.candidate, self.baseline, self.ablation


def wilson_95(successes: int, trials: int) -> tuple[float, float] | None:
    """Two-sided 95% Wilson interval; samples are independent *groups*."""
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials < 0
        or not 0 <= successes <= trials
    ):
        raise ValueError("invalid interval counts")
    if trials == 0:
        return None
    z = 1.959963984540054
    fraction = successes / trials
    correction = z * z / trials
    center = (fraction + correction / 2) / (1 + correction)
    radius = (
        z
        * math.sqrt(fraction * (1 - fraction) / trials + correction / (4 * trials))
        / (1 + correction)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _paired_bootstrap(
    differences: tuple[float, ...], *, seed: int, repeats: int
) -> dict[str, float]:
    randomizer = random.Random(seed)
    size = len(differences)
    samples = sorted(
        sum(differences[randomizer.randrange(size)] for _ in range(size)) / size
        for _ in range(repeats)
    )
    return {
        "mean_difference": sum(differences) / size,
        "percentile_95_lower": samples[int(0.025 * (repeats - 1))],
        "percentile_95_upper": samples[int(0.975 * (repeats - 1))],
        "independent_groups": size,
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
    }


def evaluate_open(
    cases: tuple[EvaluationCase, ...], plan: EvaluationPlan
) -> dict[str, Any]:
    """Score all paired methods with one independent vote per history family.

    The local scorer refuses sealed/calibration/future data, so changing open
    development code cannot be advertised as a sealed result. Incomplete gold
    is refused instead of turning an unannotated claim into a false positive.
    """
    if type(cases) is not tuple or not cases or not isinstance(plan, EvaluationPlan):
        raise ValueError("an explicit plan and open cases are required")
    by_group: dict[str, list[dict[str, Any]]] = {}
    seen_cases: set[str] = set()
    for case in cases:
        if not isinstance(case, EvaluationCase):
            raise ValueError("invalid evaluation case")
        if case.split != "development" or not case.gold_complete:
            raise ValueError("only completely labeled open development can be scored")
        if case.case_id in seen_cases:
            raise ValueError("duplicate case ID")
        if case.history_group_id in plan.trained_history_groups:
            raise ValueError("evaluation history overlaps training")
        if case.relation_id not in (
            plan.baseline_relation_ids | plan.newly_trained_relation_ids
        ):
            raise ValueError("relation outside the pinned training vocabulary")
        if set(case.predictions) != set(plan.methods):
            raise ValueError("all paired methods must predict every case")
        seen_cases.add(case.case_id)
        scored: dict[str, dict[str, Any]] = {}
        for method in plan.methods:
            prediction = case.predictions[method]
            accepted = bool(prediction.claims)
            exact = accepted and prediction.claims == case.gold_claims
            structurally_grounded = accepted and {
                pointer.claim for pointer in prediction.evidence
            } == set(prediction.claims)
            scored[method] = {
                "accepted": accepted,
                "exact": exact,
                "evidence_pointers_present": structurally_grounded,
                "useful": exact and structurally_grounded,
                "unverified_or_wrong": accepted
                and not (exact and structurally_grounded),
            }
        by_group.setdefault(case.history_group_id, []).append(
            {
                "case_id": case.case_id,
                "relation_id": case.relation_id,
                "novel_to_baseline": case.relation_id
                in plan.newly_trained_relation_ids,
                "gold_claims": list(case.gold_claims),
                "methods": scored,
            }
        )
    if not any(
        case["novel_to_baseline"] for group in by_group.values() for case in group
    ):
        raise ValueError("no case tests a newly trained relation")

    metrics: dict[str, Any] = {}
    for method in plan.methods:
        groups = list(by_group.values())
        accepted_groups = sum(
            any(case["methods"][method]["accepted"] for case in group)
            for group in groups
        )
        erroneous_groups = sum(
            any(case["methods"][method]["unverified_or_wrong"] for case in group)
            for group in groups
        )
        useful_groups = sum(
            all(case["methods"][method]["useful"] for case in group) for group in groups
        )
        all_cases = [case for group in groups for case in group]
        novel_cases = [case for case in all_cases if case["novel_to_baseline"]]
        metrics[method] = {
            "independent_groups": len(groups),
            "cases": len(all_cases),
            "accepted_cases": sum(
                case["methods"][method]["accepted"] for case in all_cases
            ),
            "exact_cases": sum(case["methods"][method]["exact"] for case in all_cases),
            "useful_groups": useful_groups,
            "useful_group_fraction": useful_groups / len(groups),
            "useful_group_wilson_95": wilson_95(useful_groups, len(groups)),
            "accepted_groups": accepted_groups,
            "unverified_or_wrong_accepted_groups": erroneous_groups,
            "accepted_group_risk_upper_95": (
                interval[1]
                if (interval := wilson_95(erroneous_groups, accepted_groups))
                is not None
                else None
            ),
            "new_relation_exact_cases": sum(
                case["methods"][method]["exact"] for case in novel_cases
            ),
            "new_relation_cases": len(novel_cases),
        }

    def paired(other: str) -> dict[str, Any]:
        differences = tuple(
            sum(
                case["methods"][plan.candidate]["useful"]
                - case["methods"][other]["useful"]
                for case in group
            )
            / len(group)
            for group in by_group.values()
        )
        descriptive: dict[str, Any] = _paired_bootstrap(
            differences, seed=plan.bootstrap_seed, repeats=plan.bootstrap_repeats
        )
        wins = sum(delta > 0 for delta in differences)
        losses = sum(delta < 0 for delta in differences)
        descriptive.update(
            {
                "winner_groups": wins,
                "loser_groups": losses,
                "tie_groups": len(differences) - wins - losses,
                "win_fraction_wilson_95_non_ties": wilson_95(wins, wins + losses),
                "warning": "Few groups can collapse the bootstrap interval.",
            }
        )
        return descriptive

    return {
        "schema": "ai2-p22-open-grouped-v1",
        "split": "development",
        "unit": "history_group",
        "status": "open_diagnostic_not_independent_pilot",
        "methods": list(plan.methods),
        "trained_history_groups": sorted(plan.trained_history_groups),
        "newly_trained_relation_ids": sorted(plan.newly_trained_relation_ids),
        "per_method": metrics,
        "paired_candidate_minus_baseline": paired(plan.baseline),
        "paired_candidate_minus_ablation": paired(plan.ablation),
        "groups": [
            {"history_group_id": group_id, "cases": group}
            for group_id, group in sorted(by_group.items())
        ],
        "limits": [
            "Input gold and source permission are not independently verified here.",
            "Evidence pointers record structure only; access needs P18/P20 receipts.",
            "Development confidence intervals do not establish pilot acceptance.",
        ],
    }

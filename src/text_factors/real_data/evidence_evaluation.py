"""Open, grouped comparison of two equally simple evidence selection rules.

Neither rule is trained here or reads the gold choice. A source group counts
once for grounded selection, while the control counts raw evidence roots. The
caller supplies human-reviewed correct choices and source-family identifiers;
no sealed or future split may be passed to this development-only evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import RealDataEngine


@dataclass(frozen=True, slots=True)
class EvidenceChoiceCase:
    case_id: str
    group_id: str
    split: str
    engine: RealDataEngine
    candidate_ids: tuple[str, ...]
    correct_claim_id: str


def _choice(scores: dict[str, int]) -> str | None:
    best = max(scores.values())
    winners = [claim_id for claim_id, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def evaluate_evidence_choices(
    cases: tuple[EvidenceChoiceCase, ...],
    *,
    training_group_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Score open evidence choices for facts.

    Each independent source family contributes exactly one case. Selection
    always uses the same candidate set and zero learned parameters in both
    modes. Ties abstain instead of being broken using answer IDs or gold.
    """
    if not cases:
        raise ValueError("open evaluation requires at least one case")
    if type(training_group_ids) is not frozenset or any(
        type(group) is not str or not group for group in training_group_ids
    ):
        raise ValueError("training groups must be pinned IDs")
    seen_cases: set[str] = set()
    seen_groups: set[str] = set()
    rows: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, EvidenceChoiceCase):
            raise ValueError("invalid evidence choice case")
        if (
            type(case.case_id) is not str
            or not case.case_id
            or type(case.group_id) is not str
            or not case.group_id
            or case.case_id in seen_cases
            or case.group_id in seen_groups
        ):
            raise ValueError("duplicate case or dependent source-family group")
        if case.split != "development" or case.group_id in training_group_ids:
            raise ValueError("evaluation only accepts disjoint open development")
        if (
            not isinstance(case.engine, RealDataEngine)
            or type(case.candidate_ids) is not tuple
            or len(case.candidate_ids) < 2
            or len(set(case.candidate_ids)) != len(case.candidate_ids)
            or case.correct_claim_id not in case.candidate_ids
        ):
            raise ValueError("invalid candidate set or reviewed gold choice")
        independent: dict[str, int] = {}
        raw: dict[str, int] = {}
        for claim_id in case.candidate_ids:
            if type(claim_id) is not str:
                raise ValueError("candidate ID must be a string")
            receipt = case.engine.receipt(
                question_id=case.case_id,
                answer_text="candidate",
                claim_ids=(claim_id,),
                model_version="development-only",
            )
            if not receipt.complete:
                raise ValueError("candidate has no current grounded permission")
            independent[claim_id] = case.engine.evidence.independent_support(claim_id)
            raw[claim_id] = case.engine.evidence.support_count(claim_id)
        selected = _choice(independent)
        control = _choice(raw)
        rows.append(
            {
                "case_id": case.case_id,
                "group_id": case.group_id,
                "grounded_choice": selected,
                "raw_count_choice": control,
                "grounded_correct": selected == case.correct_claim_id,
                "raw_count_correct": control == case.correct_claim_id,
                "grounded_support": independent,
                "raw_count_support": raw,
            }
        )
        seen_cases.add(case.case_id)
        seen_groups.add(case.group_id)

    def metrics(key: str, correct_key: str) -> dict[str, int | float | None]:
        selected = [row for row in rows if row[key] is not None]
        correct = sum(bool(row[correct_key]) for row in rows)
        return {
            "correct": correct,
            "total": len(rows),
            "answered": len(selected),
            "coverage": len(selected) / len(rows),
            "accuracy_all": correct / len(rows),
            "accuracy_answered": correct / len(selected) if selected else None,
        }

    return {
        "split": "development",
        "unit": "independent_source_family",
        "tuning_parameters_per_rule": 0,
        "fact_choice": {
            "grounded": metrics("grounded_choice", "grounded_correct"),
            "raw_root_count_control": metrics("raw_count_choice", "raw_count_correct"),
            "paired_grounded_wins": sum(
                bool(row["grounded_correct"]) and not row["raw_count_correct"]
                for row in rows
            ),
            "paired_control_wins": sum(
                bool(row["raw_count_correct"]) and not row["grounded_correct"]
                for row in rows
            ),
        },
        "cases": rows,
    }

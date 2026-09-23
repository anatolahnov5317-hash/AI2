"""Open synthetic controls; these cases are not acceptance evidence for P12."""

from __future__ import annotations

import unittest

from text_factors.real_data import Claim, ClaimStatus, EvidenceRoot, RealDataEngine
from text_factors.real_data.contracts import SourceSlice
from text_factors.real_data.evidence_evaluation import (
    EvidenceChoiceCase,
    evaluate_evidence_choices,
)


def _open_choice(group: str) -> EvidenceChoiceCase:
    engine = RealDataEngine()
    for index in range(5):
        # Three report copies share one source family; the two independent
        # reports use separate families, all within this one example.
        root_group = f"{group}-true-{index}" if index < 2 else f"{group}-copy"
        engine.register_evidence(
            EvidenceRoot(
                f"{group}-r{index}",
                root_group,
                f"{group}-s{index}",
                1,
                "default",
                SourceSlice(f"{group}-s{index}", 1, 0, 1, "0" * 64),
            )
        )
        engine.add_claim(
            Claim(
                f"{group}-base{index}",
                "state",
                (),
                ClaimStatus.OBSERVED,
                source=SourceSlice(f"{group}-s{index}", 1, 0, 1, "0" * 64),
                evidence_roots=(f"{group}-r{index}",),
            )
        )
    for candidate, indices in (("independent", range(2)), ("copies", range(2, 5))):
        engine.add_claim(
            Claim(
                f"{group}-{candidate}",
                "state",
                (),
                ClaimStatus.INFERRED,
                source=SourceSlice(f"{group}-s{indices.start}", 1, 0, 1, "0" * 64),
            ),
            parent_claim_ids=tuple(f"{group}-base{i}" for i in indices),
        )
    return EvidenceChoiceCase(
        case_id=f"case-{group}",
        group_id=group,
        split="development",
        engine=engine,
        candidate_ids=(f"{group}-independent", f"{group}-copies"),
        correct_claim_id=f"{group}-independent",
    )


class EvidenceEvaluationDevelopmentTests(unittest.TestCase):
    def test_duplicate_reports_can_beat_raw_count_on_synthetic_cases(self) -> None:
        report = evaluate_evidence_choices(
            (_open_choice("family-a"), _open_choice("family-b"))
        )
        facts = report["fact_choice"]
        self.assertEqual(facts["grounded"]["correct"], 2)
        self.assertEqual(facts["raw_root_count_control"]["correct"], 0)
        self.assertEqual(facts["paired_grounded_wins"], 2)
        self.assertEqual(facts["paired_control_wins"], 0)
        self.assertEqual(report["tuning_parameters_per_rule"], 0)
        self.assertNotIn("forecast", report)

    def test_no_sealed_split_or_dependent_family_can_be_evaluated(self) -> None:
        case = _open_choice("family")
        with self.assertRaisesRegex(ValueError, "open development"):
            evaluate_evidence_choices(
                (
                    EvidenceChoiceCase(
                        case.case_id,
                        case.group_id,
                        "sealed_test",
                        case.engine,
                        case.candidate_ids,
                        case.correct_claim_id,
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "open development"):
            evaluate_evidence_choices((case,), training_group_ids=frozenset({"family"}))
        with self.assertRaisesRegex(ValueError, "dependent"):
            evaluate_evidence_choices((case, case))

    def test_ungrounded_candidates_cannot_pad_an_accuracy_result(self) -> None:
        case = _open_choice("family")
        case.engine.correct_claim(
            "family-base0",
            Claim(
                "fresh",
                "state",
                (),
                ClaimStatus.OBSERVED,
                source=SourceSlice("family-s0", 1, 0, 1, "0" * 64),
                evidence_roots=("family-r0",),
            ),
        )
        with self.assertRaisesRegex(ValueError, "no current grounded permission"):
            evaluate_evidence_choices((case,))


if __name__ == "__main__":
    unittest.main()

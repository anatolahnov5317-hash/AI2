"""Open synthetic checks for P24; none are real pilot observations."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from text_factors.real_data.pilot import _exact_binomial_upper
from text_factors.real_data.pilot_readiness import PilotTrial, score_p24_trials


def sample(index: int) -> PilotTrial:
    return PilotTrial(
        group_id=f"independent-family-{index}",
        query_id=f"preselected-question-{index}",
        answerable=True,
        accepted=True,
        useful_supported=True,
        unsupported_or_wrong=False,
        authorized_source_and_version=True,
    )


class P24GroupedPilotTests(unittest.TestCase):
    def test_exact_one_sided_bound_at_the_pilot_threshold(self) -> None:
        self.assertAlmostEqual(_exact_binomial_upper(1, 2, 0.95), math.sqrt(0.95))
        self.assertFalse(
            score_p24_trials(tuple(sample(i) for i in range(298))).numeric_gate_passed
        )
        result = score_p24_trials(tuple(sample(i) for i in range(299)))
        self.assertTrue(result.numeric_gate_passed)
        self.assertEqual(result.independent_groups, 299)
        upper = result.unsupported_upper_bound
        assert upper is not None
        self.assertLessEqual(upper, 0.01)

    def test_repeats_cannot_raise_effective_sample_size(self) -> None:
        original = sample(0)
        repeated = replace(original, query_id="another-question-same-history")
        with self.assertRaisesRegex(ValueError, "one preselected query"):
            score_p24_trials((original, repeated))
        with self.assertRaisesRegex(ValueError, "before model output"):
            score_p24_trials((replace(original, preselected=False),))

    def test_unknown_annotations_are_not_negatives_or_successes(self) -> None:
        trials = tuple(sample(i) for i in range(299))
        for changed, reason in (
            (replace(trials[0], answerable=None), "unknown_answerability"),
            (
                replace(trials[0], unsupported_or_wrong=None),
                "unknown_accepted_answer_support",
            ),
            (
                replace(trials[0], authorized_source_and_version=None),
                "unknown_accepted_answer_attribution",
            ),
            (
                replace(trials[0], useful_supported=None),
                "unknown_accepted_answer_usefulness",
            ),
        ):
            report = score_p24_trials((changed, *trials[1:]))
            self.assertFalse(report.numeric_gate_passed)
            self.assertIn(reason, report.blockers)
        self.assertIsNone(
            score_p24_trials(
                (replace(trials[0], unsupported_or_wrong=None),)
            ).unsupported_upper_bound
        )

    def test_abstentions_reduce_useful_coverage_and_zero_is_undefined(self) -> None:
        trials = tuple(sample(i) for i in range(299))
        answered = tuple(
            replace(
                t,
                accepted=False,
                useful_supported=None,
                unsupported_or_wrong=None,
                authorized_source_and_version=None,
            )
            if i < 91
            else t
            for i, t in enumerate(trials)
        )
        result = score_p24_trials(answered)
        self.assertFalse(result.numeric_gate_passed)
        coverage = result.useful_coverage
        assert coverage is not None
        self.assertLess(coverage, 0.70)
        empty = score_p24_trials(())
        self.assertIsNone(empty.unsupported_upper_bound)
        self.assertFalse(empty.numeric_gate_passed)

    def test_receipt_and_critical_error_cannot_be_hidden_by_volume(self) -> None:
        trials = tuple(sample(i) for i in range(299))
        for changed, reason in (
            (
                replace(trials[0], authorized_source_and_version=False),
                "missing_authorized_source_or_exact_version",
            ),
            (replace(trials[0], critical_error=True), "critical_error"),
            (
                replace(trials[0], unsupported_or_wrong=True),
                "unsupported_risk_upper_bound_above_1_percent",
            ),
        ):
            result = score_p24_trials((changed, *trials[1:]))
            self.assertFalse(result.numeric_gate_passed)
            self.assertIn(reason, result.blockers)

    def test_unanswerable_and_unattributed_answers_cannot_be_called_safe(self) -> None:
        trials = tuple(sample(i) for i in range(299))
        unanswerable = replace(
            sample(299),
            answerable=False,
            useful_supported=False,
            unsupported_or_wrong=False,
        )
        result = score_p24_trials((*trials, unanswerable))
        self.assertEqual(result.unsupported_answers, 1)
        self.assertIn("accepted_unanswerable_label_conflict", result.blockers)
        self.assertFalse(result.numeric_gate_passed)

        contradictory = replace(trials[0], authorized_source_and_version=False)
        report = score_p24_trials((contradictory, *trials[1:]))
        self.assertEqual(report.useful_supported_answers, 298)
        self.assertEqual(report.unsupported_answers, 1)
        self.assertIn("contradictory_usefulness_label", report.blockers)

    def test_real_p01_draft_blocks_even_a_perfect_synthetic_score(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = {
            "schema": "ai2-p24-trials-v1",
            "phase": "shadow",
            "frozen_selection_reference": "example-only",
            "independence_review_reference": "example-only",
            "adjudication_reference": "example-only",
            "code_and_thresholds_freeze_reference": "example-only",
            "trials": [
                {
                    "group_id": f"synthetic-{i}",
                    "query_id": f"question-{i}",
                    "answerable": True,
                    "accepted": True,
                    "useful_supported": True,
                    "unsupported_or_wrong": False,
                    "authorized_source_and_version": True,
                    "preselected": True,
                }
                for i in range(299)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "trials.json"
            input_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/score_p24_pilot.py",
                    "--input",
                    str(input_path),
                    "--contract",
                    "docs/real_data/pilot_contract.yaml",
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["score"]["numeric_gate_passed"])
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["pilot_activated"])
        self.assertIn("p01_contract_not_finalized", report["blockers"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from text_factors.evaluation.coactivation_structure import (
    METHODS,
    SCENARIOS,
    CoactivationStructureConfig,
    evaluate_history,
    make_coactivation_histories,
    run_coactivation_structure,
)


class CoactivationStructureTests(unittest.TestCase):
    def test_column_shuffle_preserves_exact_marginals(self) -> None:
        for seed in (101, 211, 307):
            histories = make_coactivation_histories(seed)
            for scenario in SCENARIOS:
                variants = {
                    item["variant"]: item["history"]
                    for item in histories
                    if item["scenario"] == scenario
                }
                joint = variants["joint"]
                control = variants["marginal_shuffle"]
                np.testing.assert_array_equal(
                    np.count_nonzero(joint, axis=0),
                    np.count_nonzero(control, axis=0),
                )
                self.assertFalse(np.array_equal(joint, control))
                self.assertLessEqual(len(joint), 32)
                self.assertLessEqual(joint.shape[1], 12)

    def test_evaluation_does_not_mutate_history(self) -> None:
        item = make_coactivation_histories(101)[0]
        history = item["history"]
        before = history.copy()
        result = evaluate_history(history, item["true_groups"])
        np.testing.assert_array_equal(history, before)
        self.assertEqual(set(result["methods"]), set(METHODS))
        for method in METHODS:
            self.assertEqual(
                len(result["methods"][method]["raw_mask"]), history.shape[1]
            )

    def test_probes_change_only_row_order(self) -> None:
        reference = make_coactivation_histories(101)
        for seed in (211, 307):
            candidate = make_coactivation_histories(seed)
            for first, second in zip(reference, candidate, strict=True):
                self.assertEqual(first["scenario"], second["scenario"])
                self.assertEqual(first["variant"], second["variant"])
                self.assertEqual(
                    sorted(map(tuple, first["history"])),
                    sorted(map(tuple, second["history"])),
                )

    def test_report_is_deterministic_and_auditable(self) -> None:
        config = CoactivationStructureConfig(seconds=5.0)
        first = run_coactivation_structure(config)
        second = run_coactivation_structure(config)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(len(first["runs"]), 12)
        self.assertIsNone(first["timeout"])
        self.assertEqual(len(first["aggregate"]), 8)
        for run in first["runs"]:
            evaluation = run["evaluation"]
            history = np.asarray(evaluation["history"], dtype=np.bool_)
            np.testing.assert_array_equal(
                np.count_nonzero(history, axis=0),
                evaluation["counts"]["column_active"],
            )
            for method in METHODS:
                details = evaluation["methods"][method]
                self.assertEqual(
                    details["effective_prune_valid"], details["retained_count"] >= 3
                )
                self.assertIn("f1", details["best_true_group"])

    def test_timeout_returns_partial_status_without_aggregate(self) -> None:
        calls = 0

        def interrupt() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("simulated external budget")

        report = run_coactivation_structure(
            CoactivationStructureConfig(seconds=5.0), check_budget=interrupt
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertIsNotNone(report["timeout"])
        self.assertIsNone(report["aggregate"])
        self.assertIsNone(report["order_sensitivity"])
        self.assertLess(len(report["runs"]), 12)


if __name__ == "__main__":
    unittest.main()

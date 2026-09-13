import unittest

from text_factors.evaluation.statistics import (
    NoveltyGate,
    classification_metrics,
    paired_accuracy_interval,
    roc_auc,
    wilson_interval,
)


class EvaluationStatisticsTests(unittest.TestCase):
    def test_gate_rejects_ties_and_uses_negative_dev_only(self):
        gate = NoveltyGate.fit([0.0, 1.0, 1.0, 3.0], target_fpr=0.25)
        self.assertEqual(gate.threshold, 1.0)
        self.assertFalse(gate.accepts(1.0))
        self.assertTrue(gate.accepts(3.0))
        self.assertEqual(sum(gate.accepts(s) for s in [0, 1, 1, 3]), 1)

    def test_zero_fpr_gate(self):
        gate = NoveltyGate.fit([1.0, 2.0], target_fpr=0.0)
        self.assertEqual(gate.threshold, 2.0)

    def test_invalid_scores_rejected(self):
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                NoveltyGate.fit(scores)
        with self.assertRaises(ValueError):
            NoveltyGate.fit([1.0], target_fpr=1.0)

    def test_auc_perfect_reversed_and_tied(self):
        self.assertEqual(roc_auc([False, True], [0, 1]), 1.0)
        self.assertEqual(roc_auc([False, True], [1, 0]), 0.0)
        self.assertEqual(roc_auc([False, True], [1, 1]), 0.5)
        self.assertEqual(roc_auc([False, True, False, True], [0, 1, 1, 2]), 0.875)

    def test_auc_validation(self):
        for labels, scores in [([True], [1]), ([1, 0], [1, 0]), ([True], [1, 2])]:
            with self.assertRaises(ValueError):
                roc_auc(labels, scores)

    def test_counts_are_recomputable(self):
        gate = NoveltyGate.fit([0, 1])
        metrics = classification_metrics([True, True, False, False], [2, 0, 3, 1], gate)
        self.assertEqual([metrics[k] for k in ["tp", "fn", "fp", "tn"]], [1] * 4)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_interval_has_nonzero_uncertainty_with_zero_errors(self):
        lower, upper = wilson_interval(0, 32)
        self.assertAlmostEqual(lower, 0.0)
        self.assertGreater(upper, 0.05)

    def test_bootstrap_pairs_and_seed(self):
        args = ([True, False], [True, False], [False, False])
        first = paired_accuracy_interval(*args, seed=7, resamples=100)
        self.assertEqual(first, paired_accuracy_interval(*args, seed=7, resamples=100))
        self.assertEqual(first["accuracy_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()

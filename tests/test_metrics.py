import unittest

import numpy as np

from text_factors import bit_precision_recall, jaccard_similarity


class MetricTests(unittest.TestCase):
    def test_jaccard_similarity(self) -> None:
        first = np.asarray([True, True, False, False], dtype=np.bool_)
        second = np.asarray([True, False, True, False], dtype=np.bool_)
        self.assertAlmostEqual(jaccard_similarity(first, second), 1 / 3)

    def test_two_empty_codes_are_identical(self) -> None:
        empty = np.zeros(4, dtype=np.bool_)
        self.assertEqual(jaccard_similarity(empty, empty), 1.0)

    def test_bit_precision_and_recall(self) -> None:
        prediction = np.asarray([True, True, False, False], dtype=np.bool_)
        target = np.asarray([True, False, True, False], dtype=np.bool_)
        self.assertEqual(bit_precision_recall(prediction, target), (0.5, 0.5))

    def test_metrics_reject_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            jaccard_similarity(np.zeros(2, dtype=np.bool_), np.zeros(3, dtype=np.bool_))


if __name__ == "__main__":
    unittest.main()

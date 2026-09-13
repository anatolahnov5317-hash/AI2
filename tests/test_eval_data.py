import unittest
from collections import Counter

from text_factors.evaluation.data import (
    ALPHABET,
    POSITIVE_ENDPOINT_PAIRS,
    WINDOW_LENGTH,
    dataset_hash,
    make_dataset,
)


class EvaluationDataTests(unittest.TestCase):
    def test_reproducible_and_seed_sensitive(self) -> None:
        first = make_dataset(seed=17)
        second = make_dataset(seed=17)
        third = make_dataset(seed=18)

        self.assertEqual(first, second)
        self.assertEqual(dataset_hash(first), dataset_hash(second))
        self.assertNotEqual(first, third)
        self.assertRegex(dataset_hash(first), r"^[0-9a-f]{64}$")

    def test_windows_are_well_formed_and_globally_disjoint(self) -> None:
        dataset = make_dataset()
        partitions = [
            set(dataset.train),
            {case.text for case in dataset.dev},
            {case.text for case in dataset.test},
            set(dataset.noise),
        ]
        expected_total = sum(len(partition) for partition in partitions)

        self.assertEqual(len(set().union(*partitions)), expected_total)
        for partition in partitions:
            for text in partition:
                self.assertEqual(len(text), WINDOW_LENGTH)
                self.assertLessEqual(set(text), set(ALPHABET))

    def test_train_is_positive_only_and_exposes_all_known_pairs(self) -> None:
        dataset = make_dataset(seed=4)
        pairs = Counter((text[0], text[-1]) for text in dataset.train)

        self.assertEqual(set(pairs), set(POSITIVE_ENDPOINT_PAIRS))
        self.assertLessEqual(max(pairs.values()) - min(pairs.values()), 1)

    def test_eval_splits_balance_classes_and_match_endpoint_marginals(self) -> None:
        dataset = make_dataset(seed=9, dev_size=34, test_size=66)
        valid_pairs = set(POSITIVE_ENDPOINT_PAIRS)

        for split in (dataset.dev, dataset.test):
            positives = [case for case in split if case.label]
            negatives = [case for case in split if not case.label]
            self.assertEqual(len(positives), len(negatives))
            self.assertTrue(all(case.family == "positive" for case in positives))
            self.assertTrue(all(case.family == "negative" for case in negatives))
            self.assertTrue(
                all((case.text[0], case.text[-1]) in valid_pairs for case in positives)
            )
            self.assertTrue(
                all(
                    (case.text[0], case.text[-1]) not in valid_pairs
                    for case in negatives
                )
            )
            self.assertEqual(
                Counter(case.text[0] for case in positives),
                Counter(case.text[0] for case in negatives),
            )
            self.assertEqual(
                Counter(case.text[-1] for case in positives),
                Counter(case.text[-1] for case in negatives),
            )

    def test_default_splits_cover_all_wrong_endpoint_pairs(self) -> None:
        valid_pairs = set(POSITIVE_ENDPOINT_PAIRS)
        all_left = {left for left, _ in POSITIVE_ENDPOINT_PAIRS}
        all_right = {right for _, right in POSITIVE_ENDPOINT_PAIRS}
        expected_wrong = {
            (left, right)
            for left in all_left
            for right in all_right
            if (left, right) not in valid_pairs
        }

        for seed in (7, 17, 42):
            dataset = make_dataset(seed=seed)
            for split in (dataset.dev, dataset.test):
                observed = {
                    (case.text[0], case.text[-1]) for case in split if not case.label
                }
                self.assertEqual(observed, expected_wrong)

    def test_nonmultiple_balanced_blocks_preserve_marginals(self) -> None:
        for size in (10, 12, 14):
            dataset = make_dataset(seed=size, dev_size=size, test_size=size)
            for split in (dataset.dev, dataset.test):
                positives = [case.text for case in split if case.label]
                negatives = [case.text for case in split if not case.label]
                self.assertEqual(
                    Counter(text[0] for text in positives),
                    Counter(text[0] for text in negatives),
                )
                self.assertEqual(
                    Counter(text[-1] for text in positives),
                    Counter(text[-1] for text in negatives),
                )
                self.assertTrue(
                    all(
                        (text[0], text[-1]) not in set(POSITIVE_ENDPOINT_PAIRS)
                        for text in negatives
                    )
                )

    def test_invalid_and_impossible_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_size"):
            make_dataset(train_size=3)
        with self.assertRaisesRegex(ValueError, "dev_size must be even"):
            make_dataset(dev_size=9)
        with self.assertRaisesRegex(ValueError, "positive capacity"):
            make_dataset(train_size=2048, dev_size=8, test_size=8)
        with self.assertRaisesRegex(ValueError, "complete unique window capacity"):
            make_dataset(noise_size=len(ALPHABET) ** WINDOW_LENGTH)
        with self.assertRaisesRegex(TypeError, "integer"):
            make_dataset(test_size=32.0)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_dataset(seed=-1)


if __name__ == "__main__":
    unittest.main()

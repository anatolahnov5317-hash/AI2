import unittest
from itertools import product
from math import isfinite, log

import numpy as np

from text_factors import ModelConfig, SparseSymbolEncoder
from text_factors.evaluation.baselines import (
    ExactMemory,
    NearestSDRMemory,
    NGramMemory,
    PairAssociationMemory,
)
from text_factors.evaluation.data import make_dataset


def cyclic_encoder() -> SparseSymbolEncoder:
    config = ModelConfig(
        input_bits=6,
        active_bits_per_symbol=1,
        positions=2,
        frame_size=2,
        context_count=2,
        receptive_bits=2,
        point_count=2,
        output_bits=2,
        create_threshold=2,
        activation_threshold=1,
        min_active_points=1,
        probation_after=2,
        stable_after=3,
    )
    codebook = np.arange(6, dtype=np.int32).reshape(3, 2, 1)
    return SparseSymbolEncoder(config, alphabet="ab", codebook=codebook)


class ExactMemoryTests(unittest.TestCase):
    def test_exact_scores_and_errors(self) -> None:
        memory = ExactMemory()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            memory.score("seen")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            memory.fit([])

        memory.fit(["seen", "also-seen"])
        self.assertEqual(memory.score("seen"), 1.0)
        self.assertEqual(memory.score("unseen"), 0.0)

    def test_empty_query_has_a_finite_exact_score(self) -> None:
        memory = ExactMemory()
        memory.fit(["not-empty"])
        self.assertEqual(memory.score(""), 0.0)


class NGramMemoryTests(unittest.TestCase):
    def test_smoothed_bigram_scores_have_exact_values(self) -> None:
        memory = NGramMemory(n=2)
        memory.fit(["ab"])

        # Training grams are <START>-a, a-b, and b-<END>.  Together with the
        # unknown bucket, add-one smoothing gives denominator 3 + 4 = 7.
        self.assertAlmostEqual(memory.score("ab"), log(2.0 / 7.0))
        self.assertAlmostEqual(memory.score("xy"), log(1.0 / 7.0))
        expected_partial = (log(2.0 / 7.0) + 2 * log(1.0 / 7.0)) / 3
        self.assertAlmostEqual(memory.score("ax"), expected_partial)

    def test_unseen_ties_and_empty_text_are_finite(self) -> None:
        memory = NGramMemory(n=2)
        memory.fit(["ab", "cd"])

        self.assertAlmostEqual(memory.score("ax"), memory.score("cy"))
        self.assertTrue(isfinite(memory.score("")))

    def test_configuration_and_fit_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            NGramMemory(n=0)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            NGramMemory(smoothing=float("inf"))
        memory = NGramMemory()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            memory.score("x")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            memory.fit([])


def planted_pair_training(positions: tuple[int, int]) -> list[str]:
    """Make a full factorial design with dependence only at ``positions``."""

    other_positions = [position for position in range(5) if position not in positions]
    texts: list[str] = []
    for left_symbol, right_symbol in zip("abcd", "wxyz", strict=True):
        for filler in product("01", repeat=len(other_positions)):
            characters = [""] * 5
            characters[positions[0]] = left_symbol
            characters[positions[1]] = right_symbol
            for position, symbol in zip(other_positions, filler, strict=True):
                characters[position] = symbol
            texts.append("".join(characters))
    return texts


class PairAssociationMemoryTests(unittest.TestCase):
    def test_recovers_endpoint_pair_and_scores_unseen_fillers(self) -> None:
        memory = PairAssociationMemory()
        training = planted_pair_training((0, 4))
        memory.fit(training)

        self.assertEqual(memory.selected_positions, (0, 4))
        self.assertAlmostEqual(memory.dependence_score or 0.0, log(4.0))
        familiar = "a!@#w"
        wrong = "a!@#x"
        self.assertNotIn(familiar, training)
        self.assertNotIn(wrong, training)
        self.assertGreater(memory.score(familiar), memory.score(wrong))

    def test_recovers_nonendpoint_pair_without_position_constants(self) -> None:
        memory = PairAssociationMemory()
        training = planted_pair_training((1, 3))
        memory.fit(training)

        self.assertEqual(memory.selected_positions, (1, 3))
        familiar = "!a@w#"
        wrong = "!a@x#"
        self.assertNotIn(familiar, training)
        self.assertGreater(memory.score(familiar), memory.score(wrong))

    def test_smoothed_scores_have_exact_values_and_unseen_is_finite(self) -> None:
        memory = PairAssociationMemory()
        memory.fit(["ax", "ax", "by", "by"])

        self.assertEqual(memory.selected_positions, (0, 1))
        self.assertAlmostEqual(memory.score("ax"), log(3.0 / 5.0))
        self.assertAlmostEqual(memory.score("ay"), log(1.0 / 5.0))
        self.assertAlmostEqual(memory.score("zz"), log(1.0 / 3.0))
        self.assertTrue(isfinite(memory.score("zz")))

    def test_ties_validation_and_unfitted_use(self) -> None:
        memory = PairAssociationMemory()
        with self.assertRaisesRegex(RuntimeError, "fit"):
            memory.score("ab")
        with self.assertRaisesRegex(ValueError, "at least two training"):
            memory.fit(["ab"])
        with self.assertRaisesRegex(ValueError, "fixed-length"):
            memory.fit(["ab", "abc"])
        with self.assertRaisesRegex(ValueError, "length of at least two"):
            memory.fit(["a", "b"])

        memory.fit(["abc", "abc"])
        self.assertEqual(memory.selected_positions, (0, 1))
        with self.assertRaisesRegex(ValueError, "fitted length"):
            memory.score("ab")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            PairAssociationMemory(smoothing=0)


class NearestSDRMemoryTests(unittest.TestCase):
    def test_all_cyclic_interpretations_and_jaccard_are_used(self) -> None:
        memory = NearestSDRMemory(cyclic_encoder())
        memory.fit(["ab"])

        self.assertEqual(memory.score("ab"), 1.0)
        # With two positions, "ba" is the cyclic position shift of "ab".
        self.assertEqual(memory.score("ba"), 1.0)
        self.assertEqual(memory.score("a"), 0.5)
        self.assertEqual(memory.score(""), 0.0)

    def test_empty_training_and_unfitted_use_are_rejected(self) -> None:
        memory = NearestSDRMemory(cyclic_encoder())
        with self.assertRaisesRegex(RuntimeError, "fit"):
            memory.score("ab")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            memory.fit([])

    def test_baselines_accept_the_same_positive_only_input_budget(self) -> None:
        dataset = make_dataset(seed=2, train_size=8, dev_size=8, test_size=8)
        encoder = SparseSymbolEncoder(ModelConfig(seed=2), alphabet="abcdefgh")
        baselines = [ExactMemory(), NGramMemory(), NearestSDRMemory(encoder)]

        for baseline in baselines:
            baseline.fit(dataset.train)
            scores = [baseline.score(case.text) for case in dataset.dev]
            self.assertEqual(len(scores), len(dataset.dev))
            self.assertTrue(all(isfinite(score) for score in scores))


if __name__ == "__main__":
    unittest.main()

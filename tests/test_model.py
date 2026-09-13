import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from text_factors import ModelConfig, TextFactorModel
from text_factors.memory import CombinatorialMemory


def model_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 64,
        "active_bits_per_symbol": 4,
        "positions": 5,
        "frame_size": 3,
        "context_count": 5,
        "receptive_bits": 16,
        "point_count": 256,
        "output_bits": 16,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
        "max_clusters_per_point": 8,
        "seed": 7,
    }
    values.update(overrides)
    return ModelConfig(**values)


class TextFactorModelTests(unittest.TestCase):
    def test_invalid_text_and_epochs_do_not_change_model(self) -> None:
        model = TextFactorModel(model_config(), alphabet="abc")
        before = model.summary()
        for value in (None, [], b"", 0, False):
            with self.assertRaises(ValueError):
                model.fit_text(cast(Any, value))
            with self.assertRaises(ValueError):
                model.transform_text(cast(Any, value))
            with self.assertRaises(ValueError):
                model.partial_fit_window(cast(Any, value))
            self.assertEqual(model.summary(), before)
        for epochs in (True, 1.5, 0, -1):
            with self.assertRaises(ValueError):
                model.fit_text("abc", epochs=cast(Any, epochs))
            self.assertEqual(model.summary(), before)

    def test_repeated_window_produces_stable_explainable_factors(self) -> None:
        model = TextFactorModel(model_config(), alphabet="abc")
        for _ in range(3):
            model.partial_fit_window("abc")

        self.assertGreater(model.memory.stats()["stable_clusters"], 0)
        result = model.transform_window("abc")
        self.assertGreater(len(result.output_bits), 0)
        factors = model.top_factors(5)
        self.assertGreater(len(factors), 0)
        self.assertGreater(len(model.explain_factor(factors[0].output_bit)), 0)

    def test_transform_does_not_mutate_memory(self) -> None:
        model = TextFactorModel(model_config(), alphabet="abc")
        for _ in range(3):
            model.partial_fit_window("abc")
        before = model.memory.stats()
        model.transform_window("abc")
        after = model.memory.stats()
        self.assertEqual(before, after)

    def test_same_seed_produces_same_model_behavior(self) -> None:
        first = TextFactorModel(model_config(seed=11), alphabet="abc")
        second = TextFactorModel(model_config(seed=11), alphabet="abc")
        for model in (first, second):
            model.fit_text("abcabcabc", epochs=3)

        np.testing.assert_array_equal(first.memory.receptors, second.memory.receptors)
        self.assertEqual(first.summary(), second.summary())
        self.assertEqual(
            first.transform_window("abc").to_dict(),
            second.transform_window("abc").to_dict(),
        )

    def test_model_round_trip_preserves_predictions(self) -> None:
        model = TextFactorModel(model_config(), alphabet="abc")
        model.fit_text("abcabcabc", epochs=3)
        expected = model.transform_window("abc").to_dict()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = TextFactorModel.load(path)

        self.assertEqual(loaded.transform_window("abc").to_dict(), expected)
        self.assertEqual(loaded.summary(), model.summary())

    def test_context_transform_mode_learns_only_target_bits(self) -> None:
        config = model_config(
            input_bits=32,
            output_bits=32,
            active_bits_per_symbol=4,
            receptive_bits=8,
            point_count=512,
            create_threshold=2,
            activation_threshold=2,
            prediction_vote_threshold=1,
        )
        model = TextFactorModel(config, alphabet="ab")
        for _ in range(3):
            model.learn_context_transform("ab")

        prediction = model.predict_context_transform("ab")
        target = model.encoder.encode_window("ab", context=1)
        self.assertGreater(int(np.count_nonzero(prediction.output)), 0)
        self.assertTrue(np.all(np.logical_not(prediction.output) | target))

    def test_training_modes_and_context_operator_cannot_be_mixed(self) -> None:
        config = model_config(input_bits=64, output_bits=64)
        unsupervised = TextFactorModel(config, alphabet="abc")
        unsupervised.partial_fit_window("abc")
        self.assertEqual(unsupervised.training_mode, "unsupervised")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            unsupervised.predict_context_transform("abc")
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            unsupervised.learn_context_transform("abc")

        supervised = TextFactorModel(config, alphabet="abc")
        supervised.learn_context_transform("abc", source_context=0, target_context=1)
        self.assertEqual(supervised.training_mode, "supervised")
        self.assertEqual(supervised.context_pair, (0, 1))
        with self.assertRaisesRegex(ValueError, "operators"):
            supervised.learn_context_transform(
                "abc", source_context=1, target_context=2
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            supervised.predict_context_transform("abc", source_context=1)
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            supervised.partial_fit_window("abc")

    def test_populated_injected_memory_requires_explicit_mode(self) -> None:
        config = model_config()
        original = TextFactorModel(config, alphabet="abc")
        original.partial_fit_window("abc")
        injected = TextFactorModel(
            config,
            alphabet="abc",
            memory=original.memory,
        )

        self.assertEqual(injected.training_mode, "unknown")
        with self.assertRaisesRegex(ValueError, "assume_training_mode"):
            injected.partial_fit_window("abc")
        injected.assume_training_mode("unsupervised")
        injected.partial_fit_window("abc")

        empty = TextFactorModel(
            config,
            alphabet="abc",
            memory=CombinatorialMemory(config),
        )
        self.assertEqual(empty.training_mode, "untrained")


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from text_factors import ModelConfig, TextFactorModel


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


if __name__ == "__main__":
    unittest.main()

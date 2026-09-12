import unittest

from text_factors import ModelConfig


class ModelConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        config = ModelConfig()
        self.assertEqual(config.input_bits, 256)
        self.assertEqual(config.receptive_bits, 32)

    def test_rejects_invalid_threshold_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation_threshold"):
            ModelConfig(activation_threshold=7, create_threshold=6)

    def test_rejects_impossible_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_size"):
            ModelConfig(frame_size=11, positions=10)

    def test_round_trip_dict(self) -> None:
        config = ModelConfig(point_count=123, seed=9)
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from text_factors import ModelConfig, SparseSymbolEncoder


def encoder_config(**overrides: int) -> ModelConfig:
    values = {
        "input_bits": 64,
        "active_bits_per_symbol": 4,
        "positions": 5,
        "frame_size": 3,
        "context_count": 5,
        "receptive_bits": 8,
        "point_count": 16,
        "output_bits": 8,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
    }
    values.update(overrides)
    return ModelConfig(**values)


class SparseSymbolEncoderTests(unittest.TestCase):
    def test_codebook_is_reproducible_and_sparse(self) -> None:
        first = SparseSymbolEncoder(encoder_config(seed=17), alphabet="abc")
        second = SparseSymbolEncoder(encoder_config(seed=17), alphabet="abc")
        third = SparseSymbolEncoder(encoder_config(seed=18), alphabet="abc")

        np.testing.assert_array_equal(first.codebook, second.codebook)
        self.assertFalse(np.array_equal(first.codebook, third.codebook))
        for bits in first.codebook.reshape(-1, 4):
            self.assertEqual(len(np.unique(bits)), 4)

    def test_context_is_a_cyclic_position_shift(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), alphabet="abc")
        shifted = encoder.encode_window("ab", context=1)
        explicit = encoder.encode_positions([("a", 1), ("b", 2)])
        np.testing.assert_array_equal(shifted, explicit)

    def test_unknown_characters_are_separators(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), alphabet="abc")
        encoded = encoder.encode_window(" ! ")
        self.assertEqual(int(np.count_nonzero(encoded)), 0)

    def test_window_length_is_checked(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), alphabet="abc")
        with self.assertRaisesRegex(ValueError, "maximum"):
            encoder.encode_window("abca")

    def test_iter_windows_keeps_absolute_cyclic_offset(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), alphabet="abc")
        self.assertEqual(
            list(encoder.iter_windows("abcabc", stride=2)),
            [("abc", 0), ("cab", 2)],
        )


if __name__ == "__main__":
    unittest.main()

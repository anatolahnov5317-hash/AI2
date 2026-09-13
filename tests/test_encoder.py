import unittest
from typing import Any, cast

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

    def test_default_preserves_digits_and_operators(self) -> None:
        encoder = SparseSymbolEncoder(ModelConfig())
        for first, second in [("a+b", "a-b"), ("a1b", "a9b"), ("кто?", "кто!")]:
            self.assertFalse(
                np.array_equal(
                    encoder.encode_window(first), encoder.encode_window(second)
                )
            )
        self.assertGreater(int(encoder.encode_window("12345").sum()), 0)

    def test_impossible_codebook_fails_before_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity"):
            SparseSymbolEncoder(ModelConfig(active_bits_per_symbol=256))

    def test_small_full_code_space_terminates(self) -> None:
        config = ModelConfig(
            input_bits=3,
            active_bits_per_symbol=1,
            positions=1,
            frame_size=1,
            context_count=1,
            receptive_bits=1,
            point_count=1,
            output_bits=1,
            create_threshold=1,
            activation_threshold=1,
            min_active_points=1,
        )
        encoder = SparseSymbolEncoder(config, "ab")
        self.assertEqual(len(np.unique(encoder.codebook)), 3)

    def test_extension_preserves_known_codes(self) -> None:
        first = SparseSymbolEncoder(encoder_config(), "abc")
        second = SparseSymbolEncoder(encoder_config(), "abc")
        expected = first.encode_window("abc").copy()
        first.extend_alphabet("12+")
        second.extend_alphabet("12+")
        np.testing.assert_array_equal(first.encode_window("abc"), expected)
        np.testing.assert_array_equal(first.codebook, second.codebook)
        self.assertGreater(int(first.encode_window("1+2").sum()), 0)

    def test_supplied_codebook_rejects_invalid_indices(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), "abc")
        for mode in ("float", "duplicate_bit", "duplicate_code", "overflow"):
            array = encoder.codebook.astype(np.int64).copy()
            if mode == "float":
                array = array.astype(float) + 0.1
            elif mode == "duplicate_bit":
                array[0, 0, 1] = array[0, 0, 0]
            elif mode == "duplicate_code":
                array[0, 1] = array[0, 0]
            else:
                array[0, 0, 0] = 2**32
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SparseSymbolEncoder(encoder.config, "abc", codebook=cast(Any, array))

    def test_explicit_positions_and_offsets_validated(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), "abc")
        for position in (-1, 5, 0.5, True):
            with self.assertRaises(ValueError):
                encoder.encode_positions([("a", cast(Any, position))])
        with self.assertRaises(ValueError):
            encoder.encode_positions([("a", 0)], context=5)
        with self.assertRaises(ValueError):
            encoder.encode_window("a", offset=-1)

    def test_streaming_chunks_equal_one_document(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), "abc")
        for text in ("", "a", "ab", "abc", "abcabcabcabc"):
            for stride in (1, 2, 4):
                expected = list(encoder.iter_windows(text, stride=stride))
                for split in range(len(text) + 1):
                    actual = list(
                        encoder.iter_chunked_windows(
                            [text[:split], "", text[split:]], stride=stride
                        )
                    )
                    self.assertEqual(actual, expected)

    def test_text_apis_reject_non_strings_including_empty_values(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), "abc")
        for value in (None, [], ["a"], b"", b"a", 0, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encoder.encode_window(cast(Any, value))
                with self.assertRaises(ValueError):
                    list(encoder.iter_windows(cast(Any, value)))
                with self.assertRaises(ValueError):
                    encoder.normalize_char(cast(Any, value))

    def test_unknown_symbol_does_not_skip_position_validation(self) -> None:
        encoder = SparseSymbolEncoder(encoder_config(), "abc")
        with self.assertRaises(ValueError):
            encoder.concept_bits("?", -1)


if __name__ == "__main__":
    unittest.main()

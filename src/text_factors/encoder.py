"""Deterministic sparse symbol and position encoding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from string import ascii_lowercase

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig

CYRILLIC_LOWERCASE = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
DEFAULT_ALPHABET = ascii_lowercase + CYRILLIC_LOWERCASE
UNKNOWN_SYMBOL = "_"


@dataclass(frozen=True, slots=True)
class EncodedWindow:
    """A text window and its position-aware sparse representation."""

    text: str
    offset: int
    context: int
    bits: NDArray[np.bool_]


class SparseSymbolEncoder:
    """Map ``(symbol, position)`` pairs to reproducible sparse vectors.

    Unknown characters are treated as separators and produce no active bits,
    matching the useful behavior of the source experiment without conflating
    every punctuation mark and space into one learned symbol.
    """

    def __init__(
        self,
        config: ModelConfig,
        alphabet: str = DEFAULT_ALPHABET,
        *,
        codebook: NDArray[np.integer] | None = None,
    ) -> None:
        self.config = config
        normalized = tuple(dict.fromkeys(ch.casefold() for ch in alphabet))
        if any(len(ch) != 1 for ch in normalized):
            raise ValueError("alphabet must contain single-character symbols")
        if UNKNOWN_SYMBOL in normalized:
            normalized = tuple(ch for ch in normalized if ch != UNKNOWN_SYMBOL)
        if not normalized:
            raise ValueError("alphabet cannot be empty")

        self.alphabet = "".join(normalized)
        self.symbols = normalized + (UNKNOWN_SYMBOL,)
        self.symbol_to_index = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }

        expected_shape = (
            len(self.symbols),
            config.positions,
            config.active_bits_per_symbol,
        )
        if codebook is None:
            self.codebook = self._make_codebook(expected_shape)
        else:
            converted = np.asarray(codebook, dtype=np.int32)
            if converted.shape != expected_shape:
                raise ValueError(
                    f"codebook shape must be {expected_shape}, got {converted.shape}"
                )
            if np.any(converted < 0) or np.any(converted >= config.input_bits):
                raise ValueError("codebook contains an out-of-range bit index")
            self.codebook = converted.copy()

    def _make_codebook(self, shape: tuple[int, int, int]) -> NDArray[np.int32]:
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 0]))
        result = np.empty(shape, dtype=np.int32)
        signatures: set[tuple[int, ...]] = set()

        for symbol_index in range(shape[0]):
            for position in range(shape[1]):
                while True:
                    bits = np.sort(
                        rng.choice(
                            self.config.input_bits,
                            size=self.config.active_bits_per_symbol,
                            replace=False,
                        )
                    ).astype(np.int32)
                    signature = tuple(int(bit) for bit in bits)
                    if signature not in signatures:
                        signatures.add(signature)
                        result[symbol_index, position] = bits
                        break
        return result

    def normalize_char(self, character: str) -> str:
        if len(character) != 1:
            raise ValueError("normalize_char expects exactly one character")
        normalized = character.casefold()
        if len(normalized) == 1 and normalized in self.symbol_to_index:
            return normalized
        return UNKNOWN_SYMBOL

    def encode_window(
        self,
        text: str,
        *,
        context: int = 0,
        offset: int = 0,
    ) -> NDArray[np.bool_]:
        """Encode one window, applying a cyclic positional context shift."""

        if len(text) > self.config.frame_size:
            maximum = self.config.frame_size
            raise ValueError(f"window length {len(text)} exceeds maximum {maximum}")
        if not 0 <= context < self.config.context_count:
            raise ValueError(f"context must be in [0, {self.config.context_count})")

        encoded = np.zeros(self.config.input_bits, dtype=np.bool_)
        for local_position, character in enumerate(text):
            symbol = self.normalize_char(character)
            if symbol == UNKNOWN_SYMBOL:
                continue
            position = (offset + local_position + context) % self.config.positions
            symbol_index = self.symbol_to_index[symbol]
            encoded[self.codebook[symbol_index, position]] = True
        return encoded

    def interpretations(self, text: str, *, offset: int = 0) -> NDArray[np.bool_]:
        return np.stack(
            [
                self.encode_window(text, context=context, offset=offset)
                for context in range(self.config.context_count)
            ],
            axis=0,
        )

    def iter_windows(
        self,
        text: str,
        *,
        stride: int = 1,
    ) -> Iterator[tuple[str, int]]:
        """Yield overlapping windows and their cyclic absolute offsets."""

        if stride <= 0:
            raise ValueError("stride must be positive")
        if not text:
            return
        if len(text) <= self.config.frame_size:
            yield text, 0
            return

        final_start = len(text) - self.config.frame_size
        for start in range(0, final_start + 1, stride):
            yield (
                text[start : start + self.config.frame_size],
                start % self.config.positions,
            )

    def concept_bits(self, symbol: str, position: int) -> NDArray[np.int32]:
        normalized = self.normalize_char(symbol)
        if normalized == UNKNOWN_SYMBOL:
            return np.empty(0, dtype=np.int32)
        if not 0 <= position < self.config.positions:
            raise ValueError(f"position must be in [0, {self.config.positions})")
        return self.codebook[self.symbol_to_index[normalized], position].copy()

    def concepts(self) -> Iterator[tuple[str, int, NDArray[np.int32]]]:
        for symbol in self.symbols[:-1]:
            for position in range(self.config.positions):
                yield symbol, position, self.concept_bits(symbol, position)

    def encode_positions(
        self,
        symbols: Sequence[tuple[str, int]],
        *,
        context: int = 0,
    ) -> NDArray[np.bool_]:
        """Encode explicitly positioned symbols; useful in controlled experiments."""

        encoded = np.zeros(self.config.input_bits, dtype=np.bool_)
        for symbol, position in symbols:
            normalized = self.normalize_char(symbol)
            if normalized == UNKNOWN_SYMBOL:
                continue
            shifted = (position + context) % self.config.positions
            encoded[self.codebook[self.symbol_to_index[normalized], shifted]] = True
        return encoded

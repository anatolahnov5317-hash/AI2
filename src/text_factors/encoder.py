"""Deterministic sparse symbol and position encoding."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from string import ascii_lowercase, digits, punctuation

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig

CYRILLIC_LOWERCASE = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
LEGACY_ALPHABET = ascii_lowercase + CYRILLIC_LOWERCASE
# '_' remains the reserved unknown/separator marker for v1 compatibility.
# Whitespace remains a positional separator; case is folded deliberately.
DEFAULT_ALPHABET = LEGACY_ALPHABET + digits + punctuation.replace("_", "")
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
        normalized = self._normalize_alphabet(alphabet)
        self._set_alphabet(normalized)

        expected_shape = (
            len(self.symbols),
            config.positions,
            config.active_bits_per_symbol,
        )
        self._check_capacity(len(self.symbols))
        if codebook is None:
            self.codebook = self._make_codebook(expected_shape)
        else:
            raw = np.asarray(codebook)
            if raw.dtype.kind not in "iu":
                raise ValueError("codebook must contain integer indices")
            if raw.shape != expected_shape:
                raise ValueError(
                    f"codebook shape must be {expected_shape}, got {raw.shape}"
                )
            if np.any(raw < 0) or np.any(raw >= config.input_bits):
                raise ValueError("codebook contains an out-of-range bit index")
            converted = raw.astype(np.int32)
            if np.any(np.diff(converted, axis=2) <= 0):
                raise ValueError("codebook rows must be sorted and unique")
            signatures = converted.reshape(-1, config.active_bits_per_symbol)
            if len(np.unique(signatures, axis=0)) != len(signatures):
                raise ValueError("codebook contains duplicate symbol-position codes")
            self.codebook = converted.copy()

    @staticmethod
    def _normalize_alphabet(alphabet: str) -> tuple[str, ...]:
        if not isinstance(alphabet, str):
            raise ValueError("alphabet must be a string")
        normalized = tuple(dict.fromkeys(ch.casefold() for ch in alphabet))
        if any(len(ch) != 1 for ch in normalized):
            raise ValueError("alphabet must contain single-character symbols")
        if UNKNOWN_SYMBOL in normalized:
            normalized = tuple(ch for ch in normalized if ch != UNKNOWN_SYMBOL)
        if not normalized:
            raise ValueError("alphabet cannot be empty")
        return normalized

    def _set_alphabet(self, normalized: tuple[str, ...]) -> None:
        self.alphabet = "".join(normalized)
        self.symbols = normalized + (UNKNOWN_SYMBOL,)
        self.symbol_to_index = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }

    def _check_capacity(self, symbol_count: int) -> None:
        capacity = comb(self.config.input_bits, self.config.active_bits_per_symbol)
        required = symbol_count * self.config.positions
        if required > capacity:
            raise ValueError(
                f"codebook requires {required} unique codes, but capacity is {capacity}"
            )

    def _make_codebook(self, shape: tuple[int, int, int]) -> NDArray[np.int32]:
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 0]))
        result = np.empty(shape, dtype=np.int32)
        signatures: set[tuple[int, ...]] = set()
        fallback = combinations(range(self.config.input_bits), shape[2])
        for symbol_index in range(shape[0]):
            for position in range(shape[1]):
                for _ in range(64):
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
                else:
                    # Bounded rejection sampling, even near full code-space use.
                    signature = next(row for row in fallback if row not in signatures)
                    signatures.add(signature)
                    result[symbol_index, position] = signature
        return result

    def extend_alphabet(self, alphabet: str) -> None:
        """Append symbols without changing existing known-symbol codes or indices.

        Extension is deterministic for the same sequence of calls. Constructing
        a fresh encoder with another alphabet is NOT a migration procedure.
        Existing models should use this method and persist the full codebook.
        """
        additions = tuple(
            ch
            for ch in self._normalize_alphabet(alphabet)
            if ch not in self.symbol_to_index
        )
        if not additions:
            return
        self._check_capacity(len(self.symbols) + len(additions))
        signatures = {
            tuple(int(bit) for bit in row)
            for row in self.codebook.reshape(-1, self.config.active_bits_per_symbol)
        }
        rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 2, len(self.symbols)])
        )
        extra = np.empty(
            (len(additions), self.config.positions, self.config.active_bits_per_symbol),
            dtype=np.int32,
        )
        fallback = combinations(
            range(self.config.input_bits), self.config.active_bits_per_symbol
        )
        for symbol_index in range(len(additions)):
            for position in range(self.config.positions):
                for _ in range(64):
                    row = tuple(
                        sorted(
                            int(bit)
                            for bit in rng.choice(
                                self.config.input_bits,
                                self.config.active_bits_per_symbol,
                                replace=False,
                            )
                        )
                    )
                    if row not in signatures:
                        break
                else:
                    row = next(row for row in fallback if row not in signatures)
                signatures.add(row)
                extra[symbol_index, position] = row
        self.codebook = np.concatenate(
            [self.codebook[:-1], extra, self.codebook[-1:]], axis=0
        )
        self._set_alphabet(tuple(self.alphabet) + additions)

    def normalize_char(self, character: str) -> str:
        if not isinstance(character, str) or len(character) != 1:
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

        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if len(text) > self.config.frame_size:
            maximum = self.config.frame_size
            raise ValueError(f"window length {len(text)} exceeds maximum {maximum}")
        if type(context) is not int or not 0 <= context < self.config.context_count:
            raise ValueError(f"context must be in [0, {self.config.context_count})")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")

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

        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if type(stride) is not int or stride <= 0:
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
        if type(position) is not int or not 0 <= position < self.config.positions:
            raise ValueError(f"position must be in [0, {self.config.positions})")
        normalized = self.normalize_char(symbol)
        if normalized == UNKNOWN_SYMBOL:
            return np.empty(0, dtype=np.int32)
        return self.codebook[self.symbol_to_index[normalized], position].copy()

    def iter_chunked_windows(
        self,
        chunks: Iterable[str],
        *,
        stride: int = 1,
    ) -> Iterator[tuple[str, int]]:
        """Stream chunks as one document, preserving boundaries and cyclic offsets.

        Equivalent to iter_windows(''.join(chunks)), without materializing the
        whole document. Separate fit_text calls instead represent separate texts.
        """
        if type(stride) is not int or stride <= 0:
            raise ValueError("stride must be positive")
        window: deque[str] = deque(maxlen=self.config.frame_size)
        count = 0
        for chunk in chunks:
            if not isinstance(chunk, str):
                raise ValueError("chunks must contain strings")
            for character in chunk:
                window.append(character)
                count += 1
                start = count - self.config.frame_size
                if start >= 0 and start % stride == 0:
                    yield "".join(window), start % self.config.positions
        if 0 < count < self.config.frame_size:
            yield "".join(window), 0

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

        if type(context) is not int or not 0 <= context < self.config.context_count:
            raise ValueError(f"context must be in [0, {self.config.context_count})")
        encoded = np.zeros(self.config.input_bits, dtype=np.bool_)
        for symbol, position in symbols:
            if type(position) is not int or not 0 <= position < self.config.positions:
                raise ValueError(f"position must be in [0, {self.config.positions})")
            normalized = self.normalize_char(symbol)
            if normalized == UNKNOWN_SYMBOL:
                continue
            shifted = (position + context) % self.config.positions
            encoded[self.codebook[self.symbol_to_index[normalized], shifted]] = True
        return encoded

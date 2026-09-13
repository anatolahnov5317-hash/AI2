"""Label-free familiarity baselines for the controlled evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from ..encoder import SparseSymbolEncoder


def _training_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not one string")
    converted = tuple(texts)
    if not converted:
        raise ValueError("training texts cannot be empty")
    if any(not isinstance(text, str) for text in converted):
        raise TypeError("every training text must be a string")
    return converted


class ExactMemory:
    """Score one for a memorized string and zero for every unseen string."""

    def __init__(self) -> None:
        self._texts: frozenset[str] | None = None

    def fit(self, texts: Sequence[str]) -> None:
        self._texts = frozenset(_training_texts(texts))

    def score(self, text: str) -> float:
        if self._texts is None:
            raise RuntimeError("ExactMemory must be fit before scoring")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return float(text in self._texts)


NGram = tuple[str, ...]
_START = "<START>"
_END = "<END>"
_EMPTY: NGram = ("<EMPTY>",)


class NGramMemory:
    """Mean additively-smoothed log probability of character n-grams.

    Each training string is padded with ``n - 1`` start and end tokens before
    its n-grams are counted.  Counts form one empirical distribution over
    observed n-gram types plus a single unknown type.  A score is the arithmetic
    mean of its n-gram log probabilities, avoiding a length advantage.  Empty
    strings use boundary-only n-grams (or a dedicated event when ``n == 1``),
    which keeps every score finite.  This local statistic does not encode the
    task's distant endpoint-pair rule.
    """

    def __init__(self, n: int = 2, *, smoothing: float = 1.0) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be a positive finite number")
        self.n = n
        self.smoothing = float(smoothing)
        self._counts: Counter[NGram] | None = None
        self._total = 0
        self._denominator = 0.0

    def _ngrams(self, text: str) -> tuple[NGram, ...]:
        padding = self.n - 1
        tokens = (_START,) * padding + tuple(text) + (_END,) * padding
        grams = tuple(
            tuple(tokens[start : start + self.n])
            for start in range(len(tokens) - self.n + 1)
        )
        return grams or (_EMPTY,)

    def fit(self, texts: Sequence[str]) -> None:
        checked = _training_texts(texts)
        counts: Counter[NGram] = Counter()
        for text in checked:
            counts.update(self._ngrams(text))
        self._counts = counts
        self._total = sum(counts.values())
        # One additional bucket gives every unobserved n-gram the same finite
        # probability without presuming a task-specific character vocabulary.
        vocabulary_size = len(counts) + 1
        self._denominator = self._total + self.smoothing * vocabulary_size

    def score(self, text: str) -> float:
        if self._counts is None:
            raise RuntimeError("NGramMemory must be fit before scoring")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        grams = self._ngrams(text)
        log_probability = 0.0
        for gram in grams:
            numerator = self._counts.get(gram, 0) + self.smoothing
            log_probability += log(numerator / self._denominator)
        return float(log_probability / len(grams))


class PairAssociationMemory:
    """Learn and score the strongest positional association in training text.

    ``fit`` computes empirical mutual information for every pair of character
    positions and selects the maximum, breaking exact ties lexicographically.
    ``score`` returns the mean of the two additively smoothed conditional log
    probabilities for the symbols at that pair.  Selection and counts use only
    the unlabelled, positive training strings; no endpoint positions or symbols
    are built in.

    Empirical mutual information is optimistic when the training sample is
    small or the character vocabulary is large.  This is an exploratory
    relational baseline, not a reliable structure-selection procedure from a
    handful of examples.
    """

    def __init__(self, *, smoothing: float = 1.0) -> None:
        if (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, Real)
            or not np.isfinite(smoothing)
            or smoothing <= 0
        ):
            raise ValueError("smoothing must be a positive finite number")
        self.smoothing = float(smoothing)
        self.selected_positions: tuple[int, int] | None = None
        self.dependence_score: float | None = None
        self._length: int | None = None
        self._joint_counts: Counter[tuple[str, str]] | None = None
        self._left_counts: Counter[str] | None = None
        self._right_counts: Counter[str] | None = None
        self._left_vocabulary_size = 0
        self._right_vocabulary_size = 0

    @staticmethod
    def _mutual_information(
        texts: tuple[str, ...], left_position: int, right_position: int
    ) -> float:
        joint = Counter((text[left_position], text[right_position]) for text in texts)
        left = Counter(text[left_position] for text in texts)
        right = Counter(text[right_position] for text in texts)
        total = len(texts)
        information = 0.0
        for (left_symbol, right_symbol), count in joint.items():
            # Algebraically identical to p(x,y) log(p(x,y)/(p(x)p(y))),
            # arranged in integer counts to avoid unnecessary rounding.
            information += (count / total) * log(
                count * total / (left[left_symbol] * right[right_symbol])
            )
        return information

    def fit(self, texts: Sequence[str]) -> None:
        checked = _training_texts(texts)
        if len(checked) < 2:
            raise ValueError("pair association requires at least two training texts")
        lengths = {len(text) for text in checked}
        if len(lengths) != 1:
            raise ValueError("pair association requires fixed-length training texts")
        length = lengths.pop()
        if length < 2:
            raise ValueError("pair association requires text length of at least two")

        selected = (0, 1)
        best_information = self._mutual_information(checked, *selected)
        for left_position in range(length):
            for right_position in range(left_position + 1, length):
                positions = (left_position, right_position)
                information = self._mutual_information(checked, *positions)
                if information > best_information:
                    selected = positions
                    best_information = information

        left_position, right_position = selected
        pairs = [(text[left_position], text[right_position]) for text in checked]
        self.selected_positions = selected
        self.dependence_score = float(best_information)
        self._length = length
        self._joint_counts = Counter(pairs)
        self._left_counts = Counter(left for left, _ in pairs)
        self._right_counts = Counter(right for _, right in pairs)
        # Include one unknown-symbol bucket on each side, keeping scores finite
        # for characters that never occurred at a selected position.
        self._left_vocabulary_size = len(self._left_counts) + 1
        self._right_vocabulary_size = len(self._right_counts) + 1

    def score(self, text: str) -> float:
        if (
            self.selected_positions is None
            or self._length is None
            or self._joint_counts is None
            or self._left_counts is None
            or self._right_counts is None
        ):
            raise RuntimeError("PairAssociationMemory must be fit before scoring")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if len(text) != self._length:
            raise ValueError(f"text length must equal fitted length {self._length}")

        left_position, right_position = self.selected_positions
        left_symbol = text[left_position]
        right_symbol = text[right_position]
        joint_count = self._joint_counts.get((left_symbol, right_symbol), 0)
        left_denominator = self._left_counts.get(left_symbol, 0) + (
            self.smoothing * self._right_vocabulary_size
        )
        right_denominator = self._right_counts.get(right_symbol, 0) + (
            self.smoothing * self._left_vocabulary_size
        )
        right_given_left = (joint_count + self.smoothing) / left_denominator
        left_given_right = (joint_count + self.smoothing) / right_denominator
        return float((log(right_given_left) + log(left_given_right)) / 2)


class NearestSDRMemory:
    """Maximum Jaccard similarity to any stored cyclic SDR interpretation."""

    def __init__(self, encoder: SparseSymbolEncoder) -> None:
        if not isinstance(encoder, SparseSymbolEncoder):
            raise TypeError("encoder must be a SparseSymbolEncoder")
        self.encoder = encoder
        self._representations: NDArray[np.bool_] | None = None

    def fit(self, texts: Sequence[str]) -> None:
        checked = _training_texts(texts)
        encoded = [self.encoder.interpretations(text) for text in checked]
        self._representations = np.concatenate(encoded, axis=0)

    def score(self, text: str) -> float:
        if self._representations is None:
            raise RuntimeError("NearestSDRMemory must be fit before scoring")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        best = 0.0
        for query in self.encoder.interpretations(text):
            intersections = np.count_nonzero(self._representations & query, axis=1)
            unions = np.count_nonzero(self._representations | query, axis=1)
            similarities = np.divide(
                intersections,
                unions,
                out=np.ones_like(intersections, dtype=np.float64),
                where=unions != 0,
            )
            best = max(best, float(np.max(similarities)))
        return best

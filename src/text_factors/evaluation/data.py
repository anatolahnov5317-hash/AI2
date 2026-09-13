"""Controlled data for evaluating familiarity without supervised labels.

The rule is deliberately known during training: a positive window starts and
ends with one of four :data:`POSITIVE_ENDPOINT_PAIRS`.  Development and test
positives are held-out *strings* satisfying those same four associations; this
is therefore a test of held-out combinations within a known rule, not unseen
rule discovery.  Negative examples use a balanced stratified design over the
three nonidentity cyclic endpoint pairings while matching the positive class's
left- and right-endpoint marginals exactly.  This labelled design is not IID.

Interior characters are sampled independently of the class.  Noise is sampled
uniformly without replacement from the window space remaining after the
globally disjoint labelled partitions are drawn.  It is not filtered by the
endpoint rule, so accidental rule matches are possible.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from numbers import Integral

import numpy as np
from numpy.random import Generator

WINDOW_LENGTH = 5
"""Length of every generated text window."""

ALPHABET = "abcdefgh"
"""The complete, explicit alphabet used at every window position."""

POSITIVE_ENDPOINT_PAIRS: tuple[tuple[str, str], ...] = (
    ("a", "e"),
    ("b", "f"),
    ("c", "g"),
    ("d", "h"),
)
"""The four endpoint associations exposed by positive-only training."""

POSITIVE_FAMILY = "positive"
NEGATIVE_FAMILY = "negative"

_INTERIORS = tuple("".join(chars) for chars in product(ALPHABET, repeat=3))
_FULL_SPACE_SIZE = len(ALPHABET) ** WINDOW_LENGTH
_VALID_PAIR_SET = frozenset(POSITIVE_ENDPOINT_PAIRS)


@dataclass(frozen=True, slots=True)
class TextCase:
    """One labelled evaluation case."""

    text: str
    label: bool
    family: str


@dataclass(frozen=True, slots=True)
class TextDataset:
    """Positive-only training data and disjoint evaluation/noise partitions."""

    train: tuple[str, ...]
    dev: tuple[TextCase, ...]
    test: tuple[TextCase, ...]
    noise: tuple[str, ...]


def _checked_size(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return converted


def _balanced_pair_indices(size: int, rng: Generator) -> list[int]:
    """Use every valid pair, with counts differing by at most one."""

    pair_count = len(POSITIVE_ENDPOINT_PAIRS)
    quotient, remainder = divmod(size, pair_count)
    counts = np.full(pair_count, quotient, dtype=np.int64)
    if remainder:
        extra = rng.permutation(pair_count)[:remainder]
        counts[extra] += 1
    indices = np.repeat(np.arange(pair_count), counts)
    rng.shuffle(indices)
    return [int(index) for index in indices]


def _wrong_pairs_with_matched_marginals(
    positive_indices: list[int], rng: Generator
) -> list[tuple[str, str]]:
    """Stratify invalid pairings while preserving both endpoint marginals."""

    pair_count = len(POSITIVE_ENDPOINT_PAIRS)
    complete_blocks, remainder = divmod(len(positive_indices), pair_count)
    positive_counts = Counter(positive_indices)
    pairs: list[tuple[str, str]] = []

    # When there is a remainder, reserve one complete block plus the distinct
    # extra indices as a bounded tail.  A remainder of one cannot be deranged
    # by itself, whereas the resulting five-to-seven-item tail always can be.
    stratified_blocks = complete_blocks - int(bool(remainder))
    offset_counts = np.full(3, stratified_blocks // 3, dtype=np.int64)
    offset_remainder = stratified_blocks % 3
    if offset_remainder:
        chosen = rng.permutation(3)[:offset_remainder]
        offset_counts[chosen] += 1
    offsets = np.repeat(np.arange(1, 4), offset_counts)
    rng.shuffle(offsets)
    for offset_value in offsets:
        offset = int(offset_value)
        pairs.extend(
            (
                POSITIVE_ENDPOINT_PAIRS[index][0],
                POSITIVE_ENDPOINT_PAIRS[(index + offset) % pair_count][1],
            )
            for index in range(pair_count)
        )

    if remainder:
        tail_indices = list(range(pair_count))
        tail_indices.extend(
            index
            for index, count in sorted(positive_counts.items())
            if count > complete_blocks
        )
        ordered = sorted(tail_indices)
        maximum_count = max(Counter(ordered).values())
        if int(rng.integers(2)):
            shifted = ordered[maximum_count:] + ordered[:maximum_count]
        else:
            shifted = ordered[-maximum_count:] + ordered[:-maximum_count]
        pairs.extend(
            (
                POSITIVE_ENDPOINT_PAIRS[left][0],
                POSITIVE_ENDPOINT_PAIRS[right][1],
            )
            for left, right in zip(ordered, shifted, strict=True)
        )

    if any(pair in _VALID_PAIR_SET for pair in pairs):  # defensive invariant
        raise RuntimeError("failed to construct wrong endpoint pairings")
    expected_left = Counter(
        POSITIVE_ENDPOINT_PAIRS[index][0] for index in positive_indices
    )
    expected_right = Counter(
        POSITIVE_ENDPOINT_PAIRS[index][1] for index in positive_indices
    )
    if (
        Counter(left for left, _ in pairs) != expected_left
        or Counter(right for _, right in pairs) != expected_right
    ):  # defensive invariant
        raise RuntimeError("wrong pairings did not preserve endpoint marginals")
    rng.shuffle(pairs)
    return pairs


def _draw_texts(
    endpoints: list[tuple[str, str]],
    rng: Generator,
    used: set[str],
) -> list[str]:
    """Draw unique interiors for fixed endpoints, with a finite pool check."""

    result: list[str] = []
    by_pair = Counter(endpoints)
    for pair, required in sorted(by_pair.items()):
        left, right = pair
        available = [
            interior
            for interior in _INTERIORS
            if f"{left}{interior}{right}" not in used
        ]
        if required > len(available):
            raise ValueError(
                "requested split sizes exhaust the unique strings available "
                f"for endpoint pair {pair!r}"
            )
        chosen = rng.choice(len(available), size=required, replace=False)
        for index in np.atleast_1d(chosen):
            text = f"{left}{available[int(index)]}{right}"
            used.add(text)
            result.append(text)
    rng.shuffle(result)
    return result


def _make_evaluation_split(
    size: int,
    rng: Generator,
    used: set[str],
) -> tuple[TextCase, ...]:
    class_size = size // 2
    positive_indices = _balanced_pair_indices(class_size, rng)
    positive_endpoints = [POSITIVE_ENDPOINT_PAIRS[index] for index in positive_indices]
    negative_endpoints = _wrong_pairs_with_matched_marginals(positive_indices, rng)

    positives = _draw_texts(positive_endpoints, rng, used)
    negatives = _draw_texts(negative_endpoints, rng, used)
    cases = [TextCase(text, True, POSITIVE_FAMILY) for text in positives]
    cases.extend(TextCase(text, False, NEGATIVE_FAMILY) for text in negatives)
    rng.shuffle(cases)
    return tuple(cases)


def make_dataset(
    seed: int = 0,
    train_size: int = 24,
    dev_size: int = 32,
    test_size: int = 64,
    noise_size: int = 64,
) -> TextDataset:
    """Build deterministic, globally disjoint controlled evaluation data.

    Each evaluation split must contain at least four cases per class so every
    known association is represented.  Separate ``SeedSequence`` child streams
    drive train, development, test, and noise draws.  Selection is always from
    finite candidate pools, so impossible requests raise instead of retrying.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    train_size = _checked_size(
        "train_size", train_size, minimum=len(POSITIVE_ENDPOINT_PAIRS)
    )
    dev_size = _checked_size("dev_size", dev_size, minimum=8)
    test_size = _checked_size("test_size", test_size, minimum=8)
    noise_size = _checked_size("noise_size", noise_size)
    if dev_size % 2:
        raise ValueError("dev_size must be even for exact class balance")
    if test_size % 2:
        raise ValueError("test_size must be even for exact class balance")

    labelled_positive_count = train_size + dev_size // 2 + test_size // 2
    positive_capacity = len(POSITIVE_ENDPOINT_PAIRS) * len(_INTERIORS)
    if labelled_positive_count > positive_capacity:
        raise ValueError(
            "requested positive examples exceed the finite positive capacity "
            f"of {positive_capacity}"
        )
    labelled_negative_count = dev_size // 2 + test_size // 2
    negative_capacity = (
        len(POSITIVE_ENDPOINT_PAIRS)
        * (len(POSITIVE_ENDPOINT_PAIRS) - 1)
        * len(_INTERIORS)
    )
    if labelled_negative_count > negative_capacity:
        raise ValueError(
            "requested negative examples exceed the finite negative capacity "
            f"of {negative_capacity}"
        )
    requested_total = train_size + dev_size + test_size + noise_size
    if requested_total > _FULL_SPACE_SIZE:
        raise ValueError(
            "requested examples exceed the complete unique window capacity "
            f"of {_FULL_SPACE_SIZE}"
        )

    train_seed, dev_seed, test_seed, noise_seed = np.random.SeedSequence(
        int(seed)
    ).spawn(4)
    train_rng = np.random.default_rng(train_seed)
    dev_rng = np.random.default_rng(dev_seed)
    test_rng = np.random.default_rng(test_seed)
    noise_rng = np.random.default_rng(noise_seed)
    used: set[str] = set()

    train_indices = _balanced_pair_indices(train_size, train_rng)
    train_endpoints = [POSITIVE_ENDPOINT_PAIRS[index] for index in train_indices]
    train = tuple(_draw_texts(train_endpoints, train_rng, used))
    dev = _make_evaluation_split(dev_size, dev_rng, used)
    test = _make_evaluation_split(test_size, test_rng, used)

    if noise_size:
        available_noise = [
            "".join(chars)
            for chars in product(ALPHABET, repeat=WINDOW_LENGTH)
            if "".join(chars) not in used
        ]
        if noise_size > len(available_noise):
            raise ValueError(
                "noise_size exceeds the remaining globally disjoint window "
                f"capacity of {len(available_noise)}"
            )
        noise_indices = noise_rng.choice(
            len(available_noise), size=noise_size, replace=False
        )
        noise = tuple(available_noise[int(index)] for index in noise_indices)
    else:
        noise = ()

    return TextDataset(train=train, dev=dev, test=test, noise=noise)


def dataset_hash(dataset: TextDataset) -> str:
    """Return a stable SHA-256 hash of a dataset's canonical JSON form."""

    payload = {
        "train": list(dataset.train),
        "dev": [
            {"text": case.text, "label": case.label, "family": case.family}
            for case in dataset.dev
        ],
        "test": [
            {"text": case.text, "label": case.label, "family": case.family}
            for case in dataset.test
        ],
        "noise": list(dataset.noise),
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return sha256(serialized.encode("utf-8")).hexdigest()

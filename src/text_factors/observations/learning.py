"""A small supervised candidate baseline, not a language or semantic model.

All offsets refer to the original Python string (Unicode code points). Features
use Unicode categories and hashed character n-grams, never a vocabulary of names
or pronouns. Sigmoid scores are *not calibrated probabilities*. Span negatives
require an explicit complete-coverage declaration for every training document;
the declaration itself cannot prove annotation quality. Pair supervision uses
only two explicitly known entity identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schema import canonical_json, fields, text_field

MODEL_SCHEMA = "ai2-open-candidate-model-v1"
_ALGORITHM = "unicode-char-shape-sparse-logistic-v3"


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if (
        type(value) not in (int, float)
        or not low <= value <= high
        or not math.isfinite(value)
    ):
        raise ValueError(f"invalid finite {name}")
    return float(value)


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Explicit resource limits; exceeding a limit raises instead of truncating."""

    seed: int = 17
    epochs: int = 12
    feature_dim: int = 8192
    max_span_tokens: int = 16
    max_tokens: int = 4096
    max_antecedents: int = 64
    negative_ratio: int = 3
    max_documents: int = 128
    max_document_chars: int = 262_144
    max_training_tokens: int = 100_000
    max_candidates: int = 65_536
    max_training_examples: int = 100_000
    max_pair_examples: int = 200_000
    max_feature_chars: int = 128
    max_char_ngrams: int = 128
    learning_rate: float = 0.25
    l2: float = 0.0001

    def __post_init__(self) -> None:
        bounds = {
            "seed": (0, 2**32 - 1),
            "epochs": (1, 100),
            "feature_dim": (128, 65_536),
            "max_span_tokens": (1, 16),
            "max_tokens": (1, 100_000),
            "max_antecedents": (1, 256),
            "negative_ratio": (1, 100),
            "max_documents": (1, 1000),
            "max_document_chars": (1, 4_000_000),
            "max_training_tokens": (1, 1_000_000),
            "max_candidates": (1, 1_000_000),
            "max_training_examples": (1, 1_000_000),
            "max_pair_examples": (1, 1_000_000),
            "max_feature_chars": (4, 2048),
            "max_char_ngrams": (1, 512),
        }
        for name, (low, high) in bounds.items():
            _integer(getattr(self, name), name, low, high)
        _number(self.learning_rate, "learning_rate", 0.001, 2.0)
        _number(self.l2, "l2", 0.0, 0.1)


@dataclass(frozen=True, slots=True)
class _Token:
    start: int
    end: int


def _word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LMN" or category == "Pc"


def _tokens(text: str, config: LearningConfig) -> list[_Token]:
    text_field(text, "document text", cap=config.max_document_chars, empty=True)
    result: list[_Token] = []
    offset = 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        start = offset
        if _word_char(text[offset]):
            offset += 1
            while offset < len(text) and _word_char(text[offset]):
                offset += 1
        else:
            offset += 1
        result.append(_Token(start, offset))
        if len(result) > config.max_tokens:
            raise ValueError("document exceeds max_tokens")
    width = min(len(result), config.max_span_tokens)
    candidate_count = width * len(result) - width * (width - 1) // 2
    if candidate_count > config.max_candidates:
        raise ValueError("document exceeds max_candidates")
    return result


def _span_indices(tokens: list[_Token], config: LearningConfig):
    for begin in range(len(tokens)):
        for stop in range(
            begin + 1, min(len(tokens), begin + config.max_span_tokens) + 1
        ):
            yield begin, stop


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + text[-(limit - half) :]


def _shape(text: str) -> str:
    # Unicode categories, not a list of linguistic classes or known words.
    result: list[str] = []
    previous = None
    for char in text:
        category = unicodedata.category(char)
        if category != previous:
            result.append(category)
            previous = category
    return ":".join(result)


def _grams(text: str, config: LearningConfig) -> list[str]:
    bounded = "^" + _clip(text.casefold(), config.max_feature_chars) + "$"
    grams = [
        bounded[index : index + width]
        for width in (1, 2, 3)
        for index in range(len(bounded) - width + 1)
    ]
    if len(grams) <= config.max_char_ngrams:
        return grams
    # Sample across n-gram widths and positions, rather than dropping the tail.
    count = config.max_char_ngrams
    return [grams[index * len(grams) // count] for index in range(count)]


@lru_cache(maxsize=65_536)
def _hash_feature(name: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest[:4], "little") % (dim - 1), (
        1.0 if digest[4] & 1 else -1.0
    )


def _vector(
    features: list[tuple[str, float]], config: LearningConfig
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    hashed: dict[int, float] = defaultdict(float)
    for name, value in features:
        index, sign = _hash_feature(name, config.feature_dim)
        hashed[index] += sign * value
    length = math.sqrt(sum(value * value for value in hashed.values())) or 1.0
    # A separate bias is not diluted by the variable number of character grams.
    indices = np.array([0, *hashed.keys()], dtype=np.int64)
    values = np.array([1.0, *(v / length for v in hashed.values())], dtype=np.float64)
    return indices, values


def _bucket(value: int) -> int:
    return max(0, value).bit_length()


def _span_features(
    text: str, tokens: list[_Token], begin: int, stop: int, config: LearningConfig
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    start, end = tokens[begin].start, tokens[stop - 1].end
    surface = text[start:end]
    bounded = _clip(surface, config.max_feature_chars)
    features = [
        (f"width:{stop - begin}", 1.0),
        (f"length:{_bucket(len(surface))}", 1.0),
        (f"shape:{_shape(bounded)}", 1.0),
        (f"upper-initial:{surface[0].isupper()}", 1.0),
        (f"all-upper:{surface.isupper()}", 1.0),
        (f"title:{surface.istitle()}", 1.0),
        (f"starts-document:{begin == 0}", 1.0),
        (f"ends-document:{stop == len(tokens)}", 1.0),
    ]
    features.extend((f"span:{gram}", 1.0) for gram in _grams(surface, config))
    for prefix, position in (("before", begin - 1), ("after", stop)):
        if 0 <= position < len(tokens):
            token = tokens[position]
            context = _clip(text[token.start : token.end], 32)
            features.append((f"{prefix}-shape:{_shape(context)}", 1.0))
            features.extend(
                (f"{prefix}:{gram}", 0.5) for gram in _grams(context, config)
            )
    return _vector(features, config)


def _coordinates(mention: Any, text: str) -> tuple[int, int]:
    if type(mention) is not dict or not {"start", "end"} <= mention.keys():
        raise ValueError("mention needs start and end")
    start = _integer(mention["start"], "mention start", 0, len(text))
    end = _integer(mention["end"], "mention end", start + 1, len(text))
    return start, end


def _overlapping_token_indices(
    tokens: list[_Token],
    span: tuple[int, int],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, token in enumerate(tokens)
        if token.end > span[0] and token.start < span[1]
    )


def _pair_features(
    text: str,
    tokens: list[_Token],
    left: tuple[int, int],
    right: tuple[int, int],
    config: LearningConfig,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    left_text, right_text = text[slice(*left)], text[slice(*right)]
    left_folded, right_folded = left_text.casefold(), right_text.casefold()
    left_bounded = _clip(left_text, config.max_feature_chars)
    right_bounded = _clip(right_text, config.max_feature_chars)
    left_grams, right_grams = (
        set(_grams(left_text, config)),
        set(_grams(right_text, config)),
    )
    overlap = len(left_grams & right_grams) / max(1, len(left_grams | right_grams))
    left_shape, right_shape = _shape(left_bounded), _shape(right_bounded)

    left_indices = _overlapping_token_indices(tokens, left)
    right_indices = _overlapping_token_indices(tokens, right)
    left_tokens = tuple(
        text[tokens[index].start : tokens[index].end].casefold()
        for index in left_indices
    )
    right_tokens = tuple(
        text[tokens[index].start : tokens[index].end].casefold()
        for index in right_indices
    )
    left_set, right_set = set(left_tokens), set(right_tokens)
    token_union = left_set | right_set
    token_overlap = len(left_set & right_set) / len(token_union) if token_union else 0.0
    token_gap = (
        max(0, right_indices[0] - left_indices[-1] - 1)
        if left_indices and right_indices
        else 0
    )
    if left[1] <= right[0]:
        between = text[left[1] : right[0]]
    elif right[1] <= left[0]:
        between = text[right[1] : left[0]]
    else:
        between = ""
    punctuation = sum(unicodedata.category(char).startswith("P") for char in between)
    line_breaks = between.count("\n") + between.count("\r")
    left_digits, right_digits = _digit_runs(left_text), _digit_runs(right_text)
    same_first_token = bool(
        left_tokens and right_tokens and left_tokens[0] == right_tokens[0]
    )
    same_last_token = bool(
        left_tokens and right_tokens and left_tokens[-1] == right_tokens[-1]
    )

    features = [
        (f"exact-equal:{left_text == right_text}", 1.0),
        (f"folded-equal:{left_folded == right_folded}", 1.0),
        (f"shape-equal:{left_shape == right_shape}", 1.0),
        (f"left-shape:{left_shape}", 1.0),
        (f"right-shape:{right_shape}", 1.0),
        (f"distance:{_bucket(abs(right[0] - left[1]))}", 1.0),
        (f"token-gap:{_bucket(token_gap)}", 1.0),
        (f"punctuation-gap:{_bucket(punctuation)}", 1.0),
        (f"line-break-gap:{_bucket(line_breaks)}", 1.0),
        (f"left-token-width:{_bucket(len(left_tokens))}", 1.0),
        (f"right-token-width:{_bucket(len(right_tokens))}", 1.0),
        (f"same-first-token:{same_first_token}", 1.0),
        (f"same-last-token:{same_last_token}", 1.0),
        (
            "surface-contained:"
            f"{left_folded in right_folded or right_folded in left_folded}",
            1.0,
        ),
        (
            f"digit-pattern-equal:{bool(left_digits) and left_digits == right_digits}",
            1.0,
        ),
        (f"both-have-digits:{bool(left_digits) and bool(right_digits)}", 1.0),
        (f"overlapping:{max(left[0], right[0]) < min(left[1], right[1])}", 1.0),
        (f"rightward:{left[0] <= right[0]}", 1.0),
        (f"similarity-bucket:{int(overlap * 10)}", 1.0),
        (f"token-similarity-bucket:{int(token_overlap * 10)}", 1.0),
        ("similarity", overlap),
        ("token-similarity", token_overlap),
    ]
    features.extend((f"left:{gram}", 0.5) for gram in sorted(left_grams))
    features.extend((f"right:{gram}", 0.5) for gram in sorted(right_grams))
    for prefix, span in (("left", left), ("right", right)):
        context = text[max(0, span[0] - 32) : span[0]] + text[span[1] : span[1] + 32]
        features.extend(
            (f"{prefix}-context:{gram}", 0.25) for gram in _grams(context, config)
        )
    return _vector(features, config)


def _digit_runs(value: str) -> tuple[str, ...]:
    runs: list[str] = []
    current: list[str] = []
    for character in value:
        if character.isdigit():
            current.append(character)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return tuple(runs)


def _critical_negative(
    text: str,
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    """Return true for identity conflicts that must never be sampled away."""

    left_text, right_text = text[slice(*left)], text[slice(*right)]
    left_folded, right_folded = left_text.casefold(), right_text.casefold()
    if left_folded == right_folded:
        return True
    left_digits, right_digits = _digit_runs(left_text), _digit_runs(right_text)
    if not left_digits or not right_digits or left_digits == right_digits:
        return False
    left_skeleton = "".join(char for char in left_folded if not char.isdigit())
    right_skeleton = "".join(char for char in right_folded if not char.isdigit())
    return bool(left_skeleton) and left_skeleton == right_skeleton


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _score(
    weights: NDArray[np.float64], vector: tuple[NDArray[np.int64], NDArray[np.float64]]
) -> float:
    indices, values = vector
    return _sigmoid(float(np.dot(weights[indices], values)))


_COUNT_KEYS = {
    "train_document_count",
    "train_token_count",
    "span_positive_examples",
    "span_negative_examples",
    "pair_positive_examples",
    "pair_negative_examples",
    "pair_eligible_positive_examples",
    "pair_eligible_negative_examples",
    "pair_all_same_entity_examples",
    "pair_nonpreferred_positive_examples_ignored",
    "pair_critical_negative_examples",
    "pair_eligible_critical_negative_examples",
    "unsupported_gold_spans",
    "unknown_identity_pairs_ignored",
}
_SUMMARY_KEYS = _COUNT_KEYS | {
    "train_document_ids",
    "train_group_ids",
    "train_data_sha256",
    "span_training_enabled",
    "pair_training_enabled",
    "span_disabled_reason",
    "pair_disabled_reason",
    "pair_negative_sampling",
    "score_interpretation",
}


def _summary(value: Any, config: LearningConfig) -> dict[str, Any]:
    value = fields(value, _SUMMARY_KEYS)
    for key in _COUNT_KEYS:
        _integer(value[key], key, 0, 1_000_000_000)
    for key in ("train_document_ids", "train_group_ids"):
        entries = value[key]
        if type(entries) is not list or not 1 <= len(entries) <= config.max_documents:
            raise ValueError(f"invalid {key}")
        for entry in entries:
            text_field(entry, key)
        if entries != sorted(set(entries)):
            raise ValueError(f"{key} must be sorted and unique")
    if value["train_document_count"] != len(value["train_document_ids"]):
        raise ValueError("training document count mismatch")
    if len(value["train_group_ids"]) > value["train_document_count"]:
        raise ValueError("training group count mismatch")
    if value["train_token_count"] > config.max_training_tokens:
        raise ValueError("training token count exceeds model budget")
    if (
        value["pair_positive_examples"] != value["pair_eligible_positive_examples"]
        or value["pair_all_same_entity_examples"] < value["pair_positive_examples"]
        or value["pair_nonpreferred_positive_examples_ignored"]
        != value["pair_all_same_entity_examples"] - value["pair_positive_examples"]
        or value["pair_negative_examples"] > value["pair_eligible_negative_examples"]
        or value["pair_critical_negative_examples"]
        > value["pair_eligible_critical_negative_examples"]
        or value["pair_critical_negative_examples"] > value["pair_negative_examples"]
    ):
        raise ValueError("sampled pair counts disagree with eligible examples")
    for kind, cap in (
        ("span", config.max_training_examples),
        ("pair", config.max_pair_examples),
    ):
        positive = value[f"{kind}_positive_examples"]
        negative = value[f"{kind}_negative_examples"]
        if positive + negative > cap:
            raise ValueError("training example count exceeds model budget")
        enabled = value[f"{kind}_training_enabled"]
        if type(enabled) is not bool or enabled != (positive > 0 and negative > 0):
            raise ValueError("training enabled flag disagrees with labels")
        expected_reason = None if enabled else "requires_positive_and_negative_examples"
        if value[f"{kind}_disabled_reason"] != expected_reason:
            raise ValueError("invalid training disabled reason")
    digest = value["train_data_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid train_data_sha256")
    if value["pair_negative_sampling"] != "retain_identity_conflicts_then_reservoir_v2":
        raise ValueError("invalid pair negative sampling")
    if value["score_interpretation"] != "uncalibrated_sigmoid":
        raise ValueError("invalid score interpretation")
    return json.loads(canonical_json(value))


class CandidateModel:
    """Learned local proposals only; inference never creates identity decisions."""

    def __init__(
        self,
        config: LearningConfig,
        span_weights: NDArray[np.float64],
        link_weights: NDArray[np.float64],
        training_summary: dict[str, Any],
    ) -> None:
        self.config = config
        self._training_summary = _summary(training_summary, config)
        self._span_weights = self._weights(span_weights)
        self._link_weights = self._weights(link_weights)
        for weights, enabled in (
            (self._span_weights, self.span_training_enabled),
            (self._link_weights, self.pair_training_enabled),
        ):
            if not enabled and np.any(weights):
                raise ValueError("disabled model must have zero weights")
        self._last_text: str | None = None
        self._last_tokens: list[_Token] = []

    def _weights(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        if weights.shape != (self.config.feature_dim,):
            raise ValueError("invalid model weight dimensions")
        if not np.all(np.isfinite(weights)) or np.any(np.abs(weights) > 1_000_000):
            raise ValueError("invalid finite model weights")
        copied = np.array(weights, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        return copied

    @property
    def training_summary(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._training_summary))

    @property
    def span_training_enabled(self) -> bool:
        return self._training_summary["span_training_enabled"]

    @property
    def pair_training_enabled(self) -> bool:
        return self._training_summary["pair_training_enabled"]

    def _prepare(self, text: str) -> list[_Token]:
        if type(text) is not str:
            raise ValueError("document text must be a string")
        if text != self._last_text:
            tokens = _tokens(text, self.config)
            self._last_text, self._last_tokens = text, tokens
        return self._last_tokens

    def span_scores(self, text: str) -> list[dict[str, Any]]:
        """Return every permitted token-aligned span with an uncalibrated score."""
        tokens = self._prepare(text)
        return [
            {
                "start": tokens[begin].start,
                "end": tokens[stop - 1].end,
                "score": (
                    _score(
                        self._span_weights,
                        _span_features(text, tokens, begin, stop, self.config),
                    )
                    if self.span_training_enabled
                    else 0.5
                ),
            }
            for begin, stop in _span_indices(tokens, self.config)
        ]

    def link_score(self, text: str, left: dict, right: dict) -> float:
        """Score a pair without consulting supplied IDs, labels or gold metadata.

        Even equal strings are just a feature. A disabled pair learner returns
        0.5 and must never be enabled by a downstream decision gate.
        """
        self._prepare(text)
        left_span, right_span = _coordinates(left, text), _coordinates(right, text)
        if left_span == right_span:
            raise ValueError("a mention cannot link to itself")
        if not self.pair_training_enabled:
            return 0.5
        tokens = self._prepare(text)
        return _score(
            self._link_weights,
            _pair_features(text, tokens, left_span, right_span, self.config),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": MODEL_SCHEMA,
            "algorithm": _ALGORITHM,
            "config": asdict(self.config),
            "span_weights": self._span_weights.tolist(),
            "link_weights": self._link_weights.tolist(),
            "training_summary": self.training_summary,
        }
        return {
            **payload,
            "fingerprint": hashlib.sha256(
                canonical_json(payload).encode("utf-8")
            ).hexdigest(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> CandidateModel:
        value = fields(
            value,
            {
                "schema",
                "algorithm",
                "config",
                "span_weights",
                "link_weights",
                "training_summary",
                "fingerprint",
            },
        )
        if value["schema"] != MODEL_SCHEMA or value["algorithm"] != _ALGORITHM:
            raise ValueError("unsupported candidate model schema or algorithm")
        config_fields = set(LearningConfig.__dataclass_fields__)
        config = LearningConfig(**fields(value["config"], config_fields))
        summary = _summary(value["training_summary"], config)
        arrays = []
        for name in ("span_weights", "link_weights"):
            weights = value[name]
            if type(weights) is not list or len(weights) != config.feature_dim:
                raise ValueError("invalid model weight dimensions")
            for weight in weights:
                _number(weight, name, -1_000_000, 1_000_000)
            arrays.append(np.array(weights, dtype=np.float64))
        fingerprint = hashlib.sha256(
            canonical_json(
                {key: item for key, item in value.items() if key != "fingerprint"}
            ).encode("utf-8")
        ).hexdigest()
        if type(value["fingerprint"]) is not str or value["fingerprint"] != fingerprint:
            raise ValueError("candidate model fingerprint mismatch")
        return cls(config, arrays[0], arrays[1], summary)


@dataclass(slots=True)
class _Document:
    document_id: str
    group_id: str
    text: str
    tokens: list[_Token]
    mentions: list[dict[str, Any]]


def _documents(train_documents: list[dict], config: LearningConfig) -> list[_Document]:
    if (
        type(train_documents) is not list
        or not 1 <= len(train_documents) <= config.max_documents
    ):
        raise ValueError("training document count exceeds configured bounds")
    documents = []
    ids: set[str] = set()
    token_count = 0
    for document in train_documents:
        if type(document) is not dict or document.get("split") != "train":
            raise ValueError("training accepts only explicit train documents")
        if document.get("coverage") != "complete":
            raise ValueError("training requires coverage='complete' for every document")
        document_id = text_field(document.get("document_id"), "document_id")
        group_id = text_field(document.get("group_id"), "group_id")
        if document_id in ids:
            raise ValueError("duplicate training document_id")
        ids.add(document_id)
        text = document.get("text")
        if type(text) is not str:
            raise ValueError("document text must be a string")
        tokens = _tokens(text, config)
        token_count += len(tokens)
        if token_count > config.max_training_tokens:
            raise ValueError("corpus exceeds max_training_tokens")
        mentions = document.get("mentions")
        if type(mentions) is not list or len(mentions) > config.max_candidates:
            raise ValueError("invalid or oversized mention annotations")
        cleaned = []
        seen: set[tuple[int, int]] = set()
        for mention in mentions:
            start, end = _coordinates(mention, text)
            if (start, end) in seen:
                raise ValueError("duplicate gold mention span")
            seen.add((start, end))
            if "entity_id" not in mention:
                raise ValueError("gold mention requires entity_id (null if unknown)")
            entity_id = mention["entity_id"]
            if entity_id is not None:
                text_field(entity_id, "entity_id")
            cleaned.append({"start": start, "end": end, "entity_id": entity_id})
        cleaned.sort(key=lambda item: (item["start"], item["end"]))
        documents.append(_Document(document_id, group_id, text, tokens, cleaned))
    return sorted(documents, key=lambda document: document.document_id)


def _fit(
    examples: list[tuple],
    documents: list[_Document],
    config: LearningConfig,
    *,
    pair: bool,
    progress: Callable[[dict[str, Any]], None] | None,
) -> NDArray[np.float64]:
    weights = np.zeros(config.feature_dim, dtype=np.float64)
    if not any(example[-1] == 1 for example in examples) or not any(
        example[-1] == 0 for example in examples
    ):
        return weights
    rng = random.Random(config.seed + int(pair))
    order = list(range(len(examples)))
    updates = 0
    for epoch in range(config.epochs):
        if progress is not None:
            progress(
                {
                    "phase": "pair_training" if pair else "span_training",
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "completed_updates": updates,
                    "total_updates": len(examples) * config.epochs,
                }
            )
        rng.shuffle(order)
        # Epoch-wise L2 keeps the working set sparse between individual updates.
        weights[1:] *= 1.0 - config.learning_rate * config.l2
        for index in order:
            document_index, left, right, target = examples[index]
            document = documents[document_index]
            vector = (
                _pair_features(document.text, document.tokens, left, right, config)
                if pair
                else _span_features(document.text, document.tokens, left, right, config)
            )
            indices, values = vector
            error = _score(weights, vector) - target
            rate = config.learning_rate / math.sqrt(1.0 + updates / 2000.0)
            weights[indices] -= rate * error * values
            updates += 1
            if progress is not None and updates % 2000 == 0:
                progress(
                    {
                        "phase": "pair_training" if pair else "span_training",
                        "epoch": epoch + 1,
                        "epochs": config.epochs,
                        "completed_updates": updates,
                        "total_updates": len(examples) * config.epochs,
                    }
                )
    if progress is not None:
        progress(
            {
                "phase": "pair_training" if pair else "span_training",
                "epoch": config.epochs,
                "epochs": config.epochs,
                "completed_updates": updates,
                "total_updates": updates,
            }
        )
    return weights


def _antecedent_pairs(mentions: list[dict[str, Any]], config: LearningConfig):
    for right_index, right in enumerate(mentions):
        for left in mentions[
            max(0, right_index - config.max_antecedents) : right_index
        ]:
            yield left, right


def train_model(
    train_documents: list[dict],
    config: LearningConfig | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> CandidateModel:
    """Fit using train annotations only, retaining no entity IDs in feature weights.

    Each document must explicitly declare coverage='complete'. Negative spans
    are reproducibly reservoir sampled up to negative_ratio times the document's
    total gold span count, including unsupported gold spans. Pair negatives that
    represent explicit identity conflicts with the same surface, or with the same
    non-digit identifier skeleton and different digit runs, are always retained.
    For each mention, only the nearest previous mention of the same known entity
    is a positive antecedent target. Older same-entity mentions remain valid
    alternatives but are ignored by the pair loss instead of becoming duplicate
    positives or false negatives. Different known entities are negatives.
    Remaining pair negatives are reservoir sampled up to negative_ratio times the
    preferred-positive count. Pair counts remain in the summary. Unaligned gold
    spans are counted, not relabelled as negatives. Pair labels are generated
    only within a document and the configured antecedent window.
    No corpus-wide feature matrix is materialized: SGD computes one sparse vector
    at a time.
    """
    config = config or LearningConfig()
    documents = _documents(train_documents, config)
    span_examples: list[tuple] = []
    pair_examples: list[tuple] = []
    unsupported = 0
    ignored_unknown_pairs = 0
    eligible_pair_positives = 0
    eligible_pair_negatives = 0
    all_same_entity_pairs = 0
    ignored_nonpreferred_positives = 0
    eligible_critical_negatives = 0
    retained_critical_negatives = 0
    digest = hashlib.sha256()
    for document_index, document in enumerate(documents):
        if progress is not None:
            progress(
                {
                    "phase": "prepare_examples",
                    "document_id": document.document_id,
                    "completed_documents": document_index,
                    "total_documents": len(documents),
                }
            )
        digest.update(
            canonical_json(
                {
                    "document_id": document.document_id,
                    "group_id": document.group_id,
                    "split": "train",
                    "text": document.text,
                    "mentions": document.mentions,
                }
            ).encode("utf-8")
            + b"\n"
        )
        gold = {(mention["start"], mention["end"]) for mention in document.mentions}
        positives = []
        negatives: list[tuple] = []
        negative_limit = max(1, len(gold)) * config.negative_ratio
        rng = random.Random(f"{config.seed}:{document.document_id}")
        negative_seen = 0
        for begin, stop in _span_indices(document.tokens, config):
            span = (document.tokens[begin].start, document.tokens[stop - 1].end)
            if span in gold:
                positives.append((document_index, begin, stop, 1))
            else:
                negative_seen += 1
                entry = (document_index, begin, stop, 0)
                if len(negatives) < negative_limit:
                    negatives.append(entry)
                else:
                    replace = rng.randrange(negative_seen)
                    if replace < negative_limit:
                        negatives[replace] = entry
            if (
                len(span_examples) + len(positives) + len(negatives)
                > config.max_training_examples
            ):
                raise ValueError("corpus exceeds max_training_examples")
        unsupported += len(gold) - len(positives)
        if (
            len(span_examples) + len(positives) + len(negatives)
            > config.max_training_examples
        ):
            raise ValueError("corpus exceeds max_training_examples")
        span_examples.extend(positives)
        span_examples.extend(negatives)
        pair_positives, pair_negatives, critical_negatives = 0, 0, 0
        document_all_same = 0
        document_ignored_same = 0
        for right_index, right in enumerate(document.mentions):
            previous = document.mentions[
                max(0, right_index - config.max_antecedents) : right_index
            ]
            right_entity = right["entity_id"]
            if right_entity is None:
                ignored_unknown_pairs += len(previous)
                continue
            known_previous = []
            for left in previous:
                if left["entity_id"] is None:
                    ignored_unknown_pairs += 1
                else:
                    known_previous.append(left)
            same = [
                left for left in known_previous if left["entity_id"] == right_entity
            ]
            if same:
                pair_positives += 1
                document_all_same += len(same)
                document_ignored_same += len(same) - 1
            for left in known_previous:
                if left["entity_id"] == right_entity:
                    continue
                pair_negatives += 1
                if _critical_negative(
                    document.text,
                    (left["start"], left["end"]),
                    (right["start"], right["end"]),
                ):
                    critical_negatives += 1

        eligible_pair_positives += pair_positives
        eligible_pair_negatives += pair_negatives
        all_same_entity_pairs += document_all_same
        ignored_nonpreferred_positives += document_ignored_same
        eligible_critical_negatives += critical_negatives

        ordinary_negatives = pair_negatives - critical_negatives
        ordinary_limit = min(
            ordinary_negatives, max(1, pair_positives) * config.negative_ratio
        )
        selected_negative_count = critical_negatives + ordinary_limit
        if (
            len(pair_examples) + pair_positives + selected_negative_count
            > config.max_pair_examples
        ):
            raise ValueError("corpus exceeds max_pair_examples")

        sampled_negatives = []
        negative_seen = 0
        rng = random.Random(f"{config.seed}:{document.document_id}:pair")
        for right_index, right in enumerate(document.mentions):
            previous = document.mentions[
                max(0, right_index - config.max_antecedents) : right_index
            ]
            right_entity = right["entity_id"]
            if right_entity is None:
                continue
            known_previous = [
                left for left in previous if left["entity_id"] is not None
            ]
            same = [
                left for left in known_previous if left["entity_id"] == right_entity
            ]
            preferred = same[-1] if same else None
            if preferred is not None:
                pair_examples.append(
                    (
                        document_index,
                        (preferred["start"], preferred["end"]),
                        (right["start"], right["end"]),
                        1,
                    )
                )
            for left in known_previous:
                if left["entity_id"] == right_entity:
                    continue
                left_span = (left["start"], left["end"])
                right_span = (right["start"], right["end"])
                entry = (document_index, left_span, right_span, 0)
                if _critical_negative(document.text, left_span, right_span):
                    pair_examples.append(entry)
                    retained_critical_negatives += 1
                else:
                    negative_seen += 1
                    if len(sampled_negatives) < ordinary_limit:
                        sampled_negatives.append(entry)
                    elif ordinary_limit:
                        replace = rng.randrange(negative_seen)
                        if replace < ordinary_limit:
                            sampled_negatives[replace] = entry
        pair_examples.extend(sampled_negatives)
    summary: dict[str, Any] = {
        "train_document_count": len(documents),
        "train_token_count": sum(len(document.tokens) for document in documents),
        "train_document_ids": [document.document_id for document in documents],
        "train_group_ids": sorted({document.group_id for document in documents}),
        "train_data_sha256": digest.hexdigest(),
        "unsupported_gold_spans": unsupported,
        "unknown_identity_pairs_ignored": ignored_unknown_pairs,
        "pair_eligible_positive_examples": eligible_pair_positives,
        "pair_eligible_negative_examples": eligible_pair_negatives,
        "pair_all_same_entity_examples": all_same_entity_pairs,
        "pair_nonpreferred_positive_examples_ignored": (ignored_nonpreferred_positives),
        "pair_critical_negative_examples": retained_critical_negatives,
        "pair_eligible_critical_negative_examples": eligible_critical_negatives,
        "pair_negative_sampling": "retain_identity_conflicts_then_reservoir_v2",
        "score_interpretation": "uncalibrated_sigmoid",
    }
    for kind, examples in (("span", span_examples), ("pair", pair_examples)):
        positive_count = sum(example[-1] == 1 for example in examples)
        negative_count = len(examples) - positive_count
        enabled = positive_count > 0 and negative_count > 0
        summary.update(
            {
                f"{kind}_positive_examples": positive_count,
                f"{kind}_negative_examples": negative_count,
                f"{kind}_training_enabled": enabled,
                f"{kind}_disabled_reason": None
                if enabled
                else "requires_positive_and_negative_examples",
            }
        )
    return CandidateModel(
        config,
        _fit(span_examples, documents, config, pair=False, progress=progress),
        _fit(pair_examples, documents, config, pair=True, progress=progress),
        summary,
    )

"""Domain-independent learned attention for real-data revision.

The ranker consumes arbitrary numeric feature maps supplied by learned encoders
or adapters. It has no built-in entity, pronoun, predicate, or keyword list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np


def _text(value: str, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    return value


def _feature_map(value: dict[str, float]) -> dict[str, float]:
    if type(value) is not dict or len(value) > 4096:
        raise ValueError("attention features must be a bounded object")
    result: dict[str, float] = {}
    for name, raw in value.items():
        _text(name, "feature name")
        if type(raw) not in (int, float) or not isfinite(float(raw)):
            raise ValueError("attention feature value must be finite")
        number = float(raw)
        if abs(number) > 1_000_000:
            raise ValueError("attention feature magnitude is too large")
        if number:
            result[name] = number
    return result


@dataclass(frozen=True, slots=True)
class AttentionExample:
    group_id: str
    candidate_id: str
    features: dict[str, float]
    label: int

    def __post_init__(self) -> None:
        _text(self.group_id, "group_id")
        _text(self.candidate_id, "candidate_id")
        object.__setattr__(self, "features", _feature_map(self.features))
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("attention label must be 0 or 1")


@dataclass(frozen=True, slots=True)
class AttentionCandidate:
    candidate_id: str
    features: dict[str, float]

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        object.__setattr__(self, "features", _feature_map(self.features))


class HashedAttentionRanker:
    """Bounded logistic ranker over open sparse feature names."""

    def __init__(
        self,
        *,
        dimension: int = 2048,
        seed: int = 17,
        weights: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    ) -> None:
        if type(dimension) is not int or not 64 <= dimension <= 65_536:
            raise ValueError("attention dimension must be in [64, 65536]")
        if type(seed) is not int or seed < 0:
            raise ValueError("attention seed must be non-negative")
        self.dimension = dimension
        self.seed = seed
        if weights is None:
            self.weights = np.zeros(dimension, dtype=np.float64)
        else:
            converted = np.asarray(weights, dtype=np.float64)
            if converted.shape != (dimension,) or not np.isfinite(converted).all():
                raise ValueError("invalid attention weights")
            self.weights = converted.copy()

    def _slot(self, name: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            f"{self.seed}:{name}".encode(), digest_size=16
        ).digest()
        slot = int.from_bytes(digest[:8], "little") % self.dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        return slot, sign

    def vector(
        self, features: dict[str, float]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        checked = _feature_map(features)
        vector = np.zeros(self.dimension, dtype=np.float64)
        bias_slot, bias_sign = self._slot("__bias__")
        vector[bias_slot] += bias_sign
        for name, value in checked.items():
            slot, sign = self._slot(name)
            vector[slot] += sign * value
        return vector

    def score(self, features: dict[str, float]) -> float:
        raw = float(self.vector(features) @ self.weights)
        return float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))

    def fit(
        self,
        examples: list[AttentionExample],
        *,
        steps: int = 400,
        learning_rate: float = 0.2,
        l2: float = 0.002,
    ) -> None:
        if not 1 <= len(examples) <= 10_000:
            raise ValueError("attention training example capacity")
        if type(steps) is not int or not 1 <= steps <= 2_000:
            raise ValueError("attention training steps must be in [1, 2000]")
        for name, value in (
            ("learning_rate", learning_rate),
            ("l2", l2),
        ):
            if type(value) not in (int, float) or not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 < learning_rate <= 2 or not 0 <= l2 <= 1:
            raise ValueError("invalid attention optimization parameters")

        matrix = np.stack([self.vector(item.features) for item in examples])
        labels = np.asarray([item.label for item in examples], dtype=np.float64)
        for _ in range(steps):
            logits = np.clip(matrix @ self.weights, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = matrix.T @ (probabilities - labels) / len(labels)
            gradient += float(l2) * self.weights
            self.weights -= float(learning_rate) * gradient

    def rank(
        self,
        candidates: list[AttentionCandidate],
        *,
        limit: int = 8,
    ) -> tuple[tuple[str, float], ...]:
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("attention rank limit must be in [1, 128]")
        if len(candidates) > 4096:
            raise ValueError("attention candidate capacity")
        scored = [
            (candidate.candidate_id, self.score(candidate.features))
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(scored[:limit])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ai2-open-attention-v1",
            "dimension": self.dimension,
            "seed": self.seed,
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HashedAttentionRanker:
        expected = {"schema", "dimension", "seed", "weights"}
        if (
            type(value) is not dict
            or set(value) != expected
            or value["schema"] != "ai2-open-attention-v1"
            or type(value["weights"]) is not list
        ):
            raise ValueError("invalid open attention model")
        return cls(
            dimension=value["dimension"],
            seed=value["seed"],
            weights=np.asarray(value["weights"], dtype=np.float64),
        )

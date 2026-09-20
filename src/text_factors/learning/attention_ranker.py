"""Small supervised retrieval ranker. Retrieval is not evidence of truth."""

from __future__ import annotations

import json
from importlib.resources import files
from math import isfinite
from typing import Any

import numpy as np

from .hypotheses import digest
from .schema import Event, Meaning

FEATURES = ("bias", "target", "object", "predicate", "participants", "scope", "time")


def features(target: str, meaning: Meaning, record: dict[str, Any]) -> list[float]:
    event = meaning.event
    other = record.get("index_event")
    candidate = Event.from_dict(other) if other is not None else None
    if event is None or candidate is None:
        return [
            1.0,
            float(target == record["observation"]["observation_id"]),
            *([0.0] * 5),
        ]
    people = {event.actor, event.recipient} - {""}
    previous = {candidate.actor, candidate.recipient} - {""}
    return [
        1.0,
        float(bool(target) and target == record["observation"]["observation_id"]),
        float(bool(event.object) and event.object == candidate.object),
        float(event.predicate == candidate.predicate),
        len(people & previous) / max(1, len(people | previous)),
        float(
            event.modality == candidate.modality and event.content == candidate.content
        ),
        float(event.time == candidate.time),
    ]


def fit(rows: list[list[float]], labels: list[int], *, steps: int = 400) -> list[float]:
    """Deterministic bounded logistic regression, called only by offline training."""
    if not 1 <= len(rows) <= 4096 or len(rows) != len(labels) or not 1 <= steps <= 1000:
        raise ValueError("attention training capacity")
    x, y = np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if x.shape != (len(rows), len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError("invalid attention features")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("invalid attention labels")
    # Center non-bias features so an irrelevant feature cannot act as an
    # accidental intercept during a finite training run. Export raw-space
    # weights, keeping runtime scoring independent of the training dataset.
    center = x.mean(axis=0)
    center[0] = 0.0
    x = x - center
    weights = np.zeros(len(FEATURES))
    for _ in range(steps):
        probabilities = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
        weights -= 0.25 * (x.T @ (probabilities - y) / len(y) + 0.002 * weights)
    weights[0] -= float(center @ weights)
    return weights.tolist()


class AttentionRanker:
    def __init__(self, weights: list[float]) -> None:
        if len(weights) != len(FEATURES) or any(
            type(w) not in (int, float) or not isfinite(w) or abs(w) > 100
            for w in weights
        ):
            raise ValueError("invalid attention ranker weights")
        self.weights = tuple(float(w) for w in weights)
        self.fingerprint = digest(
            {"features": list(FEATURES), "weights": list(self.weights)}
        )

    @classmethod
    def load(cls) -> AttentionRanker:
        value = json.loads(
            files("text_factors.learning")
            .joinpath("attention_weights.json")
            .read_text()
        )
        if value["features"] != list(FEATURES):
            raise ValueError("attention feature schema mismatch")
        return cls(value["weights"])

    def score(self, target: str, meaning: Meaning, record: dict[str, Any]) -> float:
        return sum(
            w * x
            for w, x in zip(
                self.weights, features(target, meaning, record), strict=True
            )
        )

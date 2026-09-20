"""Bounded, serializable proposals. Observations are never overwritten by a choice.

These are engineering contracts, not new trained parameters. A partial candidate
is retained for clarification and must not be projected into the factual world.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ..conversation.persistence import encode_json
from .schema import Meaning, bounded_text, exact_fields

SCHEMA = "ai2-hypotheses-v1"
MAX_CANDIDATES = 8
MAX_EXPANSIONS = 512


def digest(value: Any) -> str:
    return hashlib.sha256(encode_json({"value": value})).hexdigest()


@dataclass(frozen=True)
class Observation:
    observation_id: str
    text: str
    turn_id: int = 0
    source: str = "input"

    def __post_init__(self) -> None:
        bounded_text(self.observation_id, "observation id", empty=False)
        bounded_text(self.text, "observed text", cap=2048)
        bounded_text(self.source, "observation source", empty=False)
        if type(self.turn_id) is not int or not 0 <= self.turn_id < 2**53:
            raise ValueError("invalid observation turn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "text": self.text,
            "turn_id": self.turn_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Observation:
        return cls(
            **exact_fields(
                value, {"observation_id", "text", "turn_id", "source"}, "observation"
            )
        )


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    observation_id: str
    meaning: Meaning | None
    language_regret: float
    bindings: tuple[tuple[str, int, str], ...] = ()
    missing: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        bounded_text(self.observation_id, "hypothesis observation", empty=False)
        if self.meaning is not None and not isinstance(self.meaning, Meaning):
            raise ValueError("invalid candidate meaning")
        if (
            type(self.language_regret) not in (float, int)
            or not isfinite(self.language_regret)
            or not 0 <= self.language_regret <= 1000
        ):
            raise ValueError("invalid candidate language regret")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported hypothesis version")
        if type(self.missing) is not tuple or len(self.missing) > 16:
            raise ValueError("invalid missing structure")
        for name in self.missing:
            bounded_text(name, "missing role", empty=False)
        if self.meaning is None and not self.missing:
            raise ValueError("meaningless candidate must be partial")
        if type(self.bindings) is not tuple or len(self.bindings) > 16:
            raise ValueError("invalid bindings")
        for binding in self.bindings:
            if type(binding) is not tuple or len(binding) != 3:
                raise ValueError("invalid binding")
            role, index, entity = binding
            bounded_text(role, "bound role", empty=False)
            bounded_text(entity, "bound entity", empty=False)
            if type(index) is not int or not -2 <= index < 96:
                raise ValueError("invalid source token index")
        if self.hypothesis_id != self.identity(
            self.observation_id, self.meaning, self.missing
        ):
            raise ValueError("candidate identity does not match its meaning")

    @staticmethod
    def identity(
        observation_id: str, meaning: Meaning | None, missing: tuple[str, ...]
    ) -> str:
        return "h-" + digest(
            [observation_id, meaning.to_dict() if meaning else None, list(missing)]
        )

    @classmethod
    def create(
        cls,
        observation_id: str,
        meaning: Meaning | None,
        regret: float,
        bindings: tuple[tuple[str, int, str], ...] = (),
        missing: tuple[str, ...] = (),
    ) -> Hypothesis:
        return cls(
            cls.identity(observation_id, meaning, missing),
            observation_id,
            meaning,
            regret,
            bindings,
            missing,
        )

    @property
    def complete(self) -> bool:
        return self.meaning is not None and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "observation_id": self.observation_id,
            "meaning": self.meaning.to_dict() if self.meaning else None,
            "language_regret": float(self.language_regret),
            "bindings": [list(item) for item in self.bindings],
            "missing": list(self.missing),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Hypothesis:
        v = exact_fields(
            value,
            {
                "hypothesis_id",
                "observation_id",
                "meaning",
                "language_regret",
                "bindings",
                "missing",
                "version",
            },
            "hypothesis",
        )
        if type(v["bindings"]) is not list or type(v["missing"]) is not list:
            raise ValueError("invalid candidate collections")
        if any(type(item) is not list for item in v["bindings"]):
            raise ValueError("invalid saved binding")
        return cls(
            v["hypothesis_id"],
            v["observation_id"],
            Meaning.from_dict(v["meaning"]) if v["meaning"] is not None else None,
            v["language_regret"],
            tuple(tuple(b) for b in v["bindings"]),
            tuple(v["missing"]),
            v["version"],
        )


@dataclass(frozen=True)
class CandidateSet:
    observation: Observation
    context_digest: str
    candidates: tuple[Hypothesis, ...]
    complete: bool = True
    stop_reason: str = ""
    expansions: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise ValueError("invalid candidate observation")
        if (
            type(self.context_digest) is not str
            or len(self.context_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.context_digest)
        ):
            raise ValueError("invalid context digest")
        if type(self.candidates) is not tuple or len(self.candidates) > MAX_CANDIDATES:
            raise ValueError("candidate capacity")
        if any(
            not isinstance(c, Hypothesis)
            or c.observation_id != self.observation.observation_id
            for c in self.candidates
        ):
            raise ValueError("candidate provenance mismatch")
        if len({c.hypothesis_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate hypothesis")
        if type(self.complete) is not bool:
            raise ValueError("invalid completion status")
        bounded_text(self.stop_reason, "search stop reason")
        if self.complete == bool(self.stop_reason):
            raise ValueError("inconsistent search completion")
        if (
            type(self.expansions) is not int
            or not 0 <= self.expansions <= MAX_EXPANSIONS
        ):
            raise ValueError("expansion capacity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "observation": self.observation.to_dict(),
            "context_digest": self.context_digest,
            "candidates": [c.to_dict() for c in self.candidates],
            "complete": self.complete,
            "stop_reason": self.stop_reason,
            "expansions": self.expansions,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSet:
        v = exact_fields(
            value,
            {
                "schema",
                "observation",
                "context_digest",
                "candidates",
                "complete",
                "stop_reason",
                "expansions",
            },
            "candidate set",
        )
        if v["schema"] != SCHEMA or type(v["candidates"]) is not list:
            raise ValueError("invalid candidate set schema")
        return cls(
            Observation.from_dict(v["observation"]),
            v["context_digest"],
            tuple(Hypothesis.from_dict(c) for c in v["candidates"]),
            v["complete"],
            v["stop_reason"],
            v["expansions"],
        )

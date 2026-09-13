"""Strict, JSON-friendly contracts shared by the dialogue components."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from math import isfinite
from time import perf_counter
from typing import Any


def _json_dict(value: Any) -> dict[str, Any]:
    """Detach dataclasses and convert tuples into genuine JSON arrays."""
    return json.loads(json.dumps(asdict(value), ensure_ascii=False, allow_nan=False))


ACTS = frozenset(
    {
        "inform",
        "ask",
        "correct",
        "retract",
        "hypothesis",
        "greet",
        "thanks",
        "help",
        "teach",
        "topic",
        "unknown",
    }
)
PREDICATES = frozenset({"", "locate", "move", "give", "have"})
QUERIES = frozenset({"", "where", "who_has", "what_has", "verify", "why"})
MODALITIES = frozenset({"asserted", "hypothetical", "reported"})


class BudgetExceeded(RuntimeError):
    """A cooperative deadline or a bounded-work constraint was reached."""


@dataclass(frozen=True, slots=True)
class ConversationLimits:
    max_chars: int = 2048
    max_tokens: int = 128
    max_clauses: int = 8
    max_events: int = 512
    max_entities: int = 256
    max_history: int = 128
    max_candidates: int = 8
    max_training_examples: int = 256
    max_state_bytes: int = 4_000_000
    turn_seconds: float = 2.0

    def __post_init__(self) -> None:
        caps = {
            "max_chars": 16384,
            "max_tokens": 2048,
            "max_clauses": 32,
            "max_events": 4096,
            "max_entities": 1024,
            "max_history": 1024,
            "max_candidates": 32,
            "max_training_examples": 1024,
            "max_state_bytes": 16_000_000,
        }
        for name, cap in caps.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} must be an integer in [1, {cap}]")
        if (
            type(self.turn_seconds) not in (int, float)
            or not isfinite(self.turn_seconds)
            or not 0 < self.turn_seconds <= 30
        ):
            raise ValueError("turn_seconds must be finite and in (0, 30]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> ConversationLimits:
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("invalid conversation limits")
        return cls(**value)


class Budget:
    def __init__(self, seconds: float) -> None:
        if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        self.started = perf_counter()
        self.deadline = self.started + seconds

    def check(self) -> None:
        if perf_counter() >= self.deadline:
            raise BudgetExceeded("turn_time_budget")


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    """Raw cue is interpreted by the learned bridge, never by the role parser.

    Pronoun markers may be @object, @person, @place, @ambiguous or @speaker.
    Canonical entity strings identify objects in a bounded topic-local world.
    """

    act: str
    cue: str = ""
    predicate: str = ""
    actor: str = ""
    object: str = ""
    object_kind: str = "unknown"
    place: str = ""
    spatial: str = "in"
    recipient: str = ""
    query: str = ""
    negated: bool = False
    modality: str = "asserted"
    tense: str = "current"
    topic: str = ""
    reference: int | None = None
    raw: str = ""

    def __post_init__(self) -> None:
        for name in (
            "act",
            "cue",
            "predicate",
            "actor",
            "object",
            "object_kind",
            "place",
            "spatial",
            "recipient",
            "query",
            "modality",
            "tense",
            "topic",
            "raw",
        ):
            value = getattr(self, name)
            cap = 2048 if name == "raw" else 128
            if type(value) is not str or len(value) > cap or "\x00" in value:
                raise ValueError(f"invalid frame field {name}")
        if (
            self.act not in ACTS
            or self.predicate not in PREDICATES
            or self.query not in QUERIES
            or self.modality not in MODALITIES
        ):
            raise ValueError("invalid semantic frame category")
        if self.tense not in {"current", "past", "future"}:
            raise ValueError("invalid tense")
        if self.spatial not in {"in", "on"}:
            raise ValueError("invalid spatial relation")
        if self.object_kind not in {"unknown", "person", "thing"}:
            raise ValueError("invalid object kind")
        if type(self.negated) is not bool:
            raise ValueError("negated must be a boolean")
        if self.reference is not None and (
            type(self.reference) is not int or self.reference < 1
        ):
            raise ValueError("reference must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SemanticFrame:
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("invalid semantic frame fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ParseResult:
    frames: tuple[SemanticFrame, ...] = ()
    complete: bool = True
    reason: str = ""
    alternatives: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(self)


@dataclass(frozen=True, slots=True)
class Assertion:
    subject: str
    relation: str
    value: str
    event_id: int
    negated: bool = False
    source: str = "user"
    topic: str = "default"
    qualifier: str = "in"

    def __post_init__(self) -> None:
        for name in ("subject", "relation", "value", "source", "topic", "qualifier"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not 0 < len(value) <= 128
                or any(ord(c) < 32 or ord(c) == 127 for c in value)
            ):
                raise ValueError(f"invalid assertion {name}")
        if self.relation not in {"location", "holder"} or self.qualifier not in {
            "in",
            "on",
        }:
            raise ValueError("invalid assertion relation")
        if type(self.negated) is not bool:
            raise ValueError("invalid assertion polarity")
        if type(self.event_id) is not int or not 1 <= self.event_id < 2**53:
            raise ValueError("invalid assertion event identifier")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> Assertion:
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("invalid assertion fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StateOutcome:
    action: str
    assertions: tuple[Assertion, ...] = ()
    reason: str = ""
    event_ids: tuple[int, ...] = ()
    alternatives: tuple[str, ...] = ()
    resolved_frame: SemanticFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(self)


@dataclass(frozen=True, slots=True)
class TurnResponse:
    turn_id: int
    text: str
    action: str
    frames: tuple[SemanticFrame, ...] = ()
    assertions: tuple[Assertion, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    complete: bool = True
    reason: str = ""
    elapsed_seconds: float = 0.0
    alternatives: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(self)

    @classmethod
    def from_dict(cls, value: Any) -> TurnResponse:
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("invalid response fields")
        if type(value["turn_id"]) is not int or not 0 <= value["turn_id"] < 2**53:
            raise ValueError("invalid response turn identifier")
        for name, cap in (("text", 16384), ("action", 64), ("reason", 128)):
            if (
                type(value[name]) is not str
                or len(value[name]) > cap
                or "\x00" in value[name]
            ):
                raise ValueError(f"invalid response {name}")
        if type(value["complete"]) is not bool:
            raise ValueError("invalid response completeness")
        if value["action"] not in {
            "ack",
            "answer",
            "unknown",
            "clarify",
            "hypothetical",
            "retracted",
            "topic",
            "help",
            "greet",
            "thanks",
            "taught",
        } or value["complete"] == (value["action"] == "clarify"):
            raise ValueError("invalid response action/completeness")
        elapsed = value["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid response elapsed time")
        for name, cap in (
            ("frames", 32),
            ("assertions", 4096),
            ("evidence", 32),
            ("alternatives", 32),
        ):
            if type(value[name]) is not list or len(value[name]) > cap:
                raise ValueError(f"invalid response {name}")
        if any(type(item) is not dict for item in value["evidence"]):
            raise ValueError("invalid response evidence")
        if any(
            type(item) is not str or len(item) > 128 for item in value["alternatives"]
        ):
            raise ValueError("invalid response alternatives")
        try:
            if len(json.dumps(value, allow_nan=False).encode()) > 1_000_000:
                raise ValueError("response is too large")
        except (TypeError, RecursionError) as error:
            raise ValueError("invalid response data") from error
        return cls(
            **{
                k: v
                for k, v in value.items()
                if k not in {"frames", "assertions", "evidence", "alternatives"}
            },
            frames=tuple(SemanticFrame.from_dict(f) for f in value["frames"]),
            assertions=tuple(Assertion.from_dict(a) for a in value["assertions"]),
            evidence=tuple(json.loads(json.dumps(value["evidence"], allow_nan=False))),
            alternatives=tuple(value["alternatives"]),
        )

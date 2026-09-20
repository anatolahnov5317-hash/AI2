"""Bounded meaning trees and evidence contracts for the learned dialogue track.

These types are an engineering ontology, not a claim that an ontology alone
implements Redozubov's theory. Non-actual scopes wrap events; their contents
must never be silently projected into the factual world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from ..conversation.persistence import decode_json, encode_json

ACTS = frozenset(
    {"inform", "ask", "correct", "retract", "greet", "thanks", "help", "unknown"}
)
PREDICATES = frozenset(
    {"locate", "move", "give", "have", "promise", "report", "conditional"}
)
MODALITIES = frozenset({"actual", "possible", "intended", "reported", "conditional"})
TIMES = frozenset({"past", "present", "future", "unspecified"})
QUERIES = frozenset({"where", "who_has", "what_has", "verify", "why"})
_MAX_DIAGNOSTIC_BYTES = 65_536


def bounded_text(value: Any, name: str, *, cap: int = 128, empty: bool = True) -> str:
    if type(value) is not str or len(value) > cap or (not empty and not value):
        raise ValueError(f"invalid {name}")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"invalid control character in {name}")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"invalid Unicode in {name}") from exc
    return value


def exact_fields(value: Any, names: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != names:
        raise ValueError(f"invalid {name} fields")
    return value


def _category(value: Any, choices: frozenset[str] | set[str], name: str) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True, slots=True)
class Entity:
    name: str
    kind: str = "unknown"
    gender: str = "unknown"

    def __post_init__(self) -> None:
        bounded_text(self.name, "entity name", empty=False)
        _category(self.kind, {"person", "thing", "place", "unknown"}, "entity kind")
        _category(self.gender, {"male", "female", "neuter", "unknown"}, "entity gender")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> Entity:
        return cls(**exact_fields(value, {"name", "kind", "gender"}, "entity"))


@dataclass(frozen=True, slots=True)
class Event:
    predicate: str
    actor: str = ""
    object: str = ""
    recipient: str = ""
    place: str = ""
    spatial: str = "in"
    negated: bool = False
    modality: str = "actual"
    time: str = "past"
    content: Event | None = None
    condition: Event | None = None

    def __post_init__(self) -> None:
        _category(self.predicate, PREDICATES, "event predicate")
        _category(self.modality, MODALITIES, "event modality")
        _category(self.time, TIMES, "event time")
        for name in ("actor", "object", "recipient", "place"):
            bounded_text(getattr(self, name), name)
        _category(self.spatial, {"in", "on"}, "event spatial relation")
        if type(self.negated) is not bool:
            raise ValueError("invalid event qualifier")
        if self.predicate in {"promise", "report", "conditional"}:
            if not isinstance(self.content, Event):
                raise ValueError("scoped events require content")
        elif self.content is not None or self.condition is not None:
            raise ValueError("atomic event cannot contain nested content")
        if self.condition is not None and (
            self.predicate != "conditional" or not isinstance(self.condition, Event)
        ):
            raise ValueError("invalid event condition")
        if self.depth() > 4:
            raise ValueError("meaning tree depth exceeds 4")

    def depth(self) -> int:
        return 1 + max(
            (
                node.depth()
                for node in (self.content, self.condition)
                if node is not None
            ),
            default=0,
        )

    @property
    def actual(self) -> bool:
        return (
            self.content is None and self.modality == "actual" and self.time != "future"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any, *, _depth: int = 0) -> Event:
        if type(_depth) is not int or not 0 <= _depth < 4:
            raise ValueError("meaning tree depth exceeds 4")
        value = dict(
            exact_fields(
                value,
                {
                    "predicate",
                    "actor",
                    "object",
                    "recipient",
                    "place",
                    "spatial",
                    "negated",
                    "modality",
                    "time",
                    "content",
                    "condition",
                },
                "event",
            )
        )
        for name in ("content", "condition"):
            if value[name] is not None:
                value[name] = cls.from_dict(value[name], _depth=_depth + 1)
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Query:
    kind: str
    subject: str = ""
    value: str = ""
    relation: str = ""
    time: str = "present"
    spatial: str = "in"
    negated: bool = False

    def __post_init__(self) -> None:
        if type(self.negated) is not bool:
            raise ValueError("invalid query polarity")
        _category(self.kind, QUERIES, "query kind")
        _category(self.time, TIMES, "query time")
        for name in ("subject", "value", "relation"):
            bounded_text(getattr(self, name), name)
        _category(self.relation, {"", "location", "holder"}, "query relation")
        _category(self.spatial, {"in", "on"}, "query spatial relation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> Query:
        return cls(
            **exact_fields(
                value,
                {"kind", "subject", "value", "relation", "time", "spatial", "negated"},
                "query",
            )
        )


@dataclass(frozen=True, slots=True)
class Meaning:
    act: str
    event: Event | None = None
    query: Query | None = None
    entities: tuple[Entity, ...] = ()

    def __post_init__(self) -> None:
        _category(self.act, ACTS, "meaning act")
        if self.event is not None and not isinstance(self.event, Event):
            raise ValueError("invalid meaning event")
        if self.query is not None and not isinstance(self.query, Query):
            raise ValueError("invalid meaning query")
        if (
            type(self.entities) is not tuple
            or len(self.entities) > 32
            or any(not isinstance(entity, Entity) for entity in self.entities)
        ):
            raise ValueError("invalid meaning entities")
        if self.act in {"inform", "correct"} and (
            self.event is None or self.query is not None
        ):
            raise ValueError("inform/correct require only an event")
        if self.act == "ask" and (self.query is None or self.event is not None):
            raise ValueError("ask requires only a query")
        if self.act not in {"inform", "correct", "ask"} and (
            self.event is not None or self.query is not None
        ):
            raise ValueError("unexpected semantic content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "act": self.act,
            "event": self.event.to_dict() if self.event else None,
            "query": self.query.to_dict() if self.query else None,
            "entities": [e.to_dict() for e in self.entities],
        }

    @classmethod
    def from_dict(cls, value: Any) -> Meaning:
        value = exact_fields(value, {"act", "event", "query", "entities"}, "meaning")
        if type(value["entities"]) is not list or len(value["entities"]) > 32:
            raise ValueError("invalid meaning entities")
        return cls(
            value["act"],
            Event.from_dict(value["event"]) if value["event"] is not None else None,
            Query.from_dict(value["query"]) if value["query"] is not None else None,
            tuple(Entity.from_dict(e) for e in value["entities"]),
        )


@dataclass(frozen=True, slots=True)
class Interpretation:
    meaning: Meaning | None
    score: float = 0.0
    alternatives: tuple[Meaning, ...] = ()
    reason: str = ""
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.meaning is not None and not isinstance(self.meaning, Meaning):
            raise ValueError("invalid interpreted meaning")
        if type(self.score) not in (float, int):
            raise ValueError("invalid interpretation score")
        try:
            finite = isfinite(self.score)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("invalid interpretation score")
        if (
            type(self.alternatives) is not tuple
            or len(self.alternatives) > 8
            or any(not isinstance(m, Meaning) for m in self.alternatives)
        ):
            raise ValueError("invalid interpretation alternatives")
        bounded_text(self.reason, "interpretation reason", cap=256)
        if self.diagnostics is not None:
            # Validate strict types, finite numbers, tree depth and byte size;
            # retain a bounded copy rather than a caller-owned mutable mapping.
            raw = encode_json(self.diagnostics, max_bytes=_MAX_DIAGNOSTIC_BYTES)
            object.__setattr__(
                self,
                "diagnostics",
                decode_json(raw, max_bytes=_MAX_DIAGNOSTIC_BYTES),
            )


@dataclass(frozen=True, slots=True)
class DialogueContext:
    """Bounded preceding turns and typed referents; no hidden training targets."""

    turns: tuple[str, ...] = ()
    entities: tuple[Entity, ...] = ()
    focus: tuple[str, ...] = ()
    pending: str = ""

    def __post_init__(self) -> None:
        if type(self.turns) is not tuple or len(self.turns) > 16:
            raise ValueError("invalid context turns")
        for turn in self.turns:
            bounded_text(turn, "context turn", cap=2048)
        if (
            type(self.entities) is not tuple
            or len(self.entities) > 64
            or any(not isinstance(e, Entity) for e in self.entities)
        ):
            raise ValueError("invalid context entities")
        if type(self.focus) is not tuple or len(self.focus) > 16:
            raise ValueError("invalid context focus")
        for name in self.focus:
            bounded_text(name, "context focus", empty=False)
        bounded_text(self.pending, "pending intent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": list(self.turns),
            "entities": [e.to_dict() for e in self.entities],
            "focus": list(self.focus),
            "pending": self.pending,
        }

    @classmethod
    def from_dict(cls, value: Any) -> DialogueContext:
        value = exact_fields(
            value, {"turns", "entities", "focus", "pending"}, "context"
        )
        if any(type(value[key]) is not list for key in ("turns", "entities", "focus")):
            raise ValueError("invalid context collections")
        if (
            len(value["turns"]) > 16
            or len(value["entities"]) > 64
            or len(value["focus"]) > 16
        ):
            raise ValueError("context collection exceeds its capacity")
        return cls(
            tuple(value["turns"]),
            tuple(Entity.from_dict(e) for e in value["entities"]),
            tuple(value["focus"]),
            value["pending"],
        )

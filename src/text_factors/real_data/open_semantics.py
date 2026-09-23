"""Bounded, supervised event graph for externally identified mentions.

The lexical cues, relation IDs, role names, polarity cues, and time labels
come from annotated training examples. This is an intentionally narrow
research baseline: it does not perform mention detection, infer coreference,
or turn a parsed event into an asserted fact in the evidence ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_BREAK = re.compile(r"[.;!?\n]")
_MAX_CHARS = 2048
_MAX_EXAMPLES = 512
_MAX_MENTIONS = 64
_MAX_CLAUSES = 32
_MAX_TOKENS = 256
_MAX_EVENTS = 32
_Signature = tuple[str, int, str, str | None]


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must be a nonempty half-open range")

    @classmethod
    def from_dict(cls, value: Any) -> Span:
        if type(value) is not dict or set(value) != {"start", "end"}:
            raise ValueError("invalid span record")
        return cls(value["start"], value["end"])


@dataclass(frozen=True, slots=True)
class IdentifiedMention:
    mention_id: str
    instance_id: str
    span: Span
    kind: str = "instance"
    morphology: str | None = None

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.mention_id, self.instance_id, self.kind)
        ):
            raise ValueError("mention IDs and kind must be nonempty strings")
        if self.morphology is not None and (
            type(self.morphology) is not str or not self.morphology
        ):
            raise ValueError("invalid externally annotated morphology")


@dataclass(frozen=True, slots=True)
class LabeledEvent:
    relation_id: str
    trigger: Span
    roles: tuple[tuple[str, str], ...]  # (role, mention_id)
    negation_cue: Span | None = None
    time_cue: Span | None = None
    time_label: str | None = None

    def __post_init__(self) -> None:
        if type(self.relation_id) is not str or not self.relation_id:
            raise ValueError("relation_id must be nonempty")
        if not self.roles or len({role for role, _ in self.roles}) != len(self.roles):
            raise ValueError("event needs unique annotated roles")
        if any(
            type(role) is not str or not role or type(mid) is not str or not mid
            for role, mid in self.roles
        ):
            raise ValueError("invalid role annotation")
        if (self.time_cue is None) != (self.time_label is None):
            raise ValueError("time cue and label must both be given")
        if self.time_label is not None and (
            type(self.time_label) is not str or not self.time_label
        ):
            raise ValueError("invalid time label")


@dataclass(frozen=True, slots=True)
class LabeledLink:
    """Directed relation between two annotated event triggers."""

    relation_id: str
    source_trigger: Span
    target_trigger: Span
    cue: Span

    def __post_init__(self) -> None:
        if type(self.relation_id) is not str or not self.relation_id:
            raise ValueError("link relation ID required")


@dataclass(frozen=True, slots=True)
class LabeledText:
    text: str
    mentions: tuple[IdentifiedMention, ...]
    events: tuple[LabeledEvent, ...]
    links: tuple[LabeledLink, ...] = ()


@dataclass(frozen=True, slots=True)
class EventRole:
    role: str
    mention_id: str
    instance_id: str

    @classmethod
    def from_dict(cls, value: Any) -> EventRole:
        if type(value) is not dict or set(value) != {
            "role",
            "mention_id",
            "instance_id",
        }:
            raise ValueError("invalid graph role")
        if any(type(item) is not str or not item for item in value.values()):
            raise ValueError("invalid graph role ID")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    event_id: str
    relation_id: str
    trigger: Span
    source_span: Span
    roles: tuple[EventRole, ...]
    negated: bool
    time_label: str | None

    def __post_init__(self) -> None:
        if type(self.event_id) is not str or not self.event_id:
            raise ValueError("invalid event ID")
        if type(self.relation_id) is not str or not self.relation_id:
            raise ValueError("invalid relation ID")
        if type(self.negated) is not bool:
            raise ValueError("invalid negation")
        if self.time_label is not None and (
            type(self.time_label) is not str or not self.time_label
        ):
            raise ValueError("invalid time label")
        if not _within(self.trigger, self.source_span):
            raise ValueError("event trigger outside source span")
        if not self.roles or len({item.role for item in self.roles}) != len(self.roles):
            raise ValueError("invalid graph roles")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "relation_id": self.relation_id,
            "trigger": {"start": self.trigger.start, "end": self.trigger.end},
            "source_span": {
                "start": self.source_span.start,
                "end": self.source_span.end,
            },
            "roles": [
                {
                    "role": r.role,
                    "mention_id": r.mention_id,
                    "instance_id": r.instance_id,
                }
                for r in self.roles
            ],
            "negated": self.negated,
            "time_label": self.time_label,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SemanticEvent:
        if (
            type(value) is not dict
            or set(value)
            != {
                "event_id",
                "relation_id",
                "trigger",
                "source_span",
                "roles",
                "negated",
                "time_label",
            }
            or type(value["roles"]) is not list
        ):
            raise ValueError("invalid graph event")
        return cls(
            event_id=value["event_id"],
            relation_id=value["relation_id"],
            trigger=Span.from_dict(value["trigger"]),
            source_span=Span.from_dict(value["source_span"]),
            roles=tuple(EventRole.from_dict(item) for item in value["roles"]),
            negated=value["negated"],
            time_label=value["time_label"],
        )


@dataclass(frozen=True, slots=True)
class SemanticLink:
    relation_id: str
    source_event_id: str
    target_event_id: str
    cue: Span

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str or not value
                for value in (
                    self.relation_id,
                    self.source_event_id,
                    self.target_event_id,
                )
            )
            or self.source_event_id == self.target_event_id
        ):
            raise ValueError("invalid event link")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "cue": {"start": self.cue.start, "end": self.cue.end},
        }

    @classmethod
    def from_dict(cls, value: Any) -> SemanticLink:
        if type(value) is not dict or set(value) != {
            "relation_id",
            "source_event_id",
            "target_event_id",
            "cue",
        }:
            raise ValueError("invalid event link record")
        return cls(
            value["relation_id"],
            value["source_event_id"],
            value["target_event_id"],
            Span.from_dict(value["cue"]),
        )


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    """Versioned candidate graph. Residual text prevents false completeness."""

    source_id: str
    source_version: int
    text_sha256: str
    model_fingerprint: str
    events: tuple[SemanticEvent, ...]
    unexplained: tuple[Span, ...]
    links: tuple[SemanticLink, ...] = ()
    schema: str = "ai2-supervised-event-graph-v1"

    def __post_init__(self) -> None:
        if type(self.source_id) is not str or not self.source_id:
            raise ValueError("missing source ID")
        if type(self.source_version) is not int or self.source_version <= 0:
            raise ValueError("invalid source version")
        for digest in (self.text_sha256, self.model_fingerprint):
            if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid graph digest")
        ids = {event.event_id for event in self.events}
        if len(ids) != len(self.events) or any(
            link.source_event_id not in ids or link.target_event_id not in ids
            for link in self.links
        ):
            raise ValueError("invalid graph event/link IDs")

    @property
    def complete(self) -> bool:
        return bool(self.events) and not self.unexplained

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "text_sha256": self.text_sha256,
            "model_fingerprint": self.model_fingerprint,
            "events": [event.to_dict() for event in self.events],
            "links": [link.to_dict() for link in self.links],
            "unexplained": [
                {"start": span.start, "end": span.end} for span in self.unexplained
            ],
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SemanticGraph:
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "source_id",
                "source_version",
                "text_sha256",
                "model_fingerprint",
                "events",
                "links",
                "unexplained",
                "complete",
            }
            or value["schema"] != "ai2-supervised-event-graph-v1"
        ):
            raise ValueError("invalid graph schema")
        if (
            type(value["events"]) is not list
            or type(value["links"]) is not list
            or type(value["unexplained"]) is not list
        ):
            raise ValueError("invalid graph collections")
        result = cls(
            source_id=value["source_id"],
            source_version=value["source_version"],
            text_sha256=value["text_sha256"],
            model_fingerprint=value["model_fingerprint"],
            events=tuple(SemanticEvent.from_dict(item) for item in value["events"]),
            unexplained=tuple(Span.from_dict(item) for item in value["unexplained"]),
            links=tuple(SemanticLink.from_dict(item) for item in value["links"]),
        )
        if type(value["complete"]) is not bool or value["complete"] != result.complete:
            raise ValueError("graph completeness mismatch")
        return result


def _clauses(text: str) -> tuple[Span, ...]:
    start = 0
    parts: list[Span] = []
    for match in _BREAK.finditer(text):
        left = start
        right = match.start()
        while left < right and text[left].isspace():
            left += 1
        while left < right and text[right - 1].isspace():
            right -= 1
        if left < right:
            parts.append(Span(left, right))
        start = match.end()
    left = start
    right = len(text)
    while left < right and text[left].isspace():
        left += 1
    while left < right and text[right - 1].isspace():
        right -= 1
    if left < right:
        parts.append(Span(left, right))
    if len(parts) > _MAX_CLAUSES:
        raise ValueError("clause limit")
    return tuple(parts)


def _within(inner: Span, outer: Span) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def _words(text: str, part: Span) -> list[tuple[str, Span]]:
    return [
        (match.group().casefold(), Span(match.start(), match.end()))
        for match in _WORD.finditer(text, part.start, part.end)
    ]


def _cue(text: str, span: Span) -> str:
    words = _words(text, span)
    if not words or words[0][1].start != span.start or words[-1][1].end != span.end:
        raise ValueError("cue must have word boundaries")
    if any(
        not text[left.end : right.start].isspace()
        for (_, left), (_, right) in zip(words, words[1:], strict=False)
    ):
        raise ValueError("cue words must be separated by whitespace")
    return " ".join(word for word, _ in words)


def _residual(text: str, part: Span, covered: list[Span]) -> list[Span]:
    """Keep unknown punctuation as well as unknown words in the residual."""
    result: list[Span] = []
    pending: int | None = None
    for at in range(part.start, part.end):
        consumed = text[at].isspace() or any(
            span.start <= at < span.end for span in covered
        )
        if consumed:
            if pending is not None:
                result.append(Span(pending, at))
                pending = None
        elif pending is None:
            pending = at
    if pending is not None:
        result.append(Span(pending, part.end))
    return result


def _validate_text(
    text: str, mentions: tuple[IdentifiedMention, ...]
) -> tuple[Span, ...]:
    if type(text) is not str or not text or len(text) > _MAX_CHARS:
        raise ValueError("text must be nonempty and within character limit")
    text.encode("utf-8", errors="strict")
    if len(mentions) > _MAX_MENTIONS:
        raise ValueError("mention limit")
    if len({m.mention_id for m in mentions}) != len(mentions):
        raise ValueError("duplicate mention ID")
    parts = _clauses(text)
    ordered = sorted(mentions, key=lambda m: m.span.start)
    for index, mention in enumerate(ordered):
        if mention.span.end > len(text) or not any(
            _within(mention.span, part) for part in parts
        ):
            raise ValueError("mention outside a clause")
        if index and ordered[index - 1].span.end > mention.span.start:
            raise ValueError("overlapping mentions are not supported")
    if sum(len(_words(text, part)) for part in parts) > _MAX_TOKENS:
        raise ValueError("token limit")
    return parts


def _matches(words: list[tuple[str, Span]], cue: str) -> list[Span]:
    bits = cue.split(" ")
    matches = []
    for at in range(len(words) - len(bits) + 1):
        if [word for word, _ in words[at : at + len(bits)]] == bits:
            matches.append(Span(words[at][1].start, words[at + len(bits) - 1][1].end))
    return matches


def _role_signature(
    mention: IdentifiedMention, trigger: Span, mentions: tuple[IdentifiedMention, ...]
) -> _Signature:
    if mention.span.end <= trigger.start:
        earlier = [m for m in mentions if m.span.end <= trigger.start]
        rank = len(earlier) - earlier.index(mention)
        return "before", rank, mention.kind, mention.morphology
    if mention.span.start >= trigger.end:
        later = [m for m in mentions if m.span.start >= trigger.end]
        return "after", later.index(mention) + 1, mention.kind, mention.morphology
    raise ValueError("trigger overlaps a role mention")


class OpenSemanticModel:
    """Memorizes annotated cue forms and unique relation-specific role layouts.

    Forms and layouts are acquired only in ``fit``. A conflicting cue, role
    layout or time label abstains locally. This baseline has no data-independent
    verb semantics or case/morphology inference.
    """

    def __init__(self) -> None:
        self._relations: dict[str, set[str]] = {}
        self._layouts: dict[
            tuple[str, str], set[tuple[tuple[str, _Signature], ...]]
        ] = {}
        self._negations: set[str] = set()
        self._times: dict[str, set[str]] = {}
        self._event_links: dict[str, set[str]] = {}
        self._fitted = False
        self.model_fingerprint: str | None = None

    def fit(self, examples: tuple[LabeledText, ...]) -> OpenSemanticModel:
        if not 1 <= len(examples) <= _MAX_EXAMPLES:
            raise ValueError("training requires 1..512 annotated texts")
        relations: dict[str, set[str]] = {}
        layouts: dict[tuple[str, str], set[tuple[tuple[str, _Signature], ...]]] = {}
        negations: set[str] = set()
        times: dict[str, set[str]] = {}
        links: dict[str, set[str]] = {}
        for example in examples:
            if type(example) is not LabeledText:
                raise ValueError("invalid training example")
            parts = _validate_text(example.text, example.mentions)
            if not 1 <= len(example.events) <= _MAX_EVENTS:
                raise ValueError("training example needs 1..32 events")
            by_id = {m.mention_id: m for m in example.mentions}
            triggers = set()
            for event in example.events:
                part = next(
                    (part for part in parts if _within(event.trigger, part)), None
                )
                if part is None or (event.trigger.start, event.trigger.end) in triggers:
                    raise ValueError("duplicate or out-of-clause trigger")
                triggers.add((event.trigger.start, event.trigger.end))
                cue = _cue(example.text, event.trigger)
                relations.setdefault(cue, set()).add(event.relation_id)
                local = tuple(
                    sorted(
                        (m for m in example.mentions if _within(m.span, part)),
                        key=lambda m: m.span.start,
                    )
                )
                slots = []
                for role, mention_id in event.roles:
                    if mention_id not in by_id or by_id[mention_id] not in local:
                        raise ValueError("role mention is absent from event clause")
                    slots.append(
                        (role, _role_signature(by_id[mention_id], event.trigger, local))
                    )
                layouts.setdefault((cue, event.relation_id), set()).add(
                    tuple(sorted(slots))
                )
                for marker in (event.negation_cue, event.time_cue):
                    if marker is not None and (
                        not _within(marker, part)
                        or any(
                            not (
                                marker.end <= m.span.start or marker.start >= m.span.end
                            )
                            for m in local
                        )
                        or not (
                            marker.end <= event.trigger.start
                            or marker.start >= event.trigger.end
                        )
                    ):
                        raise ValueError("modifier cue overlaps an argument or trigger")
                if event.negation_cue is not None:
                    negations.add(_cue(example.text, event.negation_cue))
                if event.time_cue is not None and event.time_label is not None:
                    times.setdefault(_cue(example.text, event.time_cue), set()).add(
                        event.time_label
                    )
            if len(example.links) > _MAX_EVENTS:
                raise ValueError("link limit")
            for link in example.links:
                if (link.source_trigger.start, link.source_trigger.end) not in triggers:
                    raise ValueError("link source event missing")
                if (link.target_trigger.start, link.target_trigger.end) not in triggers:
                    raise ValueError("link target event missing")
                if not (
                    link.source_trigger.end <= link.cue.start
                    and link.cue.end <= link.target_trigger.start
                ):
                    raise ValueError("link cue must be between ordered triggers")
                links.setdefault(_cue(example.text, link.cue), set()).add(
                    link.relation_id
                )
        self._relations = relations
        self._layouts = layouts
        self._negations = negations
        self._times = times
        self._event_links = links
        self._fitted = True
        self.model_fingerprint = hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return self

    def to_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("fit the model before serializing")
        return {
            "schema": "ai2-open-semantic-model-v1",
            "relations": {
                cue: sorted(ids) for cue, ids in sorted(self._relations.items())
            },
            "layouts": [
                {
                    "cue": cue,
                    "relation_id": relation,
                    "options": [
                        [[role, list(sig)] for role, sig in layout]
                        for layout in sorted(options, key=repr)
                    ],
                }
                for (cue, relation), options in sorted(self._layouts.items())
            ],
            "negation_cues": sorted(self._negations),
            "time_cues": {
                cue: sorted(labels) for cue, labels in sorted(self._times.items())
            },
            "event_link_cues": {
                cue: sorted(labels) for cue, labels in sorted(self._event_links.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> OpenSemanticModel:
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "relations",
                "layouts",
                "negation_cues",
                "time_cues",
                "event_link_cues",
            }
            or value["schema"] != "ai2-open-semantic-model-v1"
        ):
            raise ValueError("invalid semantic model schema")
        relations = value["relations"]
        raw_layouts = value["layouts"]
        negations = value["negation_cues"]
        times = value["time_cues"]
        links = value["event_link_cues"]
        if (
            type(relations) is not dict
            or not 1 <= len(relations) <= _MAX_EXAMPLES * 32
            or type(raw_layouts) is not list
            or len(raw_layouts) > _MAX_EXAMPLES * 32
            or type(negations) is not list
            or len(negations) > _MAX_EXAMPLES * 32
            or type(times) is not dict
            or len(times) > _MAX_EXAMPLES * 32
            or type(links) is not dict
            or len(links) > _MAX_EXAMPLES * 32
        ):
            raise ValueError("invalid semantic model collections")

        def word_cue(cue: Any) -> bool:
            return (
                type(cue) is str
                and bool(cue)
                and len(cue) <= 128
                and all(
                    _WORD.fullmatch(word) and word == word.casefold()
                    for word in cue.split(" ")
                )
            )

        def labels(items: Any) -> bool:
            return (
                type(items) is list
                and bool(items)
                and len(items) <= 32
                and all(type(item) is str and 0 < len(item) <= 128 for item in items)
                and items == sorted(set(items))
            )

        if not all(word_cue(cue) and labels(ids) for cue, ids in relations.items()):
            raise ValueError("invalid relation cues")
        if not all(type(cue) is str for cue in negations):
            raise ValueError("invalid negation cues")
        if negations != sorted(set(negations)) or not all(
            word_cue(cue) for cue in negations
        ):
            raise ValueError("invalid negation cues")
        if not all(word_cue(cue) and labels(ids) for cue, ids in times.items()):
            raise ValueError("invalid time cues")
        if not all(word_cue(cue) and labels(ids) for cue, ids in links.items()):
            raise ValueError("invalid event-link cues")
        reconstructed: dict[
            tuple[str, str], set[tuple[tuple[str, _Signature], ...]]
        ] = {}
        for row in raw_layouts:
            if (
                type(row) is not dict
                or set(row) != {"cue", "relation_id", "options"}
                or not word_cue(row["cue"])
            ):
                raise ValueError("invalid role layout")
            cue, relation = row["cue"], row["relation_id"]
            if (
                relation not in relations.get(cue, [])
                or type(row["options"]) is not list
            ):
                raise ValueError("role layout has unknown cue/relation")
            options = set()
            for raw_layout in row["options"]:
                if type(raw_layout) is not list or not 1 <= len(raw_layout) <= 32:
                    raise ValueError("invalid role layout options")
                slots = []
                for raw_slot in raw_layout:
                    if (
                        type(raw_slot) is not list
                        or len(raw_slot) != 2
                        or type(raw_slot[0]) is not str
                        or not raw_slot[0]
                        or type(raw_slot[1]) is not list
                        or len(raw_slot[1]) != 4
                    ):
                        raise ValueError("invalid role slot")
                    direction, rank, kind, morphology = raw_slot[1]
                    if (
                        direction not in ("before", "after")
                        or type(rank) is not int
                        or not 1 <= rank <= _MAX_MENTIONS
                        or type(kind) is not str
                        or not kind
                        or (
                            morphology is not None
                            and (type(morphology) is not str or not morphology)
                        )
                    ):
                        raise ValueError("invalid role signature")
                    slots.append((raw_slot[0], (direction, rank, kind, morphology)))
                if len({role for role, _ in slots}) != len(slots):
                    raise ValueError("duplicate role in layout")
                options.add(tuple(sorted(slots)))
            key = cue, relation
            if key in reconstructed or not options:
                raise ValueError("duplicate or empty role layout")
            reconstructed[key] = options
        expected = {
            (cue, relation)
            for cue, choices in relations.items()
            for relation in choices
        }
        if set(reconstructed) != expected:
            raise ValueError("missing role layout")
        model = cls()
        model._relations = {cue: set(ids) for cue, ids in relations.items()}
        model._layouts = reconstructed
        model._negations = set(negations)
        model._times = {cue: set(ids) for cue, ids in times.items()}
        model._event_links = {cue: set(ids) for cue, ids in links.items()}
        model._fitted = True
        if model.to_dict() != value:
            raise ValueError("noncanonical model encoding")
        model.model_fingerprint = hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return model

    def parse(
        self,
        text: str,
        mentions: tuple[IdentifiedMention, ...],
        *,
        source_id: str,
        source_version: int,
    ) -> SemanticGraph:
        if not self._fitted:
            raise RuntimeError("fit the model before parsing")
        if type(source_id) is not str or not source_id:
            raise ValueError("source ID required")
        if type(source_version) is not int or source_version <= 0:
            raise ValueError("positive source version required")
        parts = _validate_text(text, mentions)
        part_index = {(part.start, part.end): index for index, part in enumerate(parts)}
        events: list[SemanticEvent] = []
        unexplained: list[Span] = []
        for part in parts:
            words = _words(text, part)
            found = [
                (cue, span) for cue in self._relations for span in _matches(words, cue)
            ]
            # Multiple triggers in one clause need clause-level argument scope.
            if len(found) != 1 or len(events) >= _MAX_EVENTS:
                unexplained.append(part)
                continue
            cue, trigger = found[0]
            choices = self._relations[cue]
            if len(choices) != 1:
                unexplained.append(part)
                continue
            relation = next(iter(choices))
            layouts = self._layouts[(cue, relation)]
            local = tuple(
                m
                for m in sorted(mentions, key=lambda m: m.span.start)
                if _within(m.span, part)
            )
            keyed: dict[_Signature, IdentifiedMention] = {}
            for mention in local:
                try:
                    key = _role_signature(mention, trigger, local)
                except ValueError:
                    continue
                keyed[key] = mention
            matching: list[tuple[EventRole, ...]] = []
            for layout in layouts:
                roles = []
                for role, key in layout:
                    mention = keyed.get(key)
                    if mention is None:
                        break
                    roles.append(
                        EventRole(role, mention.mention_id, mention.instance_id)
                    )
                else:
                    # Position alone cannot distinguish same-kind role swaps.
                    # Require external morphology for every participant of a
                    # kind used by more than one distinct role.
                    used = [keyed[key] for _, key in layout]
                    unsafe = any(
                        mention.morphology is None
                        and sum(other.kind == mention.kind for other in used) > 1
                        for mention in used
                    )
                    if not unsafe:
                        matching.append(tuple(roles))
            if len(matching) != 1:
                unexplained.append(part)
                continue
            roles = list(matching[0])
            negation = [
                span for marker in self._negations for span in _matches(words, marker)
            ]
            times = [
                (labels, span)
                for marker, labels in self._times.items()
                for span in _matches(words, marker)
            ]
            if (
                len(negation) > 1
                or len(times) > 1
                or any(len(labels) != 1 for labels, _ in times)
            ):
                unexplained.append(part)
                continue
            event = SemanticEvent(
                event_id=(
                    f"{source_id}:v{source_version}:event:{trigger.start}:{trigger.end}"
                ),
                relation_id=relation,
                trigger=trigger,
                source_span=part,
                roles=tuple(roles),
                negated=bool(negation),
                time_label=next(iter(times[0][0])) if times else None,
            )
            events.append(event)
            covered = [
                trigger,
                *(
                    m.span
                    for m in local
                    if m.mention_id in {role.mention_id for role in roles}
                ),
                *negation,
                *(span for _, span in times),
            ]
            unexplained.extend(_residual(text, part, covered))
        event_links: list[SemanticLink] = []
        consumed_link_cues: list[Span] = []
        for previous, current in zip(events, events[1:], strict=False):
            if (
                part_index[(current.source_span.start, current.source_span.end)]
                != part_index[(previous.source_span.start, previous.source_span.end)]
                + 1
            ):
                continue
            between = Span(previous.trigger.end, current.trigger.start)
            words = _words(text, between)
            found = [
                (labels, span)
                for cue, labels in self._event_links.items()
                for span in _matches(words, cue)
            ]
            if len(found) != 1 or len(found[0][0]) != 1:
                continue
            labels, span = found[0]
            event_links.append(
                SemanticLink(
                    next(iter(labels)), previous.event_id, current.event_id, span
                )
            )
            consumed_link_cues.append(span)
        return SemanticGraph(
            source_id=source_id,
            source_version=source_version,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            model_fingerprint=self.model_fingerprint or "",
            events=tuple(events),
            unexplained=tuple(
                span
                for span in unexplained
                if not any(_within(span, consumed) for consumed in consumed_link_cues)
            ),
            links=tuple(event_links),
        )

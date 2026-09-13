"""Bounded, event-sourced *engineering scaffold* for a small dialogue world.

This module does not learn language or physical laws.  The upstream learned
bridge must supply an assertion predicate; no cue-to-predicate lookup is hidden
here.  Locations and holders are functional, alternative whereabouts in this
microworld.  A positive whereabouts update replaces the old whereabouts; a
negative state excludes only its exact value.  Not performing a movement or
transfer says nothing about the resulting location or holder.

Events are append-only JSON dictionaries.  ``input_frame`` retains the original
reference and ``frame`` its resolved interpretation.  A correction is one event
that suppresses previous assertion events and asserts its replacement.  Reverse
replay computes which events are active: retracting a correction therefore also
undoes its suppression and restores the facts that preceded that correction.
An explicit reference names an event ID; an implicit correction/retraction names
the latest active assertion *turn* belonging to the same source and topic.
Queries add only dialogue-focus events, never facts.  Nonasserted modalities add
neither.  The enclosing engine owns atomicity across multiple clauses; each
individual apply is atomic.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .schema import Assertion, ConversationLimits, SemanticFrame, StateOutcome

_VERSION = 1
_MAX_ID = (1 << 53) - 1
_EVENT_KEYS = frozenset(
    {
        "event_id",
        "turn_id",
        "source",
        "topic",
        "kind",
        "input_frame",
        "frame",
        "retracts",
    }
)
_ASSERTION_KINDS = frozenset({"assert", "correct"})
_EVENT_KINDS = _ASSERTION_KINDS | {"retract", "topic", "query"}
_ROLES = ("actor", "object", "place", "recipient")
_MARKERS = frozenset({"@object", "@person", "@place", "@ambiguous", "@speaker"})


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _name(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        type(value) is str
        and len(value) <= 128
        and (bool(value) or allow_empty)
        and value == value.strip()
        and not any(
            ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        )
    )


def _identifier(value: Any) -> bool:
    return type(value) is int and 1 <= value <= _MAX_ID


class WorldState:
    """Small topic-local state with provenance and strictly validated snapshots.

    ``events`` is exposed for inspection, not direct modification.  Entity and
    byte limits count retained history, including retracted events, so switching
    topics or undoing assertions cannot bypass saturation limits.
    """

    def __init__(self, limits: ConversationLimits | None = None) -> None:
        if limits is not None and not isinstance(limits, ConversationLimits):
            raise TypeError("limits must be ConversationLimits")
        self.limits = limits or ConversationLimits()
        self.topic = "default"
        self.events: list[dict[str, Any]] = []
        self._entities: set[str] = set()
        self._event_bytes = 0
        self._identities: dict[tuple[int, str, str, str], int] = {}
        self._current_turn_count = 0

    def clone(self) -> WorldState:
        """Copy mutable storage without sharing events with the transaction."""
        other = WorldState(self.limits)
        other.topic = self.topic
        other.events = deepcopy(self.events)
        other._entities = self._entities.copy()
        other._event_bytes = self._event_bytes
        other._identities = self._identities.copy()
        other._current_turn_count = self._current_turn_count
        return other

    def _active_events(self, *, include_queries: bool = False) -> list[dict[str, Any]]:
        suppressed: set[int] = set()
        active: list[dict[str, Any]] = []
        for event in reversed(self.events):
            if event["event_id"] in suppressed:
                continue
            suppressed.update(event["retracts"])
            if event["kind"] in _ASSERTION_KINDS or (
                include_queries and event["kind"] == "query"
            ):
                active.append(event)
        active.reverse()
        return active

    @staticmethod
    def _event_assertions(event: dict[str, Any]) -> tuple[Assertion, ...]:
        frame = event["frame"]
        predicate = frame["predicate"]
        if frame["negated"] and predicate in {"move", "give"}:
            return ()
        relation = "location" if predicate in {"locate", "move"} else "holder"
        value = (
            frame["place"]
            if relation == "location"
            else frame["recipient"]
            if predicate == "give"
            else frame["actor"]
        )
        return (
            Assertion(
                subject=frame["object"],
                relation=relation,
                value=value,
                event_id=event["event_id"],
                negated=frame["negated"],
                source=event["source"],
                topic=event["topic"],
                qualifier=frame["spatial"] if relation == "location" else "in",
            ),
        )

    def facts(self) -> tuple[Assertion, ...]:
        """Current positive/negative facts in the current topic, not old claims."""
        facts: list[Assertion] = []
        for event in self._active_events():
            if event["topic"] != self.topic:
                continue
            for assertion in self._event_assertions(event):
                if not assertion.negated:
                    facts = [f for f in facts if f.subject != assertion.subject]
                else:
                    facts = [
                        f
                        for f in facts
                        if (f.subject, f.relation, f.value, f.qualifier)
                        != (
                            assertion.subject,
                            assertion.relation,
                            assertion.value,
                            assertion.qualifier,
                        )
                    ]
                facts.append(assertion)
        return tuple(facts)

    def _candidates(self, marker: str) -> tuple[str, ...]:
        roles = {
            "@object": ("object",),
            "@person": ("actor", "recipient"),
            "@place": ("place",),
        }[marker]
        # A question explicitly shifts focus even when its answer is unknown.
        # Between question boundaries, whole-turn assertion salience avoids
        # choosing whichever of two entities happened to be written last.
        selected_turn: int | None = None
        candidates: set[str] = set()
        for event in reversed(self._active_events(include_queries=True)):
            if event["topic"] != self.topic:
                continue
            frame = event["frame"]
            values = {frame[role] for role in roles}
            values.discard("")
            unknown_person = False
            if frame["object"]:
                if marker == "@object" and frame["object_kind"] == "person":
                    values.discard(frame["object"])
                elif marker == "@person":
                    if frame["object_kind"] == "person":
                        values.add(frame["object"])
                    elif frame["object_kind"] == "unknown":
                        # The latest untyped subject could itself be a person.
                        # Do not bind a later "he/she" to a stale older actor.
                        # Explicit people in this same turn still count.
                        unknown_person = True
            if selected_turn is not None and event["turn_id"] != selected_turn:
                break
            if event["kind"] == "query" and values:
                if selected_turn is None:
                    return tuple(sorted(values))
                if not (marker == "@person" and not candidates):
                    break
            if selected_turn is None and (values or unknown_person):
                selected_turn = event["turn_id"]
            candidates.update(values)
        return tuple(sorted(candidates))

    def _entity_kind(self, entity: str) -> str:
        """Recover declared type hints; never guess a type from a name suffix."""
        for event in reversed(self._active_events(include_queries=True)):
            if event["topic"] != self.topic:
                continue
            frame = event["frame"]
            if entity in {frame["actor"], frame["recipient"]}:
                return "person"
            if frame["object"] == entity and frame["object_kind"] != "unknown":
                return frame["object_kind"]
        return "unknown"

    def _resolve(
        self,
        frame: SemanticFrame,
        source: str,
    ) -> SemanticFrame | StateOutcome:
        changes: dict[str, str] = {"topic": self.topic}
        for role in _ROLES:
            value = getattr(frame, role)
            if not value.startswith("@"):
                continue
            if value not in _MARKERS or value == "@ambiguous":
                return StateOutcome("clarify", reason="ambiguous_reference")
            if value == "@speaker":
                changes[role] = "я" if source == "user" else source
                if role == "object":
                    changes["object_kind"] = "person"
                continue
            candidates = self._candidates(value)
            if len(candidates) != 1:
                return StateOutcome(
                    "clarify",
                    reason="ambiguous_reference" if candidates else "unknown_reference",
                    alternatives=candidates[: self.limits.max_candidates],
                )
            changes[role] = candidates[0]
            if role == "object" and value == "@person":
                changes["object_kind"] = "person"
        resolved_object = changes.get("object", frame.object)
        if (
            resolved_object
            and changes.get("object_kind", frame.object_kind) == "unknown"
        ):
            changes["object_kind"] = self._entity_kind(resolved_object)
        return replace(frame, **changes)

    def _targets(
        self,
        frame: SemanticFrame,
        source: str,
        turn_id: int,
    ) -> list[int] | StateOutcome:
        active = self._active_events()
        if frame.reference is not None:
            if frame.reference > len(self.events):
                return StateOutcome("clarify", reason="unknown_reference")
            target = self.events[frame.reference - 1]
            if target["topic"] != self.topic:
                return StateOutcome("clarify", reason="reference_topic_mismatch")
            if target["source"] != source:
                return StateOutcome("clarify", reason="reference_source_mismatch")
            if target["kind"] not in _ASSERTION_KINDS or target not in active:
                return StateOutcome("clarify", reason="reference_inactive")
            if target["turn_id"] == turn_id:
                return StateOutcome("clarify", reason="same_turn_retraction")
            return [target["event_id"]]
        for event in reversed(self.events):
            if event["turn_id"] != turn_id:
                break
            if event["topic"] == self.topic and event["kind"] in {"correct", "retract"}:
                # Later replacement clauses should be ordinary assertions, not
                # another implicit undo that could reach an older user turn.
                return StateOutcome("clarify", reason="multiple_implicit_retractions")
        relevant = [
            event
            for event in active
            if event["topic"] == self.topic
            and event["source"] == source
            and event["turn_id"] < turn_id
        ]
        if not relevant:
            return StateOutcome("clarify", reason="no_retractable_assertion")
        latest = relevant[-1]["turn_id"]
        return [e["event_id"] for e in relevant if e["turn_id"] == latest]

    @staticmethod
    def _required(frame: SemanticFrame) -> bool:
        if not frame.object:
            return False
        if frame.predicate == "locate":
            return bool(frame.place)
        if frame.predicate == "move":
            return bool(frame.actor and frame.place)
        if frame.predicate == "have":
            return bool(frame.actor)
        return frame.predicate == "give" and bool(frame.actor and frame.recipient)

    @staticmethod
    def _compatible(frame: SemanticFrame) -> bool:
        """A predicate must not discard a supplied place or recipient role."""
        if frame.predicate == "locate":
            return not frame.actor and not frame.recipient
        if frame.predicate == "move":
            return not frame.recipient
        if frame.predicate == "have":
            return not frame.place and not frame.recipient
        if frame.predicate == "give":
            return not frame.place
        return False

    def _append(
        self,
        input_frame: SemanticFrame,
        frame: SemanticFrame,
        *,
        turn_id: int,
        source: str,
        kind: str,
        retracts: list[int] | None = None,
    ) -> dict[str, Any] | StateOutcome:
        event_topic = frame.topic
        event = {
            "event_id": len(self.events) + 1,
            "turn_id": turn_id,
            "source": source,
            "topic": event_topic,
            "kind": kind,
            "input_frame": input_frame.to_dict(),
            "frame": frame.to_dict(),
            "retracts": retracts or [],
        }
        new_entities = {getattr(frame, role) for role in _ROLES if getattr(frame, role)}
        serialized_bytes = len(_json(event).encode("utf-8"))
        topic_after = event_topic if kind == "topic" else self.topic
        envelope_bytes = len(
            _json({"version": _VERSION, "topic": topic_after, "events": []}).encode(
                "utf-8"
            )
        )
        size = envelope_bytes + self._event_bytes + serialized_bytes + len(self.events)
        if (
            len(self.events) >= self.limits.max_events
            or len(self._entities | new_entities) > self.limits.max_entities
            or size > self.limits.max_state_bytes
            or (
                self.events
                and self.events[-1]["turn_id"] == turn_id
                and self._current_turn_count >= self.limits.max_clauses
            )
        ):
            return StateOutcome("clarify", reason="state_capacity")
        if self.events and self.events[-1]["turn_id"] == turn_id:
            self._current_turn_count += 1
        else:
            self._current_turn_count = 1
        self.events.append(event)
        self.topic = topic_after
        self._entities.update(new_entities)
        self._event_bytes += serialized_bytes
        if kind != "query":
            self._identities[
                (turn_id, source, event_topic, _json(input_frame.to_dict()))
            ] = event["event_id"]
        return event

    def _query(self, frame: SemanticFrame) -> StateOutcome:
        facts = self.facts()
        positive = tuple(f for f in facts if not f.negated)
        evidence: tuple[Assertion, ...] = ()
        reason = ""
        if frame.negated and frame.query != "verify":
            return StateOutcome(
                "clarify",
                reason="unsupported_query_negation",
                resolved_frame=frame,
            )
        if frame.query in {"where", "who_has", "why"}:
            if not frame.object:
                return StateOutcome("clarify", reason="missing_object")
            selected = facts if frame.query == "why" else positive
            evidence = tuple(f for f in selected if f.subject == frame.object)
            if frame.query == "who_has":
                evidence = tuple(f for f in evidence if f.relation == "holder")
            elif frame.query == "where":
                locations = tuple(f for f in evidence if f.relation == "location")
                # A known holder is a useful grounded answer; it does not assert
                # that either participant is at an invented physical location.
                evidence = locations or tuple(
                    f for f in evidence if f.relation == "holder"
                )
            elif frame.reference is not None:
                evidence = tuple(f for f in evidence if f.event_id == frame.reference)
            reason = "provenance" if frame.query == "why" else ""
        elif frame.query == "what_has":
            if not frame.actor:
                return StateOutcome("clarify", reason="missing_actor")
            evidence = tuple(
                f for f in positive if f.relation == "holder" and f.value == frame.actor
            )
        elif frame.query == "verify":
            if not self._required(frame):
                return StateOutcome("clarify", reason="missing_query_roles")
            if not self._compatible(frame):
                return StateOutcome("clarify", reason="incompatible_roles")
            # Action completion is not itself represented by a whereabouts fact.
            # A past negative move/give cannot be answered from static state.
            if frame.predicate in {"move", "give"}:
                return StateOutcome(
                    "unknown",
                    reason="unknown",
                    resolved_frame=frame,
                )
            relation = "location" if frame.predicate == "locate" else "holder"
            value = frame.place if relation == "location" else frame.actor
            qualifier = frame.spatial if relation == "location" else "in"
            matching = tuple(
                f
                for f in facts
                if (f.subject, f.relation, f.value, f.qualifier)
                == (frame.object, relation, value, qualifier)
            )
            if matching:
                evidence = matching
                truth = not matching[-1].negated
            else:
                evidence = tuple(
                    f
                    for f in positive
                    if f.subject == frame.object and f.relation == relation
                )
                truth = False
            if evidence:
                reason = "true" if truth != frame.negated else "false"
        else:
            return StateOutcome("clarify", reason="unsupported_query")
        if not evidence:
            return StateOutcome("unknown", reason="unknown", resolved_frame=frame)
        return StateOutcome(
            "answer",
            assertions=evidence,
            reason=reason,
            event_ids=tuple(dict.fromkeys(f.event_id for f in evidence)),
            resolved_frame=frame,
        )

    def apply(
        self,
        frame: SemanticFrame,
        *,
        turn_id: int,
        source: str = "user",
    ) -> StateOutcome:
        """Interpret one already-decoded frame; errors never partially append.

        A repeated identical assertion/control frame in the same source/topic/
        turn is idempotent. Questions can recur between distinct clauses and
        shift focus each time; the session owns whole-request idempotence. An
        older turn cannot acquire events, nor one turn have two sources.
        """
        if not isinstance(frame, SemanticFrame):
            raise TypeError("frame must be SemanticFrame")
        if not _identifier(turn_id):
            raise ValueError("turn_id must be a positive JSON-safe integer")
        if not _name(source) or source.startswith("@"):
            raise ValueError("invalid source")
        if len(frame.raw) > self.limits.max_chars:
            return StateOutcome("clarify", reason="input_capacity")
        if any(0xD800 <= ord(c) <= 0xDFFF for c in frame.raw + frame.cue):
            return StateOutcome("clarify", reason="invalid_unicode")
        if any(not _name(getattr(frame, r), allow_empty=True) for r in _ROLES):
            return StateOutcome("clarify", reason="invalid_entity")
        if not _name(frame.topic, allow_empty=True) or frame.topic.startswith("@"):
            return StateOutcome("clarify", reason="invalid_topic")
        requested_topic = frame.topic if frame.act == "topic" else self.topic
        identity = (turn_id, source, requested_topic, _json(frame.to_dict()))
        if identity in self._identities:
            event_id = self._identities[identity]
            event = self.events[event_id - 1]
            return StateOutcome(
                "ack",
                reason="duplicate_event",
                event_ids=(event_id,),
                resolved_frame=SemanticFrame.from_dict(event["frame"]),
            )
        if self.events and turn_id < self.events[-1]["turn_id"]:
            return StateOutcome("clarify", reason="nonmonotonic_turn")
        if (
            self.events
            and turn_id == self.events[-1]["turn_id"]
            and source != self.events[-1]["source"]
        ):
            return StateOutcome("clarify", reason="turn_source_mismatch")
        if frame.act in {"inform", "correct"} and not frame.predicate:
            return StateOutcome("clarify", reason="predicate_required")
        if (
            frame.act == "hypothesis"
            or frame.modality != "asserted"
            or frame.tense == "future"
        ):
            reason = (
                frame.modality
                if frame.modality != "asserted"
                else "future"
                if frame.tense == "future"
                else "hypothetical"
            )
            return StateOutcome("hypothetical", reason=reason, resolved_frame=frame)
        if frame.act == "topic":
            if not frame.topic:
                return StateOutcome("clarify", reason="missing_topic")
            if frame.topic == self.topic:
                return StateOutcome("topic", reason="unchanged", resolved_frame=frame)
            event = self._append(
                frame,
                frame,
                turn_id=turn_id,
                source=source,
                kind="topic",
            )
            if isinstance(event, StateOutcome):
                return event
            return StateOutcome(
                "topic",
                event_ids=(event["event_id"],),
                resolved_frame=frame,
            )
        if frame.topic and frame.topic != self.topic:
            return StateOutcome("clarify", reason="topic_mismatch")
        if frame.act in {"help", "greet", "thanks"}:
            return StateOutcome(frame.act, resolved_frame=frame)
        if frame.act == "teach":
            return StateOutcome("help", reason="teaching_requires_engine")
        if frame.act not in {"inform", "correct", "retract", "ask"}:
            return StateOutcome("clarify", reason="unsupported_intent")
        resolved = self._resolve(frame, source)
        if isinstance(resolved, StateOutcome):
            return resolved
        if frame.act == "ask":
            outcome = self._query(resolved)
            if outcome.action == "clarify":
                return outcome
            event = self._append(
                frame,
                resolved,
                turn_id=turn_id,
                source=source,
                kind="query",
            )
            return event if isinstance(event, StateOutcome) else outcome
        if frame.act in {"inform", "correct"} and not self._required(resolved):
            return StateOutcome("clarify", reason="missing_assertion_roles")
        if frame.act in {"inform", "correct"} and not self._compatible(resolved):
            return StateOutcome("clarify", reason="incompatible_roles")
        if frame.act == "inform" and frame.reference is not None:
            return StateOutcome("clarify", reason="unexpected_reference")
        targets: list[int] = []
        if frame.act in {"correct", "retract"}:
            selected = self._targets(resolved, source, turn_id)
            if isinstance(selected, StateOutcome):
                return selected
            targets = selected
        kind = "assert" if frame.act == "inform" else frame.act
        event = self._append(
            frame,
            resolved,
            turn_id=turn_id,
            source=source,
            kind=kind,
            retracts=targets,
        )
        if isinstance(event, StateOutcome):
            return event
        if kind == "retract":
            return StateOutcome(
                "retracted",
                reason="retracted",
                event_ids=tuple(targets),
                resolved_frame=resolved,
            )
        return StateOutcome(
            "ack",
            assertions=self._event_assertions(event),
            reason=(
                "negated_action_not_state"
                if resolved.negated and resolved.predicate in {"move", "give"}
                else "corrected"
                if kind == "correct"
                else "asserted"
            ),
            event_ids=(event["event_id"],),
            resolved_frame=resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "topic": self.topic,
            "events": deepcopy(self.events),
        }

    @classmethod
    def from_dict(
        cls,
        value: Any,
        limits: ConversationLimits | None = None,
    ) -> WorldState:
        """Validate the entire bounded log, replay, and return only on success.

        Replay rechecks input interpretation, resolved roles, event identities,
        retraction targets, source/topic boundaries and the final current topic.
        A malformed suffix is not silently discarded, even after a valid prefix.
        """
        state = cls(limits)
        if (
            type(value) is not dict
            or set(value) != {"version", "topic", "events"}
            or type(value["version"]) is not int
            or value["version"] != _VERSION
            or not _name(value["topic"])
            or value["topic"].startswith("@")
            or type(value["events"]) is not list
            or len(value["events"]) > state.limits.max_events
        ):
            raise ValueError("invalid world snapshot")
        # Validate shallow structure and scalar types before serializing.  This
        # rejects recursive or enormous nested payloads without recursive replay.
        for expected_id, event in enumerate(value["events"], start=1):
            if (
                type(event) is not dict
                or set(event) != _EVENT_KEYS
                or not _identifier(event["event_id"])
                or event["event_id"] != expected_id
                or not _identifier(event["turn_id"])
                or not _name(event["source"])
                or event["source"].startswith("@")
                or not _name(event["topic"])
                or event["topic"].startswith("@")
                or type(event["kind"]) is not str
                or event["kind"] not in _EVENT_KINDS
                or type(event["retracts"]) is not list
                or len(event["retracts"]) > state.limits.max_events
                or any(
                    not _identifier(ref) or ref >= expected_id
                    for ref in event["retracts"]
                )
                or len(set(event["retracts"])) != len(event["retracts"])
            ):
                raise ValueError("invalid world event")
            try:
                SemanticFrame.from_dict(event["input_frame"])
                SemanticFrame.from_dict(event["frame"])
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid world event frame") from exc
        if len(_json(value).encode("utf-8")) > state.limits.max_state_bytes:
            raise ValueError("world snapshot exceeds byte limit")
        for event in value["events"]:
            frame = SemanticFrame.from_dict(event["input_frame"])
            previous_count = len(state.events)
            try:
                state.apply(frame, turn_id=event["turn_id"], source=event["source"])
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid world event replay") from exc
            if len(state.events) != previous_count + 1 or state.events[-1] != event:
                raise ValueError("world event does not match validated replay")
        if state.topic != value["topic"]:
            raise ValueError("world snapshot topic does not match replay")
        return state

"""Event-sourced evidence memory for learned interpretations and state deltas.

The model proposes effects. Fixed domain invariants veto unsupported bindings,
scope leaks and impossible deltas; they never supply a missing model prediction.
Statements are user-provided evidence, not independently verified reality.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..conversation.schema import Budget, BudgetExceeded
from .schema import Event, Meaning, Query, bounded_text, exact_fields

_SCHEMA = "ai2-learned-world-v1"
_FACT_KEYS = {
    "subject",
    "relation",
    "value",
    "negated",
    "spatial",
    "source",
    "event_id",
}
_EFFECT_KEYS = {"op", "subject", "relation", "value", "spatial"}


def validate_effects(
    event: Event, effects: Any, *, before: tuple[dict[str, Any], ...] = ()
) -> tuple[dict[str, Any], ...]:
    """Check a learned delta against the meaning ontology; never infer a delta."""
    if type(effects) not in (list, tuple) or len(effects) > 1:
        raise ValueError("invalid number of learned effects")
    result = []
    for raw in effects:
        effect = dict(exact_fields(raw, _EFFECT_KEYS, "learned effect"))
        for name in ("subject", "value"):
            bounded_text(effect[name], name, empty=False)
        if (
            any(type(effect[k]) is not str for k in ("op", "relation", "spatial"))
            or effect["op"] not in {"set", "exclude"}
            or effect["relation"] not in {"location", "holder"}
            or effect["spatial"] not in {"in", "on"}
        ):
            raise ValueError("invalid learned effect category")
        result.append(effect)
    if not event.actual or (event.negated and event.predicate in {"move", "give"}):
        if result:
            raise ValueError("nonactual or negated action cannot project an effect")
        return ()
    relation = "location" if event.predicate in {"locate", "move"} else "holder"
    value = (
        event.place
        if relation == "location"
        else event.recipient
        if event.predicate == "give"
        else event.actor
    )
    if not event.object or not value:
        raise ValueError("meaning lacks a required role")
    if not result and any(
        f["subject"] == event.object
        and f["relation"] == relation
        and f["value"] == value
        and f["negated"] == event.negated
        and f["spatial"] == (event.spatial if relation == "location" else "in")
        for f in before
    ):
        # An observed unchanged transition cannot identify an effect rule. A
        # learned no-op is safe only when the stated relation already holds.
        return ()
    if len(result) != 1:
        raise ValueError("actual atomic meaning requires a learned effect")
    effect = result[0]
    if (
        effect["subject"],
        effect["relation"],
        effect["value"],
        effect["op"],
        effect["spatial"],
    ) != (
        event.object,
        relation,
        value,
        "exclude" if event.negated else "set",
        event.spatial if relation == "location" else "in",
    ):
        raise ValueError("learned effect is not licensed by the interpreted roles")
    return tuple(result)


class ExperienceWorld:
    def __init__(self, *, max_events: int = 512, max_entities: int = 256) -> None:
        if type(max_events) is not int or not 1 <= max_events <= 4096:
            raise ValueError("invalid event capacity")
        if type(max_entities) is not int or not 1 <= max_entities <= 1024:
            raise ValueError("invalid entity capacity")
        self.max_events = max_events
        self.max_entities = max_entities
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)

    def clone(self) -> ExperienceWorld:
        world = ExperienceWorld(
            max_events=self.max_events, max_entities=self.max_entities
        )
        world._events = deepcopy(self._events)
        return world

    def _active(self, *, through_turn: int | None = None) -> list[dict[str, Any]]:
        events = (
            self._events
            if through_turn is None
            else [e for e in self._events if e["turn_id"] <= through_turn]
        )
        removed = {e["target"] for e in events if e["meaning"]["act"] == "retract"}
        return [
            e
            for e in events
            if e["id"] not in removed and e["meaning"]["act"] != "retract"
        ]

    def facts(self, *, through_turn: int | None = None) -> tuple[dict[str, Any], ...]:
        facts: list[dict[str, Any]] = []
        for record in self._active(through_turn=through_turn):
            for effect in record["effects"]:
                subject, relation, value = (
                    effect[k] for k in ("subject", "relation", "value")
                )
                if effect["op"] == "set":
                    # In this world a known holder replaces a known location and
                    # vice versa. No physical location is inferred from a holder.
                    facts = [
                        f
                        for f in facts
                        if not (
                            f["subject"] == subject
                            and (
                                not f["negated"]
                                or (
                                    f["relation"] == relation
                                    and f["value"] == value
                                    and f["spatial"] == effect["spatial"]
                                )
                            )
                        )
                    ]
                else:
                    facts = [
                        f
                        for f in facts
                        if not (
                            f["subject"] == subject
                            and f["relation"] == relation
                            and f["value"] == value
                            and f["spatial"] == effect["spatial"]
                        )
                    ]
                facts.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "value": value,
                        "negated": effect["op"] == "exclude",
                        "spatial": effect["spatial"],
                        "source": record["source"],
                        "event_id": record["id"],
                    }
                )
        return tuple(deepcopy(facts))

    def _check_capacity(self, record: dict[str, Any]) -> None:
        if len(self._events) >= self.max_events:
            raise BudgetExceeded("event_capacity")
        names: set[str] = set()
        for existing in [*self._events, record]:
            meaning = Meaning.from_dict(existing["meaning"])
            names.update(e.name for e in meaning.entities)
            stack = [meaning.event] if meaning.event else []
            while stack:
                event = stack.pop()
                names.update(
                    name
                    for name in (
                        event.actor,
                        event.object,
                        event.recipient,
                        event.place,
                    )
                    if name
                )
                stack.extend(
                    node
                    for node in (event.content, event.condition)
                    if node is not None
                )
        if len(names) > self.max_entities:
            raise BudgetExceeded("entity_capacity")

    def apply(
        self,
        meaning: Meaning,
        effects: Any,
        *,
        turn_id: int,
        source: str = "user",
        budget: Budget | None = None,
    ) -> dict[str, Any]:
        if not isinstance(meaning, Meaning):
            raise ValueError("expected Meaning")
        if type(turn_id) is not int or not 1 <= turn_id < 2**53:
            raise ValueError("invalid event turn")
        if self._events and turn_id <= self._events[-1]["turn_id"]:
            raise ValueError("event turns must increase")
        bounded_text(source, "evidence source", empty=False)
        if budget:
            budget.check()
        target = None
        if meaning.act == "retract":
            if effects:
                raise ValueError("retraction cannot carry a learned effect")
            active = self._active()
            if not active:
                return {
                    "action": "unknown",
                    "reason": "nothing_to_retract",
                    "assertions": [],
                    "evidence": [],
                }
            target = active[-1]["id"]
            validated: tuple[dict[str, Any], ...] = ()
        elif meaning.act in {"inform", "correct"} and meaning.event:
            validated = validate_effects(meaning.event, effects, before=self.facts())
        else:
            raise ValueError("only evidence statements or retractions may mutate world")
        record = {
            "id": len(self._events) + 1,
            "turn_id": turn_id,
            "meaning": meaning.to_dict(),
            "source": source,
            "effects": list(validated),
            "target": target,
        }
        self._check_capacity(record)
        if budget:
            budget.check()
        self._events.append(record)
        assertions = [f for f in self.facts() if f["event_id"] == record["id"]]
        nonactual = bool(
            meaning.event
            and (
                not meaning.event.actual
                or (
                    meaning.event.negated
                    and meaning.event.predicate in {"move", "give"}
                )
            )
        )
        action = (
            "retracted"
            if meaning.act == "retract"
            else "nonactual"
            if nonactual
            else "corrected"
            if meaning.act == "correct"
            else "ack"
        )
        return {
            "action": action,
            "reason": "scoped_evidence_only" if nonactual else "",
            "assertions": assertions,
            "evidence": [deepcopy(record)],
        }

    def query(self, query: Query) -> dict[str, Any]:
        if not isinstance(query, Query):
            raise ValueError("expected Query")
        if query.time not in {"present", "unspecified"}:
            return {
                "action": "clarify",
                "reason": "temporal_anchor_required",
                "assertions": [],
                "evidence": [],
            }
        if query.kind == "why":
            return {
                "action": "unknown",
                "reason": "explanation_requires_previous_answer",
                "assertions": [],
                "evidence": [],
            }
        facts = list(self.facts())
        truth = ""
        if query.kind == "where":
            found = [
                f for f in facts if f["subject"] == query.subject and not f["negated"]
            ]
        elif query.kind == "who_has":
            found = [
                f
                for f in facts
                if f["subject"] == query.subject
                and f["relation"] == "holder"
                and not f["negated"]
            ]
        elif query.kind == "what_has":
            found = [
                f
                for f in facts
                if f["relation"] == "holder"
                and f["value"] == query.subject
                and not f["negated"]
            ]
        else:
            matching = [
                f
                for f in facts
                if f["subject"] == query.subject and f["relation"] == query.relation
            ]
            exact = [
                f
                for f in matching
                if f["value"] == query.value
                and (query.relation != "location" or f["spatial"] == query.spatial)
            ]
            positive = [f for f in matching if not f["negated"]]
            found = exact if exact else positive
            if found:
                truth = (
                    "no"
                    if found[0]["negated"]
                    or found[0]["value"] != query.value
                    or (
                        query.relation == "location"
                        and found[0]["spatial"] != query.spatial
                    )
                    else "yes"
                )
                if query.negated:
                    truth = "no" if truth == "yes" else "yes"
        if not found:
            return {
                "action": "unknown",
                "reason": "missing_evidence",
                "assertions": [],
                "evidence": [],
            }
        ids = {f["event_id"] for f in found}
        return {
            "action": "answer",
            "reason": "",
            "assertions": found,
            "evidence": [deepcopy(e) for e in self._events if e["id"] in ids],
            "truth": truth,
        }

    def explain(
        self, assertions: list[dict[str, Any]], *, subject: str = ""
    ) -> dict[str, Any]:
        bounded_text(subject, "explanation subject")
        if subject:
            assertions = [f for f in assertions if f["subject"] == subject]
        current = self.facts()
        if not assertions or any(f not in current for f in assertions):
            return {
                "action": "unknown",
                "reason": "no_current_answer_to_explain",
                "assertions": [],
                "evidence": [],
            }
        ids = {f["event_id"] for f in assertions}
        return {
            "action": "explain",
            "reason": "source_not_physical_cause",
            "assertions": deepcopy(assertions),
            "evidence": [deepcopy(e) for e in self._events if e["id"] in ids],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "max_events": self.max_events,
            "max_entities": self.max_entities,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExperienceWorld:
        value = exact_fields(
            value, {"schema", "max_events", "max_entities", "events"}, "world"
        )
        if value["schema"] != _SCHEMA or type(value["events"]) is not list:
            raise ValueError("invalid world schema")
        world = cls(max_events=value["max_events"], max_entities=value["max_entities"])
        if len(value["events"]) > world.max_events:
            raise ValueError("event capacity exceeded")
        for record in value["events"]:
            exact_fields(
                record,
                {"id", "turn_id", "meaning", "source", "effects", "target"},
                "world event",
            )
            if type(record["id"]) is not int or record["id"] != len(world._events) + 1:
                raise ValueError("invalid world event order")
            if record["target"] is not None and (
                type(record["target"]) is not int
                or not 1 <= record["target"] < record["id"]
            ):
                raise ValueError("invalid retraction target")
            meaning = Meaning.from_dict(record["meaning"])
            try:
                outcome = world.apply(
                    meaning,
                    record["effects"],
                    turn_id=record["turn_id"],
                    source=record["source"],
                )
            except BudgetExceeded as exc:
                raise ValueError(str(exc)) from exc
            if not outcome["evidence"] or world._events[-1] != record:
                raise ValueError("world event does not match deterministic replay")
        return world


def validate_assertion(value: Any) -> dict[str, Any]:
    value = dict(exact_fields(value, _FACT_KEYS, "assertion"))
    for name in ("subject", "value", "source"):
        bounded_text(value[name], name, empty=False)
    if (
        type(value["relation"]) is not str
        or value["relation"] not in {"location", "holder"}
        or type(value["spatial"]) is not str
        or value["spatial"] not in {"in", "on"}
        or type(value["negated"]) is not bool
    ):
        raise ValueError("invalid assertion categories")
    if type(value["event_id"]) is not int or not 1 <= value["event_id"] < 2**53:
        raise ValueError("invalid assertion event")
    return value

"""Event-sourced evidence memory for learned interpretations and state deltas.

The model proposes effects. Fixed domain invariants veto unsupported bindings,
scope leaks and impossible deltas; they never supply a missing model prediction.
Statements are user-provided evidence, not independently verified reality.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..conversation.schema import Budget, BudgetExceeded
from .hypotheses import Hypothesis, Observation, digest
from .schema import Event, Meaning, Query, bounded_text, exact_fields

_SCHEMA = "ai2-learned-world-v1"
_REVISION_SCHEMA = "ai2-learned-world-v2"
REVISION_TRACE_SCHEMA = "ai2-history-revision-v1"
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
        self._revisions: list[dict[str, Any]] = []

    @property
    def events(self) -> list[dict[str, Any]]:
        return deepcopy(self._events)

    @property
    def revisions(self) -> list[dict[str, Any]]:
        return deepcopy(self._revisions)

    @property
    def effective_events(self) -> list[dict[str, Any]]:
        """Current interpretations, with stable IDs and original effective turns."""
        return deepcopy([r for r in self._records() if r["meaning"] is not None])

    @property
    def snapshot_id(self) -> str:
        return digest(self.to_dict())

    def clone(self) -> ExperienceWorld:
        world = ExperienceWorld(
            max_events=self.max_events, max_entities=self.max_entities
        )
        world._events = deepcopy(self._events)
        world._revisions = deepcopy(self._revisions)
        return world

    def at_turn(self, turn_id: int) -> ExperienceWorld:
        """Knowledge available at this turn, before any later reinterpretation."""
        if type(turn_id) is not int or not 0 <= turn_id < 2**53:
            raise ValueError("invalid historical world turn")
        world = self.clone()
        world._events = [e for e in world._events if e["turn_id"] <= turn_id]
        world._revisions = [r for r in world._revisions if r["turn_id"] <= turn_id]
        return world

    def _last_turn(self) -> int:
        return max((r["turn_id"] for r in [*self._events, *self._revisions]), default=0)

    def _next_id(self) -> int:
        return 1 + max(
            [e["id"] for e in self._events]
            + [u["event_id"] for r in self._revisions for u in r["updates"]],
            default=0,
        )

    def _active_revisions(self) -> list[dict[str, Any]]:
        removed = {e["revision_target"] for e in self._events if "revision_target" in e}
        return [r for r in self._revisions if r["id"] not in removed]

    def _records(self) -> list[dict[str, Any]]:
        records = {e["id"]: deepcopy(e) for e in self._events}
        # A newly resolved old observation keeps its allocated ID even after
        # its resolving revision is retracted. Its unselected version is inert.
        for revision in self._revisions:
            for update in revision["updates"]:
                origin = update["observation"]
                records.setdefault(
                    update["event_id"],
                    {
                        "id": update["event_id"],
                        "turn_id": origin["turn_id"],
                        "meaning": None,
                        "source": origin["source"],
                        "effects": [],
                        "target": None,
                    },
                )
        for revision in self._active_revisions():
            for update in revision["updates"]:
                origin = update["observation"]
                records[update["event_id"]] = {
                    "id": update["event_id"],
                    "turn_id": origin["turn_id"],
                    "meaning": deepcopy(update["meaning"]),
                    "source": origin["source"],
                    "effects": deepcopy(update["effects"]),
                    "target": None,
                }
        return sorted(records.values(), key=lambda r: (r["turn_id"], r["id"]))

    def event_for_observation(self, observation: Observation) -> dict[str, Any] | None:
        if not isinstance(observation, Observation):
            raise ValueError("expected observation")
        record = next(
            (e for e in self._records() if e["turn_id"] == observation.turn_id), None
        )
        if record is not None and record["source"] != observation.source:
            raise ValueError("observation source does not match world event")
        return deepcopy(record)

    def _active(self, *, through_turn: int | None = None) -> list[dict[str, Any]]:
        if through_turn is not None:
            return self.at_turn(through_turn)._active()
        events = self._records()
        removed = {
            e["target"]
            for e in events
            if e["meaning"] is not None and e["meaning"]["act"] == "retract"
        }
        return [
            e
            for e in events
            if e["id"] not in removed
            and e["meaning"] is not None
            and e["meaning"]["act"] != "retract"
        ]

    def facts(self, *, through_turn: int | None = None) -> tuple[dict[str, Any], ...]:
        return self._facts(self._active(through_turn=through_turn))

    def facts_before(self, effective_turn: int) -> tuple[dict[str, Any], ...]:
        """A past event's inputs under the current interpretation versions."""
        if type(effective_turn) is not int or not 1 <= effective_turn < 2**53:
            raise ValueError("invalid effective turn")
        return self._facts([e for e in self._active() if e["turn_id"] < effective_turn])

    @staticmethod
    def _facts(records: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        facts: list[dict[str, Any]] = []
        for record in records:
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
        if len(self._events) + len(self._revisions) >= self.max_events:
            raise BudgetExceeded("event_capacity")
        self._check_entities(record)

    def _check_entities(self, record: dict[str, Any]) -> None:
        names: set[str] = set()
        versions = [u for r in self._revisions for u in r["updates"]]
        for existing in [*self._events, *versions, record]:
            if existing["meaning"] is None:
                continue
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
        if turn_id <= self._last_turn():
            raise ValueError("event turns must increase")
        bounded_text(source, "evidence source", empty=False)
        if budget:
            budget.check()
        target = None
        revision_target = None
        if meaning.act == "retract":
            if effects:
                raise ValueError("retraction cannot carry a learned effect")
            active = self._active()
            revisions = self._active_revisions()
            if revisions and (
                not active or revisions[-1]["turn_id"] > active[-1]["turn_id"]
            ):
                revision_target = revisions[-1]["id"]
            elif not active:
                return {
                    "action": "unknown",
                    "reason": "nothing_to_retract",
                    "assertions": [],
                    "evidence": [],
                }
            else:
                target = active[-1]["id"]
            validated: tuple[dict[str, Any], ...] = ()
        elif meaning.act in {"inform", "correct"} and meaning.event:
            validated = validate_effects(meaning.event, effects, before=self.facts())
        else:
            raise ValueError("only evidence statements or retractions may mutate world")
        record = {
            "id": self._next_id(),
            "turn_id": turn_id,
            "meaning": meaning.to_dict(),
            "source": source,
            "effects": list(validated),
            "target": target,
        }
        if revision_target is not None:
            record["revision_target"] = revision_target
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
            "reason": "historical_revision_retracted"
            if revision_target is not None
            else "scoped_evidence_only"
            if nonactual
            else "",
            "assertions": assertions,
            "evidence": [deepcopy(record)],
        }

    def apply_revision(
        self,
        updates: list[dict[str, Any]],
        reviews: list[dict[str, Any]],
        *,
        turn_id: int,
        cue_id: str,
        source: str,
        budget: Budget | None = None,
    ) -> dict[str, Any]:
        """Validate deltas and atomically append a historical revision.

        Callers must obtain effects from the learned dynamics. This boundary
        checks their licence, origin, dependency order and complete projection;
        a serialized replacement world is never accepted as evidence.
        """
        if type(turn_id) is not int or not self._last_turn() < turn_id < 2**53:
            raise ValueError("revision turns must increase")
        bounded_text(cue_id, "revision cue", empty=False)
        bounded_text(source, "revision source", empty=False)
        if (
            type(updates) is not list
            or not 1 <= len(updates) <= 128
            or type(reviews) is not list
            or not 1 <= len(reviews) <= 8
        ):
            raise ValueError("invalid revision update count")
        if budget:
            budget.check()
        candidate = self.clone()
        known = {e["id"]: e for e in self._records()}
        origins = {
            u["event_id"]: u["observation"]
            for r in self._revisions
            for u in r["updates"]
        }
        next_id = self._next_id()
        seen: set[int] = set()
        observations: set[str] = set()
        prior_turn = 0
        for update in updates:
            exact_fields(
                update,
                {
                    "event_id",
                    "observation",
                    "previous_digest",
                    "meaning",
                    "effects",
                    "dependencies",
                    "interpretation_dependencies",
                },
                "event revision",
            )
            event_id = update["event_id"]
            if (
                type(event_id) is not int
                or not 1 <= event_id < 2**53
                or event_id in seen
            ):
                raise ValueError("invalid revised event identity")
            seen.add(event_id)
            observation = Observation.from_dict(update["observation"])
            if (
                not prior_turn < observation.turn_id < turn_id
                or observation.observation_id in observations
            ):
                raise ValueError("invalid revision origin chronology")
            prior_turn = observation.turn_id
            observations.add(observation.observation_id)
            old = known.get(event_id)
            if old is None:
                if (
                    event_id != next_id
                    or update["previous_digest"] is not None
                    or any(e["turn_id"] == observation.turn_id for e in known.values())
                ):
                    raise ValueError("invalid newly resolved event identity")
                next_id += 1
            elif (
                old["turn_id"] != observation.turn_id
                or old["source"] != observation.source
                or old["target"] is not None
                or (old["meaning"] is not None and old["meaning"]["act"] == "retract")
                or update["previous_digest"] != digest(old)
            ):
                raise ValueError("revision does not match its previous event")
            if event_id in origins and origins[event_id] != observation.to_dict():
                raise ValueError("revision rewrites original observation")
            meaning = (
                Meaning.from_dict(update["meaning"])
                if update["meaning"] is not None
                else None
            )
            if meaning is not None and (
                meaning.act not in {"inform", "correct"} or meaning.event is None
            ):
                raise ValueError("revision requires an evidence statement")
            if type(update["effects"]) is not list or len(update["effects"]) > 1:
                raise ValueError("invalid revised effects")
            if meaning is None and update["effects"]:
                raise ValueError("unresolved interpretation cannot project effects")
            for field in ("dependencies", "interpretation_dependencies"):
                dependencies = update[field]
                if (
                    type(dependencies) is not list
                    or len(dependencies) > 128
                    or any(
                        type(i) is not int or not 1 <= i < 2**53 for i in dependencies
                    )
                    or sorted(set(dependencies)) != dependencies
                    or any(
                        i not in known or known[i]["turn_id"] >= observation.turn_id
                        for i in dependencies
                    )
                ):
                    raise ValueError("invalid revised event dependencies")
            known[event_id] = {
                "id": event_id,
                "turn_id": observation.turn_id,
                "meaning": update["meaning"],
                "source": observation.source,
                "effects": update["effects"],
                "target": None,
            }
        roots: set[int] = set()
        old_records = {e["id"]: e for e in self._records()}
        by_observation = {u["observation"]["observation_id"]: u for u in updates}
        for review in reviews:
            exact_fields(
                review,
                {
                    "target",
                    "previous_id",
                    "selected_id",
                    "archive_revision",
                    "archive_after_revision",
                    "archive_before",
                    "archive_after",
                },
                "revision review link",
            )
            update = by_observation.get(review["target"])
            if update is None or update["event_id"] in roots:
                raise ValueError("revision review has no unique event")
            roots.add(update["event_id"])
            old = old_records.get(update["event_id"])
            old_meaning = (
                Meaning.from_dict(old["meaning"])
                if old is not None and old["meaning"] is not None
                else None
            )
            new_meaning = (
                Meaning.from_dict(update["meaning"])
                if update["meaning"] is not None
                else None
            )
            if (
                type(review["archive_revision"]) is not int
                or not 1 <= review["archive_revision"] < 2**53
                or type(review["archive_after_revision"]) is not int
                or review["archive_after_revision"] != review["archive_revision"]
                or review["previous_id"]
                != (
                    Hypothesis.identity(review["target"], old_meaning, ())
                    if old_meaning is not None
                    else None
                )
                or review["selected_id"]
                != (
                    Hypothesis.identity(review["target"], new_meaning, ())
                    if new_meaning is not None
                    else None
                )
            ):
                raise ValueError("revision choice does not match its interpretation")
            # This is an undo receipt for the archive, never a replacement
            # world snapshot. Validate its own semantic/provenance contracts.
            from .attention import AttentionState

            archived = review["archive_before"]
            reviewed = review["archive_after"]
            archive_model = archived["original_snapshot"]["model_fingerprint"]
            archive = AttentionState(archive_model)
            archive.records = [deepcopy(archived)]
            archive.generation = 1
            AttentionState.from_dict(archive.to_dict(), archive_model)
            archive.records = [deepcopy(reviewed)]
            AttentionState.from_dict(archive.to_dict(), archive_model)
            immutable_fields = (
                "observation",
                "speaker",
                "context",
                "before",
                "original_snapshot",
                "index_events",
                "original_selected",
                "original_meaning",
                "initial_suppressed",
            )
            old_cues = {c["observation"]["observation_id"]: c for c in archived["cues"]}
            new_cues = {c["observation"]["observation_id"]: c for c in reviewed["cues"]}
            cue = new_cues.get(cue_id)
            if (
                archived["observation"] != update["observation"]
                or archived["revision"] + 1 != review["archive_revision"]
                or reviewed["revision"] != review["archive_after_revision"]
                or any(archived[key] != reviewed[key] for key in immutable_fields)
                or reviewed["selected_id"] != review["selected_id"]
                or reviewed["selected_meaning"] != update["meaning"]
                or set(new_cues) != set(old_cues) | {cue_id}
                or any(new_cues[key] != value for key, value in old_cues.items())
                or cue_id in old_cues
                or cue is None
                or cue["observation"]["turn_id"] != turn_id
                or cue["observation"]["source"] != source
                or any(
                    c["observation"]["turn_id"] >= turn_id for c in old_cues.values()
                )
                or reviewed["review"] is None
                or reviewed["review"]["cue_id"] != cue_id
                or reviewed["review"]["previous_id"] != archived["selected_id"]
                or reviewed["review"]["comparison"]["snapshot"]["dependency_event_ids"]
                != archived["original_snapshot"]["dependency_event_ids"]
            ):
                raise ValueError("archived review proof does not match revision delta")
            original = next(
                (e for e in self._events if e["id"] == update["event_id"]), None
            )
            historical = self.at_turn(update["observation"]["turn_id"] - 1).facts()
            semantic = [
                {
                    key: fact[key]
                    for key in ("subject", "relation", "value", "negated", "spatial")
                }
                for fact in historical
            ]
            if (
                original is not None
                and archived["original_meaning"] != original["meaning"]
            ) or (
                archived["before"] != semantic
                or archived["original_snapshot"]["dependency_event_ids"]
                != sorted({f["event_id"] for f in historical})
            ):
                raise ValueError("archive origin does not match historical world")
        affected_subjects: set[str] = set()
        for update in updates:
            old = old_records.get(update["event_id"])
            old_meaning = old["meaning"] if old else None
            new_meaning = update["meaning"]
            subjects = {
                m["event"]["object"]
                for m in (old_meaning, new_meaning)
                if m is not None and m["event"] is not None
            }
            if update["event_id"] not in roots and (
                not subjects.intersection(affected_subjects)
                and not set(update["interpretation_dependencies"]).intersection(seen)
            ):
                raise ValueError("revision changes an independent event")
            if (
                update["event_id"] not in roots
                and old_meaning != new_meaning
                and not update["interpretation_dependencies"]
            ):
                raise ValueError("changed interpretation lacks dependency provenance")
            affected_subjects.update(subjects)
        revision = {
            "id": len(self._revisions) + 1,
            "turn_id": turn_id,
            "cue_id": cue_id,
            "source": source,
            "base_digest": self.snapshot_id,
            "reviews": deepcopy(reviews),
            "updates": deepcopy(updates),
        }
        candidate._revisions.append(revision)
        update_by_id = {u["event_id"]: u for u in updates}
        previous: list[dict[str, Any]] = []
        for record in candidate._active():
            if budget:
                budget.check()
            meaning = Meaning.from_dict(record["meaning"])
            assert meaning.event is not None
            before = self._facts(previous)
            validate_effects(meaning.event, record["effects"], before=before)
            update = update_by_id.get(record["id"])
            if update is not None and update["dependencies"] != sorted(
                {
                    f["event_id"]
                    for f in before
                    if meaning.event.actual and f["subject"] == meaning.event.object
                }
            ):
                raise ValueError("revision state dependencies do not match replay")
            previous.append(record)
        # Check the same resource gates as a normal mutation, without charging
        # each changed interpretation as an additional source observation.
        for update in updates:
            self._check_capacity(known[update["event_id"]])
        # Check the union of new names rather than each delta in isolation.
        candidate._check_entities(known[updates[0]["event_id"]])
        if len(known) > self.max_events:
            raise BudgetExceeded("event_capacity")
        if budget:
            budget.check()
        outcome = candidate.revision_outcome(turn_id)
        assert outcome is not None
        if budget:
            budget.check()
        self._revisions = candidate._revisions
        return outcome

    def revision_trace(self, turn_id: int) -> dict[str, Any] | None:
        revision = next((r for r in self._revisions if r["turn_id"] == turn_id), None)
        if revision is None:
            return None
        before, after = self.at_turn(turn_id - 1), self.at_turn(turn_id)
        old_facts, new_facts = before.facts(), after.facts()
        changed = [f for f in old_facts if f not in new_facts]
        changed.extend(f for f in new_facts if f not in old_facts)
        ids = {u["event_id"] for u in revision["updates"]}
        ids.update(f["event_id"] for f in changed)
        targets = {r["target"] for r in revision["reviews"]}
        return {
            "schema": REVISION_TRACE_SCHEMA,
            "revision_id": revision["id"],
            "turn_id": turn_id,
            "cue_id": revision["cue_id"],
            "target_observation_ids": sorted(targets),
            "affected_event_ids": sorted(ids),
            "affected_turn_ids": sorted(
                {
                    e["turn_id"]
                    for e in [*before._records(), *after._records()]
                    if e["id"] in ids
                }
            ),
            "replayed_event_ids": [
                u["event_id"]
                for u in revision["updates"]
                if u["observation"]["observation_id"] not in targets
            ],
            "base_digest": revision["base_digest"],
        }

    def revision_outcome(self, turn_id: int) -> dict[str, Any] | None:
        trace = self.revision_trace(turn_id)
        if trace is None:
            return None
        world = self.at_turn(turn_id)
        uncertain = set(world.uncertain_subjects)
        assertions = [
            f
            for f in world.facts()
            if f["event_id"] in trace["affected_event_ids"]
            and f["subject"] not in uncertain
        ]
        ids = {f["event_id"] for f in assertions}
        return {
            "action": "corrected",
            "reason": "historical_interpretation_uncertain"
            if uncertain
            else "historical_interpretation_revised",
            "assertions": assertions,
            "evidence": [e for e in world._records() if e["id"] in ids],
            "revision": trace,
        }

    def retracted_revision(self, turn_id: int) -> dict[str, Any] | None:
        record = next((e for e in self._events if e["turn_id"] == turn_id), None)
        if record is None or "revision_target" not in record:
            return None
        return deepcopy(self._revisions[record["revision_target"] - 1])

    @property
    def uncertain_subjects(self) -> tuple[str, ...]:
        """Local unresolved historical alternatives, superseded by later evidence."""
        latest = {
            u["event_id"]: (r, u)
            for r in self._active_revisions()
            for u in r["updates"]
        }
        removed = {
            e["target"] for e in self._events if e["meaning"]["act"] == "retract"
        }
        active = self._active()
        uncertain: set[str] = set()
        for event_id, (revision, update) in latest.items():
            if update["meaning"] is not None or event_id in removed:
                continue
            review = next(
                (
                    r
                    for r in revision["reviews"]
                    if r["target"] == update["observation"]["observation_id"]
                ),
                None,
            )
            events = review["archive_before"]["index_events"] if review else []
            previous = self.at_turn(revision["turn_id"] - 1).event_for_observation(
                Observation.from_dict(update["observation"])
            )
            if previous is not None and previous["meaning"] is not None:
                events = [*events, previous["meaning"]["event"]]
            for raw in events:
                if raw is None:
                    continue
                event = Event.from_dict(raw)
                if not event.actual or not event.object:
                    continue
                later = any(
                    record["turn_id"] > update["observation"]["turn_id"]
                    and any(
                        effect["subject"] == event.object and effect["op"] == "set"
                        for effect in record["effects"]
                    )
                    for record in active
                )
                if not later:
                    uncertain.add(event.object)
        return tuple(sorted(uncertain))

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
        uncertain = set(self.uncertain_subjects)
        if query.subject in uncertain or (
            query.kind == "what_has"
            and any(
                f["subject"] in uncertain
                and f["relation"] == "holder"
                and f["value"] == query.subject
                for f in self.facts()
            )
        ):
            return {
                "action": "clarify",
                "reason": "historical_interpretation_uncertain",
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
            "evidence": [deepcopy(e) for e in self._records() if e["id"] in ids],
            "truth": truth,
        }

    def explain(
        self, assertions: list[dict[str, Any]], *, subject: str = ""
    ) -> dict[str, Any]:
        bounded_text(subject, "explanation subject")
        if subject:
            assertions = [f for f in assertions if f["subject"] == subject]
        if any(f["subject"] in self.uncertain_subjects for f in assertions):
            return {
                "action": "clarify",
                "reason": "historical_interpretation_uncertain",
                "assertions": [],
                "evidence": [],
            }
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
            "evidence": [deepcopy(e) for e in self._records() if e["id"] in ids],
        }

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": _REVISION_SCHEMA if self._revisions else _SCHEMA,
            "max_events": self.max_events,
            "max_entities": self.max_entities,
            "events": self.events,
        }
        if self._revisions:
            value["revisions"] = self.revisions
        return value

    @classmethod
    def from_dict(cls, value: Any) -> ExperienceWorld:
        fields = {"schema", "max_events", "max_entities", "events"}
        if type(value) is dict and value.get("schema") == _REVISION_SCHEMA:
            fields.add("revisions")
        value = exact_fields(value, fields, "world")
        if (
            type(value["schema"]) is not str
            or value["schema"] not in {_SCHEMA, _REVISION_SCHEMA}
            or type(value["events"]) is not list
        ):
            raise ValueError("invalid world schema")
        world = cls(max_events=value["max_events"], max_entities=value["max_entities"])
        revisions = value.get("revisions", [])
        if type(revisions) is not list or (
            value["schema"] == _REVISION_SCHEMA and not revisions
        ):
            raise ValueError("invalid world revision journal")
        if len(value["events"]) + len(revisions) > world.max_events:
            raise ValueError("event capacity exceeded")
        for revision in revisions:
            exact_fields(
                revision,
                {
                    "id",
                    "turn_id",
                    "cue_id",
                    "source",
                    "base_digest",
                    "reviews",
                    "updates",
                },
                "world revision",
            )
            if type(revision["id"]) is not int:
                raise ValueError("invalid revision identity")
        if any(type(e) is not dict for e in value["events"]):
            raise ValueError("invalid world event record")
        operations = [("event", e) for e in value["events"]]
        operations.extend(("revision", r) for r in revisions)
        if any(type(r.get("turn_id")) is not int for _, r in operations):
            raise ValueError("invalid world operation turn")
        operations.sort(key=lambda item: item[1]["turn_id"])
        for kind, record in operations:
            if kind == "revision":
                try:
                    world.apply_revision(
                        record["updates"],
                        record["reviews"],
                        turn_id=record["turn_id"],
                        cue_id=record["cue_id"],
                        source=record["source"],
                    )
                except BudgetExceeded as exc:
                    raise ValueError(str(exc)) from exc
                if world._revisions[-1] != record:
                    raise ValueError("revision does not match deterministic replay")
                continue
            event_fields = {"id", "turn_id", "meaning", "source", "effects", "target"}
            if "revision_target" in record:
                event_fields.add("revision_target")
            exact_fields(
                record,
                event_fields,
                "world event",
            )
            if type(record["id"]) is not int or record["id"] != world._next_id():
                raise ValueError("invalid world event order")
            if record["target"] is not None and (
                type(record["target"]) is not int
                or not 1 <= record["target"] < record["id"]
            ):
                raise ValueError("invalid retraction target")
            if "revision_target" in record and (
                type(record["revision_target"]) is not int
                or not 1 <= record["revision_target"] <= len(world._revisions)
            ):
                raise ValueError("invalid revision retraction target")
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
        if world._events != value["events"] or world._revisions != revisions:
            raise ValueError("world journals are not in chronological order")
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

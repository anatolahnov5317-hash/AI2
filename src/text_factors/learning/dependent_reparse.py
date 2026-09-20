"""Bounded, provenance-bearing rereads of historical dependent language.

The archive keeps its original choices. This hook reconstructs a private
language context from observed turns and current meaning versions, then asks
the learned parser to propose again. A later correction is never inserted as
an earlier utterance, and literal participant names are never substituted.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any

from ..conversation.schema import Budget, BudgetExceeded
from .attention import AttentionState, memory_token
from .candidate_search import SearchLimits
from .candidate_selection import select, validate_trace
from .hypotheses import Observation, digest
from .language_data import PRONOUN_FORMS, tokenize
from .revision import Reinterpretation
from .schema import DialogueContext, Entity, Interpretation, Meaning
from .world import ExperienceWorld

_SEMANTIC = ("subject", "relation", "value", "negated", "spatial")


def _meaning(value: Any) -> Meaning | None:
    return Meaning.from_dict(value) if value is not None else None


def _advance(
    context: DialogueContext,
    observation: Observation,
    meaning: Meaning | None,
    pending: str,
) -> DialogueContext:
    """Advance the session's bounded referents using one actual observation."""
    entities = {entity.name: entity for entity in context.entities}
    mentioned: list[str] = []
    if meaning is not None:
        entities.update({entity.name: entity for entity in meaning.entities})
        event = meaning.event
        while event is not None and event.content is not None:
            event = event.content
        if event is not None:
            mentioned = [
                name
                for name in (event.object, event.actor, event.recipient, event.place)
                if name
            ]
        if meaning.query is not None and meaning.query.subject:
            mentioned = [meaning.query.subject]
        for name in mentioned:
            entities.setdefault(name, Entity(name))
    focus = list(dict.fromkeys([*mentioned, *context.focus]))[:16]
    order = list(dict.fromkeys([*focus, *reversed(entities)]))[:64]
    return DialogueContext(
        (*context.turns, observation.text)[-16:],
        tuple(entities[name] for name in order if name in entities),
        tuple(focus),
        pending,
    )


class DependentReparser:
    """A single revision's chronological reparse hook and detached audit trace."""

    def __init__(
        self,
        world: ExperienceWorld,
        before_attention: AttentionState,
        after_attention: AttentionState,
        understanding: Any,
        dynamics: Any,
        *,
        model_fingerprint: str,
        budget: Budget,
        max_reparses: int = 2,
    ) -> None:
        if (
            not isinstance(world, ExperienceWorld)
            or not isinstance(before_attention, AttentionState)
            or not isinstance(after_attention, AttentionState)
            or not isinstance(budget, Budget)
            or before_attention.model_fingerprint != model_fingerprint
            or after_attention.model_fingerprint != model_fingerprint
        ):
            raise ValueError("invalid dependent reparse inputs")
        if type(max_reparses) is not int or not 0 <= max_reparses <= 128:
            raise ValueError("invalid dependent reparse capacity")
        self._world = world
        self._understanding = understanding
        self._dynamics = dynamics
        self._fingerprint = model_fingerprint
        self._budget = budget
        self._limit = max_reparses
        self._reads: list[dict[str, Any]] = []
        self._attempts = 0
        self._last_turn = 0
        self._overrides: dict[int, tuple[int, Meaning | None]] = {}
        self._records = {
            record["observation"]["turn_id"]: deepcopy(record)
            for record in before_attention.records
        }
        self._events = {record["turn_id"]: record for record in world._records()}
        self._roots: dict[int, dict[str, Any]] = {}
        next_id = world._next_id()
        for record in after_attention.records:
            observation = Observation.from_dict(record["observation"])
            previous = self._records.get(observation.turn_id)
            if previous is None or previous["revision"] == record["revision"]:
                continue
            if previous["observation"] != record["observation"]:
                raise ValueError("dependent reparse observation changed")
            existing = self._events.get(observation.turn_id)
            chosen = _meaning(record["selected_meaning"])
            if chosen is not None and chosen.event is None:
                continue
            if existing is None and chosen is None:
                continue
            event_id = existing["id"] if existing is not None else next_id
            if existing is None:
                next_id += 1
            self._roots[event_id] = deepcopy(record)

    @property
    def trace(self) -> dict[str, Any]:
        return {
            "attempts": self._attempts,
            "max_reparses": self._limit,
            "rereads": deepcopy(self._reads),
        }

    def _remaining(self, cap: float) -> float:
        self._budget.check()
        return max(0.000001, min(cap, self._budget.deadline - perf_counter()))

    def _contexts(
        self,
        observation: Observation,
        revised: tuple[int, ...],
    ) -> tuple[DialogueContext, DialogueContext, tuple[int, ...]] | None:
        # The revision core's IDs license roots, including newly allocated IDs
        # for observations that did not previously have a selected world event.
        roots = {
            record["observation"]["turn_id"]: (event_id, record)
            for event_id, record in self._roots.items()
            if event_id in revised
            and record["observation"]["turn_id"] < observation.turn_id
        }
        if not roots:
            return None
        first = min(roots)
        turns = sorted(t for t in self._records if first <= t <= observation.turn_id)
        if (
            len(turns) != observation.turn_id - first + 1
            or not turns
            or turns[0] != first
            or turns[-1] != observation.turn_id
            or self._records[observation.turn_id]["observation"]
            != observation.to_dict()
        ):
            raise BudgetExceeded("revision_source_not_retained")
        old = DialogueContext.from_dict(self._records[first]["context"])
        new = old
        dependencies: set[int] = set()
        for turn, following in zip(turns[:-1], turns[1:], strict=True):
            self._budget.check()
            record = self._records[turn]
            source = Observation.from_dict(record["observation"])
            version = self._events.get(turn)
            original = _meaning(
                version["meaning"]
                if version is not None
                else record["original_meaning"]
            )
            meaning = original
            if turn in roots:
                event_id, revised_record = roots[turn]
                meaning = _meaning(revised_record["selected_meaning"])
                dependencies.add(event_id)
            elif turn in self._overrides:
                event_id, meaning = self._overrides[turn]
                if event_id not in revised:
                    raise ValueError("dependent reparse version was not replayed")
                dependencies.add(event_id)
            # Preserve the actually observed pending dialogue intent. It is
            # not inferred from a replacement meaning or from a future cue.
            pending = self._records[following]["context"]["pending"]
            old = _advance(old, source, original, pending)
            new = _advance(new, source, meaning, pending)
        return old, new, tuple(sorted(dependencies))

    def __call__(
        self,
        observation: Observation,
        old_meaning: Meaning,
        before_semantic_facts: list[dict[str, Any]],
        revised_event_ids: tuple[int, ...],
    ) -> Reinterpretation | None:
        if not self._roots:
            return None
        self._budget.check()
        if observation.turn_id <= self._last_turn:
            raise ValueError("dependent reparses must be chronological")
        self._last_turn = observation.turn_id
        if not any(token in PRONOUN_FORMS for token in tokenize(observation.text)):
            return None
        contexts = self._contexts(observation, revised_event_ids)
        if contexts is None:
            return None
        old_context, context, dependencies = contexts
        old_facts = [
            {key: fact[key] for key in _SEMANTIC}
            for fact in self._world.facts_before(observation.turn_id)
        ]
        if context == old_context and before_semantic_facts == old_facts:
            return None
        if self._attempts >= self._limit:
            raise BudgetExceeded("revision_reparse_capacity")
        self._attempts += 1
        memory = memory_token(self._dynamics)
        batch = self._understanding.propose(
            observation.text,
            context,
            observation=observation,
            limits=SearchLimits(seconds=self._remaining(0.4)),
        )
        self._budget.check()
        if not batch.complete:
            raise BudgetExceeded("revision_incomplete_candidate_search")
        if batch.observation != observation or batch.context_digest != digest(
            context.to_dict()
        ):
            raise ValueError("dependent reparse candidate provenance mismatch")
        selected = select(
            batch,
            Interpretation(None),
            self._dynamics,
            deepcopy(before_semantic_facts),
            model_fingerprint=self._fingerprint,
            seconds=self._remaining(0.8),
            dependency_event_ids=dependencies,
        )
        self._budget.check()
        if memory_token(self._dynamics) != memory:
            raise ValueError("stale dependent reparse memory")
        if selected.meaning is not None and (
            selected.meaning.event is None
            or selected.meaning.act not in {"inform", "correct"}
        ):
            raise BudgetExceeded("revision_non_event_reparse")
        assert selected.diagnostics is not None
        comparison = selected.diagnostics["hypotheses"]
        validate_trace(comparison, selected.meaning)
        changed = selected.meaning != old_meaning
        self._reads.append(
            {
                "observation": observation.to_dict(),
                "old_context_digest": digest(old_context.to_dict()),
                "context": context.to_dict(),
                "dependency_event_ids": list(dependencies),
                "memory_version": list(memory),
                "comparison": comparison,
                "changed": changed,
            }
        )
        if not changed:
            return None
        version = self._events.get(observation.turn_id)
        if version is None:
            raise ValueError("dependent reparse has no original world event")
        self._overrides[observation.turn_id] = (version["id"], selected.meaning)
        return Reinterpretation(selected.meaning, dependencies)


def make_reparser(
    world: ExperienceWorld,
    before_attention: AttentionState,
    after_attention: AttentionState,
    understanding: Any,
    dynamics: Any,
    *,
    model_fingerprint: str,
    budget: Budget,
    max_reparses: int = 2,
) -> DependentReparser:
    """Build a fresh, bounded hook for one staged archive revision."""
    return DependentReparser(
        world,
        before_attention,
        after_attention,
        understanding,
        dynamics,
        model_fingerprint=model_fingerprint,
        budget=budget,
        max_reparses=max_reparses,
    )

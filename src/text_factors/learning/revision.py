"""Bounded historical revision of learned event effects.

Original observations and events are immutable. A plan replays affected object
transitions on a private projection, then commits licensed deltas together. A
caller may provide a provenance-bearing hook to reparse dependent references;
this module never changes a participant merely because its name is similar.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ..conversation.schema import Budget, BudgetExceeded
from .attention import AttentionState, memory_token
from .hypotheses import Hypothesis, Observation, digest
from .schema import Meaning
from .world import ExperienceWorld

_SEMANTIC = {"subject", "relation", "value", "negated", "spatial"}


@dataclass(frozen=True)
class Reinterpretation:
    """Explicit result of a caller's dependent-language reparse."""

    meaning: Meaning | None
    dependency_event_ids: tuple[int, ...]


Reparse = Callable[
    [Observation, Meaning, list[dict[str, Any]], tuple[int, ...]],
    Reinterpretation | None,
]


@dataclass(frozen=True)
class RevisionPlan:
    complete: bool
    reason: str = ""
    applicable: bool = False
    world: ExperienceWorld | None = None
    outcome: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    affected_event_ids: tuple[int, ...] = ()
    affected_turn_ids: tuple[int, ...] = ()
    base_digest: str = ""
    result_digest: str = ""
    memory_version: tuple[Any, Any] = (None, None)
    _dynamics: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any] | None:
        return deepcopy(self.trace)


def _semantic(facts: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [{key: fact[key] for key in _SEMANTIC} for fact in facts]


def _identity(observation: Observation, meaning: Any) -> str | None:
    return (
        Hypothesis.identity(observation.observation_id, Meaning.from_dict(meaning), ())
        if meaning is not None
        else None
    )


def _expressible(record: dict[str, Any], cue_id: str) -> bool:
    """A literal incompatible statement is a normal correction, not a reparse."""
    if record["selected_meaning"] is not None:
        return True
    if record["reason"] != "ambiguous_no_supported_candidate":
        return True
    cue = next(
        (c for c in record["cues"] if c["observation"]["observation_id"] == cue_id),
        None,
    )
    review = record["review"]
    if cue is None or review is None:
        return False
    return any(
        candidate["meaning"] is not None
        and candidate["meaning"]["event"] == cue["meaning"]["event"]
        and not candidate["missing"]
        for candidate in review["comparison"]["batch"]["candidates"]
    )


def prepare_archive_revision(
    world: ExperienceWorld,
    before_attention: AttentionState,
    after_attention: AttentionState,
    dynamics: Any,
    *,
    turn_id: int,
    source: str,
    budget: Budget | None = None,
    max_replayed: int = 128,
    reparse: Reparse | None = None,
) -> RevisionPlan:
    """Prepare archive changes without changing either input world or archive.

    ``reparse`` receives the immutable observation, previous meaning, semantic
    facts before its effective turn and already revised event IDs. It must return
    explicit earlier event dependencies for any changed interpretation. Without
    this hook, only dynamics dependencies are replayed; reference bindings stay
    as originally interpreted.
    """
    if not isinstance(world, ExperienceWorld) or not all(
        isinstance(a, AttentionState) for a in (before_attention, after_attention)
    ):
        raise ValueError("invalid revision inputs")
    if type(max_replayed) is not int or not 1 <= max_replayed <= 128:
        raise ValueError("invalid revision replay capacity")
    if type(turn_id) is not int or not world._last_turn() < turn_id < 2**53:
        raise ValueError("invalid revision turn")
    work_budget = budget or Budget(5.0)
    base = world.snapshot_id
    archives = (before_attention.snapshot_id, after_attention.snapshot_id)
    memory = memory_token(dynamics)
    cue_id = f"turn:{turn_id}"
    roots: dict[int, tuple[dict[str, Any], dict[str, Any], Observation]] = {}
    next_id = world._next_id()
    try:
        work_budget.check()
        for record in after_attention.records:
            observation = Observation.from_dict(record["observation"])
            old = before_attention.get(observation.observation_id)
            if old is None or record["revision"] == old["revision"]:
                continue
            if (
                old["observation"] != record["observation"]
                or record["revision"] != old["revision"] + 1
                or record["review"] is None
                or record["review"]["cue_id"] != cue_id
                or observation.turn_id >= turn_id
            ):
                raise ValueError("archive change is not this turn's revision")
            if not _expressible(record, cue_id):
                continue
            existing = world.event_for_observation(observation)
            if existing is None and record["selected_meaning"] is None:
                continue
            if existing is not None and existing["meaning"] is not None:
                meaning = Meaning.from_dict(existing["meaning"])
                if meaning.event is None or meaning.act not in {"inform", "correct"}:
                    raise ValueError("review targets a non-evidence world event")
            if record["selected_meaning"] is not None:
                meaning = Meaning.from_dict(record["selected_meaning"])
                if meaning.event is None:
                    continue
            event_id = existing["id"] if existing is not None else next_id
            if existing is None:
                next_id += 1
            roots[event_id] = (old, record, observation)
        if not roots:
            return RevisionPlan(True, "no_applicable_historical_revision")
        records = {e["id"]: e for e in world._records()}
        removed = {
            e["target"] for e in world.events if e["meaning"]["act"] == "retract"
        }
        active = {e["id"] for e in world._active()}
        for event_id, (_, _, observation) in roots.items():
            records.setdefault(
                event_id,
                {
                    "id": event_id,
                    "turn_id": observation.turn_id,
                    "meaning": None,
                    "source": observation.source,
                    "effects": [],
                    "target": None,
                },
            )
        archived = {r["observation"]["turn_id"]: r for r in before_attention.records}
        prefix: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        reviews: list[dict[str, Any]] = []
        subjects: set[str] = set()
        revised: list[int] = []
        reparsed = 0
        first_turn = min(item[2].turn_id for item in roots.values())
        for current in sorted(records.values(), key=lambda e: (e["turn_id"], e["id"])):
            work_budget.check()
            event_id = current["id"]
            root = roots.get(event_id)
            if root is None and event_id not in active:
                continue
            old_meaning = (
                Meaning.from_dict(current["meaning"])
                if current["meaning"] is not None
                else None
            )
            meaning = old_meaning
            observation = (
                root[2]
                if root is not None
                else Observation.from_dict(archived[current["turn_id"]]["observation"])
                if current["turn_id"] in archived
                else None
            )
            before = ExperienceWorld._facts(prefix)
            interpretation_dependencies: tuple[int, ...] = ()
            if root is not None:
                meaning = (
                    Meaning.from_dict(root[1]["selected_meaning"])
                    if root[1]["selected_meaning"] is not None
                    else None
                )
            elif reparse is not None and current["turn_id"] > first_turn:
                if reparsed >= max_replayed:
                    raise BudgetExceeded("revision_reparse_capacity")
                reparsed += 1
                if observation is None:
                    raise BudgetExceeded("revision_source_not_retained")
                assert old_meaning is not None
                result = reparse(
                    observation, old_meaning, _semantic(before), tuple(revised)
                )
                if result is not None:
                    if not isinstance(result, Reinterpretation):
                        raise ValueError("invalid dependent reparse result")
                    meaning = result.meaning
                    interpretation_dependencies = result.dependency_event_ids
                    if meaning != old_meaning and not interpretation_dependencies:
                        raise ValueError("dependent reparse has no event provenance")
            event = meaning.event if meaning is not None else None
            affected = (
                root is not None
                or bool(interpretation_dependencies)
                or bool(event is not None and event.object in subjects)
            )
            if not affected:
                prefix.append(current)
                continue
            if len(updates) >= max_replayed:
                raise BudgetExceeded("revision_replay_capacity")
            if observation is None:
                raise BudgetExceeded("revision_source_not_retained")
            if observation.source != current["source"]:
                raise ValueError("replay source does not match its original event")
            effects = []
            if event is not None:
                prediction = dynamics.predict(
                    _semantic(before), event, seconds=min(1.0, _remaining(work_budget))
                )
                work_budget.check()
                if not prediction.supported:
                    raise BudgetExceeded("revision_unsupported_transition")
                effects = list(prediction.effects)
            elif meaning is not None:
                raise ValueError("dependent interpretation is not an event")
            existing = world.event_for_observation(observation)
            update = {
                "event_id": event_id,
                "observation": observation.to_dict(),
                "previous_digest": digest(existing) if existing is not None else None,
                "meaning": meaning.to_dict() if meaning is not None else None,
                "effects": effects,
                "dependencies": sorted(
                    {
                        f["event_id"]
                        for f in before
                        if event is not None
                        and event.actual
                        and f["subject"] == event.object
                    }
                ),
                "interpretation_dependencies": list(interpretation_dependencies),
            }
            updates.append(update)
            revised.append(event_id)
            for interpreted in (old_meaning, meaning):
                if interpreted is not None and interpreted.event is not None:
                    subjects.add(interpreted.event.object)
            if root is not None:
                reviews.append(
                    {
                        "target": observation.observation_id,
                        "previous_id": _identity(observation, existing["meaning"])
                        if existing is not None
                        else None,
                        "selected_id": _identity(observation, update["meaning"]),
                        "archive_revision": root[1]["revision"],
                        "archive_after_revision": root[1]["revision"],
                        "archive_before": deepcopy(root[0]),
                        "archive_after": deepcopy(root[1]),
                    }
                )
            if meaning is not None and event_id not in removed:
                prefix.append(
                    {**current, "meaning": update["meaning"], "effects": effects}
                )
        candidate = world.clone()
        outcome = candidate.apply_revision(
            updates,
            reviews,
            turn_id=turn_id,
            cue_id=cue_id,
            source=source,
            budget=work_budget,
        )
        if (
            world.snapshot_id != base
            or archives != (before_attention.snapshot_id, after_attention.snapshot_id)
            or memory_token(dynamics) != memory
        ):
            raise ValueError("stale revision preparation")
        trace = candidate.revision_trace(turn_id)
        assert trace is not None
        return RevisionPlan(
            True,
            "",
            True,
            candidate,
            outcome,
            trace,
            tuple(trace["affected_event_ids"]),
            tuple(trace["affected_turn_ids"]),
            base,
            candidate.snapshot_id,
            memory,
            dynamics,
        )
    except (BudgetExceeded, TimeoutError) as exc:
        return RevisionPlan(False, str(exc), bool(roots), base_digest=base)


def _remaining(budget: Budget) -> float:
    from time import perf_counter

    budget.check()
    return max(0.000001, budget.deadline - perf_counter())


def commit_revision(
    world: ExperienceWorld, plan: RevisionPlan
) -> dict[str, Any] | None:
    """Commit a complete, unchanged prepared projection; otherwise change nothing."""
    if not isinstance(plan, RevisionPlan) or not plan.complete:
        raise ValueError("incomplete revision plan")
    if not plan.applicable:
        return None
    if (
        plan.world is None
        or plan.trace is None
        or world.snapshot_id != plan.base_digest
        or plan.world.snapshot_id != plan.result_digest
        or memory_token(plan._dynamics) != plan.memory_version
    ):
        raise ValueError("stale revision commit")
    checked = ExperienceWorld.from_dict(plan.world.to_dict())
    if checked.revision_trace(plan.trace["turn_id"]) != plan.trace:
        raise ValueError("revision trace does not match prepared deltas")
    outcome = checked.revision_outcome(plan.trace["turn_id"])
    if outcome != plan.outcome:
        raise ValueError("revision outcome does not match prepared deltas")
    world._events, world._revisions = checked._events, checked._revisions
    return deepcopy(outcome)

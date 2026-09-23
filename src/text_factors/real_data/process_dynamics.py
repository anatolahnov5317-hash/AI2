"""Research-only, group-aware process transition forecasts.

This component consumes already verified, structured observations. It does not
parse text, verify source permissions, or claim that its support fraction is a
calibrated probability. Development is frozen; only a separately designated
future stream can update after its outcome has been scored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from .contracts import RoleValue

Split = Literal["train", "development", "future_stream"]
Effect = tuple[str, str, str, bool]
Signature = tuple[
    str, tuple[tuple[str, str], ...], tuple[tuple[str, str, str, bool], ...]
]
Template = tuple[tuple[str, Effect], ...]


class StageMetrics(TypedDict):
    episodes: int
    independent_groups: int
    supported_episodes: int
    abstained_episodes: int
    exact_episodes: int
    groups_with_supported: int
    group_coverage_numerator: float
    group_exact_numerator: float
    supported_group_exact_numerator: float
    coverage: float | None
    exact_per_group: float | None
    exact_among_supported: float | None


class StreamMetrics(TypedDict):
    split: str
    episodes: int
    independent_groups: int
    forecast: dict[str, StageMetrics]
    current_fact: StageMetrics
    updates_after_reveal: int


def _name(value: str, field: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"invalid {field}")
    value.encode("utf-8", errors="strict")
    return value


def _moment(value: datetime, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ProcessFact:
    """A positive or explicitly negative observed fact; missing means unknown."""

    subject_id: str
    relation_id: str
    value_id: str
    negated: bool = False

    def __post_init__(self) -> None:
        for field in ("subject_id", "relation_id", "value_id"):
            _name(getattr(self, field), field)
        if type(self.negated) is not bool:
            raise ValueError("invalid fact polarity")


@dataclass(frozen=True, slots=True)
class ProcessEvent:
    relation_id: str
    roles: tuple[RoleValue, ...]

    def __post_init__(self) -> None:
        _name(self.relation_id, "event relation")
        if type(self.roles) is not tuple or not 1 <= len(self.roles) <= 16:
            raise ValueError("event needs bounded roles")
        if any(not isinstance(role, RoleValue) for role in self.roles):
            raise ValueError("invalid event role")
        if len({role.role for role in self.roles}) != len(self.roles):
            raise ValueError("duplicate event role")


def _facts(value: tuple[ProcessFact, ...]) -> tuple[ProcessFact, ...]:
    if type(value) is not tuple or len(value) > 128:
        raise ValueError("facts must be a bounded tuple")
    if any(not isinstance(fact, ProcessFact) for fact in value):
        raise ValueError("invalid fact")
    if len(set(value)) != len(value):
        raise ValueError("duplicate fact")
    return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class TransitionPrompt:
    """Information available at the cutoff; contains no future outcome."""

    episode_id: str
    group_id: str
    split: Split
    before_at: datetime
    cutoff_at: datetime
    outcome_due_at: datetime
    before: tuple[ProcessFact, ...]
    event: ProcessEvent
    tracked_relations: tuple[str, ...] = ()
    before_complete: bool = False

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        _name(self.group_id, "group_id")
        if self.split not in ("train", "development", "future_stream"):
            raise ValueError(
                "sealed or unknown split is not available to this component"
            )
        if not isinstance(self.event, ProcessEvent):
            raise ValueError("invalid event")
        if (
            type(self.tracked_relations) is not tuple
            or not self.tracked_relations
            or len(set(self.tracked_relations)) != len(self.tracked_relations)
            or any(type(item) is not str or not item for item in self.tracked_relations)
        ):
            raise ValueError("complete observation scope needs relation IDs")
        if self.before_complete is not True:
            raise ValueError("before must be complete within the declared scope")
        if not (
            _moment(self.before_at, "before_at")
            <= _moment(self.cutoff_at, "cutoff_at")
            < _moment(self.outcome_due_at, "outcome_due_at")
        ):
            raise ValueError("observation, cutoff and outcome times must be ordered")
        object.__setattr__(self, "before", _facts(self.before))
        if any(fact.relation_id not in self.tracked_relations for fact in self.before):
            raise ValueError("before fact outside complete observation scope")


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """Revealed independently after a prediction, at the registered horizon."""

    episode_id: str
    observed_at: datetime
    after: tuple[ProcessFact, ...]
    after_complete: bool = False

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        _moment(self.observed_at, "observed_at")
        object.__setattr__(self, "after", _facts(self.after))
        if self.after_complete is not True:
            raise ValueError("after must be complete within the declared scope")


@dataclass(frozen=True, slots=True)
class TransitionCase:
    prompt: TransitionPrompt
    outcome: TransitionOutcome
    # Independent verified current-state labels. Without these, factual quality
    # is undefined; evaluating 'before' against itself would be tautological.
    fact_gold: tuple[ProcessFact, ...] | None = None
    fact_prediction: tuple[ProcessFact, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, TransitionPrompt) or not isinstance(
            self.outcome, TransitionOutcome
        ):
            raise ValueError("invalid transition case")
        if self.prompt.episode_id != self.outcome.episode_id:
            raise ValueError("outcome belongs to another episode")
        if self.prompt.outcome_due_at != self.outcome.observed_at:
            raise ValueError("outcome does not match the registered forecast horizon")
        if any(
            fact.relation_id not in self.prompt.tracked_relations
            for fact in self.outcome.after
        ):
            raise ValueError("after fact outside complete observation scope")
        if (self.fact_gold is None) != (self.fact_prediction is None):
            raise ValueError("fact gold and prediction must both be present or absent")
        if self.fact_gold is not None and self.fact_prediction is not None:
            object.__setattr__(self, "fact_gold", _facts(self.fact_gold))
            object.__setattr__(self, "fact_prediction", _facts(self.fact_prediction))


@dataclass(frozen=True, slots=True)
class Forecast:
    after: tuple[ProcessFact, ...] | None
    support_groups: int
    competing_groups: int
    reason: str

    @property
    def supported(self) -> bool:
        return self.after is not None


def _role_for(value: str, event: ProcessEvent) -> str | None:
    matches = [role.role for role in event.roles if role.value_id == value]
    return matches[0] if len(matches) == 1 else None


def _signature(prompt: TransitionPrompt) -> Signature:
    related = []
    for fact in prompt.before:
        subject = _role_for(fact.subject_id, prompt.event)
        if subject is not None:
            related.append(
                (
                    fact.relation_id,
                    subject,
                    _role_for(fact.value_id, prompt.event) or "<other>",
                    fact.negated,
                )
            )
    return (
        prompt.event.relation_id,
        tuple(sorted((role.role, role.value_type) for role in prompt.event.roles)),
        tuple(sorted(related)),
    )


def _effect(fact: ProcessFact, event: ProcessEvent) -> Effect:
    subject = _role_for(fact.subject_id, event)
    value = _role_for(fact.value_id, event)
    if subject is None or value is None:
        raise ValueError(
            "unbound or ambiguous outcome role: abstain rather than invent a transition"
        )
    return (subject, fact.relation_id, value, fact.negated)


def _template(case: TransitionCase) -> Template:
    before, after = set(case.prompt.before), set(case.outcome.after)
    changes = [("remove", _effect(fact, case.prompt.event)) for fact in before - after]
    changes += [("add", _effect(fact, case.prompt.event)) for fact in after - before]
    if len(changes) > 16:
        raise ValueError("transition exceeds effect budget")
    return tuple(sorted(changes))


def _project(
    prompt: TransitionPrompt, template: Template
) -> tuple[ProcessFact, ...] | None:
    roles = {role.role: role.value_id for role in prompt.event.roles}
    state = set(prompt.before)
    try:
        for operation, (subject, relation, value, negated) in template:
            fact = ProcessFact(roles[subject], relation, roles[value], negated)
            if operation == "remove":
                if fact not in state:
                    return None
                state.remove(fact)
            elif operation == "add":
                state.add(fact)
            else:
                raise ValueError("invalid effect operation")
    except KeyError:
        return None
    return tuple(sorted(state))


class ProcessDynamics:
    """Learn role-bound deltas; one independent vote per signature and group."""

    def __init__(self, *, min_support_groups: int = 2) -> None:
        if type(min_support_groups) is not int or min_support_groups < 1:
            raise ValueError("invalid minimum independent support")
        self.min_support_groups = min_support_groups
        self._votes: dict[Signature, dict[str, Template | None]] = defaultdict(dict)
        self._frequency: dict[str, dict[str, Template | None]] = defaultdict(dict)
        self._trained_groups: set[str] = set()
        self._development_groups: set[str] = set()
        self._development_ids: set[str] = set()
        self._future_groups: set[str] = set()
        self._seen_ids: set[str] = set()
        self._max_train_outcome_at: datetime | None = None
        self._last_future_outcome_at: datetime | None = None
        self._pending_future: TransitionPrompt | None = None

    @property
    def trained_groups(self) -> frozenset[str]:
        return frozenset(self._trained_groups)

    @staticmethod
    def _vote(
        table: dict[str, Template | None], group_id: str, template: Template
    ) -> None:
        if group_id not in table:
            table[group_id] = template
        elif table[group_id] != template:
            # Contradictory repeats within a family do not manufacture votes.
            table[group_id] = None

    def learn(self, case: TransitionCase) -> None:
        prompt = case.prompt
        if prompt.split not in ("train", "future_stream"):
            raise ValueError("development and calibration are frozen for fitting")
        if prompt.split == "train" and (
            self._development_groups or self._future_groups
        ):
            raise ValueError("train is frozen after development or future evaluation")
        if prompt.split == "future_stream" and self._pending_future != prompt:
            raise ValueError("future outcome requires a recorded predict call first")
        if prompt.episode_id in self._seen_ids | self._development_ids:
            raise ValueError("duplicate episode ID")
        foreign = (
            self._future_groups | self._development_groups
            if prompt.split == "train"
            else self._trained_groups | self._development_groups
        )
        if prompt.group_id in foreign:
            raise ValueError("group crosses open split boundary")
        template = _template(case)  # Check whole event before mutating any vote.
        signature = _signature(prompt)
        self._vote(self._votes[signature], prompt.group_id, template)
        self._vote(self._frequency[prompt.event.relation_id], prompt.group_id, template)
        (self._trained_groups if prompt.split == "train" else self._future_groups).add(
            prompt.group_id
        )
        self._seen_ids.add(prompt.episode_id)
        if prompt.split == "train":
            if (
                self._max_train_outcome_at is None
                or case.outcome.observed_at > self._max_train_outcome_at
            ):
                self._max_train_outcome_at = case.outcome.observed_at
        else:
            self._pending_future = None
            self._last_future_outcome_at = case.outcome.observed_at

    def fit_train(self, cases: tuple[TransitionCase, ...]) -> None:
        if type(cases) is not tuple:
            raise ValueError("train cases must be a tuple")
        for case in cases:
            if case.prompt.split != "train":
                raise ValueError("train fitting received another split")
            self.learn(case)

    def _chosen(
        self, prompt: TransitionPrompt, table: dict[str, Template | None]
    ) -> Forecast:
        counts: dict[Template, int] = defaultdict(int)
        for template in table.values():
            if template is not None:
                counts[template] += 1
        if not counts:
            return Forecast(None, 0, 0, "untrained_or_conflicted")
        ranking = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        best, support = ranking[0]
        runner_up = ranking[1][1] if len(ranking) > 1 else 0
        if support < self.min_support_groups or support <= runner_up:
            return Forecast(
                None,
                support,
                runner_up,
                "insufficient_or_ambiguous_independent_support",
            )
        after = _project(prompt, best)
        if after is None:
            return Forecast(None, support, runner_up, "unbound_current_roles")
        return Forecast(after, support, runner_up, "learned_role_bound_transition")

    def predict(self, prompt: TransitionPrompt) -> Forecast:
        if not isinstance(prompt, TransitionPrompt):
            raise ValueError("invalid prompt")
        if prompt.split == "development" and (
            self._future_groups or prompt.group_id in self._trained_groups
        ):
            raise ValueError(
                "development must precede future updates and exclude train"
            )
        if (
            self._max_train_outcome_at is not None
            and prompt.cutoff_at < self._max_train_outcome_at
        ):
            raise ValueError("forecast cutoff predates a training outcome")
        forecast = self._chosen(prompt, self._votes.get(_signature(prompt), {}))
        if prompt.split == "future_stream":
            if (
                self._last_future_outcome_at is not None
                and prompt.cutoff_at < self._last_future_outcome_at
            ):
                raise ValueError("future prompt predates the last revealed outcome")
            self._pending_future = prompt
        return forecast

    def frequency_control(self, prompt: TransitionPrompt) -> Forecast:
        return self._chosen(prompt, self._frequency.get(prompt.event.relation_id, {}))

    @staticmethod
    def persistence_control(prompt: TransitionPrompt) -> Forecast:
        return Forecast(prompt.before, 0, 0, "persist_unchanged")


def _summary(rows: list[tuple[str, bool, bool]]) -> StageMetrics:
    """Macro average by independent family; events remain descriptive only."""
    by_group: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for group, supported, exact in rows:
        by_group[group].append((supported, exact))
    if not by_group:
        return {
            "episodes": 0,
            "independent_groups": 0,
            "supported_episodes": 0,
            "abstained_episodes": 0,
            "exact_episodes": 0,
            "groups_with_supported": 0,
            "group_coverage_numerator": 0.0,
            "group_exact_numerator": 0.0,
            "supported_group_exact_numerator": 0.0,
            "coverage": None,
            "exact_per_group": None,
            "exact_among_supported": None,
        }
    group_coverage_numerator = sum(
        sum(x for x, _ in values) / len(values) for values in by_group.values()
    )
    group_exact_numerator = sum(
        sum(y for _, y in values) / len(values) for values in by_group.values()
    )
    accepted = [
        (sum(y for x, y in values if x), sum(x for x, _ in values))
        for values in by_group.values()
    ]
    eligible = [(right / total) for right, total in accepted if total]
    supported = sum(supported for _, supported, _ in rows)
    return {
        "episodes": len(rows),
        "independent_groups": len(by_group),
        "supported_episodes": supported,
        "abstained_episodes": len(rows) - supported,
        "exact_episodes": sum(exact for _, _, exact in rows),
        "groups_with_supported": len(eligible),
        "group_coverage_numerator": group_coverage_numerator,
        "group_exact_numerator": group_exact_numerator,
        "supported_group_exact_numerator": sum(eligible),
        "coverage": group_coverage_numerator / len(by_group),
        "exact_per_group": group_exact_numerator / len(by_group),
        "exact_among_supported": sum(eligible) / len(eligible) if eligible else None,
    }


def evaluate_open_stream(
    model: ProcessDynamics,
    cases: tuple[TransitionCase, ...],
    *,
    split: Literal["development", "future_stream"],
) -> StreamMetrics:
    """Predict before reveal; development is frozen, future learns after scoring.

    The caller must supply outcomes in a separate record; this function is an
    engineering contract, not a substitute for independent annotation custody.
    """
    if not isinstance(model, ProcessDynamics) or type(cases) is not tuple:
        raise ValueError("invalid evaluator inputs")
    if split not in ("development", "future_stream"):
        raise ValueError("only open development or registered future stream")
    if split == "development" and model._future_groups:
        raise ValueError("development is frozen before the first future update")
    ids: set[str] = set()
    groups: set[str] = set()
    last_cutoff: datetime | None = None
    for case in cases:
        if not isinstance(case, TransitionCase) or case.prompt.split != split:
            raise ValueError("case split does not match evaluation mode")
        forbidden = (
            model._trained_groups | model._future_groups
            if split == "development"
            else model._trained_groups | model._development_groups
        )
        if case.prompt.group_id in forbidden:
            raise ValueError("evaluation group overlaps train or another split")
        if case.prompt.episode_id in ids | model._seen_ids | model._development_ids:
            raise ValueError("duplicate evaluation episode")
        if last_cutoff is not None and case.prompt.cutoff_at < last_cutoff:
            raise ValueError("events must arrive in chronological order")
        if (
            model._max_train_outcome_at is not None
            and case.prompt.cutoff_at < model._max_train_outcome_at
        ):
            raise ValueError("evaluation cutoff predates a training outcome")
        ids.add(case.prompt.episode_id)
        groups.add(case.prompt.group_id)
        last_cutoff = case.prompt.cutoff_at
    # Prior outcomes may be consulted, but later outcomes are not opened here.
    if (
        split == "future_stream"
        and cases
        and model._last_future_outcome_at is not None
        and cases[0].prompt.cutoff_at < model._last_future_outcome_at
    ):
        raise ValueError("future stream regresses behind a revealed outcome")
    results: dict[str, list[tuple[str, bool, bool]]] = {
        "learned": [],
        "frequency": [],
        "persistence": [],
        "current_fact": [],
    }
    for case in cases:
        prompt = case.prompt
        # The model and both controls receive only the prompt, never the outcome.
        forecasts = {
            "learned": model.predict(prompt),
            "frequency": model.frequency_control(prompt),
            "persistence": model.persistence_control(prompt),
        }
        for name, forecast in forecasts.items():
            results[name].append(
                (
                    prompt.group_id,
                    forecast.supported,
                    forecast.after == case.outcome.after
                    if forecast.supported
                    else False,
                )
            )
        if case.fact_gold is not None and case.fact_prediction is not None:
            results["current_fact"].append(
                (prompt.group_id, True, case.fact_gold == case.fact_prediction)
            )
        if split == "future_stream":
            model.learn(case)  # Outcome revealed and scored before any update.
    if split == "development":
        model._development_groups.update(groups)
        model._development_ids.update(ids)
    return {
        "split": split,
        "episodes": len(cases),
        "independent_groups": len(groups),
        "forecast": {
            key: _summary(results[key])
            for key in ("learned", "frequency", "persistence")
        },
        "current_fact": _summary(results["current_fact"]),
        "updates_after_reveal": len(cases) if split == "future_stream" else 0,
    }

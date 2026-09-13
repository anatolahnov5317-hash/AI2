"""A transparent active-learning microworld for protocol discovery.

This module is intentionally small and hand specified.  It compares two candidate
rules; it does not make a claim that the text-factor SDR learns either rule.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

RuleName = Literal["color_match", "shape_match"]


@dataclass(frozen=True)
class Entity:
    """A structured object or lock; identifiers are not predictive features."""

    identifier: str
    color: str
    shape: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.identifier,
            "color": self.color,
            "shape": self.shape,
        }


@dataclass(frozen=True)
class Interaction:
    """An unlabeled possible interaction between an object and a lock."""

    object: Entity
    lock: Entity

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"object": self.object.to_dict(), "lock": self.lock.to_dict()}


@dataclass(frozen=True)
class Observation:
    """The result exposed after a real interaction is performed."""

    interaction: Interaction
    outcome: bool

    def to_dict(self) -> dict[str, Any]:
        return {**self.interaction.to_dict(), "outcome": self.outcome}


def _color_match(interaction: Interaction) -> bool:
    return interaction.object.color == interaction.lock.color


def _shape_match(interaction: Interaction) -> bool:
    return interaction.object.shape == interaction.lock.shape


# These are the complete, hand-specified hypothesis class.  They are deliberately
# module-visible so that the experiment's inductive bias cannot be mistaken for a
# learned rule language.
HYPOTHESES: Mapping[RuleName, Callable[[Interaction], bool]] = MappingProxyType(
    {"color_match": _color_match, "shape_match": _shape_match}
)


class HiddenMatchWorld:
    """Authoritative outcome source; learners receive results only through actions.

    ``act`` is used for paid interventions. ``score`` is reserved for held-out
    evaluation after selection/training.  Neither method is supplied to learners.
    """

    def __init__(self, rule: RuleName) -> None:
        if rule not in HYPOTHESES:
            raise ValueError(f"unknown hidden rule: {rule!r}")
        self._rule = rule

    def _evaluate(self, interaction: Interaction) -> bool:
        # IDs are intentionally absent from the authoritative outcome calculation.
        if self._rule == "color_match":
            return interaction.object.color == interaction.lock.color
        return interaction.object.shape == interaction.lock.shape

    def act(self, interaction: Interaction) -> Observation:
        """Perform a real action and expose its outcome."""

        return Observation(interaction, self._evaluate(interaction))

    def score(self, interaction: Interaction) -> bool:
        """Return a held-out outcome for evaluation only."""

        return self._evaluate(interaction)


@dataclass(frozen=True)
class Prediction:
    probability: float | None
    abstained: bool
    state: Literal["resolved", "disagreement", "unknown"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "abstained": self.abstained,
            "state": self.state,
        }


@dataclass(frozen=True)
class QueryChoice:
    interaction: Interaction
    information_gain_bits: float


class TwoRuleLearner:
    """Version-space learner over exactly two hand-written match predicates."""

    def __init__(self) -> None:
        self._candidates: list[RuleName] = list(HYPOTHESES)

    @property
    def candidates(self) -> tuple[RuleName, ...]:
        return tuple(self._candidates)

    def observe(self, observation: Observation) -> None:
        """Retain only hypotheses consistent with an exposed action outcome."""

        self._candidates = [
            name
            for name in self._candidates
            if HYPOTHESES[name](observation.interaction) == observation.outcome
        ]

    def predict(self, interaction: Interaction) -> Prediction:
        if not self._candidates:
            return Prediction(None, True, "unknown")
        votes = [HYPOTHESES[name](interaction) for name in self._candidates]
        probability = sum(votes) / len(votes)
        if all(vote == votes[0] for vote in votes):
            return Prediction(float(probability), False, "resolved")
        return Prediction(float(probability), True, "disagreement")

    def information_gain(self, interaction: Interaction) -> float:
        """Expected uniform-version-space entropy reduction, in bits."""

        count = len(self._candidates)
        if count <= 1:
            return 0.0
        true_count = sum(HYPOTHESES[name](interaction) for name in self._candidates)
        false_count = count - true_count
        posterior = 0.0
        for branch_count in (false_count, true_count):
            if branch_count:
                posterior += (branch_count / count) * math.log2(branch_count)
        return math.log2(count) - posterior

    def select_query(self, allowed: list[Interaction]) -> QueryChoice | None:
        """Choose maximum information gain; semantic attributes break ties."""

        if not allowed or len(self._candidates) <= 1:
            return None

        def tie_key(interaction: Interaction) -> tuple[str, ...]:
            return (
                interaction.object.color,
                interaction.object.shape,
                interaction.lock.color,
                interaction.lock.shape,
                interaction.object.identifier,
                interaction.lock.identifier,
            )

        ordered = sorted(allowed, key=tie_key)
        gains = [(self.information_gain(item), item) for item in ordered]
        gain, interaction = max(gains, key=lambda pair: pair[0])
        if gain <= 0.0:
            return None
        return QueryChoice(interaction, float(gain))


class ExactEpisodicLearner:
    """Baseline that recalls only exactly identical entity episodes."""

    def __init__(self) -> None:
        self._outcomes: dict[tuple[str, ...], bool] = {}

    @staticmethod
    def _key(interaction: Interaction) -> tuple[str, ...]:
        return (
            interaction.object.identifier,
            interaction.object.color,
            interaction.object.shape,
            interaction.lock.identifier,
            interaction.lock.color,
            interaction.lock.shape,
        )

    def observe(self, observation: Observation) -> None:
        self._outcomes[self._key(observation.interaction)] = observation.outcome

    def predict(self, interaction: Interaction) -> Prediction:
        outcome = self._outcomes.get(self._key(interaction))
        if outcome is None:
            return Prediction(None, True, "unknown")
        return Prediction(float(outcome), False, "resolved")


@dataclass(frozen=True)
class _Episode:
    initial: tuple[Interaction, ...]
    candidates: tuple[Interaction, ...]
    held_out: tuple[Interaction, ...]


def _entity(identifier: str, color: str, shape: str) -> Entity:
    return Entity(identifier=identifier, color=color, shape=shape)


def _make_episode(index: int, rng: random.Random) -> _Episode:
    colors = ["amber", "blue", "coral"]
    shapes = ["circle", "square", "triangle"]
    rng.shuffle(colors)
    rng.shuffle(shapes)
    c0, c1, c2 = colors
    s0, s1, s2 = shapes
    prefix = f"e{index}"

    def item(number: int, oc: str, os: str, lc: str, ls: str) -> Interaction:
        return Interaction(
            _entity(f"{prefix}-object-{number}", oc, os),
            _entity(f"{prefix}-lock-{number}", lc, ls),
        )

    # Both hypotheses agree on each initial example: one positive and one negative.
    initial = (
        item(0, c0, s0, c0, s0),
        item(1, c0, s0, c1, s1),
    )
    # Candidates contain two color-only, two shape-only, and two uninformative
    # actions. Labels remain hidden until a strategy actually acts.
    candidates = (
        item(10, c0, s0, c0, s1),
        item(11, c0, s0, c1, s0),
        item(12, c1, s1, c1, s2),
        item(13, c1, s1, c2, s1),
        item(14, c2, s2, c2, s2),
        item(15, c2, s2, c0, s0),
    )
    # New identifiers and unseen OBJECT attribute recombinations. Lock attribute
    # tuples may recur from the candidate pool. The tests require opposite
    # predictions from the two hypotheses.
    held_out = (
        item(100, c2, s0, c2, s1),
        item(101, c2, s0, c1, s0),
    )
    return _Episode(initial, candidates, held_out)


def _validate_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _prediction_record(
    interaction: Interaction, prediction: Prediction, outcome: bool
) -> dict[str, Any]:
    predicted_label = None
    if not prediction.abstained and prediction.probability is not None:
        predicted_label = prediction.probability >= 0.5
    return {
        **interaction.to_dict(),
        **prediction.to_dict(),
        "predicted_outcome": predicted_label,
        "actual_outcome": outcome,
        "correct": predicted_label == outcome if predicted_label is not None else False,
    }


def _metrics(
    records: list[dict[str, Any]], queries: int, episodes: int
) -> dict[str, float | int]:
    covered = sum(not record["abstained"] for record in records)
    correct = sum(record["correct"] for record in records)
    total = len(records)
    return {
        "success": float(correct / total),
        "coverage": float(covered / total),
        "accuracy_on_covered": float(correct / covered) if covered else 0.0,
        "query_cost": float(queries / episodes),
        "correct": int(correct),
        "covered": int(covered),
        "held_out_count": int(total),
        "queries": int(queries),
    }


def run_microworld(
    seed: int = 42, episodes: int = 32, action_budget: int = 1
) -> dict[str, Any]:
    """Run the finite active-learning pilot and return a JSON-safe report.

    The hidden scoring rule is never passed to a learner. Held-out outcomes are
    requested only after all strategies have selected and observed their actions.
    """

    _validate_int("seed", seed)
    _validate_int("episodes", episodes)
    _validate_int("action_budget", action_budget)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if action_budget < 0:
        raise ValueError("action_budget must be non-negative")
    candidate_count = 6
    if action_budget > candidate_count:
        raise ValueError(f"action_budget cannot exceed {candidate_count}")

    rng = random.Random(seed)
    episode_reports: list[dict[str, Any]] = []
    all_predictions: dict[str, list[dict[str, Any]]] = {
        "active": [],
        "random": [],
        "episodic": [],
    }
    query_counts = {name: 0 for name in all_predictions}

    for episode_index in range(episodes):
        episode = _make_episode(episode_index, rng)
        # Alternation guarantees both rules are represented whenever episodes >= 2.
        rule: RuleName = (
            "color_match" if (episode_index + seed) % 2 == 0 else "shape_match"
        )
        world = HiddenMatchWorld(rule)
        initial_observations = [world.act(item) for item in episode.initial]

        active = TwoRuleLearner()
        random_rule = TwoRuleLearner()
        episodic = ExactEpisodicLearner()
        for observation in initial_observations:
            active.observe(observation)
            random_rule.observe(observation)
            episodic.observe(observation)

        # One random ordering is shared by both non-active baselines. It is sampled
        # without labels, before either learner observes candidate outcomes.
        random_order = list(range(len(episode.candidates)))
        rng.shuffle(random_order)

        strategy_traces: dict[str, dict[str, Any]] = {
            "active": {
                "hypotheses_before_queries": list(active.candidates),
                "queries": [],
            },
            "random": {
                "hypotheses_before_queries": list(random_rule.candidates),
                "queries": [],
            },
            "episodic": {"hypotheses_before_queries": None, "queries": []},
        }

        active_remaining = list(episode.candidates)
        for _ in range(action_budget):
            choice = active.select_query(active_remaining)
            if choice is None:
                break
            candidates_before = list(active.candidates)
            observation = world.act(choice.interaction)
            active.observe(observation)
            strategy_traces["active"]["queries"].append(
                {
                    **observation.to_dict(),
                    "information_gain_bits": choice.information_gain_bits,
                    "hypotheses_before": candidates_before,
                    "hypotheses_after": list(active.candidates),
                }
            )
            active_remaining.remove(choice.interaction)
            query_counts["active"] += 1

        for candidate_index in random_order[:action_budget]:
            interaction = episode.candidates[candidate_index]
            candidates_before = list(random_rule.candidates)
            random_observation = world.act(interaction)
            episodic_observation = world.act(interaction)
            random_rule.observe(random_observation)
            episodic.observe(episodic_observation)
            strategy_traces["random"]["queries"].append(
                {
                    **random_observation.to_dict(),
                    "hypotheses_before": candidates_before,
                    "hypotheses_after": list(random_rule.candidates),
                }
            )
            strategy_traces["episodic"]["queries"].append(
                episodic_observation.to_dict()
            )
            query_counts["random"] += 1
            query_counts["episodic"] += 1

        # This is the first point at which held-out outcomes are materialized.
        held_out_outcomes = [world.score(item) for item in episode.held_out]
        learners = {
            "active": active,
            "random": random_rule,
            "episodic": episodic,
        }
        for name, learner in learners.items():
            records = [
                _prediction_record(item, learner.predict(item), outcome)
                for item, outcome in zip(
                    episode.held_out, held_out_outcomes, strict=True
                )
            ]
            strategy_traces[name]["held_out_predictions"] = records
            if isinstance(learner, TwoRuleLearner):
                strategy_traces[name]["posterior_hypotheses"] = list(learner.candidates)
            else:
                strategy_traces[name]["posterior_hypotheses"] = None
            all_predictions[name].extend(records)

        episode_reports.append(
            {
                "episode": episode_index,
                "scoring_rule_revealed_after_actions": rule,
                "initial_observations": [
                    observation.to_dict() for observation in initial_observations
                ],
                "candidate_pool": [item.to_dict() for item in episode.candidates],
                "held_out": [
                    {**item.to_dict(), "outcome": outcome}
                    for item, outcome in zip(
                        episode.held_out, held_out_outcomes, strict=True
                    )
                ],
                "strategies": strategy_traces,
            }
        )

    return {
        "manifest": {
            "name": "AI2 v0.2 transparent active-learning microworld pilot",
            "scope": (
                "single-intervention protocol identification and held-out transfer; "
                "not a planner, learned DSL, causal discovery system, or AGI claim"
            ),
            "hypothesis_classes": [
                {
                    "name": "color_match",
                    "hand_specified": True,
                    "rule": "outcome = (object.color == lock.color)",
                },
                {
                    "name": "shape_match",
                    "hand_specified": True,
                    "rule": "outcome = (object.shape == lock.shape)",
                },
            ],
            "priors": {"color_match": 0.5, "shape_match": 0.5},
            "protocol": {
                "initial_observations": 2,
                "candidate_pool_size": candidate_count,
                "held_out_interactions_per_episode": 2,
                "held_out_novelty": (
                    "fresh entity IDs and new object attribute tuples; "
                    "lock attribute tuples may already occur in the candidate pool"
                ),
                "action_budget": action_budget,
                "selection": {
                    "active": "maximum expected information gain",
                    "random": "seeded uniform ordering without replacement",
                    "episodic": "same actions as random; exact interaction lookup",
                },
                "no_test_leakage": (
                    "held-out outcomes are scored only after action selection and "
                    "observation are complete"
                ),
                "contradiction_policy": (
                    "retain no hypotheses, report state=unknown, probability=null, "
                    "and abstain; never silently reset"
                ),
            },
            "seed": seed,
            "episodes": episodes,
        },
        "episodes": episode_reports,
        "metrics": {
            name: _metrics(all_predictions[name], query_counts[name], episodes)
            for name in ("active", "random", "episodic")
        },
    }


__all__ = [
    "Entity",
    "ExactEpisodicLearner",
    "HYPOTHESES",
    "HiddenMatchWorld",
    "Interaction",
    "Observation",
    "Prediction",
    "QueryChoice",
    "TwoRuleLearner",
    "run_microworld",
]

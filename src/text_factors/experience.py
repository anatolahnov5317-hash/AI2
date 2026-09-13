"""Experimental, bounded learning from caller-confirmed observations.

The caller supplies stable encodings, context labels, actions and real outcomes.
This module neither discovers those labels nor verifies their physical origin.
Forecasts never enter memory automatically. Historical inverse lookup is
explicitly many-to-one and makes no prediction about unseen predecessors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .memory import CombinatorialMemory


def _label(value: str, name: str) -> None:
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters")


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SparseCode:
    """An immutable SDR with an explicit, caller-maintained encoding version."""

    encoding_id: str
    width: int
    active_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        _label(self.encoding_id, "encoding_id")
        _positive(self.width, "width")
        if type(self.active_bits) is not tuple or any(
            type(bit) is not int or not 0 <= bit < self.width
            for bit in self.active_bits
        ):
            raise ValueError("active_bits must be a tuple of in-range integer indices")
        if tuple(sorted(set(self.active_bits))) != self.active_bits:
            raise ValueError("active_bits must be sorted and unique")

    def to_array(self) -> NDArray[np.bool_]:
        result = np.zeros(self.width, dtype=np.bool_)
        result[list(self.active_bits)] = True
        return result

    @classmethod
    def from_array(cls, encoding_id: str, bits: NDArray[np.bool_]) -> SparseCode:
        checked = np.asarray(bits)
        if checked.ndim != 1 or checked.dtype != np.dtype(np.bool_):
            raise ValueError("bits must be a one-dimensional boolean array")
        return cls(
            encoding_id,
            int(checked.size),
            tuple(int(bit) for bit in np.flatnonzero(checked)),
        )


@dataclass(frozen=True, slots=True, order=True)
class ContextKey:
    """Caller-provided source view, destination view and operation/action label."""

    source_context: str
    target_context: str
    action: str

    def __post_init__(self) -> None:
        for name in ("source_context", "target_context", "action"):
            _label(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ObservedTransition:
    """Caller assertion that this uniquely numbered interaction was observed.

    Constructing this object is a trust boundary, not a physical-world check.
    New event IDs must increase across the bank, including after archive eviction.
    """

    event_id: int
    key: ContextKey
    source: SparseCode
    outcome: SparseCode

    def __post_init__(self) -> None:
        if type(self.event_id) is not int or self.event_id < 0:
            raise ValueError("event_id must be a non-negative integer")
        if type(self.key) is not ContextKey:
            raise TypeError("key must be a ContextKey")
        if type(self.source) is not SparseCode or type(self.outcome) is not SparseCode:
            raise TypeError("source and outcome must be SparseCode observations")


@dataclass(frozen=True, slots=True)
class Forecast:
    """A model prediction; quality and votes are not calibrated probabilities."""

    key: ContextKey
    source: SparseCode
    predicted: SparseCode
    quality: int
    active_points: int
    training_events: int


@dataclass(frozen=True, slots=True)
class Predecessor:
    """One distinct archived source and the retained event IDs supporting it."""

    source: SparseCode
    event_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionScore:
    forecast: Forecast
    utility: float


@dataclass(frozen=True, slots=True)
class ActionChoice:
    selected: ActionScore
    candidates: tuple[ActionScore, ...]
    explored: bool


class ExperienceMemory:
    """Independent supervised factor memories for a bounded set of contexts.

    Capacity exhaustion rejects a new key before learning. Archive eviction does
    not erase learned clusters; their capacity is set by ModelConfig. The archive
    supports exact inverse lookup and retained-event conflict checks. An event
    older than the most recently accepted ID is never learned again, even when
    its contents have been evicted. Out-of-order event delivery is unsupported.

    The object is experimental and single-threaded. Whole-bank persistence and
    automatic encoding migration are intentionally not provided.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        encoding_id: str,
        max_operations: int = 16,
        max_events: int = 256,
        max_action_candidates: int = 16,
        policy_seed: int = 0,
    ) -> None:
        if config.input_bits != config.output_bits:
            raise ValueError("experience memory requires equal input and output widths")
        _label(encoding_id, "encoding_id")
        for name, value in (
            ("max_operations", max_operations),
            ("max_events", max_events),
            ("max_action_candidates", max_action_candidates),
        ):
            _positive(value, name)
        if type(policy_seed) is not int or policy_seed < 0:
            raise ValueError("policy_seed must be a non-negative integer")
        self.config = config
        self.encoding_id = encoding_id
        self.max_operations = max_operations
        self.max_events = max_events
        self.max_action_candidates = max_action_candidates
        self.policy_seed = policy_seed
        self._rng = np.random.default_rng(policy_seed)
        self._memories: dict[ContextKey, CombinatorialMemory] = {}
        self._events: dict[int, ObservedTransition] = {}
        self._latest_event_id = -1
        self._observation_count = 0

    def _check(self, key: ContextKey, code: SparseCode) -> None:
        if type(key) is not ContextKey or type(code) is not SparseCode:
            raise TypeError("expected a ContextKey and a SparseCode")
        if code.width != self.config.input_bits:
            raise ValueError("code width does not match the memory")
        if code.encoding_id != self.encoding_id:
            raise ValueError("code encoding_id does not match the memory")

    def learn(self, observation: ObservedTransition) -> bool:
        """Learn a new real event, or return False for an identical retained one.

        A Forecast is rejected at runtime. Explicitly constructing a new
        ObservedTransition still relies on the caller supplying a real outcome.
        """

        if type(observation) is not ObservedTransition:
            raise TypeError(
                "learn accepts only an ObservedTransition, never a Forecast"
            )
        self._check(observation.key, observation.source)
        self._check(observation.key, observation.outcome)
        retained = self._events.get(observation.event_id)
        if retained is not None:
            if retained != observation:
                raise ValueError("event_id already has different observed contents")
            return False
        if observation.event_id <= self._latest_event_id:
            raise ValueError("stale or evicted event_id; new event IDs must increase")
        memory = self._memories.get(observation.key)
        if memory is None:
            if len(self._memories) >= self.max_operations:
                raise ValueError("operation capacity reached")
            memory = CombinatorialMemory(self.config)
            self._memories[observation.key] = memory
        memory.observe(
            observation.source.to_array(), target=observation.outcome.to_array()
        )
        self._latest_event_id = observation.event_id
        self._observation_count += 1
        self._events[observation.event_id] = observation
        if len(self._events) > self.max_events:
            del self._events[next(iter(self._events))]
        return True

    def forecast(self, key: ContextKey, source: SparseCode) -> Forecast:
        """Read a learned transformation; an unseen key returns an empty forecast."""

        self._check(key, source)
        memory = self._memories.get(key)
        if memory is None:
            return Forecast(
                key, source, SparseCode(self.encoding_id, source.width, ()), 0, 0, 0
            )
        result = memory.predict(source.to_array())
        return Forecast(
            key,
            source,
            SparseCode.from_array(self.encoding_id, result.output),
            result.quality,
            result.active_points,
            memory.step,
        )

    def predecessors(
        self, key: ContextKey, outcome: SparseCode
    ) -> tuple[Predecessor, ...]:
        """Return every distinct retained predecessor of this exact observed outcome.

        Results are context/action filtered and bounded by max_events. Empty
        results mean no retained exact match, not proof that no cause exists.
        """

        self._check(key, outcome)
        grouped: dict[SparseCode, list[int]] = {}
        for event in self._events.values():
            if event.key == key and event.outcome == outcome:
                grouped.setdefault(event.source, []).append(event.event_id)
        return tuple(Predecessor(source, tuple(ids)) for source, ids in grouped.items())

    def choose_action(
        self,
        source: SparseCode,
        candidates: Sequence[ContextKey],
        utility: Callable[[Forecast], float],
        *,
        exploration_probability: float = 0.0,
    ) -> ActionChoice:
        """Score bounded actions from forecasts, with explicit epsilon exploration.

        The caller supplies a goal utility, not hidden world outcomes. Unknown
        forecasts remain visibly empty. Ties use ContextKey lexical order;
        exploration uses the recorded policy seed and changes only policy RNG.
        """

        if not 1 <= len(candidates) <= self.max_action_candidates:
            raise ValueError("candidate count must be within the configured limit")
        if type(exploration_probability) not in (float, int) or not (
            isfinite(exploration_probability) and 0 <= exploration_probability <= 1
        ):
            raise ValueError("exploration_probability must be finite and in [0, 1]")
        for key in candidates:
            self._check(key, source)
        keys = tuple(sorted(candidates))
        if len(set(keys)) != len(keys):
            raise ValueError("candidate keys must be unique")
        if len({(key.source_context, key.target_context) for key in keys}) != 1:
            raise ValueError("action candidates must share source and target contexts")
        scored: list[ActionScore] = []
        for key in keys:
            forecast = self.forecast(key, source)
            value = float(utility(forecast))
            if not isfinite(value):
                raise ValueError("utility must return a finite number")
            scored.append(ActionScore(forecast, value))
        explored = (
            exploration_probability > 0 and self._rng.random() < exploration_probability
        )
        selected = (
            scored[int(self._rng.integers(len(scored)))]
            if explored
            else max(scored, key=lambda candidate: candidate.utility)
        )
        return ActionChoice(selected, tuple(scored), bool(explored))

    def replay_consolidation(self) -> dict[str, int]:
        """One bounded consolidation pass; no re-observation or new event support."""

        removed = {"removed_clusters": 0, "removed_bits": 0}
        for memory in self._memories.values():
            result = memory.replay_consolidation()
            for key in removed:
                removed[key] += result[key]
        return removed

    def stats(self) -> dict[str, int]:
        return {
            "observations": self._observation_count,
            "retained_events": len(self._events),
            "operations": len(self._memories),
            "memory_steps": sum(memory.step for memory in self._memories.values()),
            "clusters": sum(
                len(clusters)
                for memory in self._memories.values()
                for clusters in memory.clusters
            ),
            "cluster_capacity": (
                self.max_operations
                * self.config.point_count
                * self.config.max_clusters_per_point
            ),
            # Array payload only: not RSS, Python/container overhead, or archive size.
            "model_numpy_payload_bytes_lower_bound": sum(
                memory.receptors.nbytes
                + memory.output_map.nbytes
                + sum(
                    cluster.bits.nbytes
                    + cluster.bit_hits.nbytes
                    + sum(history.nbytes for history in cluster.activation_history)
                    for clusters in memory.clusters
                    for cluster in clusters
                )
                for memory in self._memories.values()
            ),
        }


def run_experience_demo(*, seed: int = 42) -> dict[str, Any]:
    """Small executable integration probe, not evidence of general intelligence.

    The external toy environment exposes two mutually exclusive two-bit classes plus
    one changing distractor. Action and view labels and sparse encodings are
    provided. The learner sees only individual source/outcome observations, not
    the environment's mapping. Unseen distractor combinations are scored before
    one separate real interaction closes the action/observation loop.
    """

    config = ModelConfig(
        input_bits=32,
        output_bits=32,
        active_bits_per_symbol=2,
        positions=4,
        frame_size=2,
        context_count=4,
        receptive_bits=16,
        point_count=1024,
        create_threshold=2,
        activation_threshold=2,
        min_active_points=1,
        probation_after=2,
        stable_after=4,
        max_clusters_per_point=4,
        prediction_vote_threshold=1,
        consolidation_method="coactivation",
        seed=seed,
    )
    bank = ExperienceMemory(
        config,
        encoding_id="experience-demo-v1",
        max_operations=3,
        max_events=40,
        max_action_candidates=2,
        policy_seed=seed,
    )
    left = ContextKey("view-a", "consequence", "left")
    right = ContextKey("view-a", "consequence", "right")
    align = ContextKey("view-b", "consequence", "paired-view")
    keys = (left, right, align)

    def source(key: ContextKey, factor: int, distractor: int) -> SparseCode:
        # Different views use disjoint factor coordinates, not duplicate inputs.
        first_bit = 2 * factor + (16 if key == align else 0)
        return SparseCode(
            bank.encoding_id, 32, tuple(sorted((first_bit, first_bit + 1, distractor)))
        )

    def real_outcome(key: ContextKey, factor: int) -> SparseCode:
        # Environment-only truth: never supplied as a rule or callback to the bank.
        destination = 1 - factor if key == right else factor
        return SparseCode(
            bank.encoding_id, 32, (24 + 2 * destination, 25 + 2 * destination)
        )

    before = bank.forecast(left, source(left, 0, 10))
    event_id = 0
    fit_started = perf_counter()
    for key in keys:
        for distractor in range(4, 10):
            for factor in (0, 1):
                bank.learn(
                    ObservedTransition(
                        event_id,
                        key,
                        source(key, factor, distractor),
                        real_outcome(key, factor),
                    )
                )
                event_id += 1
    fit_seconds = perf_counter() - fit_started

    stats_before_scoring = bank.stats()
    records: list[dict[str, Any]] = []
    predictions: dict[tuple[ContextKey, int, int], SparseCode] = {}
    true_positive = false_positive = false_negative = 0
    constant_matches = 0
    constant_output = real_outcome(left, 0)
    score_started = perf_counter()
    for key in keys:
        for distractor in (10, 11):
            for factor in (0, 1):
                actual_source = source(key, factor, distractor)
                prediction = bank.forecast(key, actual_source).predicted
                truth = real_outcome(key, factor)
                predictions[key, factor, distractor] = prediction
                predicted_bits, true_bits = (
                    set(prediction.active_bits),
                    set(truth.active_bits),
                )
                true_positive += len(predicted_bits & true_bits)
                false_positive += len(predicted_bits - true_bits)
                false_negative += len(true_bits - predicted_bits)
                constant_matches += int(constant_output == truth)
                records.append(
                    {
                        "source_context": key.source_context,
                        "target_context": key.target_context,
                        "action": key.action,
                        "scoring_factor_id": factor,
                        "distractor_bit": distractor,
                        "source_bits": list(actual_source.active_bits),
                        "predicted_bits": list(prediction.active_bits),
                        "observed_truth_bits_for_scoring_only": list(truth.active_bits),
                        "exact_match": prediction == truth,
                    }
                )
    score_seconds = perf_counter() - score_started
    scoring_read_only = bank.stats() == stats_before_scoring
    cross_view_inputs_distinct = all(
        source(left, factor, distractor) != source(align, factor, distractor)
        for factor in (0, 1)
        for distractor in (10, 11)
    )
    same_factor_agreement = all(
        predictions[left, factor, distractor].active_bits
        and predictions[left, factor, distractor]
        == predictions[align, factor, distractor]
        for factor in (0, 1)
        for distractor in (10, 11)
    )
    distinct_factors = all(
        predictions[align, 0, distractor].active_bits
        and predictions[align, 1, distractor].active_bits
        and predictions[align, 0, distractor] != predictions[align, 1, distractor]
        for distractor in (10, 11)
    )
    inverse_count = len(bank.predecessors(left, real_outcome(left, 0)))
    # For class 1 the useful action sorts LAST: always choosing the first action
    # cannot pass this integration check.
    choice_source = source(left, 1, 12)
    interaction_started = perf_counter()
    choice = bank.choose_action(
        choice_source,
        (right, left),
        lambda forecast: float(len(set(forecast.predicted.active_bits) & {24, 25})),
    )
    chosen_key = choice.selected.forecast.key
    # A new real interaction is learned only after the held-out evaluation above.
    bank.learn(
        ObservedTransition(
            event_id, chosen_key, choice_source, real_outcome(chosen_key, 1)
        )
    )
    interaction_seconds = perf_counter() - interaction_started
    before_replay = bank.stats()
    replay_started = perf_counter()
    replay = bank.replay_consolidation()
    replay_seconds = perf_counter() - replay_started
    after_replay = bank.stats()
    return {
        "schema_version": 1,
        "seed": seed,
        "policy_seed": bank.policy_seed,
        "config": config.to_dict(),
        "scope": "structured toy integration; context labels and encodings supplied",
        "timing": {
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "new_interaction_seconds": interaction_seconds,
            "replay_seconds": replay_seconds,
        },
        "training_observations": stats_before_scoring["observations"],
        "held_out_count": len(records),
        "held_out_exact_match": sum(record["exact_match"] for record in records)
        / len(records),
        "held_out_bit_precision": true_positive
        / max(1, true_positive + false_positive),
        "held_out_bit_recall": true_positive / max(1, true_positive + false_negative),
        "all_zero_exact_match": 0.0,
        "constant_output_exact_match": constant_matches / len(records),
        "constant_output_distinguishes_factors": False,
        "prediction_before_training": list(before.predicted.active_bits),
        "cross_view_input_codes_distinct": cross_view_inputs_distinct,
        "same_factor_cross_view_agreement_nonempty": bool(same_factor_agreement),
        "different_factor_codes_distinct_nonempty": bool(distinct_factors),
        "retained_inverse_predecessors_for_one_outcome": inverse_count,
        "scoring_did_not_learn": scoring_read_only,
        "selected_action": chosen_key.action,
        "selected_forecast_bits": list(choice.selected.forecast.predicted.active_bits),
        "actual_action_outcome_bits": list(real_outcome(chosen_key, 1).active_bits),
        "new_real_interactions_after_scoring": 1,
        "replay": replay,
        "replay_added_observations": after_replay["observations"]
        - before_replay["observations"],
        "replay_added_memory_steps": after_replay["memory_steps"]
        - before_replay["memory_steps"],
        "final_stats": after_replay,
        "held_out": records,
    }

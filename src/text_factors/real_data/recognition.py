"""Prospective context readout and explicitly partial observation coverage.

Prediction sees only the source and frozen context memory. Recognition may use
observed target bits later, but its selected responses are never scored as a
prediction made before the target was revealed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .contexts import ContextRegistry, jaccard
from .contracts import LearningEpisode


def _bits(values: tuple[int, ...], width: int, name: str) -> tuple[int, ...]:
    if (
        type(values) is not tuple
        or any(type(bit) is not int or not 0 <= bit < width for bit in values)
        or values != tuple(sorted(set(values)))
    ):
        raise ValueError(f"{name} must be sorted unique bits within width")
    return values


@dataclass(frozen=True, slots=True)
class ProspectiveResponse:
    context_id: str
    predicted_bits: tuple[int, ...]
    independent_support: int
    source_coverage: float


@dataclass(frozen=True, slots=True)
class ProspectiveReadout:
    width: int
    source_bits: tuple[int, ...]
    responses: tuple[ProspectiveResponse, ...]
    predicted_bits: tuple[int, ...]
    min_independent_groups: int

    @property
    def primary_context_id(self) -> str | None:
        return self.responses[0].context_id if self.responses else None


@dataclass(frozen=True, slots=True)
class ObservedReadout:
    prospect: ProspectiveReadout
    observed_present: tuple[int, ...]
    observed_absent: tuple[int, ...]
    selected_context_ids: tuple[str, ...]
    explained_bits: tuple[int, ...]
    unexplained_bits: tuple[int, ...]
    contradicted_context_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrequentialStep:
    """A prediction frozen before reveal, then a readout and optional update."""

    readout: ObservedReadout
    learned_context_id: str | None
    created_context: bool | None


def prospective_read(
    registry: ContextRegistry,
    source_bits: tuple[int, ...],
    *,
    output_limit: int = 64,
    max_results: int = 4,
    min_independent_groups: int = 2,
) -> ProspectiveReadout:
    """Choose a source-only response before the next observation or update.

    Source coverage counts *whether* a source bit has been learned in a context;
    replaying a row cannot increase it. Every output bit needs independent
    pair-level support, including after restoration. Legacy frequency-only
    states without group provenance abstain. Votes are not a probability.
    """
    if not isinstance(registry, ContextRegistry):
        raise ValueError("registry must be ContextRegistry")
    source = _bits(source_bits, registry.width, "source")
    if not source:
        raise ValueError("source must not be empty")
    for name, value in (
        ("output_limit", output_limit),
        ("max_results", max_results),
        ("min_independent_groups", min_independent_groups),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    ranked: list[ProspectiveResponse] = []
    for context in registry.contexts:
        if context.independent_support < min_independent_groups:
            continue
        predicted = tuple(
            bit
            for bit in context.transform.predict(source, limit=output_limit)
            if (
                (votes := context.transform.support_for(source, bit))[0]
                >= min_independent_groups
                and votes[0] > votes[1]
            )
        )
        if not predicted:
            continue
        trained_sources = context.transform.to_dict()["counts"]
        covered = sum(str(bit) in trained_sources for bit in source)
        ranked.append(
            ProspectiveResponse(
                context.context_id,
                predicted,
                context.independent_support,
                covered / len(source),
            )
        )
    ranked.sort(
        key=lambda response: (
            -response.source_coverage,
            -response.independent_support,
            response.context_id,
        )
    )
    selected = tuple(ranked[:max_results])
    return ProspectiveReadout(
        registry.width,
        source,
        selected,
        selected[0].predicted_bits if selected else (),
        min_independent_groups,
    )


def recognize_observation(
    prospect: ProspectiveReadout,
    observed_present: tuple[int, ...],
    *,
    observed_absent: tuple[int, ...] = (),
    max_results: int = 4,
) -> ObservedReadout:
    """Explain only *known present* bits, retaining every unmatched input bit.

    Bits outside the present/absent masks remain unknown, not negative labels.
    Observations select explanations after reveal; these selections must not be
    reported as prospective predictions.
    """
    if not isinstance(prospect, ProspectiveReadout):
        raise ValueError("prospect must be ProspectiveReadout")
    positive = _bits(observed_present, prospect.width, "observed present")
    negative = _bits(observed_absent, prospect.width, "observed absent")
    if set(positive) & set(negative):
        raise ValueError("a target bit cannot be both present and absent")
    if type(max_results) is not int or max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    present = set(positive)
    absent = set(negative)
    remaining = set(positive)
    selected: list[str] = []
    contradicted = tuple(
        response.context_id
        for response in prospect.responses
        if absent.intersection(response.predicted_bits)
    )
    available = [
        response
        for response in prospect.responses
        if not absent.intersection(response.predicted_bits)
    ]
    while remaining and available and len(selected) < max_results:
        ranked = sorted(
            available,
            key=lambda response: (
                -len(remaining.intersection(response.predicted_bits)),
                -response.source_coverage,
                -response.independent_support,
                response.context_id,
            ),
        )
        choice = ranked[0]
        matched = remaining.intersection(choice.predicted_bits)
        if not matched:
            break
        remaining.difference_update(matched)
        selected.append(choice.context_id)
        available.remove(choice)
    return ObservedReadout(
        prospect,
        positive,
        negative,
        tuple(selected),
        tuple(sorted(present - remaining)),
        tuple(sorted(remaining)),
        contradicted,
    )


def prequential_step(
    registry: ContextRegistry,
    source_bits: tuple[int, ...],
    reveal: Callable[[], LearningEpisode],
    *,
    learn: bool = True,
    output_limit: int = 64,
    min_independent_groups: int = 2,
) -> PrequentialStep:
    """Predict -> reveal observation -> explain -> optionally update memory.

    The callback is invoked only after the source-only prediction is fixed. A
    caller must obtain the next observation independently of this prediction;
    it cannot use held-out evaluation labels for learning or tuning.
    """
    prospect = prospective_read(
        registry,
        source_bits,
        output_limit=output_limit,
        min_independent_groups=min_independent_groups,
    )
    episode = reveal()
    if not isinstance(episode, LearningEpisode):
        raise ValueError("reveal must return a LearningEpisode")
    if _bits(episode.source_code, registry.width, "episode source") != source_bits:
        raise ValueError("revealed episode source does not match predicted source")
    target = set(_bits(episode.target_code, registry.width, "episode target"))
    if episode.observed_target_bits is None:
        present = tuple(sorted(target))
        absent: tuple[int, ...] = ()
    else:
        mask = set(
            _bits(episode.observed_target_bits, registry.width, "observation mask")
        )
        present = tuple(sorted(target & mask))
        absent = tuple(sorted(mask - target))
    observed = recognize_observation(prospect, present, observed_absent=absent)
    if not learn:
        return PrequentialStep(observed, None, None)
    context_id, created, _ = registry.learn(episode)
    return PrequentialStep(observed, context_id, created)


def evaluate_frozen_transfer(
    registry: ContextRegistry,
    training: Sequence[LearningEpisode],
    evaluation: Sequence[LearningEpisode],
    *,
    output_limit: int = 64,
    min_independent_groups: int = 2,
) -> dict[str, Any]:
    """Evaluate new, group-disjoint episodes without updating any memory.

    The source-only response and two baselines (training-group majority and
    nearest training source) are fixed before each evaluation target is read.
    Unknown target bits never count as a confirmed absent bit. This is a
    development diagnostic, not a replacement for a separately sealed test.
    """
    train_groups: dict[str, LearningEpisode] = {}
    train_ids: set[str] = set()
    for episode in training:
        if episode.episode_id in train_ids:
            raise ValueError("duplicate training episode ID")
        train_ids.add(episode.episode_id)
        train_groups.setdefault(episode.group_id, episode)
    if not train_groups or not evaluation:
        raise ValueError("nonempty training and evaluation groups are required")
    eval_groups = [episode.group_id for episode in evaluation]
    if (
        len(set(eval_groups)) != len(eval_groups)
        or set(eval_groups) & train_groups.keys()
    ):
        raise ValueError("evaluation groups must be unique and disjoint from training")
    learned_groups = {
        group for context in registry.contexts for group in context.group_ids
    }
    learned_ids = {
        episode_id
        for context in registry.contexts
        for episode_id in context.episode_ids
    }
    if (
        not learned_groups <= train_groups.keys()
        or learned_groups & set(eval_groups)
        or not learned_ids <= train_ids
    ):
        raise ValueError("context memory contains a group outside the training split")
    frozen = registry.to_dict()
    width = registry.width
    selected_training = tuple(train_groups.values())
    frequencies: dict[int, int] = {}
    for episode in selected_training:
        target = _bits(tuple(sorted(set(episode.target_code))), width, "train target")
        if episode.observed_target_bits is not None:
            mask = _bits(
                tuple(sorted(set(episode.observed_target_bits))),
                width,
                "training observation mask",
            )
            target = tuple(sorted(set(target) & set(mask)))
        for bit in target:
            frequencies[bit] = frequencies.get(bit, 0) + 1
    majority = tuple(
        sorted(
            sorted(frequencies, key=lambda bit: (-frequencies[bit], bit))[:output_limit]
        )
    )
    rows: list[dict[str, Any]] = []
    for episode in evaluation:
        source = _bits(tuple(sorted(set(episode.source_code))), width, "eval source")
        prospect = prospective_read(
            registry,
            source,
            output_limit=output_limit,
            min_independent_groups=min_independent_groups,
        )
        nearest = max(
            selected_training,
            key=lambda previous: jaccard(
                source, tuple(sorted(set(previous.source_code)))
            ),
        )
        near_target = set(nearest.target_code)
        if nearest.observed_target_bits is not None:
            near_target.intersection_update(nearest.observed_target_bits)
        nearest_bits = tuple(sorted(near_target))[:output_limit]
        predictions = {
            "context": prospect.predicted_bits,
            "majority": majority,
            "nearest": nearest_bits,
            "zero": (),
        }
        # Nothing above this line may access the evaluation target or mask.
        target = set(
            _bits(tuple(sorted(set(episode.target_code))), width, "eval target")
        )
        if episode.observed_target_bits is None:
            present = tuple(sorted(target))
            absent: tuple[int, ...] = ()
        else:
            mask = set(
                _bits(
                    tuple(sorted(set(episode.observed_target_bits))),
                    width,
                    "evaluation observation mask",
                )
            )
            present = tuple(sorted(target & mask))
            absent = tuple(sorted(mask - target))
        explained = recognize_observation(prospect, present, observed_absent=absent)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "group_id": episode.group_id,
                "source_bits": source,
                "observed_present": present,
                "observed_absent": absent,
                "predictions": predictions,
                "prospective_context_id": prospect.primary_context_id,
                "selected_context_ids_after_reveal": explained.selected_context_ids,
                "unexplained_bits": explained.unexplained_bits,
            }
        )
    if registry.to_dict() != frozen:
        raise RuntimeError("read-only transfer evaluation modified context memory")
    comparisons: dict[str, dict[str, float | int | None]] = {}
    for method in ("context", "majority", "nearest", "zero"):
        hits = sum(
            len(set(row["predictions"][method]) & set(row["observed_present"]))
            for row in rows
        )
        positives = sum(len(row["observed_present"]) for row in rows)
        contradictions = sum(
            len(set(row["predictions"][method]) & set(row["observed_absent"]))
            for row in rows
        )
        compared = hits + contradictions
        known_negatives = sum(len(row["observed_absent"]) for row in rows)
        comparisons[method] = {
            "observed_positive_recall": hits / positives if positives else None,
            "known_prediction_precision": (
                hits / compared if known_negatives and compared else None
            ),
            "matched_positive_bits": hits,
            "observed_positive_bits": positives,
            "observed_negative_bits": known_negatives,
            "contradicted_known_absent_bits": contradictions,
        }
    return {"groups": len(rows), "rows": rows, "comparisons": comparisons}

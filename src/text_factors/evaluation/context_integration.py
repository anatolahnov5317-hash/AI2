"""Fixed v0.3.0a3 integration protocol: learned views, common memory and words.

Gold mappings, canonical teaching anchors and ideal views live only in this
evaluator. Runtime pipelines and word resolvers receive no evaluator targets.
Whole-scene SDR transfer and explicitly segmented part naming are scored apart.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import ModelConfig
from ..context_affinity import ContextAffinity, ContextAffinityConfig
from ..context_pipeline import LearnedContextPipeline
from ..dialogue import (
    GroundedCandidate,
    GroundedDialogue,
    GroundingEvidence,
    LabelEvent,
)
from ..grounding import GroundingPolicy
from ..memory import CombinatorialMemory
from ..recognition import (
    ContextView,
    RecognitionCandidate,
    RecognitionLimits,
    RecognitionResult,
    memory_encoding_id,
    recognize_views,
    relate_candidates,
)
from ..transforms import LearnedSDRTransform
from .context_transfer import sdr_trace_metrics
from .runner import source_manifest
from .transform_learning import (
    BIT_COUNT,
    HiddenMapping,
    PartEncoder,
    TransformLearningConfig,
    hidden_mappings,
    make_transform_dataset,
    transform_memory_config,
)

CONTEXTS = ("primary", "neighbor", "alternative")
METHODS = ("learned", "shuffled", "untrained")
POLICIES = ("exact", "factor")


@dataclass(frozen=True, slots=True)
class ContextIntegrationConfig:
    seeds: tuple[int, ...] = (11, 23, 47)
    points: int = 512
    epochs: int = 3
    seconds: float = 90.0

    def __post_init__(self) -> None:
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonempty nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for name in ("points", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("seconds must be positive and finite")


def common_memory_config(config: ContextIntegrationConfig, seed: int) -> ModelConfig:
    return ModelConfig(
        input_bits=BIT_COUNT,
        output_bits=BIT_COUNT,
        active_bits_per_symbol=6,
        positions=3,
        frame_size=3,
        context_count=1,
        receptive_bits=32,
        point_count=config.points,
        create_threshold=3,
        activation_threshold=3,
        min_active_points=min(4, config.points),
        probation_after=2,
        stable_after=3,
        prediction_vote_threshold=2,
        consolidation_method="frequency",
        max_clusters_per_point=32,
        seed=seed,
    )


def _active(bits: NDArray[np.bool_]) -> list[int]:
    return [int(bit) for bit in np.flatnonzero(bits)]


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _memory_hash(memory: CombinatorialMemory) -> str:
    digest = hashlib.sha256(str(memory.step).encode())
    for array in (memory.receptors, memory.output_map):
        digest.update(array.tobytes())
    for point, cluster in memory.iter_clusters():
        digest.update(
            str(
                (
                    point,
                    cluster.created_at,
                    cluster.last_seen,
                    int(cluster.status),
                    cluster.partial_hits,
                    cluster.exact_hits,
                    cluster.partial_errors,
                    cluster.complete_errors,
                )
            ).encode()
        )
        digest.update(cluster.bits.tobytes())
        digest.update(cluster.bit_hits.tobytes())
        for row in cluster.activation_history:
            digest.update(row.tobytes())
    return digest.hexdigest()


def _ground(candidate: RecognitionCandidate) -> GroundedCandidate:
    return GroundedCandidate(
        candidate.candidate_id,
        candidate.context_id,
        candidate.content_key,
        candidate.observation_id,
        candidate.output_bits,
        candidate.source_positions,
        tuple(
            GroundingEvidence(
                e.point_index, e.signature, e.matched_bits, e.observations, e.output_bit
            )
            for e in candidate.evidence
        ),
    )


def _summarize(
    result: RecognitionResult,
    selected: RecognitionResult,
    dialogues: Mapping[str, GroundedDialogue],
) -> dict[str, Any]:
    raw = result.candidates + result.suppressed
    candidates = []
    for candidate in raw:
        grounded = _ground(candidate)
        resolutions = {
            policy: asdict(
                dialogue.resolve_candidate(grounded, encoding_id=result.encoding_id)
            )
            for policy, dialogue in dialogues.items()
        }
        candidates.append(
            {
                "id": candidate.candidate_id,
                "context": candidate.context_id,
                "content_key": candidate.content_key,
                "output_bits": candidate.output_bits,
                "source_positions": candidate.source_positions,
                "resolutions": resolutions,
            }
        )
    return {
        "complete": result.complete and selected.complete,
        "raw_count": len(raw),
        "exact_selected_count": len(result.candidates),
        "selected_count": len(selected.candidates),
        "selected_ids": [c.candidate_id for c in selected.candidates],
        "candidates": candidates,
    }


def evaluate_integration_query(
    source: NDArray[np.bool_],
    pipelines: Mapping[str, LearnedContextPipeline],
    common: CombinatorialMemory,
    dialogues: Mapping[str, GroundedDialogue],
    affinities: Mapping[str, ContextAffinity],
    target_factory: Callable[[], Mapping[str, tuple[NDArray[np.bool_], str | None]]]
    | None,
    *,
    observation_id: str,
    source_positions: tuple[int, ...],
    semantic: bool,
    check_budget: Callable[[], None] | None = None,
    trace_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit every ordinary prediction/resolution before constructing any gold.

    ``target_factory`` is evaluator-only. It is never passed to a pipeline,
    transform, affinity selector or dialogue. The ideal condition runs last.
    """
    row = trace_sink if trace_sink is not None else {}
    row.update({"source_bits": _active(source), "methods": {}})

    def check() -> None:
        if check_budget is not None:
            check_budget()

    for method, pipeline in pipelines.items():
        check()
        result = pipeline.recognize(
            source, observation_id=observation_id, source_positions=source_positions
        )
        selected = affinities[method].select(result.recognition)
        condition = _summarize(
            result.recognition, selected, dialogues if semantic else {}
        )
        condition["predictions"] = {
            t.context_id: list(t.predicted_bits) for t in result.transforms
        }
        row["methods"][method] = condition
        check()
        if not result.complete or not selected.complete:
            raise InterruptedError(
                result.stop_reason or selected.stop_reason or "incomplete_query"
            )
    # No ordinary method can receive a target that has not been constructed yet.
    if target_factory is None:
        return row
    check()
    targets = target_factory()
    row["targets"] = {
        context: {"bits": _active(bits), "word": word}
        for context, (bits, word) in targets.items()
    }
    if semantic:
        namespace = next(iter(pipelines.values())).memory_namespace
        ideal = recognize_views(
            common,
            tuple(
                ContextView(
                    f"ideal-{index}",
                    context,
                    bits.copy(),
                    source_positions,
                    observation_id=observation_id,
                )
                for index, (context, (bits, _)) in enumerate(targets.items())
            ),
            total_views=len(targets),
            encoding_id=namespace,
            limits=RecognitionLimits(max_views=3, max_candidates=3, seconds=2.0),
        )
        row["methods"]["ideal"] = _summarize(ideal, ideal, dialogues)
        check()
        if not ideal.complete:
            raise InterruptedError(ideal.stop_reason or "incomplete_ideal_control")
    return row


def _part_targets(
    mappings: Mapping[str, HiddenMapping],
    encoder: PartEncoder,
    position: int,
    value: int,
) -> dict[str, tuple[NDArray[np.bool_], str | None]]:
    result: dict[str, tuple[NDArray[np.bool_], str | None]] = {}
    for context, mapping in mappings.items():
        destination = mapping.position_order.index(position)
        translated = mapping.value_map[value]
        result[context] = (
            encoder.codebook[destination, translated].copy(),
            f"p{destination}v{translated}",
        )
    return result


def _teach_words(
    common: CombinatorialMemory,
    canonical: PartEncoder,
    namespace: str,
    check: Callable[[], None],
) -> tuple[dict[str, GroundedDialogue], list[dict[str, Any]]]:
    dialogues = {
        policy: GroundedDialogue(
            namespace,
            output_width=BIT_COUNT,
            grounding_policy=GroundingPolicy(
                mode=policy,
                min_atoms=4,
                min_points=2,
                min_shared_atoms=3,
                threshold=0.6,
                margin=0.1,
            ),
        )
        for policy in POLICIES
    }
    teaching: list[dict[str, Any]] = []
    for position in range(3):
        for value in range(3):
            check()
            word = f"p{position}v{value}"
            bits = canonical.codebook[position, value].copy()
            result = recognize_views(
                common,
                (
                    ContextView(
                        "anchor", "canonical", bits, (position,), observation_id=word
                    ),
                ),
                total_views=1,
                encoding_id=namespace,
                limits=RecognitionLimits(max_views=1, max_candidates=1, seconds=2.0),
            )
            if not result.complete:
                raise InterruptedError("incomplete_word_anchor")
            for dialogue in dialogues.values():
                dialogue.remember_recognition(word, result, encoding_id=namespace)
                if result.candidates:
                    dialogue.confirm(
                        LabelEvent(
                            position * 3 + value,
                            word,
                            result.candidates[0].candidate_id,
                            word,
                        )
                    )
            teaching.append(
                {
                    "word": word,
                    "bits": _active(bits),
                    "recognized": bool(result.candidates),
                    "content_keys": [c.content_key for c in result.candidates],
                }
            )
    return dialogues, teaching


def _semantic_counts(
    rows: Sequence[dict[str, Any]], method: str, policy: str
) -> dict[str, Any]:
    total = correct = named = ambiguous = selected_total = selected_correct = 0
    raw_count = selected_count = false_suppression = 0
    for row in rows:
        condition = row["methods"][method]
        raw_count += condition["raw_count"]
        selected_count += condition["selected_count"]
        selected_ids = set(condition["selected_ids"])
        retained_words = {
            row["targets"][c["context"]]["word"]
            for c in condition["candidates"]
            if c["id"] in selected_ids
        }
        by_context = {c["context"]: c for c in condition["candidates"]}
        for context, target in row["targets"].items():
            total += 1
            candidate = by_context.get(context)
            if candidate is None:
                continue
            answer = candidate["resolutions"][policy]
            confident = bool(answer["words"]) and not answer["ambiguous"]
            right = confident and list(answer["words"]) == [target["word"]]
            named += confident
            correct += right
            ambiguous += bool(answer["ambiguous"])
            if candidate["id"] in selected_ids:
                selected_total += 1
                selected_correct += right
            elif target["word"] not in retained_words:
                false_suppression += 1
    return {
        "context_queries": total,
        "correct": correct,
        "confident_named": named,
        "confident_wrong": named - correct,
        "ambiguous": ambiguous,
        "accuracy": correct / total if total else None,
        "precision_when_named": correct / named if named else None,
        "selected_candidates": selected_total,
        "selected_correct": selected_correct,
        "raw_candidates": raw_count,
        "retained_candidates": selected_count,
        "suppressed_gold_meaning_not_retained": false_suppression,
    }


def _summaries(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase, splits in (("before", ("dev", "test")), ("after", ("test",))):
        for split in splits:
            whole = [
                row
                for run in runs
                if run["status"] == "complete"
                for row in run["whole_traces"]
                if row["phase"] == phase and row["split"] == split
            ]
            parts = [
                row
                for run in runs
                if run["status"] == "complete"
                for row in run["part_traces"]
                if row["phase"] == phase and row["split"] == split
            ]
            group: dict[str, Any] = {"whole_sdr": {}, "part_words": {}}
            for method in METHODS:
                traces = [
                    {
                        "predicted_bits": row["methods"][method]["predictions"][
                            context
                        ],
                        "target_bits": target["bits"],
                    }
                    for row in whole
                    for context, target in row["targets"].items()
                ]
                group["whole_sdr"][method] = (
                    sdr_trace_metrics(traces, BIT_COUNT) if traces else None
                )
            for method in (*METHODS, "ideal"):
                group["part_words"][method] = {
                    policy: _semantic_counts(parts, method, policy)
                    for policy in POLICIES
                }
            result[f"{phase}/{split}"] = group
    return result


def _stability(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    before = {
        (row["case_index"], row["position"]): row
        for row in rows
        if row["phase"] == "before" and row["split"] == "test"
    }
    after = {
        (row["case_index"], row["position"]): row
        for row in rows
        if row["phase"] == "after"
    }
    result: dict[str, Any] = {}
    for method in (*METHODS, "ideal"):
        stats: dict[str, Any] = {
            "paired_part_queries": len(before.keys() & after.keys()),
            "context_key_sets_changed": 0,
        }
        for policy in POLICIES:
            old_correct = retained_correct = 0
            for key in before.keys() & after.keys():
                old, new = before[key], after[key]
                for context in CONTEXTS:
                    old_candidates = [
                        c
                        for c in old["methods"][method]["candidates"]
                        if c["context"] == context
                    ]
                    new_candidates = [
                        c
                        for c in new["methods"][method]["candidates"]
                        if c["context"] == context
                    ]
                    if policy == "exact":
                        stats["context_key_sets_changed"] += {
                            c["content_key"] for c in old_candidates
                        } != {c["content_key"] for c in new_candidates}
                    expected = old["targets"][context]["word"]

                    def correct(candidates, policy=policy, expected=expected):
                        return any(
                            not c["resolutions"][policy]["ambiguous"]
                            and list(c["resolutions"][policy]["words"]) == [expected]
                            for c in candidates
                        )

                    previous = correct(old_candidates)
                    old_correct += previous
                    retained_correct += previous and correct(new_candidates)
            stats[policy] = {
                "previously_correct": old_correct,
                "still_correct": retained_correct,
                "retention": retained_correct / old_correct if old_correct else None,
            }
        result[method] = stats
    return result


def _two_instance_control(
    source: NDArray[np.bool_],
    pipelines: Mapping[str, LearnedContextPipeline],
    affinities: Mapping[str, ContextAffinity],
    check: Callable[[], None],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, pipeline in pipelines.items():
        candidates: list[RecognitionCandidate] = []
        suppressed: list[RecognitionCandidate] = []
        for instance, origin in enumerate((0, 10)):
            check()
            item = pipeline.recognize(
                source, observation_id="two-instances", source_positions=(origin,)
            )
            if not item.complete:
                raise InterruptedError("incomplete_two_instance_control")
            candidates.extend(
                replace(c, candidate_id=f"i{instance}-{c.candidate_id}")
                for c in item.recognition.candidates
            )
            suppressed.extend(
                replace(c, candidate_id=f"i{instance}-{c.candidate_id}")
                for c in item.recognition.suppressed
            )
        raw = tuple(candidates)
        combined = RecognitionResult(
            raw,
            tuple(
                relate_candidates(left, right)
                for i, left in enumerate(raw)
                for right in raw[i + 1 :]
            ),
            True,
            6,
            6,
            suppressed=tuple(suppressed),
            memory_step=pipeline.memory.step,
            encoding_id=pipeline.memory_namespace,
        )
        selected = affinities[method].select(combined)
        if not selected.complete:
            raise InterruptedError(
                selected.stop_reason or "incomplete_instance_selection"
            )
        old_origins = {c.source_positions for c in raw}
        new_origins = {c.source_positions for c in selected.candidates}
        result[method] = {
            "raw_candidates": len(raw),
            "retained_candidates": len(selected.candidates),
            "recognized_origins": sorted(old_origins),
            "retained_origins": sorted(new_origins),
            "lost_recognized_origins": len(old_origins - new_origins),
            "both_instances_recognized": len(new_origins) == 2,
        }
    return result


def run_context_integration(
    config: ContextIntegrationConfig | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config = config or ContextIntegrationConfig()
    started = perf_counter()
    runs: list[dict[str, Any]] = []

    def check() -> None:
        if perf_counter() - started >= config.seconds:
            raise TimeoutError("global_time_budget")

    def emit(seed: int, phase: str) -> None:
        if progress is not None:
            progress(
                {
                    "seed": seed,
                    "phase": phase,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
        check()

    for seed in config.seeds:
        run: dict[str, Any] = {
            "seed": seed,
            "status": "running",
            "stage": "setup",
            "whole_traces": [],
            "part_traces": [],
            "unknown_traces": [],
            "training_presentations": 0,
            "partial_query": {},
        }
        runs.append(run)
        try:
            emit(seed, "setup")
            data = make_transform_dataset(seed)
            source_encoder, canonical, extra = (
                PartEncoder(seed, 0),
                PartEncoder(seed, 1),
                PartEncoder(seed, 2),
            )
            map0, map1 = hidden_mappings(seed)
            mappings = {"primary": map0, "neighbor": map0, "alternative": map1}
            memory_seeds = dict(
                zip(CONTEXTS, (seed, seed + 1009, seed + 2017), strict=True)
            )
            recipe = TransformLearningConfig(
                seeds=(seed,), points=config.points, epochs=config.epochs
            )
            banks = {
                method: {
                    context: LearnedSDRTransform(
                        transform_memory_config(recipe, memory_seeds[context])
                    )
                    for context in CONTEXTS
                }
                for method in METHODS
            }
            common = CombinatorialMemory(common_memory_config(config, seed))
            namespace = memory_encoding_id(common)
            sources = [source_encoder.encode(case) for case in data["train"]]
            targets = {
                context: [
                    canonical.encode(mapping.apply(case)) for case in data["train"]
                ]
                for context, mapping in mappings.items()
            }
            shuffled: dict[str, list[int]] = {}
            for index, context in enumerate(CONTEXTS):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 109, index]))
                shift = int(rng.integers(1, len(sources)))
                shuffled[context] = [
                    (i + shift) % len(sources) for i in range(len(sources))
                ]
            run["training_payload"] = {
                "dataset": data,
                "mappings_evaluator_only": {k: asdict(v) for k, v in mappings.items()},
                "transform_configs": {
                    c: banks["learned"][c].memory.config.to_dict() for c in CONTEXTS
                },
                "common_config": common.config.to_dict(),
                "shuffled_train_indices": shuffled,
                "canonical_anchors": [
                    {"word": f"p{p}v{v}", "bits": _active(canonical.codebook[p, v])}
                    for p in range(3)
                    for v in range(3)
                ],
                "additional_codes": [
                    _active(extra.codebook[p, v]) for p in range(3) for v in range(3)
                ],
                "codebook_sha256": {
                    "source": hashlib.sha256(
                        source_encoder.codebook.tobytes()
                    ).hexdigest(),
                    "canonical": hashlib.sha256(
                        canonical.codebook.tobytes()
                    ).hexdigest(),
                    "additional": hashlib.sha256(extra.codebook.tobytes()).hexdigest(),
                },
            }
            run["stage"] = "training_transforms"
            emit(seed, run["stage"])
            for _ in range(config.epochs):
                for i, source in enumerate(sources):
                    for context in CONTEXTS:
                        for method, target_index in (
                            ("learned", i),
                            ("shuffled", shuffled[context][i]),
                        ):
                            check()
                            banks[method][context].observe(
                                source, targets[context][target_index]
                            )
                            run["training_presentations"] += 1
            run["stage"] = "training_common_and_words"
            emit(seed, run["stage"])
            for _ in range(6):
                for position in range(3):
                    for value in range(3):
                        check()
                        bits = canonical.codebook[position, value]
                        common.observe(bits, target=bits)
            dialogues, teaching = _teach_words(common, canonical, namespace, check)
            run["teaching"] = teaching
            pipelines = {
                method: LearnedContextPipeline(
                    common,
                    transforms,
                    input_encoding_id=f"source-{seed}",
                    memory_namespace=namespace,
                    limits=RecognitionLimits(
                        max_views=3, max_candidates=3, seconds=2.0
                    ),
                )
                for method, transforms in banks.items()
            }
            affinities = {
                method: ContextAffinity(
                    namespace,
                    ContextAffinityConfig(
                        min_observations=3,
                        evidence_threshold=0.6,
                        affinity_threshold=0.8,
                    ),
                )
                for method in METHODS
            }
            run["stage"] = "training_affinity"
            emit(seed, run["stage"])
            for case_index, case in enumerate(data["train"]):
                for position, value in enumerate(case):
                    for method, pipeline in pipelines.items():
                        check()
                        observation = f"train-{case_index}-part-{position}"
                        found = pipeline.recognize(
                            source_encoder.codebook[position, value],
                            observation_id=observation,
                            source_positions=(position,),
                        )
                        if not found.complete:
                            raise InterruptedError("incomplete_affinity_training_read")
                        affinities[method].observe(observation, found.recognition)
            run["affinity_state"] = {
                m: affinity.to_dict() for m, affinity in affinities.items()
            }

            def frozen(banks=banks, dialogues=dialogues, affinities=affinities):
                return {
                    "transforms": {
                        m: {c: _memory_hash(t.memory) for c, t in bank.items()}
                        for m, bank in banks.items()
                    },
                    "dialogues": {
                        p: _json_hash(d.to_dict()) for p, d in dialogues.items()
                    },
                    "affinities": {
                        m: _json_hash(a.to_dict()) for m, a in affinities.items()
                    },
                }

            original_frozen = frozen()
            common_before = _memory_hash(common)
            run["common_steps_before"] = common.step
            for phase in ("before", "after"):
                if phase == "after":
                    run["common_unchanged_during_initial_queries"] = (
                        common_before == _memory_hash(common)
                    )
                    emit(seed, "additional_common_training")
                    for _ in range(6):
                        for position in range(3):
                            for value in range(3):
                                check()
                                bits = extra.codebook[position, value]
                                common.observe(bits, target=bits)
                frozen_common = _memory_hash(common)
                run["stage"] = phase
                emit(seed, phase)
                for case_index, case in enumerate(data["held_out"]):
                    split = "dev" if case_index < 4 else "test"
                    if phase == "after" and split == "dev":
                        continue
                    check()
                    info = {
                        "phase": phase,
                        "split": split,
                        "case_index": case_index,
                        "case": list(case),
                    }
                    sink: dict[str, Any] = dict(info)
                    run["partial_query"] = sink
                    evaluate_integration_query(
                        source_encoder.encode(case),
                        pipelines,
                        common,
                        dialogues,
                        affinities,
                        lambda case=case, canonical=canonical, mappings=mappings: {
                            c: (canonical.encode(m.apply(case)), None)
                            for c, m in mappings.items()
                        },
                        observation_id=f"{phase}-whole-{case_index}",
                        source_positions=(0, 1, 2),
                        semantic=False,
                        check_budget=check,
                        trace_sink=sink,
                    )
                    run["whole_traces"].append(sink)
                    for position, value in enumerate(case):
                        sink = {**info, "position": position, "value": value}
                        run["partial_query"] = sink
                        evaluate_integration_query(
                            source_encoder.codebook[position, value],
                            pipelines,
                            common,
                            dialogues,
                            affinities,
                            partial(
                                _part_targets, mappings, canonical, position, value
                            ),
                            observation_id=f"{phase}-part-{case_index}-{position}",
                            source_positions=(position,),
                            semantic=True,
                            check_budget=check,
                            trace_sink=sink,
                        )
                        run["part_traces"].append(sink)
                if phase == "before":
                    for position in range(3):
                        sink = {"phase": phase, "position": position, "value": position}
                        run["partial_query"] = sink
                        evaluate_integration_query(
                            extra.codebook[position, position],
                            pipelines,
                            common,
                            dialogues,
                            affinities,
                            None,
                            observation_id=f"unknown-{position}",
                            source_positions=(position,),
                            semantic=True,
                            check_budget=check,
                            trace_sink=sink,
                        )
                        run["unknown_traces"].append(sink)
                    run["two_instances"] = _two_instance_control(
                        source_encoder.codebook[0, 0], pipelines, affinities, check
                    )
                run[f"common_unchanged_during_{phase}_queries"] = (
                    frozen_common == _memory_hash(common)
                )
                run[f"frozen_components_unchanged_{phase}"] = (
                    original_frozen == frozen()
                )
                if (
                    not run[f"common_unchanged_during_{phase}_queries"]
                    or not run[f"frozen_components_unchanged_{phase}"]
                ):
                    raise InterruptedError("unexpected_read_mutation")
            run["common_steps_after"] = common.step
            run["common_changed_after_additional_training"] = (
                common_before != _memory_hash(common)
            )
            run["word_stability"] = _stability(run["part_traces"])
            run["label_events"] = {p: d.stats()["events"] for p, d in dialogues.items()}
            run["transform_observation_calls"] = {
                m: {c: t.memory.step for c, t in bank.items()}
                for m, bank in banks.items()
            }
            run["partial_query"] = {}
            run["status"] = "complete"
            emit(seed, "complete")
        except (TimeoutError, InterruptedError) as error:
            run["status"] = "incomplete"
            run["stop_reason"] = str(error)
            break
    completed = sum(run["status"] == "complete" for run in runs)
    return {
        "protocol": "ai2.context_integration.v1",
        "status": "complete" if completed == len(config.seeds) else "incomplete",
        "config": asdict(config),
        "completed_seeds": [r["seed"] for r in runs if r["status"] == "complete"],
        "elapsed_seconds": perf_counter() - started,
        "source": source_manifest(),
        "runs": runs,
        "aggregate": _summaries(runs) if completed == len(config.seeds) else None,
        "scope": {
            "given": [
                "codebooks and 3 slots",
                "mapping family and training-pair grouping",
                "part boundaries",
                "nine canonical labels",
                "fixed thresholds",
            ],
            "learned": [
                "paired SDR mappings",
                "common local clusters",
                "explicit word bindings",
                "train-only context co-responses",
            ],
            "not_tested": [
                "context discovery",
                "segmentation",
                "free grammar",
                "unseen primitive meanings",
            ],
            "repetition": (
                "parts repeat across scenes; neither epochs nor repeated parts "
                "are independent new meanings"
            ),
            "ideal": (
                "evaluator-only gold-transform control after ordinary predictions"
            ),
            "dev": "four fixed held-out cases; no fitting or selection on dev",
            "budget": (
                "cooperative checks plus per-query two-second caps; "
                "use an external process timeout"
            ),
        },
    }

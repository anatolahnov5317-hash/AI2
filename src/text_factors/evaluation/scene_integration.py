"""Frozen whole-scene recognition experiment with evaluator-only attribution.

The runtime receives the complete Boolean vector and empty source positions.
Portrait anchors are explicitly taught before queries, but their opaque IDs do
not describe parts. Mapping answers, slot masks and names are unavailable until
all ordinary predictions, portrait proposals and word resolutions are committed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..context_affinity import (
    AdaptiveContextAffinity,
    AdaptiveContextAffinityConfig,
)
from ..context_pipeline import LearnedContextPipeline
from ..dialogue import GroundedDialogue, LabelEvent
from ..grounding import GroundingPolicy
from ..memory import CombinatorialMemory
from ..recognition import (
    ContextView,
    RecognitionLimits,
    RecognitionResult,
    memory_encoding_id,
    recognize_views,
)
from ..scene_recognition import (
    FactorSceneReader,
    SceneRecognitionConfig,
    SceneRecognitionResult,
)
from ..transforms import LearnedSDRTransform
from .context_integration import (
    CONTEXTS,
    METHODS,
    ContextIntegrationConfig,
    _active,
    _ground,
    _json_hash,
    _memory_hash,
    common_memory_config,
)
from .context_transfer import sdr_trace_metrics
from .runner import source_manifest
from .transform_diagnostics import diagnose_transform_bits
from .transform_learning import (
    BIT_COUNT,
    BLOCK_BITS,
    HiddenMapping,
    PartEncoder,
    TransformLearningConfig,
    hidden_mappings,
    make_transform_dataset,
    transform_memory_config,
)


@dataclass(frozen=True, slots=True)
class SceneIntegrationConfig:
    seeds: tuple[int, ...] = (59, 71, 89)
    points: int = 512
    epochs: int = 3
    seconds: float = 120.0

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


def _bits(indices: Sequence[int]) -> NDArray[np.bool_]:
    bits = np.zeros(BIT_COUNT, dtype=np.bool_)
    bits[list(indices)] = True
    return bits


def _limits(views: int = 3) -> RecognitionLimits:
    return RecognitionLimits(
        max_views=views, max_candidates=64, max_evidence=32768, seconds=2.0
    )


def _views(
    predictions: Mapping[str, Sequence[int]], observation_id: str
) -> tuple[ContextView, ...]:
    return tuple(
        ContextView(
            f"view-{index}", context, _bits(indices), (), observation_id=observation_id
        )
        for index, (context, indices) in enumerate(predictions.items())
    )


def _candidate_rows(
    result: RecognitionResult, dialogue: GroundedDialogue
) -> list[dict[str, Any]]:
    rows = []
    for candidate in result.candidates + result.suppressed:
        resolution = dialogue.resolve_candidate(
            _ground(candidate), encoding_id=result.encoding_id
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "context": candidate.context_id,
                "content_key": candidate.content_key,
                "source_positions": candidate.source_positions,
                "output_bits": candidate.output_bits,
                "active_points": candidate.active_points,
                "evidence_count": len(candidate.evidence),
                "evidence_sha256": _json_hash(
                    [e.to_dict() for e in candidate.evidence]
                ),
                "words": resolution.words,
                "word_ambiguous": resolution.ambiguous,
                "word_method": resolution.method,
                "word_score": resolution.score,
                "word_reason": resolution.reason,
            }
        )
    return rows


def _scene_summary(
    result: SceneRecognitionResult,
    dialogue: GroundedDialogue,
    selected: RecognitionResult | None = None,
) -> dict[str, Any]:
    candidates = _candidate_rows(result.recognition, dialogue)
    by_id = {p.candidate_id: p for p in result.proposals}
    for candidate in candidates:
        proposal = by_id[candidate["candidate_id"]]
        candidate.update(asdict(proposal))
    selected = selected or result.recognition
    return {
        "complete": result.complete and selected.complete,
        "candidates": candidates,
        "selected_ids": [c.candidate_id for c in selected.candidates],
        "raw_count": len(candidates),
        "selected_count": len(selected.candidates),
        "view_traces": [asdict(t) for t in result.view_traces],
    }


def _legacy_summary(
    result: RecognitionResult, dialogue: GroundedDialogue
) -> dict[str, Any]:
    return {
        "complete": result.complete,
        "candidates": _candidate_rows(result, dialogue),
        "responses": [asdict(response) for response in result.responses],
    }


def _score_condition(
    condition: dict[str, Any],
    targets: Mapping[str, dict[str, Any]],
    *,
    matched_min_points: int | None = None,
) -> None:
    """Evaluate frozen proposals; never send answers back into recognition."""
    scores: dict[str, Any] = {}
    for context, target in targets.items():
        candidates = [
            c for c in condition["scene"]["candidates"] if c["context"] == context
        ]
        expected = set(target["portraits"])
        proposed = {c["portrait_id"] for c in candidates}
        expected_words = set(target["words"])
        words = {
            word for c in candidates if not c["word_ambiguous"] for word in c["words"]
        }
        legacy = [
            c for c in condition["legacy"]["candidates"] if c["context"] == context
        ]
        legacy_words = {
            word for c in legacy if not c["word_ambiguous"] for word in c["words"]
        }
        predicted = set(condition["predictions"][context])
        gold = set(target["bits"])
        score: dict[str, Any] = {
            "correct_portraits": sorted(proposed & expected),
            "missing_portraits": sorted(expected - proposed),
            "spurious_portraits": sorted(proposed - expected),
            "exact_portrait_set": proposed == expected,
            "correct_words": sorted(words & expected_words),
            "missing_words": sorted(expected_words - words),
            "spurious_words": sorted(words - expected_words),
            "exact_word_set": words == expected_words,
            "legacy_correct_words": sorted(legacy_words & expected_words),
            "legacy_missing_words": sorted(expected_words - legacy_words),
            "legacy_spurious_words": sorted(legacy_words - expected_words),
            "legacy_exact_word_set": legacy_words == expected_words,
            "ambiguous_proposals": sum(bool(c["ambiguous"]) for c in candidates),
            "ambiguous_word_resolutions": sum(
                bool(c["word_ambiguous"]) for c in candidates
            ),
            "tp": len(predicted & gold),
            "fp": len(predicted - gold),
            "fn": len(gold - predicted),
        }
        if "focal" in target:
            focal = target["focal"]
            focal_gold = set(focal["bits"])
            block = set(
                range(focal["slot"] * BLOCK_BITS, (focal["slot"] + 1) * BLOCK_BITS)
            )
            focal_predicted = predicted & block
            score["focal"] = {
                "tp": len(focal_predicted & focal_gold),
                "fp_in_focal_slot": len(focal_predicted - focal_gold),
                "fn": len(focal_gold - focal_predicted),
                "gold_bits": len(focal_gold),
                "portrait_found": focal["portrait"] in proposed,
                "word_found": focal["word"] in words,
                "legacy_word_found": focal["word"] in legacy_words,
                "spurious_bits_outside_target": len(predicted - gold),
            }
        scores[context] = score
    condition["scores"] = scores
    if matched_min_points is not None:
        filtered = {
            **condition,
            "scene": {
                **condition["scene"],
                "candidates": [
                    c
                    for c in condition["scene"]["candidates"]
                    if c["active_points"] >= matched_min_points
                ],
            },
        }
        _score_condition(filtered, targets)
        condition["matched_points"] = {
            "min_active_points": matched_min_points,
            "scores": filtered["scores"],
            "candidate_count": len(filtered["scene"]["candidates"]),
        }


def evaluate_scene_query(
    source: NDArray[np.bool_],
    pipelines: Mapping[str, LearnedContextPipeline],
    reader: FactorSceneReader,
    dialogues: Mapping[str, GroundedDialogue],
    target_factory: Callable[[], Mapping[str, dict[str, Any]]],
    *,
    observation_id: str,
    affinities: Mapping[str, AdaptiveContextAffinity] | None = None,
    check_budget: Callable[[], None] | None = None,
    trace_sink: dict[str, Any] | None = None,
    diagnose: bool = True,
) -> dict[str, Any]:
    """Run all ordinary methods before constructing any query gold or masks.

    Source origins are intentionally empty for whole scenes, including the
    isolated diagnostics. No supplied scope pretends to be discovered support.
    ``diagnose`` merely enables evaluator-side gate explanations after scoring.
    """
    row = trace_sink if trace_sink is not None else {}
    row.update(
        {"query_id": observation_id, "source_bits": _active(source), "methods": {}}
    )

    def check() -> None:
        if check_budget is not None:
            check_budget()

    for method, pipeline in pipelines.items():
        check()
        pipeline_result = pipeline.recognize(
            source, observation_id=observation_id, source_positions=()
        )
        predictions = {
            trace.context_id: list(trace.predicted_bits)
            for trace in pipeline_result.transforms
        }
        condition: dict[str, Any] = {"predictions": predictions}
        row["methods"][method] = condition
        if not pipeline_result.complete:
            raise InterruptedError(pipeline_result.stop_reason or "incomplete_pipeline")
        check()
        scene = reader.recognize_views(
            _views(predictions, observation_id),
            total_views=len(predictions),
            limits=_limits(len(predictions)),
        )
        selected = affinities[method].select(scene.recognition) if affinities else None
        condition["scene"] = _scene_summary(scene, dialogues["scene"], selected)
        condition["legacy"] = _legacy_summary(
            pipeline_result.recognition, dialogues["legacy"]
        )
        check()
        if not condition["scene"]["complete"]:
            raise InterruptedError(scene.stop_reason or "incomplete_scene_read")
    check()
    targets = target_factory()
    row["targets"] = targets
    matched_points = reader.memory.config.min_active_points if "split" in row else None
    for condition in row["methods"].values():
        _score_condition(condition, targets, matched_min_points=matched_points)
    ideal_predictions = {context: target["bits"] for context, target in targets.items()}
    ideal_views = _views(ideal_predictions, observation_id)
    check()
    ideal_scene = reader.recognize_views(
        ideal_views, total_views=len(ideal_views), limits=_limits(len(ideal_views))
    )
    ideal_legacy = recognize_views(
        reader.memory,
        ideal_views,
        total_views=len(ideal_views),
        limits=_limits(len(ideal_views)),
        encoding_id=reader.encoding_id,
    )
    ideal = {
        "predictions": ideal_predictions,
        "scene": _scene_summary(ideal_scene, dialogues["scene"]),
        "legacy": _legacy_summary(ideal_legacy, dialogues["legacy"]),
    }
    row["methods"]["ideal"] = ideal
    if not ideal_scene.complete or not ideal_legacy.complete:
        raise InterruptedError("incomplete_ideal_control")
    _score_condition(ideal, targets, matched_min_points=matched_points)
    if diagnose and "learned" in pipelines:
        bank = dict(pipelines["learned"].transforms)
        for context, target in targets.items():
            check()
            scope = None
            if "focal" in target:
                slot = target["focal"]["slot"]
                scope = np.arange(slot * BLOCK_BITS, (slot + 1) * BLOCK_BITS)
            row["methods"]["learned"].setdefault("fn_diagnostics", {})[context] = (
                diagnose_transform_bits(
                    bank[context],
                    source,
                    _bits(target["bits"]),
                    predicted=_bits(row["methods"]["learned"]["predictions"][context]),
                    evaluated_bits=scope,
                )
            )
    row["complete"] = True
    check()
    return row


def _portrait_ids(canonical: PartEncoder) -> dict[tuple[int, int], str]:
    return {
        (p, v): "portrait-"
        + hashlib.sha256(canonical.codebook[p, v].tobytes()).hexdigest()[:16]
        for p in range(3)
        for v in range(3)
    }


def _teach(
    common: CombinatorialMemory,
    reader: FactorSceneReader,
    canonical: PartEncoder,
    portraits: Mapping[tuple[int, int], str],
    check: Callable[[], None],
) -> tuple[dict[str, GroundedDialogue], list[dict[str, Any]]]:
    dialogues = {
        name: GroundedDialogue(
            encoding_id=reader.encoding_id,
            output_width=common.config.output_bits,
            grounding_policy=GroundingPolicy(
                mode="factor",
                min_atoms=4,
                min_points=2,
                min_shared_atoms=3,
                threshold=0.6,
                margin=0.1,
            ),
        )
        for name in ("scene", "legacy")
    }
    teaching = []
    for (position, value), portrait in portraits.items():
        check()
        bits = canonical.codebook[position, value].copy()
        registered = reader.observe_portrait(portrait, bits)
        teaching.append(
            {
                "portrait_id": portrait,
                "word": f"p{position}v{value}",
                "bits": _active(bits),
                "registered": registered,
            }
        )
    for index, anchor in enumerate(teaching):
        check()
        word = anchor["word"]
        views = _views({"canonical": anchor["bits"]}, f"anchor-{index}")
        scene = reader.recognize_views(views, total_views=1, limits=_limits(1))
        legacy = recognize_views(
            common,
            views,
            total_views=1,
            limits=_limits(1),
            encoding_id=reader.encoding_id,
        )
        if not scene.complete or not legacy.complete:
            raise InterruptedError("incomplete_teaching_read")
        own = next(
            (
                p.candidate_id
                for p in scene.proposals
                if p.portrait_id == anchor["portrait_id"]
            ),
            None,
        )
        for name, result, chosen in (
            ("scene", scene.recognition, own),
            (
                "legacy",
                legacy,
                legacy.candidates[0].candidate_id if legacy.candidates else None,
            ),
        ):
            dialogues[name].remember_recognition(
                word, result, encoding_id=reader.encoding_id
            )
            if chosen is not None:
                dialogues[name].confirm(LabelEvent(index, word, chosen, word))
            anchor[f"{name}_named"] = chosen is not None
    return dialogues, teaching


def _targets(
    case: Sequence[int],
    canonical: PartEncoder,
    mappings: Mapping[str, HiddenMapping],
    portraits: Mapping[tuple[int, int], str],
    *,
    focal: tuple[int, int] | None = None,
    isolated: bool = False,
) -> dict[str, Any]:
    result = {}
    for context, mapping in mappings.items():
        transformed = mapping.apply(tuple(case))
        positions = tuple(range(3))
        if isolated:
            if focal is None:
                raise ValueError("isolated targets need a focal part")
            positions = (mapping.position_order.index(focal[0]),)
        bits = np.logical_or.reduce(
            [canonical.codebook[p, transformed[p]] for p in positions]
        )
        target: dict[str, Any] = {
            "bits": _active(bits),
            "portraits": [portraits[(p, transformed[p])] for p in positions],
            "words": [f"p{p}v{transformed[p]}" for p in positions],
        }
        if focal is not None:
            slot = mapping.position_order.index(focal[0])
            value = mapping.value_map[focal[1]]
            target["focal"] = {
                "slot": slot,
                "bits": _active(canonical.codebook[slot, value]),
                "portrait": portraits[(slot, value)],
                "word": f"p{slot}v{value}",
            }
        result[context] = target
    return result


def _control_rows(
    reader: FactorSceneReader,
    dialogues: Mapping[str, GroundedDialogue],
    canonical: PartEncoder,
    extra: PartEncoder,
    portraits: Mapping[tuple[int, int], str],
    check: Callable[[], None],
    sink: list[dict[str, Any]],
) -> None:
    a, b, second_a = (
        canonical.codebook[0, 0],
        canonical.codebook[1, 1],
        canonical.codebook[1, 0],
    )
    weak_a = _bits(_active(a)[:3])
    unknown = extra.codebook[1, 2]
    controls = (
        ("A", a, ((0, 0),)),
        ("B", b, ((1, 1),)),
        ("A_plus_B", a | b, ((0, 0), (1, 1))),
        ("weak_A_plus_B", weak_a | b, ((0, 0), (1, 1))),
        ("same_value_distinct_slot_codes", a | second_a, ((0, 0), (1, 0))),
        ("A_plus_unlabelled", a | unknown, ((0, 0),)),
        ("unlabelled_alone", unknown, ()),
    )
    for name, source, known in controls:
        check()
        row: dict[str, Any] = {
            "control": name,
            "source_bits": _active(source),
            "complete": False,
        }
        sink.append(row)
        views = _views({"canonical": row["source_bits"]}, name)
        scene = reader.recognize_views(views, total_views=1, limits=_limits(1))
        legacy = recognize_views(
            reader.memory,
            views,
            total_views=1,
            limits=_limits(1),
            encoding_id=reader.encoding_id,
        )
        condition = {
            "predictions": {"canonical": row["source_bits"]},
            "scene": _scene_summary(scene, dialogues["scene"]),
            "legacy": _legacy_summary(legacy, dialogues["legacy"]),
        }
        row["condition"] = condition
        if not scene.complete or not legacy.complete:
            raise InterruptedError("incomplete_canonical_control")
        # Known membership is scored only after proposals and word resolutions.
        targets = {
            "canonical": {
                "bits": row["source_bits"],
                "portraits": [portraits[part] for part in known],
                "words": [f"p{p}v{v}" for p, v in known],
            }
        }
        row["targets"] = targets
        _score_condition(condition, targets)
        row["complete"] = True
        check()


def _aggregate_rows(rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    total = 0
    correct = missing = spurious = exact = 0
    word_correct = word_missing = word_spurious = word_exact = 0
    legacy_correct = legacy_missing = legacy_spurious = legacy_exact = 0
    ambiguous = raw = retained = unexplained = 0
    unique_queries: set[tuple[Any, ...]] = set()
    traces = []
    matched_scores: list[dict[str, Any]] = []
    for row in rows:
        condition = row["methods"][method]
        matched_scores.extend(
            condition.get("matched_points", {}).get("scores", {}).values()
        )
        raw += condition["scene"]["raw_count"]
        retained += condition["scene"]["selected_count"]
        unexplained += sum(
            len(t["unexplained_bits"]) for t in condition["scene"]["view_traces"]
        )
        for context, score in condition["scores"].items():
            total += 1
            unique_queries.add((row["seed"], context, tuple(row["source_bits"])))
            correct += len(score["correct_portraits"])
            missing += len(score["missing_portraits"])
            spurious += len(score["spurious_portraits"])
            exact += score["exact_portrait_set"]
            word_correct += len(score["correct_words"])
            word_missing += len(score["missing_words"])
            word_spurious += len(score["spurious_words"])
            word_exact += score["exact_word_set"]
            legacy_correct += len(score["legacy_correct_words"])
            legacy_missing += len(score["legacy_missing_words"])
            legacy_spurious += len(score["legacy_spurious_words"])
            legacy_exact += score["legacy_exact_word_set"]
            ambiguous += score["ambiguous_proposals"]
            traces.append(
                {
                    "predicted_bits": condition["predictions"][context],
                    "target_bits": row["targets"][context]["bits"],
                }
            )
    return {
        "matched_points": {
            "context_queries": len(matched_scores),
            "portrait_tp": sum(len(s["correct_portraits"]) for s in matched_scores),
            "portrait_fp": sum(len(s["spurious_portraits"]) for s in matched_scores),
            "portrait_fn": sum(len(s["missing_portraits"]) for s in matched_scores),
            "exact_portrait_scenes": sum(
                s["exact_portrait_set"] for s in matched_scores
            ),
            "word_tp": sum(len(s["correct_words"]) for s in matched_scores),
            "word_fp": sum(len(s["spurious_words"]) for s in matched_scores),
            "word_fn": sum(len(s["missing_words"]) for s in matched_scores),
            "exact_word_scenes": sum(s["exact_word_set"] for s in matched_scores),
        },
        "context_queries": total,
        "unique_source_context_queries": len(unique_queries),
        "portraits": {
            "tp": correct,
            "fp": spurious,
            "fn": missing,
            "precision": correct / (correct + spurious) if correct + spurious else None,
            "recall": correct / (correct + missing) if correct + missing else None,
            "exact_scenes": exact,
        },
        "words": {
            "tp": word_correct,
            "fp": word_spurious,
            "fn": word_missing,
            "exact_scenes": word_exact,
        },
        "legacy_words": {
            "tp": legacy_correct,
            "fp": legacy_spurious,
            "fn": legacy_missing,
            "exact_scenes": legacy_exact,
        },
        "ambiguous_proposals": ambiguous,
        "raw_proposals": raw,
        "retained_proposals": retained,
        "unexplained_active_bits": unexplained,
        "sdr": sdr_trace_metrics(traces, BIT_COUNT) if traces else None,
    }


def _aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"whole": {}, "focal": {}, "canonical_controls": {}}
    for split in ("dev", "test"):
        rows = [
            row for run in runs for row in run["whole_traces"] if row["split"] == split
        ]
        aggregate["whole"][split] = {
            method: _aggregate_rows(rows, method) for method in (*METHODS, "ideal")
        }
    for condition in ("neighbors_0", "neighbors_1", "isolated"):
        rows = [
            row
            for run in runs
            for row in run["focal_traces"]
            if row["condition"] == condition
        ]
        values = {}
        for method in (*METHODS, "ideal"):
            scores = [
                score["focal"]
                for row in rows
                for score in row["methods"][method]["scores"].values()
            ]
            values[method] = {
                "context_queries": len(scores),
                "unique_focal_context_queries": len(
                    {
                        (row["seed"], tuple(row["focal_source"]), context)
                        for row in rows
                        for context in row["targets"]
                    }
                ),
                "tp": sum(s["tp"] for s in scores),
                "fp_in_focal_slot": sum(s["fp_in_focal_slot"] for s in scores),
                "fn": sum(s["fn"] for s in scores),
                "portrait_found": sum(s["portrait_found"] for s in scores),
                "word_found": sum(s["word_found"] for s in scores),
                "legacy_word_found": sum(s["legacy_word_found"] for s in scores),
            }
        aggregate["focal"][condition] = values
    names = sorted(
        {row["control"] for run in runs for row in run["canonical_controls"]}
    )
    for name in names:
        scores = [
            row["condition"]["scores"]["canonical"]
            for run in runs
            for row in run["canonical_controls"]
            if row["control"] == name
        ]
        aggregate["canonical_controls"][name] = {
            "queries": len(scores),
            "exact_portrait_sets": sum(s["exact_portrait_set"] for s in scores),
            "correct_portraits": sum(len(s["correct_portraits"]) for s in scores),
            "missing_portraits": sum(len(s["missing_portraits"]) for s in scores),
            "spurious_portraits": sum(len(s["spurious_portraits"]) for s in scores),
            "exact_word_sets": sum(s["exact_word_set"] for s in scores),
        }
    return aggregate


def run_scene_integration(
    config: SceneIntegrationConfig | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config = config or SceneIntegrationConfig()
    started = perf_counter()
    runs: list[dict[str, Any]] = []

    def check() -> None:
        if perf_counter() - started >= config.seconds:
            raise TimeoutError("global_time_budget")

    def emit(seed: int, phase: str, completed: int = 0, total: int = 0) -> None:
        if progress is not None:
            progress(
                {
                    "seed": seed,
                    "phase": phase,
                    "completed": completed,
                    "total": total,
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
            "focal_traces": [],
            "canonical_controls": [],
            "partial_query": {},
            "training_presentations": 0,
        }
        runs.append(run)
        try:
            emit(seed, "setup")
            data = make_transform_dataset(seed)
            source_encoder, canonical, extra = (PartEncoder(seed, i) for i in range(3))
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
            common = CombinatorialMemory(
                common_memory_config(
                    ContextIntegrationConfig(
                        seeds=(seed,), points=config.points, epochs=config.epochs
                    ),
                    seed,
                )
            )
            namespace = memory_encoding_id(common)
            source_vectors = [source_encoder.encode(case) for case in data["train"]]
            target_vectors = {
                context: [
                    canonical.encode(mapping.apply(case)) for case in data["train"]
                ]
                for context, mapping in mappings.items()
            }
            shuffled = {}
            for index, context in enumerate(CONTEXTS):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 109, index]))
                shift = int(rng.integers(1, len(source_vectors)))
                shuffled[context] = [
                    (i + shift) % len(source_vectors)
                    for i in range(len(source_vectors))
                ]
            run["training_payload"] = {
                "dataset": data,
                "mappings_evaluator_only": {k: asdict(v) for k, v in mappings.items()},
                "transform_configs": {
                    c: banks["learned"][c].memory.config.to_dict() for c in CONTEXTS
                },
                "common_config": common.config.to_dict(),
                "shuffled_train_indices": shuffled,
                "codebooks": {
                    "source": [
                        [_active(source_encoder.codebook[p, v]) for v in range(3)]
                        for p in range(3)
                    ],
                    "canonical": [
                        [_active(canonical.codebook[p, v]) for v in range(3)]
                        for p in range(3)
                    ],
                    "additional": [
                        [_active(extra.codebook[p, v]) for v in range(3)]
                        for p in range(3)
                    ],
                },
            }
            run["stage"] = "training_transforms"
            emit(seed, run["stage"])
            for epoch in range(config.epochs):
                for i, source in enumerate(source_vectors):
                    for context in CONTEXTS:
                        for method, target_index in (
                            ("learned", i),
                            ("shuffled", shuffled[context][i]),
                        ):
                            check()
                            banks[method][context].observe(
                                source, target_vectors[context][target_index]
                            )
                            run["training_presentations"] += 1
                emit(seed, run["stage"], epoch + 1, config.epochs)
            run["stage"] = "training_common_and_portraits"
            emit(seed, run["stage"])
            for _ in range(6):
                for p in range(3):
                    for v in range(3):
                        check()
                        bits = canonical.codebook[p, v]
                        common.observe(bits, target=bits)
            reader = FactorSceneReader(
                common,
                config=SceneRecognitionConfig(
                    min_atoms=4, min_points=2, min_shared_atoms=3, coverage=0.6
                ),
                encoding_id=namespace,
            )
            portraits = _portrait_ids(canonical)
            dialogues, teaching = _teach(common, reader, canonical, portraits, check)
            run["teaching"] = teaching
            pipelines = {
                method: LearnedContextPipeline(
                    common,
                    bank,
                    input_encoding_id=f"source-{seed}",
                    memory_namespace=namespace,
                    limits=_limits(),
                )
                for method, bank in banks.items()
            }
            affinities = {
                method: AdaptiveContextAffinity(
                    namespace,
                    AdaptiveContextAffinityConfig(
                        window_events=32,
                        min_observations=3,
                        min_distinct_patterns=3,
                        affinity_threshold=0.8,
                        evidence_threshold=0.6,
                    ),
                )
                for method in METHODS
            }
            run["stage"] = "training_context_activity"
            emit(seed, run["stage"])
            for index, source in enumerate(source_vectors):
                for method, pipeline in pipelines.items():
                    check()
                    observation_id = f"train-scene-{index}"
                    result = pipeline.recognize(
                        source, observation_id=observation_id, source_positions=()
                    )
                    if not result.complete:
                        raise InterruptedError("incomplete_affinity_training")
                    affinities[method].observe(observation_id, result.recognition)
            run["affinity_statistics"] = {
                method: affinity.pair_statistics()
                for method, affinity in affinities.items()
            }

            def frozen(
                common=common,
                banks=banks,
                dialogues=dialogues,
                affinities=affinities,
                reader=reader,
            ) -> dict[str, Any]:
                return {
                    "common": _memory_hash(common),
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
                    "reader": _json_hash(reader.to_dict()),
                }

            run["frozen_before"] = frozen()
            run["stage"] = "whole_queries"
            emit(seed, run["stage"])
            for index, case in enumerate(data["held_out"]):
                split = "dev" if index < 4 else "test"
                row: dict[str, Any] = {"seed": seed, "split": split, "case": case}
                run["partial_query"] = row
                evaluate_scene_query(
                    source_encoder.encode(case),
                    pipelines,
                    reader,
                    dialogues,
                    partial(_targets, case, canonical, mappings, portraits),
                    observation_id=f"whole-{seed}-{index}",
                    affinities=affinities,
                    check_budget=check,
                    trace_sink=row,
                )
                run["whole_traces"].append(row)
                emit(seed, run["stage"], index + 1, 9)
            run["stage"] = "focal_diagnostics"
            emit(seed, run["stage"])
            for p in range(3):
                for v in range(3):
                    for variant in range(3):
                        isolated = variant == 2
                        case = tuple(
                            v if position == p else (v + position + variant) % 3
                            for position in range(3)
                        )
                        condition = "isolated" if isolated else f"neighbors_{variant}"
                        source = (
                            source_encoder.codebook[p, v].copy()
                            if isolated
                            else source_encoder.encode(case)
                        )
                        row = {
                            "seed": seed,
                            "focal_source": (p, v),
                            "condition": condition,
                            "case": case,
                        }
                        run["partial_query"] = row
                        evaluate_scene_query(
                            source,
                            pipelines,
                            reader,
                            dialogues,
                            partial(
                                _targets,
                                case,
                                canonical,
                                mappings,
                                portraits,
                                focal=(p, v),
                                isolated=isolated,
                            ),
                            observation_id=f"focal-{seed}-{p}-{v}-{variant}",
                            affinities=affinities,
                            check_budget=check,
                            trace_sink=row,
                        )
                        run["focal_traces"].append(row)
                    emit(seed, run["stage"], p * 3 + v + 1, 9)
            run["stage"] = "canonical_readout_controls"
            emit(seed, run["stage"])
            _control_rows(
                reader,
                dialogues,
                canonical,
                extra,
                portraits,
                check,
                run["canonical_controls"],
            )
            run["frozen_after"] = frozen()
            run["frozen_components_unchanged"] = (
                run["frozen_before"] == run["frozen_after"]
            )
            if not run["frozen_components_unchanged"]:
                raise InterruptedError("unexpected_read_mutation")
            run["transform_observation_calls"] = {
                m: {c: t.memory.step for c, t in bank.items()}
                for m, bank in banks.items()
            }
            run["common_observation_calls"] = common.step
            run["partial_query"] = {}
            run["status"] = "complete"
            emit(seed, "complete")
        except (TimeoutError, InterruptedError) as error:
            run["status"] = "incomplete"
            run["stop_reason"] = str(error)
            break
    complete = len(runs) == len(config.seeds) and all(
        r["status"] == "complete" for r in runs
    )
    return {
        "protocol": "ai2.scene_integration.v1",
        "status": "complete" if complete else "incomplete",
        "config": asdict(config),
        "elapsed_seconds": perf_counter() - started,
        "completed_seeds": [r["seed"] for r in runs if r["status"] == "complete"],
        "source": source_manifest(),
        "runs": runs,
        "aggregate": _aggregate(runs) if complete else None,
        "scope": {
            "given": [
                "three slot codebooks",
                "grouped mapping pairs",
                "nine canonical portrait anchors and labels",
                "fixed read thresholds",
            ],
            "runtime_query": (
                "one whole SDR per context, no gold source positions, boundaries, "
                "labels or masks"
            ),
            "learned": [
                "SDR transformations",
                "common stable clusters",
                "portrait evidence from teaching anchors",
                "confirmed names",
                "recent context coactivation",
            ],
            "not_claimed": [
                "unsupervised portrait discovery",
                "learned segmentation",
                "semantic context identity",
                "position invariance",
                "unseen primitive recognition",
                "learned grammar",
            ],
            "dev": "four held-out cases; no fitting, selection or threshold changes",
            "focal": (
                "nine balanced focal parts, two fixed equally dense neighbor "
                "arrangements and one isolation; diagnostic repetitions, not "
                "independent held-out meanings"
            ),
            "canonical_controls": (
                "readout-only controls after the transformation boundary; weak A "
                "is occluded but its intended membership is evaluator supplied"
            ),
            "unknown": (
                "unlabelled fixed codes diagnose unsupported familiarity; not a "
                "calibrated semantic unknown false-positive rate"
            ),
            "same_value": (
                "two independent positional codes and two portrait identities; no "
                "multiplicity can be inferred from OR of identical codes"
            ),
            "ambiguity": (
                "overlapping proposals are reported, not asserted to be semantic "
                "conflict or a proved false merge"
            ),
            "ideal": (
                "evaluator-only correct transformation after all ordinary reads; "
                "not a mathematical semantic upper bound"
            ),
            "matched_point_control": (
                "whole-scene proposals filtered at the common memory's legacy "
                "minimum active points after all word resolutions; no retraining"
            ),
            "affinity": (
                "trained on whole training inputs; empty origins give no authority "
                "to suppress scene contents"
            ),
            "budget": (
                "global cooperative deadline, two-second runtime read limits and "
                "retained incomplete prefixes; use an external process timeout"
            ),
        },
    }

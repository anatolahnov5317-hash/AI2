"""Bounded pair-supervised transformation experiment, protocol v1.

Fixed family, partition, encodings and hyperparameters precede the first run.
The learner receives two *separately grouped* streams of SDR pairs. This tests
learning mappings and recombining familiar parts, not discovering the family,
the grouping, a new context, the encoder, or an unseen kind of transformation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import permutations, product
from math import isfinite
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import ModelConfig
from ..memory import ClusterStatus
from ..transforms import LearnedSDRTransform
from .context_transfer import sdr_trace_metrics
from .runner import source_manifest

POSITIONS = 3
VALUES = 3
BLOCK_BITS = 32
ACTIVE_BITS = 6
BIT_COUNT = POSITIONS * BLOCK_BITS
METHODS = ("factor", "untrained", "shuffled_targets")
Case = tuple[int, ...]
Predictor = Callable[[NDArray[np.bool_]], NDArray[np.bool_]]


@dataclass(frozen=True, slots=True)
class TransformLearningConfig:
    """A single fixed recipe; timeout is checked between individual calls."""

    seeds: tuple[int, ...] = (7, 17, 42)
    points: int = 512
    epochs: int = 3
    seconds: float = 45.0

    def __post_init__(self) -> None:
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be non-empty, non-negative integers")
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
            raise ValueError("seconds must be a positive finite number")


@dataclass(frozen=True, slots=True)
class HiddenMapping:
    """Evaluator-only member of the given permutation/relabeling family."""

    position_order: tuple[int, ...]
    value_map: tuple[int, ...]

    def apply(self, case: Case) -> Case:
        return tuple(self.value_map[case[p]] for p in self.position_order)


class PartEncoder:
    """Given part/slot decomposition, with independently random codes per view.

    Slot blocks and codebooks are an experimental scaffold. The learner sees
    full SDRs, not these codebooks, the slot labels, or the part identities.
    """

    def __init__(self, seed: int, view: int) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 101, view]))
        self.codebook = np.zeros((POSITIONS, VALUES, BIT_COUNT), dtype=np.bool_)
        for position in range(POSITIONS):
            for value in range(VALUES):
                active = rng.choice(BLOCK_BITS, ACTIVE_BITS, replace=False)
                self.codebook[position, value, active + position * BLOCK_BITS] = True

    def encode(self, case: Case) -> NDArray[np.bool_]:
        if len(case) != POSITIONS or any(
            type(value) is not int or not 0 <= value < VALUES for value in case
        ):
            raise ValueError("case must contain three integer values in [0, 3)")
        return np.logical_or.reduce(
            [self.codebook[position, value] for position, value in enumerate(case)]
        )


def make_transform_dataset(seed: int) -> dict[str, tuple[Case, ...]]:
    """18 training / 9 held-out combinations; every part/slot is seen in train."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 103]))
    cases = [tuple(values) for values in product(range(VALUES), repeat=POSITIONS)]
    train = [case for case in cases if sum(case) % VALUES != 0]
    held_out = [case for case in cases if sum(case) % VALUES == 0]
    rng.shuffle(train)
    rng.shuffle(held_out)
    return {"train": tuple(train), "held_out": tuple(held_out)}


def hidden_mappings(seed: int) -> tuple[HiddenMapping, ...]:
    """Sample two distinct nonidentity family members, never shown to learner."""
    identity = tuple(range(VALUES))
    family = [
        HiddenMapping(position_order, value_map)
        for position_order in permutations(range(POSITIONS))
        for value_map in permutations(range(VALUES))
        if position_order != identity or value_map != identity
    ]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 107]))
    chosen = rng.choice(len(family), size=2, replace=False)
    return tuple(family[int(index)] for index in chosen)


def transform_memory_config(config: TransformLearningConfig, seed: int) -> ModelConfig:
    """Frozen recipe shared by the learned and both control memories."""
    return ModelConfig(
        input_bits=BIT_COUNT,
        output_bits=BIT_COUNT,
        active_bits_per_symbol=ACTIVE_BITS,
        positions=POSITIONS,
        frame_size=POSITIONS,
        context_count=1,
        receptive_bits=32,
        point_count=config.points,
        create_threshold=4,
        activation_threshold=3,
        min_active_points=min(4, config.points),
        probation_after=3,
        stable_after=6,
        prune_keep_ratio=0.75,
        consolidation_method="frequency",
        max_clusters_per_point=32,
        prediction_vote_threshold=2,
        seed=seed,
    )


def evaluate_transform_predictors(
    cases: Sequence[Case],
    encode_source: Callable[[Case], NDArray[np.bool_]],
    make_target: Callable[[Case], NDArray[np.bool_]],
    predictors: Mapping[str, Predictor],
    *,
    check_budget: Callable[[], None] | None = None,
    trace_sink: dict[str, list[dict[str, Any]]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Predict with every method before constructing this held-out target.

    Raw traces survive a timeout through the optional caller-owned sink. No
    metrics are returned unless all cases were evaluated for every method.
    """
    if not cases or not predictors:
        raise ValueError("cases and predictors must be non-empty")
    rows = trace_sink if trace_sink is not None else {}
    durations: dict[str, list[float]] = {}
    for name in predictors:
        rows[name] = []
        durations[name] = []
    for index, case in enumerate(cases):
        if check_budget is not None:
            check_budget()
        source = np.asarray(encode_source(case), dtype=np.bool_).copy()
        if source.shape != (BIT_COUNT,):
            raise ValueError("encoder returned the wrong source shape")
        source.flags.writeable = False
        predicted: dict[str, NDArray[np.bool_]] = {}
        for name, predict in predictors.items():
            if check_budget is not None:
                check_budget()
            started = perf_counter()
            output = np.asarray(predict(source), dtype=np.bool_).copy()
            durations[name].append(perf_counter() - started)
            if check_budget is not None:
                check_budget()
            if output.shape != (BIT_COUNT,):
                raise ValueError("predictor returned the wrong output shape")
            predicted[name] = output
        # All methods have committed their predictions before the answer exists.
        target = np.asarray(make_target(case), dtype=np.bool_)
        if target.shape != (BIT_COUNT,):
            raise ValueError("evaluator returned the wrong target shape")
        for name, output in predicted.items():
            rows[name].append(
                {
                    "case": list(case),
                    "source_bits": [int(i) for i in np.flatnonzero(source)],
                    "target_bits": [int(i) for i in np.flatnonzero(target)],
                    "predicted_bits": [int(i) for i in np.flatnonzero(output)],
                }
            )
        if progress is not None:
            progress(
                {"stage": "evaluation", "completed": index + 1, "total": len(cases)}
            )
        if check_budget is not None:
            check_budget()
    return {
        name: {
            "metrics": sdr_trace_metrics(rows[name], BIT_COUNT),
            "traces": rows[name],
            "prediction_latency": {
                "p50_ms": float(np.quantile(durations[name], 0.5)) * 1000,
                "p95_ms": float(np.quantile(durations[name], 0.95)) * 1000,
                "includes": "source SDR to predicted SDR; encoding/scoring excluded",
            },
        }
        for name in predictors
    }


def _hash_array(array: NDArray[Any]) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _predictor(model: LearnedSDRTransform) -> Predictor:
    return lambda source: model.predict(source).output


def _target_builder(
    encoder: PartEncoder, mapping: HiddenMapping
) -> Callable[[Case], NDArray[np.bool_]]:
    return lambda case: encoder.encode(mapping.apply(case))


def _coverage_diagnostics(
    training_pairs: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    output_map: NDArray[np.int32],
) -> dict[str, Any]:
    """Post-hoc trace accounting; never changes predictions or the protocol."""
    target_support = Counter(
        bit for pair in training_pairs for bit in pair["target_bits"]
    )
    known = set(target_support)
    wired = {int(bit) for bit in output_map}
    training_outputs = {tuple(pair["target_bits"]) for pair in training_pairs}
    unseen_targets: set[int] = set()
    missing_known = missing_unseen = missing_unwired = training_copies = 0
    missing_support: list[int] = []
    for row in rows:
        target = set(row["target_bits"])
        missing = target - set(row["predicted_bits"])
        unseen_targets.update(target - known)
        missing_known += len(missing & known)
        missing_unseen += len(missing - known)
        missing_unwired += len(missing - wired)
        training_copies += tuple(row["predicted_bits"]) in training_outputs
        missing_support.extend(target_support[bit] for bit in missing)
    return {
        "type": "post-hoc coverage accounting; not a new benchmark method",
        "known_target_bits_from_unique_train_pairs": len(known),
        "held_out_target_bits_absent_from_training": sorted(unseen_targets),
        "missing_known_target_bit_occurrences": missing_known,
        "missing_unseen_target_bit_occurrences": missing_unseen,
        "missing_target_bit_occurrences_without_output_receptors": missing_unwired,
        "min_unique_train_support_of_missing_bits": (
            min(missing_support) if missing_support else None
        ),
        "max_unique_train_support_of_missing_bits": (
            max(missing_support) if missing_support else None
        ),
        "predictions_identical_to_any_full_training_target": training_copies,
        "limitation": (
            "not being a training-target copy does not establish superiority "
            "over nearest-neighbour or other association methods"
        ),
    }


def _aggregate(runs: Sequence[dict[str, Any]], expected: int) -> dict[str, Any]:
    complete = [run for run in runs if run["status"] == "complete"]
    result: dict[str, Any] = {}
    for name in METHODS:
        method: dict[str, Any] = {
            "completed_tasks": [
                [run["seed"], run["mapping_index"]] for run in complete
            ],
            "all_tasks_complete": len(complete) == expected,
        }
        for metric in ("bit_precision", "bit_recall", "bit_f1", "exact_match"):
            values = [run["methods"][name]["metrics"][metric] for run in complete]
            method[metric] = {
                "mean": mean(values) if values else None,
                "stdev": stdev(values) if len(values) > 1 else None,
            }
        result[name] = method
    return result


def run_transform_learning(
    config: TransformLearningConfig | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed protocol, with visible progress and honest partial status.

    The cooperative global time budget cannot interrupt one NumPy/memory call.
    An outer process timeout of 120 seconds is recommended for default runs.
    """
    config = config or TransformLearningConfig()
    started = perf_counter()
    runs: list[dict[str, Any]] = []
    timed_out = False

    def check_budget() -> None:
        if perf_counter() - started >= config.seconds:
            raise TimeoutError("transform-learning global time budget exhausted")

    def emit(event: dict[str, Any]) -> None:
        if progress is not None:
            progress({**event, "elapsed_seconds": perf_counter() - started})

    emit({"stage": "start", "total_tasks": len(config.seeds) * 2})
    for seed in config.seeds:
        if timed_out:
            break
        data = make_transform_dataset(seed)
        source_encoder = PartEncoder(seed, 0)
        target_encoder = PartEncoder(seed, 1)
        memory_config = transform_memory_config(config, seed)
        for mapping_index, mapping in enumerate(hidden_mappings(seed)):
            run: dict[str, Any] = {
                "seed": seed,
                "mapping_index": mapping_index,
                "status": "running",
                "stage": "initialization",
                "mapping_evaluator_only": asdict(mapping),
                "data": data,
                "model_config": memory_config.to_dict(),
                "source_codebook_sha256": _hash_array(source_encoder.codebook),
                "target_codebook_sha256": _hash_array(target_encoder.codebook),
                "training_pairs": [],
                "methods": {},
                "partial_traces": {},
                "training_calls_completed": 0,
            }
            runs.append(run)
            task_info = {"seed": seed, "mapping_index": mapping_index}

            def task_progress(
                event: dict[str, Any], task_context: dict[str, int] = task_info
            ) -> None:
                emit({**task_context, **event})

            try:
                check_budget()
                task_progress({"stage": "initialization"})
                models = {name: LearnedSDRTransform(memory_config) for name in METHODS}
                check_budget()
                sources = [source_encoder.encode(case) for case in data["train"]]
                targets = [
                    target_encoder.encode(mapping.apply(case)) for case in data["train"]
                ]
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, 109, mapping_index])
                )
                shift = int(rng.integers(1, len(targets)))
                shuffled_indices = [
                    (index + shift) % len(targets) for index in range(len(targets))
                ]
                for index, case in enumerate(data["train"]):
                    run["training_pairs"].append(
                        {
                            "case": list(case),
                            "source_bits": [
                                int(i) for i in np.flatnonzero(sources[index])
                            ],
                            "target_bits": [
                                int(i) for i in np.flatnonzero(targets[index])
                            ],
                            "shuffled_target_from_train_index": shuffled_indices[index],
                            "shuffled_target_bits": [
                                int(i)
                                for i in np.flatnonzero(
                                    targets[shuffled_indices[index]]
                                )
                            ],
                        }
                    )
                run["data_sha256"] = hashlib.sha256(
                    json.dumps(run["training_pairs"], sort_keys=True).encode("utf-8")
                ).hexdigest()
                run["stage"] = "training"
                for epoch in range(config.epochs):
                    for index, source in enumerate(sources):
                        for name, target_index in (
                            ("factor", index),
                            ("shuffled_targets", shuffled_indices[index]),
                        ):
                            check_budget()
                            models[name].observe(source, targets[target_index])
                            run["training_calls_completed"] += 1
                            check_budget()
                        if (index + 1) % 6 == 0 or index + 1 == len(sources):
                            task_progress(
                                {
                                    "stage": "training",
                                    "epoch": epoch + 1,
                                    "epochs": config.epochs,
                                    "completed_pairs": index + 1,
                                    "total_pairs": len(sources),
                                }
                            )
                run["memory_state"] = {
                    name: {
                        "observation_calls": model.memory.step,
                        "unique_training_pairs": 0
                        if name == "untrained"
                        else len(sources),
                        "receptors_sha256": _hash_array(model.memory.receptors),
                        "output_map_sha256": _hash_array(model.memory.output_map),
                        "stable_clusters": sum(
                            cluster.status == ClusterStatus.STABLE
                            for _, cluster in model.memory.iter_clusters()
                        ),
                        "all_clusters": sum(1 for _ in model.memory.iter_clusters()),
                    }
                    for name, model in models.items()
                }
                run["stage"] = "evaluation"
                run["methods"] = evaluate_transform_predictors(
                    data["held_out"],
                    source_encoder.encode,
                    _target_builder(target_encoder, mapping),
                    {name: _predictor(model) for name, model in models.items()},
                    check_budget=check_budget,
                    trace_sink=run["partial_traces"],
                    progress=task_progress,
                )
                run["coverage_diagnostics"] = _coverage_diagnostics(
                    run["training_pairs"],
                    run["methods"]["factor"]["traces"],
                    models["factor"].memory.output_map,
                )
                del run["partial_traces"]
                run["status"] = "complete"
                run["stage"] = "complete"
                task_progress({"stage": "task_complete"})
            except TimeoutError:
                run["status"] = "timed_out"
                timed_out = True
                task_progress({"stage": "timed_out", "during": run["stage"]})
                break
    status = "incomplete" if timed_out else "complete"
    emit({"stage": "finished", "status": status})
    return {
        "protocol": "ai2.transform_learning.v1",
        "status": status,
        "config": asdict(config),
        "elapsed_seconds": perf_counter() - started,
        "expected_tasks": len(config.seeds) * 2,
        "completed_tasks": sum(run["status"] == "complete" for run in runs),
        "source": source_manifest(),
        "scope": {
            "learned": "SDR-to-SDR mappings from separately grouped supervised pairs",
            "given": [
                "three slots and three part values",
                "independent source and target SDR codebooks with slot blocks",
                "permutation-and-relabeling transformation family in the evaluator",
                "assignment of training pairs to each of two mapping tasks",
                "fixed hyperparameters and training/held-out split",
            ],
            "not_tested": [
                "discovery of a transformation family or a new context",
                "unseen primitive values, slots, or transformation types",
                "learning encoders, words, concepts or dialogue",
            ],
            "controls": (
                "same receptor/output wiring; no training or deranged train targets"
            ),
            "epochs": "repeated presentations; not independent new observations",
            "aggregate": "mean across mapping tasks, not a confidence interval",
            "budget": (
                "global cooperative limit between calls; use outer process timeout"
            ),
        },
        "runs": runs,
        "aggregate": _aggregate(runs, len(config.seeds) * 2),
    }

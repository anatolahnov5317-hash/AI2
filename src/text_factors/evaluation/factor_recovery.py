"""Direct recovery of planted sparse factors, with matched-marginal controls.

The learner sees binary observations only. Hidden factors and their occurrence
labels belong to the evaluator. These are structural diagnostics, not a language
benchmark or evidence of semantic understanding.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import ModelConfig
from ..consolidation import coactivation_weights
from ..memory import ClusterStatus, CombinatorialMemory
from .runner import source_manifest

SCENARIOS = (
    "single_clean",
    "multiple_clean",
    "multiple_nuisance",
    "marginal_shuffle_control",
)
METHODS = ("frequency", "coactivation")
APPROXIMATE_F1_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class FactorDataset:
    """Observed SDRs and separately held evaluator-only ground truth."""

    seed: int
    scenario: str
    training_inputs: NDArray[np.bool_]
    testing_inputs: NDArray[np.bool_]
    factors: tuple[NDArray[np.int32], ...]
    training_presence: NDArray[np.bool_]
    testing_presence: NDArray[np.bool_]
    reference_factors: tuple[NDArray[np.int32], ...]


def _check_count(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def shuffle_columns(
    inputs: NDArray[np.bool_], rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Destroy within-row associations while preserving each bit's count."""

    shuffled = inputs.copy()
    for column in range(inputs.shape[1]):
        shuffled[:, column] = rng.permutation(inputs[:, column])
    return shuffled


def make_factor_dataset(
    seed: int,
    scenario: str,
    *,
    train_samples: int = 192,
    test_samples: int = 64,
) -> FactorDataset:
    """Generate the frozen 128-bit, 12-bits-per-factor diagnostic family.

    Clean mixtures and noisy mixtures share the same factor occurrences. Noise
    activates four distinct bits outside all four planted factors. The shuffled
    control preserves noisy-data column counts separately in training and test.
    Repeated full SDRs are allowed: this test measures factor recovery, not
    compositional generalization to previously unseen full observations.
    """

    _check_count("train_samples", train_samples, 256)
    _check_count("test_samples", test_samples, 256)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown factor scenario: {scenario}")

    layout_rng = np.random.default_rng(np.random.SeedSequence([seed, 303, 0]))
    permutation = layout_rng.permutation(128)
    all_factors = tuple(
        np.sort(permutation[index * 12 : (index + 1) * 12]).astype(np.int32)
        for index in range(4)
    )
    factors = all_factors[:1] if scenario == "single_clean" else all_factors
    occurrence_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 303, 1, len(factors)])
    )
    count = train_samples + test_samples
    probability = 0.5 if scenario == "single_clean" else 0.35
    presence = occurrence_rng.random((count, len(factors))) < probability
    inputs = np.zeros((count, 128), dtype=np.bool_)
    for factor_index, factor in enumerate(factors):
        inputs[:, factor] = presence[:, factor_index, None]

    if scenario in ("multiple_nuisance", "marginal_shuffle_control"):
        nuisance_rng = np.random.default_rng(np.random.SeedSequence([seed, 303, 2]))
        for row in inputs:
            row[nuisance_rng.choice(permutation[48:], size=4, replace=False)] = True

    training_inputs = inputs[:train_samples].copy()
    testing_inputs = inputs[train_samples:].copy()
    training_presence = presence[:train_samples].copy()
    testing_presence = presence[train_samples:].copy()
    if scenario == "marginal_shuffle_control":
        shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 303, 3]))
        training_inputs = shuffle_columns(training_inputs, shuffle_rng)
        testing_inputs = shuffle_columns(testing_inputs, shuffle_rng)
        # Former factor groups remain reference masks only. The shuffled rows
        # have no planted factor-occurrence labels.
        factors = ()
        training_presence = np.zeros((train_samples, 0), dtype=np.bool_)
        testing_presence = np.zeros((test_samples, 0), dtype=np.bool_)
    return FactorDataset(
        seed,
        scenario,
        training_inputs,
        testing_inputs,
        factors,
        training_presence,
        testing_presence,
        all_factors,
    )


def fit_factor_memory(
    inputs: NDArray[np.bool_], config: ModelConfig, *, deadline: float | None = None
) -> CombinatorialMemory:
    """Train unsupervised on SDRs; this interface accepts no hidden labels."""

    if inputs.dtype != np.dtype(np.bool_) or inputs.ndim != 2:
        raise ValueError("inputs must be a two-dimensional boolean array")
    if inputs.shape[1] != config.input_bits:
        raise ValueError("input width must match config.input_bits")
    memory = CombinatorialMemory(config)
    for active in inputs:
        _check_deadline(deadline)
        memory.observe(active)
    return memory


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("factor-recovery seed exceeded its elapsed-time budget")


def _precision_recall_f1(
    matched: int, predicted: int, expected: int
) -> tuple[float, float, float]:
    precision = matched / predicted if predicted else 0.0
    recall = matched / expected if expected else 0.0
    f1 = 2.0 * matched / (predicted + expected) if predicted + expected else 0.0
    return precision, recall, f1


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_factor_recovery(
    memory: CombinatorialMemory,
    factors: Sequence[NDArray[np.int32]],
    testing_inputs: NDArray[np.bool_],
    testing_presence: NDArray[np.bool_],
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Compare stable clusters with observable *local* factor projections.

    Matching is evaluator-only, structural, and many-to-one. It is not an
    assignment learned from test outcomes or a claim that output bits uniquely
    identify concepts. A projection with fewer than activation_threshold bits
    is unobservable by the cluster activation rule and is excluded as a target.
    """

    if testing_inputs.ndim != 2 or testing_inputs.shape[1] != memory.config.input_bits:
        raise ValueError("test input shape does not match memory")
    if testing_presence.shape != (len(testing_inputs), len(factors)):
        raise ValueError("test presence shape does not match inputs and factors")
    threshold = memory.config.activation_threshold
    factor_sets = [set(int(bit) for bit in factor) for factor in factors]
    recovered_bits: list[set[int]] = [set() for _ in factors]
    observable_bits: list[set[int]] = [set() for _ in factors]
    observable_points = [0 for _ in factors]
    exact_points: list[set[int]] = [set() for _ in factors]
    structural_precision: list[float] = []
    structural_recall: list[float] = []
    structural_f1: list[float] = []
    selectivity_precision: list[float] = []
    selectivity_recall: list[float] = []
    selectivity_f1: list[float] = []
    status_counts: Counter[str] = Counter()
    exact_count = 0
    approximate_count = 0
    unsupported_count = 0
    examples: list[dict[str, Any]] = []

    for point_index, clusters in enumerate(memory.clusters):
        _check_deadline(deadline)
        receptor = set(int(bit) for bit in memory.receptors[point_index])
        projections = [receptor & factor for factor in factor_sets]
        eligible = [
            index for index, bits in enumerate(projections) if len(bits) >= threshold
        ]
        for index in eligible:
            observable_points[index] += 1
            observable_bits[index].update(projections[index])
        for cluster in clusters:
            status_counts[cluster.status.name.lower()] += 1
            if cluster.status != ClusterStatus.STABLE:
                continue
            cluster_bits = set(int(bit) for bit in cluster.bits)
            scores = [
                (
                    index,
                    _precision_recall_f1(
                        len(cluster_bits & projections[index]),
                        len(cluster_bits),
                        len(projections[index]),
                    ),
                )
                for index in eligible
            ]
            best_index: int | None = None
            best = (0.0, 0.0, 0.0)
            if scores:
                best_index, best = max(scores, key=lambda item: (item[1][2], -item[0]))
            else:
                unsupported_count += 1
            structural_precision.append(best[0])
            structural_recall.append(best[1])
            structural_f1.append(best[2])
            if best_index is not None and best[2] == 1.0:
                exact_count += 1
                recovered_bits[best_index].update(cluster_bits)
                exact_points[best_index].add(point_index)
            if best_index is not None and best[2] >= APPROXIMATE_F1_THRESHOLD:
                approximate_count += 1
                predicted = (
                    np.count_nonzero(testing_inputs[:, cluster.bits], axis=1)
                    >= threshold
                )
                expected = testing_presence[:, best_index]
                selectivity = _precision_recall_f1(
                    int(np.count_nonzero(predicted & expected)),
                    int(np.count_nonzero(predicted)),
                    int(np.count_nonzero(expected)),
                )
                selectivity_precision.append(selectivity[0])
                selectivity_recall.append(selectivity[1])
                selectivity_f1.append(selectivity[2])
            if len(examples) < 12:
                examples.append(
                    {
                        "point": point_index,
                        "bits": sorted(cluster_bits),
                        "best_factor_index": best_index,
                        "local_precision": best[0],
                        "local_recall": best[1],
                        "local_f1": best[2],
                    }
                )

    per_factor = [
        {
            "factor_index": index,
            "bits": sorted(factor),
            "observable_points": observable_points[index],
            "exact_recovery_points": len(exact_points[index]),
            "observable_bit_fraction": len(observable_bits[index]) / len(factor),
            "recovered_bit_fraction": len(recovered_bits[index]) / len(factor),
            "recovered_bits": sorted(recovered_bits[index]),
        }
        for index, factor in enumerate(factor_sets)
    ]
    stable_count = len(structural_f1)
    numpy_bytes = memory.receptors.nbytes + memory.output_map.nbytes
    for clusters in memory.clusters:
        for cluster in clusters:
            numpy_bytes += cluster.bits.nbytes + cluster.bit_hits.nbytes
            numpy_bytes += sum(row.nbytes for row in cluster.activation_history)
    # For the null control there are deliberately no planted factor labels.
    # Its stable clusters are unmatched; a latent-factor coverage is undefined.
    return {
        "cluster_status_counts": dict(sorted(status_counts.items())),
        "stable_clusters": stable_count,
        "persistent_numpy_bytes_lower_bound": numpy_bytes,
        "exact_local_clusters": exact_count,
        "unmatched_local_clusters": stable_count - exact_count,
        "approximate_local_clusters": approximate_count,
        "clusters_without_observable_target": unsupported_count,
        "exact_local_fraction": exact_count / stable_count if stable_count else 0.0,
        "best_local_precision_mean": _mean(structural_precision),
        "best_local_recall_mean": _mean(structural_recall),
        "best_local_f1_mean": _mean(structural_f1),
        "observable_factors": sum(count > 0 for count in observable_points),
        "factors_with_exact_local_recovery": sum(bool(bits) for bits in recovered_bits),
        "global_factor_bit_recall_mean": _mean(
            [
                len(recovered_bits[i]) / len(factor)
                for i, factor in enumerate(factor_sets)
                if observable_points[i] > 0
            ]
        ),
        "heldout_selectivity_cluster_count": len(selectivity_f1),
        "heldout_selectivity_precision_mean": _mean(selectivity_precision),
        "heldout_selectivity_recall_mean": _mean(selectivity_recall),
        "heldout_selectivity_f1_mean": _mean(selectivity_f1),
        "per_factor": per_factor,
        "cluster_examples_first_12": examples,
    }


def _dataset_manifest(data: FactorDataset) -> dict[str, Any]:
    train_rows = {row.tobytes() for row in data.training_inputs}
    test_rows = {row.tobytes() for row in data.testing_inputs}
    return {
        "seed": data.seed,
        "scenario": data.scenario,
        "train_samples": len(data.training_inputs),
        "test_samples": len(data.testing_inputs),
        "train_sha256": hashlib.sha256(data.training_inputs.tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(data.testing_inputs.tobytes()).hexdigest(),
        "train_unique_rows": len(train_rows),
        "test_unique_rows": len(test_rows),
        "cross_split_unique_row_overlap": len(train_rows & test_rows),
        "train_bit_counts": np.count_nonzero(data.training_inputs, axis=0).tolist(),
        "test_bit_counts": np.count_nonzero(data.testing_inputs, axis=0).tolist(),
        "train_active_bits_histogram": dict(
            sorted(
                Counter(
                    str(int(value)) for value in data.training_inputs.sum(axis=1)
                ).items()
            )
        ),
        "test_active_bits_histogram": dict(
            sorted(
                Counter(
                    str(int(value)) for value in data.testing_inputs.sum(axis=1)
                ).items()
            )
        ),
        "factor_bits_evaluator_only": [factor.tolist() for factor in data.factors],
        "reference_bits_evaluator_only": [
            factor.tolist() for factor in data.reference_factors
        ],
    }


def matched_history_control(seed: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Equal row/column counts, different coactivation; no planted candidate.

    A bounded sequence of 2x2 switches randomizes a regular incidence matrix.
    The first history alternates two exclusive four-bit groups. Its eight-bit
    union is never observed, so it cannot be an end-to-end learned candidate.
    """

    first = np.zeros((16, 8), dtype=np.bool_)
    first[::2, :4] = True
    first[1::2, 4:] = True
    control = first.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303, 4]))
    for _ in range(512):
        rows = rng.choice(16, size=2, replace=False)
        columns = rng.choice(8, size=2, replace=False)
        a, b = int(rows[0]), int(rows[1])
        c, d = int(columns[0]), int(columns[1])
        if (
            control[a, c] == control[b, d]
            and control[a, d] == control[b, c]
            and control[a, c] != control[a, d]
        ):
            control[a, c] = not control[a, c]
            control[b, d] = not control[b, d]
            control[a, d] = not control[a, d]
            control[b, c] = not control[b, c]
    return first, control


def run_filter_diagnostics(seed: int) -> dict[str, Any]:
    """Exercise the actual joint-weight kernel, separately from memory learning."""

    original, control = matched_history_control(seed)
    histories = {
        "exclusive_alternating": original,
        "exclusive_reversed": original[::-1].copy(),
        "matched_marginal_control": control,
    }
    runs: list[dict[str, Any]] = []
    for history_name, history in histories.items():
        for method in METHODS:
            if method == "frequency":
                weights = np.mean(history, axis=0)
                keep = weights >= 0.75
            else:
                weights = coactivation_weights(list(history), passes=3)
                keep = weights > 0.75
            runs.append(
                {
                    "history": history_name,
                    "method": method,
                    "weights": weights.tolist(),
                    "retained_bits": np.flatnonzero(keep).tolist(),
                    "survives_activation_threshold_3": int(np.count_nonzero(keep)) >= 3,
                }
            )
    return {
        "seed": seed,
        "scope": "isolated filter on injected union candidate; not end-to-end recovery",
        "candidate_bits": list(range(8)),
        "exclusive_factor_groups": [list(range(4)), list(range(4, 8))],
        "candidate_union_observed": False,
        "marginals_and_row_counts_equal": bool(
            np.array_equal(original.sum(axis=0), control.sum(axis=0))
            and np.array_equal(original.sum(axis=1), control.sum(axis=1))
        ),
        "histories": {
            name: value.astype(np.int8).tolist() for name, value in histories.items()
        },
        "runs": runs,
    }


def run_factor_recovery(
    seeds: Iterable[int] = (7, 17, 42),
    *,
    train_samples: int = 192,
    test_samples: int = 64,
    point_count: int = 64,
    seconds_per_seed: float = 180.0,
) -> dict[str, Any]:
    """Run all frozen scenarios and both methods without test-set selection."""

    checked_seeds = tuple(seeds)
    if not 1 <= len(checked_seeds) <= 16 or len(set(checked_seeds)) != len(
        checked_seeds
    ):
        raise ValueError("provide between 1 and 16 distinct seeds")
    if any(type(seed) is not int or seed < 0 for seed in checked_seeds):
        raise ValueError("seeds must be non-negative integers")
    _check_count("train_samples", train_samples, 256)
    _check_count("test_samples", test_samples, 256)
    _check_count("point_count", point_count, 128)
    if (
        type(seconds_per_seed) not in (int, float)
        or not isfinite(seconds_per_seed)
        or seconds_per_seed <= 0
    ):
        raise ValueError("seconds_per_seed must be a positive finite number")
    manifest = source_manifest()
    runs: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    seed_status: list[dict[str, Any]] = []
    for seed in checked_seeds:
        seed_started = perf_counter()
        deadline = seed_started + seconds_per_seed
        scenario: str | None = None
        method: str | None = None
        try:
            for scenario in SCENARIOS:
                _check_deadline(deadline)
                data = make_factor_dataset(
                    seed,
                    scenario,
                    train_samples=train_samples,
                    test_samples=test_samples,
                )
                datasets.append(_dataset_manifest(data))
                for method in METHODS:
                    _check_deadline(deadline)
                    config = ModelConfig(
                        input_bits=128,
                        active_bits_per_symbol=2,
                        positions=4,
                        frame_size=2,
                        context_count=4,
                        receptive_bits=48,
                        point_count=point_count,
                        output_bits=64,
                        create_threshold=4,
                        activation_threshold=3,
                        min_active_points=1,
                        probation_after=8,
                        stable_after=24,
                        prune_keep_ratio=0.75,
                        max_clusters_per_point=32,
                        consolidation_method=method,
                        coactivation_history_size=32,
                        coactivation_passes=3,
                        seed=seed,
                    )
                    started = perf_counter()
                    memory = fit_factor_memory(
                        data.training_inputs, config, deadline=deadline
                    )
                    fit_seconds = perf_counter() - started
                    started = perf_counter()
                    metrics = evaluate_factor_recovery(
                        memory,
                        data.factors,
                        data.testing_inputs,
                        data.testing_presence,
                        deadline=deadline,
                    )
                    evaluation_seconds = perf_counter() - started
                    runs.append(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "method": method,
                            "model_config": config.to_dict(),
                            "fit_seconds": fit_seconds,
                            "evaluation_seconds": evaluation_seconds,
                            "metrics": metrics,
                        }
                    )
        except TimeoutError as error:
            seed_status.append(
                {
                    "seed": seed,
                    "status": "timed_out",
                    "error": str(error),
                    "scenario": scenario,
                    "method": method,
                    "elapsed_seconds": perf_counter() - seed_started,
                }
            )
        else:
            seed_status.append(
                {
                    "seed": seed,
                    "status": "complete",
                    "elapsed_seconds": perf_counter() - seed_started,
                }
            )

    aggregate: list[dict[str, Any]] = []
    keys = (
        "stable_clusters",
        "exact_local_fraction",
        "unmatched_local_clusters",
        "best_local_f1_mean",
        "global_factor_bit_recall_mean",
        "heldout_selectivity_f1_mean",
    )
    for scenario in SCENARIOS:
        for method in METHODS:
            selected = [
                run
                for run in runs
                if run["scenario"] == scenario and run["method"] == method
            ]
            row: dict[str, Any] = {
                "scenario": scenario,
                "method": method,
                "seed_count": len(selected),
                "requested_seeds": list(checked_seeds),
                "completed_seeds": [run["seed"] for run in selected],
                "all_seeds_complete": len(selected) == len(checked_seeds),
            }
            for key in keys:
                values = [
                    float(run["metrics"][key])
                    for run in selected
                    if run["metrics"][key] is not None
                ]
                row[key] = _mean(values)
                row[f"{key}_contributing_seeds"] = len(values)
            aggregate.append(row)
    return {
        "schema_version": 1,
        "protocol": "factor-recovery-v03-diagnostic-v1",
        "source_manifest": manifest,
        "status": "complete"
        if all(item["status"] == "complete" for item in seed_status)
        else "incomplete",
        "seed_status": seed_status,
        "numpy_version": np.__version__,
        "config": {
            "seeds": list(checked_seeds),
            "train_samples": train_samples,
            "test_samples": test_samples,
            "point_count": point_count,
            "seconds_per_seed": seconds_per_seed,
            "approximate_local_f1_threshold": APPROXIMATE_F1_THRESHOLD,
            "factor_probability_single": 0.5,
            "factor_probability_multiple": 0.35,
            "nuisance_bits": 4,
            "training_passes": 1,
            "test_driven_parameter_selection": False,
        },
        "limitations": [
            "Repeated SDRs across train/test are allowed; "
            "this is not an unseen-composition benchmark.",
            "Stable cluster counts alone do not demonstrate useful factors.",
            "Best-local matching is evaluator-only and many-to-one; "
            "output bits are not concept IDs.",
            "Global recovery uses exact local matches; "
            "unobservable factors are excluded and counted.",
            "The shuffled control preserves column marginals, "
            "not row density or conditional candidate histories.",
            "Isolated filter diagnostics inject a candidate union "
            "that normal observation never creates.",
            "A finite-pass uncentered single component can mix "
            "equal-strength factors and depend on order.",
        ],
        "datasets": datasets,
        "runs": runs,
        "aggregate": aggregate,
        "filter_diagnostics": [run_filter_diagnostics(seed) for seed in checked_seeds],
    }

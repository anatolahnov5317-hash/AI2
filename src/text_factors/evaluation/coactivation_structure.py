"""Matched-marginal diagnosis of the existing consolidation filters.

This bounded microbenchmark inspects fixed binary histories directly. It is not
an end-to-end transfer task, a semantic benchmark, or evidence about AGI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..consolidation import coactivation_weights
from .runner import source_manifest

PROBE_SEEDS = (101, 211, 307)
SCENARIOS = ("single_joint", "competing_joint")
VARIANTS = ("joint", "marginal_shuffle")
METHODS = ("frequency", "coactivation")
ACTIVATION_THRESHOLD = 3
Progress = Callable[[dict[str, Any]], None]
BudgetCheck = Callable[[], None]


@dataclass(frozen=True, slots=True)
class CoactivationStructureConfig:
    """One frozen recipe with three order/control probes and a global budget."""

    seeds: tuple[int, int, int] = PROBE_SEEDS
    passes: int = 3
    keep_ratio: float = 0.75
    seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            len(self.seeds) != 3
            or len(set(self.seeds)) != 3
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("seeds must be three distinct non-negative integers")
        if type(self.passes) is not int or not 1 <= self.passes <= 16:
            raise ValueError("passes must be an integer in [1, 16]")
        if (
            type(self.keep_ratio) not in (int, float)
            or not isfinite(self.keep_ratio)
            or not 0.0 < self.keep_ratio <= 1.0
        ):
            raise ValueError("keep_ratio must be finite and in (0, 1]")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("seconds must be a positive finite number")


def _single_joint() -> tuple[NDArray[np.bool_], tuple[tuple[int, ...], ...]]:
    history = np.zeros((16, 8), dtype=np.bool_)
    history[:8, :3] = True
    history[:, 3] = np.arange(16) % 2 == 0
    history[:, 4] = np.tile([True, True, False, False], 4)
    history[:, 5] = np.tile([True] * 4 + [False] * 4, 2)
    parity = np.asarray([index.bit_count() % 2 == 0 for index in range(16)])
    history[:, 6] = parity
    history[:, 7] = np.roll(parity, 3)
    return history, ((0, 1, 2),)


def _competing_joint() -> tuple[NDArray[np.bool_], tuple[tuple[int, ...], ...]]:
    history = np.zeros((24, 12), dtype=np.bool_)
    history[:8, :3] = True
    history[8:16, 3:6] = True
    nuisance_rows = (
        (0, 3, 6, 9, 12, 15, 18, 21),
        (1, 4, 7, 10, 13, 16, 19, 22),
        (2, 5, 8, 11, 14, 17, 20, 23),
        (0, 1, 6, 7, 12, 13, 18, 19),
        (2, 3, 8, 9, 14, 15, 20, 21),
        (4, 5, 10, 11, 16, 17, 22, 23),
    )
    for offset, rows in enumerate(nuisance_rows, start=6):
        history[list(rows), offset] = True
    return history, ((0, 1, 2), (3, 4, 5))


def _shuffle_columns(
    history: NDArray[np.bool_], rng: np.random.Generator
) -> NDArray[np.bool_]:
    shuffled = history.copy()
    for column in range(history.shape[1]):
        shuffled[:, column] = rng.permutation(history[:, column])
    return shuffled


def make_coactivation_histories(seed: int) -> list[dict[str, Any]]:
    """Build one fixed row-order probe and its matched-marginal control."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    result: list[dict[str, Any]] = []
    for scenario_index, (scenario, builder) in enumerate(
        (("single_joint", _single_joint), ("competing_joint", _competing_joint))
    ):
        base, groups = builder()
        shuffle_rng = np.random.default_rng(
            np.random.SeedSequence([503, scenario_index])
        )
        shuffled_base = _shuffle_columns(base, shuffle_rng)
        order_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 401, scenario_index, 0])
        )
        order = order_rng.permutation(len(base))
        joint = base[order].copy()
        control = shuffled_base[order].copy()
        result.extend(
            {
                "scenario": scenario,
                "variant": variant,
                "history": history,
                "true_groups": groups,
            }
            for variant, history in (("joint", joint), ("marginal_shuffle", control))
        )
    return result


def _group_metrics(
    mask: NDArray[np.bool_], groups: Sequence[Sequence[int]]
) -> dict[str, Any]:
    selected = set(int(bit) for bit in np.flatnonzero(mask))
    candidates: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        expected = set(group)
        matched = len(selected & expected)
        precision = matched / len(selected) if selected else 0.0
        recall = matched / len(expected)
        f1 = (
            2.0 * matched / (len(selected) + len(expected))
            if selected or expected
            else 0.0
        )
        candidates.append(
            {
                "group_index": index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact": selected == expected,
            }
        )
    return max(candidates, key=lambda row: (row["f1"], -row["group_index"]))


def evaluate_history(
    history: NDArray[np.bool_],
    true_groups: Sequence[Sequence[int]],
    *,
    passes: int = 3,
    keep_ratio: float = 0.75,
) -> dict[str, Any]:
    """Apply both existing masks without mutating the supplied history."""

    if (
        not isinstance(history, np.ndarray)
        or history.dtype != np.dtype(np.bool_)
        or history.ndim != 2
        or not 1 <= len(history) <= 32
        or not 1 <= history.shape[1] <= 12
    ):
        raise ValueError("history must be a boolean array of shape [1..32, 1..12]")
    if not true_groups or any(not group for group in true_groups):
        raise ValueError("true_groups must be non-empty groups")

    frequency = np.mean(history, axis=0)
    coactivation = coactivation_weights(list(history), passes=passes)
    masks = {
        "frequency": frequency >= keep_ratio,
        "coactivation": coactivation > keep_ratio,
    }
    weights = {"frequency": frequency, "coactivation": coactivation}
    methods: dict[str, Any] = {}
    for method in METHODS:
        mask = masks[method]
        retained = int(np.count_nonzero(mask))
        methods[method] = {
            "weights": weights[method].tolist(),
            "raw_mask": mask.tolist(),
            "retained_bits": [int(bit) for bit in np.flatnonzero(mask)],
            "retained_count": retained,
            "effective_prune_valid": retained >= ACTIVATION_THRESHOLD,
            "best_true_group": _group_metrics(mask, true_groups),
        }

    group_counts = [
        int(np.count_nonzero(np.all(history[:, list(group)], axis=1)))
        for group in true_groups
    ]
    return {
        "shape": list(history.shape),
        "history": history.tolist(),
        "counts": {
            "column_active": np.count_nonzero(history, axis=0).tolist(),
            "row_active": np.count_nonzero(history, axis=1).tolist(),
            "pairwise_coactivation": (
                history.T.astype(int) @ history.astype(int)
            ).tolist(),
            "true_group_all_active": group_counts,
        },
        "true_coactivation_groups": [list(group) for group in true_groups],
        "methods": methods,
    }


def _jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _order_sensitivity(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            selected = [
                run
                for run in runs
                if run["scenario"] == scenario and run["variant"] == variant
            ]
            for method in METHODS:
                masks = [
                    run["evaluation"]["methods"][method]["retained_bits"]
                    for run in selected
                ]
                similarities = [
                    _jaccard(masks[left], masks[right])
                    for left in range(len(masks))
                    for right in range(left + 1, len(masks))
                ]
                result.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "method": method,
                        "probe_count": len(masks),
                        "all_masks_equal": all(mask == masks[0] for mask in masks[1:]),
                        "pairwise_mask_jaccard": similarities,
                        "minimum_mask_jaccard": min(similarities)
                        if similarities
                        else None,
                    }
                )
    return result


def _aggregate(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            selected = [
                run
                for run in runs
                if run["scenario"] == scenario and run["variant"] == variant
            ]
            for method in METHODS:
                details = [run["evaluation"]["methods"][method] for run in selected]
                result.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "method": method,
                        "probe_count": len(details),
                        "best_true_group_f1_mean": float(
                            np.mean([row["best_true_group"]["f1"] for row in details])
                        ),
                        "exact_true_group_fraction": float(
                            np.mean(
                                [row["best_true_group"]["exact"] for row in details]
                            )
                        ),
                        "retained_count_mean": float(
                            np.mean([row["retained_count"] for row in details])
                        ),
                        "effective_prune_valid_fraction": float(
                            np.mean([row["effective_prune_valid"] for row in details])
                        ),
                    }
                )
    return result


def run_coactivation_structure(
    config: CoactivationStructureConfig | None = None,
    *,
    progress: Progress | None = None,
    check_budget: BudgetCheck | None = None,
) -> dict[str, Any]:
    """Run the fixed diagnostic; timeout returns honest partial status."""

    config = config or CoactivationStructureConfig()
    started = perf_counter()
    runs: list[dict[str, Any]] = []

    def check() -> None:
        if check_budget is not None:
            check_budget()
        if perf_counter() - started >= config.seconds:
            raise TimeoutError("coactivation-structure global budget exhausted")

    if progress is not None:
        progress({"stage": "start", "total_probes": len(config.seeds)})
    try:
        for probe_index, seed in enumerate(config.seeds):
            check()
            histories = make_coactivation_histories(seed)
            for item in histories:
                check()
                evaluation = evaluate_history(
                    item["history"],
                    item["true_groups"],
                    passes=config.passes,
                    keep_ratio=config.keep_ratio,
                )
                runs.append(
                    {
                        "probe_index": probe_index,
                        "seed": seed,
                        "scenario": item["scenario"],
                        "variant": item["variant"],
                        "evaluation": evaluation,
                    }
                )
            if progress is not None:
                progress(
                    {
                        "stage": "probe_complete",
                        "completed_probes": probe_index + 1,
                        "total_probes": len(config.seeds),
                    }
                )
            check()
    except TimeoutError as error:
        status = "incomplete"
        timeout = {"error": str(error), "completed_runs": len(runs)}
    else:
        status = "complete"
        timeout = None

    return {
        "schema_version": 1,
        "protocol": "coactivation-structure-v03a3-diagnostic-v1",
        "status": status,
        "source_manifest": source_manifest(),
        "config": asdict(config),
        "fixed_probe_role": (
            "three predeclared order/control probes of the same structures; "
            "not train/dev/test datasets and not seed selection"
        ),
        "threshold_semantics": {
            "frequency": "absolute activation rate >= keep_ratio",
            "coactivation": "max-normalized coactivation weight > keep_ratio",
            "effective_prune_valid": f"raw retained count >= {ACTIVATION_THRESHOLD}",
        },
        "timeout": timeout,
        "runs": runs,
        "aggregate": _aggregate(runs) if status == "complete" else None,
        "order_sensitivity": _order_sensitivity(runs) if status == "complete" else None,
        "limitations": [
            "This is a post-hoc consolidation microbenchmark, not full transfer, "
            "language understanding, or AGI evidence.",
            "Seeds, passes, histories, and threshold were fixed before results; "
            "no result-driven selection is performed.",
            "Negative, mixed, and order-sensitive outcomes are accepted.",
            "Frequency and coactivation thresholds act on different scales: "
            "absolute rate versus max-normalized weight.",
            "The masks diagnose existing pruners and are not a complete "
            "competitive benchmark.",
            "A raw mask retaining fewer than activation_threshold bits makes "
            "_prune return False; an empty frequency raw mask does not mean the "
            "production kernel automatically deletes every cluster.",
            "Per-column shuffling preserves marginals but may leave accidental "
            "finite-sample coactivations.",
        ],
    }

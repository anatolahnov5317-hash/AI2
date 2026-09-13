"""Read-only diagnostics for missed outputs of a learned SDR transform.

Ground-truth targets belong here, in evaluation code.  This module classifies
each false negative by the nearest gate in the unchanged prediction readout;
it neither learns nor recommends changing a threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..memory import ClusterStatus
from ..transforms import LearnedSDRTransform

_MAX_OUTPUT_BITS = 1_000_000
_TIME_BUDGET_SECONDS = 2.0
_GATES = ("no_cluster", "no_stable", "no_exact", "below_vote")


def _checked_sdr(
    value: NDArray[Any], length: int, name: str, *, require_active: bool
) -> NDArray[np.bool_]:
    bits = np.asarray(value)
    if bits.shape != (length,):
        raise ValueError(f"{name} shape must be {(length,)}, got {bits.shape}")
    if not (
        np.issubdtype(bits.dtype, np.bool_) or np.issubdtype(bits.dtype, np.integer)
    ) or np.any((bits != 0) & (bits != 1)):
        raise ValueError(f"{name} must contain bool or integer 0/1 bits")
    if require_active and not np.any(bits):
        raise ValueError(f"{name} must have at least one active bit")
    return bits.astype(np.bool_, copy=True)


def _checked_evaluated_bits(
    value: NDArray[Any] | None, length: int
) -> NDArray[np.bool_]:
    if value is None:
        return np.ones(length, dtype=np.bool_)
    selected = np.asarray(value)
    if np.issubdtype(selected.dtype, np.bool_):
        if selected.shape != (length,):
            raise ValueError(
                f"evaluated_bits mask shape must be {(length,)}, got {selected.shape}"
            )
        mask = selected.astype(np.bool_, copy=True)
    else:
        if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("evaluated_bits must be a bool mask or integer indices")
        if np.any(selected < 0) or np.any(selected >= length):
            raise ValueError("evaluated_bits contains an out-of-range index")
        indices = selected.astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("evaluated_bits indices must be unique")
        mask = np.zeros(length, dtype=np.bool_)
        mask[indices] = True
    if not np.any(mask):
        raise ValueError("evaluated_bits must select at least one output bit")
    return mask


def _is_cancelled(cancelled: Callable[[], bool] | Any | None) -> bool:
    if cancelled is None:
        return False
    if callable(cancelled):
        return bool(cancelled())
    is_set = getattr(cancelled, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    raise ValueError("cancelled must be callable or expose is_set()")


def diagnose_transform_bits(
    transform: LearnedSDRTransform,
    source: NDArray[Any],
    target: NDArray[Any],
    predicted: NDArray[Any] | None = None,
    cancelled: Callable[[], bool] | Any | None = None,
    *,
    evaluated_bits: NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Classify false negatives by their nearest blocking prediction gate.

    ``target`` and ``predicted`` are always full-width SDRs.  ``evaluated_bits``
    can be a full-width boolean mask or unique integer indices; all confusion
    counts and missing-bit details are restricted to that selection.  This lets
    a focal-object evaluation exclude correct outputs belonging to neighbours
    instead of counting them as false positives.

    The fixed two-second and one-million-output-bit bounds keep this evaluator
    diagnostic finite.  Cancellation raises ``InterruptedError`` and expiry
    raises ``TimeoutError``; neither condition returns invented partial counts.
    """
    if not isinstance(transform, LearnedSDRTransform):
        raise TypeError("transform must be a LearnedSDRTransform")
    config = transform.memory.config
    if config.output_bits > _MAX_OUTPUT_BITS:
        raise ValueError(f"output_bits exceeds diagnostic limit {_MAX_OUTPUT_BITS}")
    checked_source = _checked_sdr(
        source, config.input_bits, "source", require_active=True
    )
    checked_target = _checked_sdr(
        target, config.output_bits, "target", require_active=True
    )
    mask = _checked_evaluated_bits(evaluated_bits, config.output_bits)
    checked_predicted = (
        None
        if predicted is None
        else _checked_sdr(
            predicted, config.output_bits, "predicted", require_active=False
        )
    )
    # Validate cancellation before potentially doing the ordinary prediction.
    if _is_cancelled(cancelled):
        raise InterruptedError("transform diagnosis cancelled")
    deadline = perf_counter() + _TIME_BUDGET_SECONDS

    readout = None
    if checked_predicted is None:
        readout = transform.predict(checked_source)
        checked_predicted = readout.output.copy()
        if perf_counter() >= deadline:
            raise TimeoutError("transform diagnosis exceeded two seconds")

    missing_mask = mask & checked_target & ~checked_predicted
    missing_indices = np.flatnonzero(missing_mask)
    missing_set = {int(bit) for bit in missing_indices}

    # output_map describes wiring even at points that have never formed a cluster.
    wired_counts = np.bincount(
        transform.memory.output_map, minlength=config.output_bits
    ).astype(np.int64, copy=False)
    scores = np.zeros(config.output_bits, dtype=np.float64)
    details = {
        bit: {
            "cluster_count": 0,
            "temporary_cluster_count": 0,
            "probation_cluster_count": 0,
            "stable_cluster_count": 0,
            "stable_exact_cluster_count": 0,
            "stable_exact_point_count": 0,
            "observed_output_support": 0,
            "best_stable_match": 0,
            "best_stable_cluster_length": 0,
            "exact_points": set(),
        }
        for bit in missing_set
    }

    for number, (point, cluster) in enumerate(transform.memory.iter_clusters()):
        if number % 256 == 0:
            if _is_cancelled(cancelled):
                raise InterruptedError("transform diagnosis cancelled")
            if perf_counter() >= deadline:
                raise TimeoutError("transform diagnosis exceeded two seconds")
        output_bit = int(transform.memory.output_map[point])
        matched = transform.memory._match_count(cluster, checked_source)
        if cluster.status == ClusterStatus.STABLE and matched == len(cluster.bits):
            scores[output_bit] += matched - config.activation_threshold + 1

        if output_bit not in missing_set:
            continue
        row = details[output_bit]
        row["cluster_count"] += 1
        status_name = cluster.status.name.lower()
        row[f"{status_name}_cluster_count"] += 1
        # This is cluster-local positive exact evidence and can count one source
        # observation at several receptive points; it is not a unique-pair count.
        row["observed_output_support"] += max(
            int(cluster.exact_hits) - int(cluster.complete_errors), 0
        )
        if cluster.status != ClusterStatus.STABLE:
            continue
        if matched > row["best_stable_match"]:
            row["best_stable_match"] = matched
            row["best_stable_cluster_length"] = len(cluster.bits)
        if matched == len(cluster.bits):
            row["stable_exact_cluster_count"] += 1
            row["exact_points"].add(point)

    if _is_cancelled(cancelled):
        raise InterruptedError("transform diagnosis cancelled")
    if perf_counter() >= deadline:
        raise TimeoutError("transform diagnosis exceeded two seconds")

    expected = scores >= config.prediction_vote_threshold
    disagreements = np.flatnonzero(expected != checked_predicted)
    if len(disagreements):
        raise AssertionError(
            "predicted disagrees with the unchanged transform readout at output "
            f"bits {[int(bit) for bit in disagreements[:16]]}"
        )
    if readout is not None and not np.array_equal(scores, readout.output_scores):
        raise AssertionError("diagnostic scores disagree with transform.predict scores")

    gate_counts = {gate: 0 for gate in _GATES}
    missing_bits: list[dict[str, Any]] = []
    for raw_bit in missing_indices:
        bit = int(raw_bit)
        row = details[bit]
        row["stable_exact_point_count"] = len(row.pop("exact_points"))
        score = float(scores[bit])
        if row["cluster_count"] == 0:
            gate = "no_cluster"
        elif row["stable_cluster_count"] == 0:
            gate = "no_stable"
        elif row["stable_exact_cluster_count"] == 0:
            gate = "no_exact"
        elif score < config.prediction_vote_threshold:
            gate = "below_vote"
        else:  # The full prediction-consistency check above makes this unreachable.
            raise AssertionError(f"missing output bit {bit} passed every readout gate")
        gate_counts[gate] += 1
        missing_bits.append(
            {
                "output_bit": bit,
                "gate": gate,
                "wired_point_count": int(wired_counts[bit]),
                **row,
                "weighted_score": score,
                "vote_threshold": config.prediction_vote_threshold,
                "score_margin": score - config.prediction_vote_threshold,
            }
        )

    scoped_target = checked_target[mask]
    scoped_predicted = checked_predicted[mask]
    counts = {
        "evaluated_bits": int(np.count_nonzero(mask)),
        "target_bits": int(np.count_nonzero(scoped_target)),
        "predicted_bits": int(np.count_nonzero(scoped_predicted)),
        "true_positive": int(np.count_nonzero(scoped_target & scoped_predicted)),
        "false_positive": int(np.count_nonzero(~scoped_target & scoped_predicted)),
        "false_negative": int(np.count_nonzero(scoped_target & ~scoped_predicted)),
        "true_negative": int(np.count_nonzero(~scoped_target & ~scoped_predicted)),
        **gate_counts,
    }
    return {
        "counts": counts,
        "missing_bits": missing_bits,
        "prediction_disagreement_count": 0,
        "gate_order": list(_GATES),
        "interpretation": "nearest blocking gate in the unchanged prediction readout",
        "interpretation_limit": (
            "Gate counts are descriptive, not causal or counterfactual evidence; "
            "false-positive counts also have no attributed cause. Counts apply "
            "only to evaluated_bits, so callers must include every output whose "
            "correctness they intend to judge."
        ),
    }

"""Direct, held-out SDR prediction for a learned positional context transform.

Protocol v1 was fixed before its first numerical run. This experiment tests
recombination of familiar symbol-position primitives, not discovery of the
encoder, an unseen transformation, language understanding, or arbitrary rules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import ModelConfig
from ..encoder import SparseSymbolEncoder
from ..model import TextFactorModel
from .runner import source_manifest

ALPHABET = "abcd"
WINDOW_SIZE = 5
METHODS = ("frequency", "coactivation", "zero", "identity", "nearest_source_sdr")


@dataclass(frozen=True, slots=True)
class ContextTransferConfig:
    """Fixed defaults plus explicit overrides for smoke and follow-up runs.

    The elapsed budget is checked between individual training and prediction
    calls; a caller should also impose an outer process timeout.
    """

    seeds: tuple[int, ...] = (7, 17, 42)
    points: int = 1024
    epochs: int = 3
    train_size: int = 128
    dev_size: int = 32
    test_size: int = 64
    seconds_per_seed: float = 180.0

    def __post_init__(self) -> None:
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be non-empty, non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for key in ("points", "epochs", "train_size", "dev_size", "test_size"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.train_size < len(ALPHABET):
            raise ValueError("train_size must cover the four primitive anchors")
        if self.train_size + self.dev_size + self.test_size > 4**WINDOW_SIZE:
            raise ValueError("requested disjoint splits exceed the 1024-string space")
        if (
            type(self.seconds_per_seed) not in (int, float)
            or not isfinite(self.seconds_per_seed)
            or self.seconds_per_seed <= 0
        ):
            raise ValueError("seconds_per_seed must be a positive finite number")


@dataclass(frozen=True, slots=True)
class ContextDataset:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]


def make_context_dataset(config: ContextTransferConfig, seed: int) -> ContextDataset:
    """Allocate whole-string-disjoint splits with complete training primitives."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    anchors = [symbol * WINDOW_SIZE for symbol in ALPHABET]
    candidates = [
        "".join(symbols)
        for symbols in product(ALPHABET, repeat=WINDOW_SIZE)
        if "".join(symbols) not in anchors
    ]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
    rng.shuffle(candidates)
    boundary = config.train_size - len(anchors)
    training = anchors + candidates[:boundary]
    rng.shuffle(training)
    test_start = boundary + config.dev_size
    return ContextDataset(
        train=tuple(training),
        dev=tuple(candidates[boundary : boundary + config.dev_size]),
        test=tuple(candidates[test_start : test_start + config.test_size]),
    )


def context_model_config(
    config: ContextTransferConfig, seed: int, method: str
) -> ModelConfig:
    """Return the v1 comparison settings, without consulting held-out data."""
    if method not in ("frequency", "coactivation"):
        raise ValueError("unknown consolidation method")
    return ModelConfig(
        input_bits=256,
        output_bits=256,
        active_bits_per_symbol=8,
        frame_size=WINDOW_SIZE,
        positions=6,
        context_count=2,
        receptive_bits=32,
        point_count=config.points,
        create_threshold=6,
        activation_threshold=4,
        min_active_points=min(4, config.points),
        probation_after=3,
        stable_after=6,
        prune_keep_ratio=0.75,
        max_clusters_per_point=32,
        prediction_vote_threshold=2,
        consolidation_method=method,
        coactivation_history_size=32,
        coactivation_passes=3,
        seed=seed,
    )


def sdr_trace_metrics(rows: Sequence[dict[str, Any]], bit_count: int) -> dict[str, Any]:
    """Recompute exact SDR metrics from active-index traces.

    Precision/recall/F1 use pooled bit counts. Undefined zero-denominator ratios
    are zero, so an all-zero prediction does not earn precision or F1 credit.
    Exact-match and mean active counts are per complete output vector.
    """
    if not rows:
        raise ValueError("at least one prediction trace is required")
    if type(bit_count) is not int or bit_count <= 0:
        raise ValueError("bit_count must be a positive integer")
    true_positive = false_positive = false_negative = exact = 0
    predicted_count = target_count = nonempty = 0
    for row in rows:
        target = set(row["target_bits"])
        predicted = set(row["predicted_bits"])
        for values in (row["target_bits"], row["predicted_bits"]):
            if len(values) != len(set(values)) or any(
                type(bit) is not int or not 0 <= bit < bit_count for bit in values
            ):
                raise ValueError("trace bits must be unique valid integer indices")
        true_positive += len(target & predicted)
        false_positive += len(predicted - target)
        false_negative += len(target - predicted)
        exact += predicted == target
        predicted_count += len(predicted)
        target_count += len(target)
        nonempty += bool(predicted)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / target_count if target_count else 0.0
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "bit_precision": precision,
        "bit_recall": recall,
        "bit_f1": 2 * true_positive / denominator if denominator else 0.0,
        "exact_match": exact / len(rows),
        "mean_predicted_active_bits": predicted_count / len(rows),
        "mean_target_active_bits": target_count / len(rows),
        "predicted_density": predicted_count / (len(rows) * bit_count),
        "target_density": target_count / (len(rows) * bit_count),
        "nonempty_rate": nonempty / len(rows),
        "true_positive_bits": true_positive,
        "false_positive_bits": false_positive,
        "false_negative_bits": false_negative,
        "cases": len(rows),
    }


def evaluate_context_predictor(
    texts: Sequence[str],
    encoder: SparseSymbolEncoder,
    predict: Callable[[NDArray[np.bool_]], NDArray[np.bool_]],
    *,
    check_budget: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Call a source-SDR-only predictor before constructing each scoring target."""
    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    for text in texts:
        if check_budget is not None:
            check_budget()
        source = encoder.encode_window(text, context=0, offset=0)
        source.flags.writeable = False
        started = perf_counter()
        prediction = np.asarray(predict(source), dtype=np.bool_)
        durations.append(perf_counter() - started)
        if prediction.shape != (encoder.config.output_bits,):
            raise ValueError("predictor returned the wrong output shape")
        # Only the evaluator knows this example's output, after prediction.
        target = encoder.encode_window(text, context=1, offset=0)
        rows.append(
            {
                "text": text,
                "source_bits": [int(bit) for bit in np.flatnonzero(source)],
                "target_bits": [int(bit) for bit in np.flatnonzero(target)],
                "predicted_bits": [int(bit) for bit in np.flatnonzero(prediction)],
            }
        )
    return {
        "metrics": sdr_trace_metrics(rows, encoder.config.output_bits),
        "prediction_latency": {
            "p50_ms": float(np.quantile(durations, 0.5)) * 1000,
            "p95_ms": float(np.quantile(durations, 0.95)) * 1000,
            "includes": "source SDR to predicted SDR; encoding and scoring excluded",
        },
        "traces": rows,
    }


def _array_hash(array: NDArray[Any]) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _numpy_payload_bytes(model: TextFactorModel) -> int:
    """Persistent NumPy payload only, excluding Python and allocator overhead."""
    size = sum(
        array.nbytes
        for array in (
            model.encoder.codebook,
            model.memory.receptors,
            model.memory.output_map,
        )
    )
    for _, cluster in model.memory.iter_clusters():
        size += cluster.bits.nbytes + cluster.bit_hits.nbytes
        size += sum(row.nbytes for row in cluster.activation_history)
    return int(size)


def _memory_predictor(
    model: TextFactorModel,
) -> Callable[[NDArray[np.bool_]], NDArray[np.bool_]]:
    def predict(source: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return model.memory.predict(source).output

    return predict


def _identity_prediction(source: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return source.copy()


def _zero_prediction(source: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return np.zeros_like(source)


def _run_seed(
    config: ContextTransferConfig,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    started = perf_counter()
    dataset = make_context_dataset(config, seed)
    data = asdict(dataset)
    report: dict[str, Any] = {
        "seed": seed,
        "status": "running",
        "data": data,
        "data_sha256": hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "unique_training_examples": len(dataset.train),
        "training_exposures": len(dataset.train) * config.epochs,
        "training_primitive_counts": {
            f"{symbol}@{position}": sum(
                text[position] == symbol for text in dataset.train
            )
            for symbol in ALPHABET
            for position in range(WINDOW_SIZE)
        },
        "methods": {},
    }

    def check_budget() -> None:
        if perf_counter() - started > config.seconds_per_seed:
            raise TimeoutError("per-seed elapsed-time budget exceeded")

    encoder = SparseSymbolEncoder(
        context_model_config(config, seed, "frequency"), ALPHABET
    )
    report["codebook_sha256"] = _array_hash(encoder.codebook)
    pair_started = perf_counter()
    train_source = np.stack(
        [encoder.encode_window(text, context=0) for text in dataset.train]
    )
    train_target = np.stack(
        [encoder.encode_window(text, context=1) for text in dataset.train]
    )
    pair_encoding_seconds = perf_counter() - pair_started
    report["shared_training_pair_encoding_seconds"] = pair_encoding_seconds
    # Keeping these raw training pairs makes the retrieval control auditable.
    report["training_pairs"] = [
        {
            "text": text,
            "source_bits": [int(bit) for bit in np.flatnonzero(source)],
            "target_bits": [int(bit) for bit in np.flatnonzero(target)],
        }
        for text, source, target in zip(
            dataset.train, train_source, train_target, strict=True
        )
    ]

    def nearest(source: NDArray[np.bool_]) -> NDArray[np.bool_]:
        intersection = np.count_nonzero(train_source & source, axis=1)
        union = np.count_nonzero(train_source | source, axis=1)
        similarities = intersection / union
        # np.argmax deterministically breaks ties by the original training order.
        return train_target[int(np.argmax(similarities))].copy()

    try:
        for method in METHODS:
            check_budget()
            if progress is not None:
                progress({"seed": seed, "method": method, "stage": "started"})
            details: dict[str, Any] = {}
            fit_started = perf_counter()
            if method in ("frequency", "coactivation"):
                model_config = context_model_config(config, seed, method)
                model = TextFactorModel(model_config, alphabet=ALPHABET)
                for _ in range(config.epochs):
                    for text in dataset.train:
                        check_budget()
                        model.learn_context_transform(text)
                fit_seconds = perf_counter() - fit_started
                details = {
                    "model_config": model_config.to_dict(),
                    "memory": model.memory.stats(),
                    "numpy_payload_bytes_lower_bound": _numpy_payload_bytes(model),
                    "receptors_sha256": _array_hash(model.memory.receptors),
                    "output_map_sha256": _array_hash(model.memory.output_map),
                    "codebook_sha256": _array_hash(model.encoder.codebook),
                }

                predict = _memory_predictor(model)
            elif method == "nearest_source_sdr":
                fit_seconds = pair_encoding_seconds
                predict = nearest
                details = {
                    "similarity": "Jaccard overlap of source SDRs",
                    "tie_break": "first item in saved training order",
                    "stored_training_pairs": len(dataset.train),
                    "numpy_payload_bytes_lower_bound": int(
                        train_source.nbytes + train_target.nbytes
                    ),
                }
            elif method == "identity":
                fit_seconds = 0.0
                predict = _identity_prediction
            else:
                fit_seconds = 0.0
                predict = _zero_prediction
            report["methods"][method] = {
                "fit_seconds": fit_seconds,
                **details,
                "dev": evaluate_context_predictor(
                    dataset.dev, encoder, predict, check_budget=check_budget
                ),
                "test": evaluate_context_predictor(
                    dataset.test, encoder, predict, check_budget=check_budget
                ),
            }
            if progress is not None:
                progress({"seed": seed, "method": method, "stage": "completed"})
        report["status"] = "complete"
    except TimeoutError as error:
        report["status"] = "timed_out"
        report["error"] = str(error)
    report["elapsed_seconds"] = perf_counter() - started
    return report


def run_context_transfer(
    config: ContextTransferConfig | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compare fixed consolidation settings and controls on all requested seeds."""
    settings = config or ContextTransferConfig()
    manifest = source_manifest()
    runs = [_run_seed(settings, seed, progress) for seed in settings.seeds]
    aggregate: dict[str, Any] = {}
    for method in METHODS:
        completed = [run for run in runs if method in run["methods"]]
        aggregate[method] = {
            "completed_seeds": [run["seed"] for run in completed],
            "requested_seeds": list(settings.seeds),
            "all_seeds_complete": len(completed) == len(settings.seeds),
        }
        for metric in (
            "bit_precision",
            "bit_recall",
            "bit_f1",
            "exact_match",
            "mean_predicted_active_bits",
            "predicted_density",
        ):
            values = [
                float(run["methods"][method]["test"]["metrics"][metric])
                for run in completed
            ]
            aggregate[method][metric] = {
                "mean": mean(values) if values else None,
                "sample_sd": stdev(values) if len(values) > 1 else None,
                "per_seed": values,
            }
    return {
        "manifest": {
            **manifest,
            "experiment": "context_transfer",
            "protocol_version": 1,
            "default_protocol_fixed_before_first_numerical_run": True,
            "protocol_variant": (
                "frozen_default"
                if settings == ContextTransferConfig()
                else "configuration_override"
            ),
            "config": asdict(settings),
            "scope": (
                "familiar-primitives recombination for one supervised context shift"
            ),
            "source_context": 0,
            "target_context": 1,
            "threshold_selection": (
                "fixed before results; development is diagnostic only"
            ),
            "prediction_interface": (
                "source SDR only; evaluator constructs target after prediction"
            ),
            "training_order": "fixed seeded order repeated for each epoch",
            "epochs_are_independent_evidence": False,
            "timeout_policy": (
                "checked between calls; incomplete seeds remain in report"
            ),
            "memory_measure": "NumPy payload lower bound, not process RAM or peak RAM",
            "limitations": [
                "The encoder and positional context operator are supplied "
                "by the experimenter.",
                "Training targets are supervised observations; latent rules "
                "are not discovered from raw text.",
                "Only combinations are held out; symbols, positions and the "
                "transformation are familiar.",
                "Repeated epochs repeat evidence and do not create "
                "independent observations.",
                "The development split does not select thresholds or "
                "hyperparameters in this protocol.",
                "This is a small synthetic diagnostic; it establishes no "
                "language, GPT or AGI advantage.",
                "Successful SDR prediction alone would not prove that "
                "individual factors are identifiable.",
            ],
        },
        "status": (
            "complete"
            if all(run["status"] == "complete" for run in runs)
            else "incomplete"
        ),
        "runs": runs,
        "aggregate": aggregate,
    }

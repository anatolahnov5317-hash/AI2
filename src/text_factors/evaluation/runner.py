"""Reproducible text-memory experiments with raw, recomputable score traces."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import numpy as np

from .. import __version__
from ..config import ModelConfig
from ..encoder import SparseSymbolEncoder
from ..model import TextFactorModel
from .baselines import ExactMemory, NearestSDRMemory, NGramMemory, PairAssociationMemory
from .data import ALPHABET, TextCase, dataset_hash, make_dataset
from .statistics import NoveltyGate, classification_metrics, paired_accuracy_interval


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (7, 17, 42)
    points: int = 128
    epochs: int = 6
    train_size: int = 24
    dev_size: int = 32
    test_size: int = 64
    noise_size: int = 64
    target_fpr: float = 0.05
    bootstrap_resamples: int = 1000

    def __post_init__(self) -> None:
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be non-empty, non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for key in (
            "points",
            "epochs",
            "train_size",
            "dev_size",
            "test_size",
            "bootstrap_resamples",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.noise_size) is not int or self.noise_size < 0:
            raise ValueError("noise_size must be a non-negative integer")
        if (
            type(self.target_fpr) not in (int, float)
            or not np.isfinite(self.target_fpr)
            or not 0 <= self.target_fpr < 1
        ):
            raise ValueError("target_fpr must be in [0, 1)")


def source_manifest() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    checkout = package.parents[1]
    revision = None
    tree = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
            tree_result = subprocess.run(
                ["git", "rev-parse", "HEAD^{tree}"],
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if tree_result.returncode == 0:
                tree = tree_result.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=checkout,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": sys.platform,
        "git_revision": revision,
        "git_tree": tree,
        "git_dirty": dirty,
        "source_sha256": source_hashes,
        "reproducibility": (
            "scores/counts deterministic for matching source and dependencies; "
            "hashes identify but do not embed source; "
            "wall-clock timings are not deterministic"
        ),
    }


def _measure_cases(
    cases: Sequence[TextCase], score: Callable[[str], float]
) -> tuple[list[float], dict[str, float]]:
    values: list[float] = []
    durations: list[float] = []
    for case in cases:
        start = perf_counter()
        values.append(float(score(case.text)))
        durations.append(perf_counter() - start)
    return values, {
        "p50_ms": float(np.quantile(durations, 0.5)) * 1000,
        "p95_ms": float(np.quantile(durations, 0.95)) * 1000,
    }


def _numpy_payload_bytes(model: TextFactorModel) -> int:
    """Lower bound only: excludes Python objects, lists, sets and allocator cost."""
    return sum(
        int(array.nbytes)
        for array in (
            model.encoder.codebook,
            model.memory.receptors,
            model.memory.output_map,
        )
    ) + sum(
        int(cluster.bits.nbytes + cluster.bit_hits.nbytes)
        for _, cluster in model.memory.iter_clusters()
    )


def _run_seed(config: EvaluationConfig, seed: int) -> dict[str, Any]:
    dataset = make_dataset(
        seed=seed,
        train_size=config.train_size,
        dev_size=config.dev_size,
        test_size=config.test_size,
        noise_size=config.noise_size,
    )
    model_config = ModelConfig(
        point_count=config.points,
        min_active_points=min(4, config.points),
        max_clusters_per_point=32,
        seed=seed,
    )
    model = TextFactorModel(model_config, alphabet=ALPHABET)
    exposures = dataset.train * config.epochs
    start = perf_counter()
    for text in exposures:
        model.partial_fit_window(text, offset=0)
    training_seconds = perf_counter() - start
    readouts: dict[str, dict[str, Any]] = {}

    def memory_score(text: str) -> float:
        result = model.transform_window(text, offset=0)
        readouts[text] = result.to_dict()
        return max(result.context_scores)

    methods: dict[str, dict[str, Any]] = {}
    baseline_specs = {
        "exact": ExactMemory(),
        "bigram": NGramMemory(n=2),
        "nearest_sdr": NearestSDRMemory(SparseSymbolEncoder(model_config, ALPHABET)),
        "pair_association": PairAssociationMemory(),
    }
    scorers: dict[str, Callable[[str], float]] = {"ai2": memory_score}
    fit_seconds: dict[str, float] = {"ai2": training_seconds}
    for name, baseline in baseline_specs.items():
        start = perf_counter()
        baseline.fit(exposures)
        fit_seconds[name] = perf_counter() - start
        scorers[name] = baseline.score

    labels = [case.label for case in dataset.test]
    decisions: dict[str, list[bool]] = {}
    for name, scorer in scorers.items():
        dev_scores, _ = _measure_cases(dataset.dev, scorer)
        gate = NoveltyGate.fit(
            [
                s
                for case, s in zip(dataset.dev, dev_scores, strict=True)
                if not case.label
            ],
            target_fpr=config.target_fpr,
        )
        test_scores, latency = _measure_cases(dataset.test, scorer)
        decisions[name] = [gate.accepts(s) for s in test_scores]
        methods[name] = {
            "threshold": asdict(gate),
            "fit_seconds": fit_seconds[name],
            "score_latency": latency,
            "metrics": classification_metrics(labels, test_scores, gate),
            "dev": [
                {**asdict(case), "score": score, "accepted": gate.accepts(score)}
                for case, score in zip(dataset.dev, dev_scores, strict=True)
            ],
            "test": [
                {**asdict(case), "score": score, "accepted": gate.accepts(score)}
                for case, score in zip(dataset.test, test_scores, strict=True)
            ],
        }
        if name == "pair_association":
            pair_model = baseline_specs[name]
            assert isinstance(pair_model, PairAssociationMemory)
            methods[name]["selected_positions"] = pair_model.selected_positions
            methods[name]["train_dependence_score"] = pair_model.dependence_score

    # A separate fresh model: one exposure per unique random window, no epochs.
    noise_model = TextFactorModel(model_config, alphabet=ALPHABET)
    for text in dataset.noise:
        noise_model.partial_fit_window(text, offset=0)
    noise_stats = noise_model.memory.stats()
    code_sizes = [len(readouts[case.text]["output_bits"]) for case in dataset.test]
    return {
        "seed": seed,
        "data_sha256": dataset_hash(dataset),
        "data": asdict(dataset),
        "model_config": model_config.to_dict(),
        "training_exposures": len(exposures),
        "methods": methods,
        "ai2_diagnostics": {
            "memory": model.memory.stats(),
            "numpy_payload_bytes_lower_bound": _numpy_payload_bytes(model),
            "test_nonempty_rate": float(np.mean(np.asarray(code_sizes) > 0)),
            "mean_test_code_bits": float(np.mean(code_sizes)),
            "test_code_density": float(np.mean(code_sizes)) / model_config.output_bits,
            "readouts": readouts,
        },
        "noise_control": {
            "memory": noise_stats,
            "numpy_payload_bytes_lower_bound": _numpy_payload_bytes(noise_model),
            "note": "stable partial clusters on noise are not semantic discoveries",
        },
        "paired_comparisons": {
            name: paired_accuracy_interval(
                labels,
                decisions["ai2"],
                decisions[name],
                seed=seed,
                resamples=config.bootstrap_resamples,
            )
            for name in baseline_specs
        },
    }


def run_evaluation(config: EvaluationConfig | None = None) -> dict[str, Any]:
    settings = config or EvaluationConfig()
    runs = [_run_seed(settings, seed) for seed in settings.seeds]
    aggregate: dict[str, dict[str, Any]] = {}
    for method in ("ai2", "exact", "bigram", "nearest_sdr", "pair_association"):
        aggregate[method] = {}
        for metric in ("roc_auc", "balanced_accuracy", "recall", "false_positive_rate"):
            values = [float(run["methods"][method]["metrics"][metric]) for run in runs]
            aggregate[method][metric] = {
                "mean": mean(values),
                "sample_sd": stdev(values) if len(values) > 1 else None,
                "per_seed": values,
            }
    return {
        "schema_version": 1,
        "manifest": {
            **source_manifest(),
            "evaluation_config": asdict(settings),
            "primary_metric": "held-out balanced_accuracy at negative-dev threshold",
            "secondary_metrics": ["roc_auc", "recall", "false_positive_rate"],
            "task": "known endpoint associations with held-out whole strings",
            "split_unit": "whole five-character string; no overlapping stream windows",
            "scope": (
                "within-family combination transfer; NOT unseen-rule learning or AGI"
            ),
            "budgets": (
                "identical training strings/exposures; actual compute costs reported"
            ),
            "threshold_policy": "negative dev only; strict score > empirical quantile",
            "statistics": (
                "per-seed paired case bootstrap; no pooling seeds as independent cases"
            ),
            "limitations": [
                "one hand-designed rule family and structured fixed-length input",
                "cyclic invariance is supplied by the encoder, not learned",
                "nonempty or stable SDR codes are not semantic labels",
                "numpy payload is a lower bound, not process RAM",
                "development results require new sealed tasks before confirmation",
            ],
        },
        "runs": runs,
        "aggregate": aggregate,
        "verdict": "diagnostic_only; no superiority claim is inferred automatically",
    }

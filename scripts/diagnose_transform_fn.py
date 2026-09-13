#!/usr/bin/env python3
"""Diagnose the immediate readout gate for frozen transform-learning FNs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import text_factors.config as config_module  # noqa: E402
import text_factors.memory as memory_module  # noqa: E402
import text_factors.transforms as transforms_module  # noqa: E402
from text_factors.config import ModelConfig  # noqa: E402
from text_factors.memory import ClusterStatus  # noqa: E402
from text_factors.transforms import LearnedSDRTransform  # noqa: E402

DEFAULT_SNAPSHOT = REPO_ROOT / "docs/results/v03a2-transform-learning.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/results/v03a2-transform-fn-diagnosis.json"
CATEGORIES = ("no_cluster", "no_stable", "no_exact_match", "below_vote")
SOURCE_MODULES = {
    "config.py": config_module,
    "memory.py": memory_module,
    "transforms.py": transforms_module,
}


def bit_vector(indices: list[int], size: int) -> np.ndarray:
    value = np.zeros(size, dtype=np.bool_)
    value[indices] = True
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_imported_source(snapshot: dict[str, Any]) -> dict[str, str]:
    expected = snapshot["source"]["source_sha256"]
    verified: dict[str, str] = {}
    for relative_name, module in SOURCE_MODULES.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"imported {relative_name} has no source file")
        actual = file_sha256(Path(module_file).resolve())
        if actual != expected.get(relative_name):
            raise RuntimeError(
                f"source hash mismatch for {relative_name}: "
                f"expected={expected.get(relative_name)} actual={actual}"
            )
        verified[relative_name] = actual
    return verified


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def diagnose(snapshot_path: Path) -> dict[str, Any]:
    raw = snapshot_path.read_bytes()
    snapshot = json.loads(raw)

    # Refuse to train before proving that the imported implementation matches
    # the implementation recorded by the frozen experiment.
    verified_source = verify_imported_source(snapshot)
    epochs = int(snapshot["config"]["epochs"])
    expected_traces = sum(
        len(run["methods"]["factor"]["traces"]) for run in snapshot["runs"]
    )
    expected_fns = sum(
        len(set(trace["target_bits"]) - set(trace["predicted_bits"]))
        for run in snapshot["runs"]
        for trace in run["methods"]["factor"]["traces"]
    )
    category_counts: Counter[str] = Counter({name: 0 for name in CATEGORIES})
    runs_out: list[dict[str, Any]] = []
    matched_traces = 0
    diagnosed_fns = 0

    for run in snapshot["runs"]:
        config = ModelConfig.from_dict(run["model_config"])
        model = LearnedSDRTransform(config)

        # Factor branch order: epochs outside, then stored train-pair order.
        # The independent controls cannot affect this model.
        for _epoch in range(epochs):
            for pair in run["training_pairs"]:
                model.observe(
                    bit_vector(pair["source_bits"], config.input_bits),
                    bit_vector(pair["target_bits"], config.output_bits),
                )

        traces_out: list[dict[str, Any]] = []
        run_categories: Counter[str] = Counter({name: 0 for name in CATEGORIES})
        for trace_index, trace in enumerate(run["methods"]["factor"]["traces"]):
            source = bit_vector(trace["source_bits"], config.input_bits)
            readout = model.predict(source)
            actual = set(readout.active_output_bits)
            expected = set(trace["predicted_bits"])
            if actual != expected:
                raise RuntimeError(
                    "prediction mismatch before diagnosis: "
                    f"seed={run['seed']} mapping={run['mapping_index']} "
                    f"trace={trace_index} expected={sorted(expected)} "
                    f"actual={sorted(actual)}"
                )
            matched_traces += 1

            fn_details: list[dict[str, Any]] = []
            for output_bit in sorted(set(trace["target_bits"]) - actual):
                wired_points = np.flatnonzero(model.memory.output_map == output_bit)
                statuses: Counter[str] = Counter()
                total_clusters = 0
                stable_exact_count = 0
                stable_exact_points: set[int] = set()
                best_stable_match = 0
                best_stable_length = 0

                for raw_point in wired_points:
                    point = int(raw_point)
                    for cluster in model.memory.clusters[point]:
                        total_clusters += 1
                        statuses[cluster.status.name.lower()] += 1
                        if cluster.status == ClusterStatus.STABLE:
                            matched = model.memory._match_count(cluster, source)
                            if matched > best_stable_match:
                                best_stable_match = matched
                                best_stable_length = len(cluster.bits)
                            if matched == len(cluster.bits):
                                stable_exact_count += 1
                                stable_exact_points.add(point)

                stable_count = statuses["stable"]
                score = float(readout.output_scores[output_bit])
                if total_clusters == 0:
                    category = "no_cluster"
                elif stable_count == 0:
                    category = "no_stable"
                elif stable_exact_count == 0:
                    category = "no_exact_match"
                elif score < config.prediction_vote_threshold:
                    category = "below_vote"
                else:
                    raise RuntimeError(
                        "false negative passed every readout gate: "
                        f"seed={run['seed']} mapping={run['mapping_index']} "
                        f"trace={trace_index} bit={output_bit} score={score}"
                    )

                diagnosed_fns += 1
                category_counts[category] += 1
                run_categories[category] += 1
                fn_details.append(
                    {
                        "output_bit": output_bit,
                        "blocking_criterion": category,
                        "wired_point_count": int(len(wired_points)),
                        "cluster_count": total_clusters,
                        "cluster_count_by_status": {
                            name: statuses[name]
                            for name in ("temporary", "probation", "stable")
                        },
                        "stable_exact_cluster_count": stable_exact_count,
                        "stable_exact_point_count": len(stable_exact_points),
                        "best_stable_match": best_stable_match,
                        "best_stable_cluster_length": best_stable_length,
                        "raw_output_score": score,
                        "vote_threshold": config.prediction_vote_threshold,
                        "score_margin": score - config.prediction_vote_threshold,
                    }
                )
            traces_out.append(
                {
                    "trace_index": trace_index,
                    "case": trace["case"],
                    "fn_count": len(fn_details),
                    "false_negatives": fn_details,
                }
            )

        runs_out.append(
            {
                "seed": run["seed"],
                "mapping_index": run["mapping_index"],
                "model_config": run["model_config"],
                "training_epochs": epochs,
                "training_pairs_per_epoch": len(run["training_pairs"]),
                "prediction_traces_matched": len(traces_out),
                "fn_categories": dict(run_categories),
                "traces": traces_out,
            }
        )

    return {
        "type": "post-hoc diagnosis of a frozen experiment; not a new benchmark",
        "snapshot": relative_to_repo(snapshot_path),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "original_source": snapshot["source"],
        "verified_source_sha256": verified_source,
        "reconstruction": {
            "method": "factor only",
            "training_order": "epochs, then stored training_pairs order",
            "controls_reconstructed": False,
            "library_code_modified": False,
        },
        "validation": {
            "expected_prediction_traces": expected_traces,
            "matched_prediction_traces": matched_traces,
            "all_predictions_matched": matched_traces == expected_traces,
            "expected_false_negatives": expected_fns,
            "diagnosed_false_negatives": diagnosed_fns,
        },
        "fn_categories": dict(category_counts),
        "category_order": list(CATEGORIES),
        "interpretation": (
            "Each category is the nearest blocking criterion in the unchanged "
            "predict readout for a saved false negative."
        ),
        "interpretation_limit": (
            "The categories do not establish which global hyperparameter or "
            "learning mechanism caused the diagnosed state. In particular, "
            "per-output signature deduplication remains an untested architectural "
            "hypothesis."
        ),
        "runs": runs_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "validation": result["validation"],
                "fn_categories": result["fn_categories"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

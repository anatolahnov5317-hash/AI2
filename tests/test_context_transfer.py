import json
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from text_factors.encoder import SparseSymbolEncoder
from text_factors.evaluation.context_transfer import (
    ALPHABET,
    ContextTransferConfig,
    context_model_config,
    evaluate_context_predictor,
    make_context_dataset,
    run_context_transfer,
    sdr_trace_metrics,
)
from text_factors.model import TextFactorModel


class ContextTransferTests(unittest.TestCase):
    def small_config(self) -> ContextTransferConfig:
        return ContextTransferConfig(
            seeds=(7,), points=8, train_size=8, dev_size=4, test_size=4, epochs=3
        )

    def test_whole_strings_disjoint_and_primitives_seen_in_training(self) -> None:
        config = ContextTransferConfig()
        for seed in config.seeds:
            dataset = make_context_dataset(config, seed)
            combined = dataset.train + dataset.dev + dataset.test
            self.assertEqual(len(combined), len(set(combined)))
            self.assertEqual(len(dataset.train), 128)
            self.assertEqual(len(dataset.dev), 32)
            self.assertEqual(len(dataset.test), 64)
            primitives = {
                (symbol, position)
                for text in dataset.train
                for position, symbol in enumerate(text)
            }
            self.assertEqual(primitives, {(s, p) for s in ALPHABET for p in range(5)})
            self.assertEqual(dataset, make_context_dataset(config, seed))

    def test_pooled_metrics_penalize_sparse_zeros_and_false_positives(self) -> None:
        rows = [
            {"target_bits": [1, 2], "predicted_bits": [1, 3]},
            {"target_bits": [2, 3], "predicted_bits": [2, 3]},
        ]
        metrics = sdr_trace_metrics(rows, 256)
        self.assertEqual(metrics["bit_precision"], 0.75)
        self.assertEqual(metrics["bit_recall"], 0.75)
        self.assertEqual(metrics["bit_f1"], 0.75)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["false_positive_bits"], 1)
        zero = sdr_trace_metrics([{"target_bits": [1, 2], "predicted_bits": []}], 256)
        self.assertEqual(zero["bit_precision"], 0)
        self.assertEqual(zero["bit_recall"], 0)
        self.assertEqual(zero["bit_f1"], 0)
        self.assertEqual(zero["exact_match"], 0)

    def test_evaluator_predicts_before_constructing_target(self) -> None:
        encoder = SparseSymbolEncoder(
            context_model_config(self.small_config(), 7, "frequency"), ALPHABET
        )
        events: list[Any] = []
        original_encode = encoder.encode_window

        def encode(text, *, context=0, offset=0):
            events.append(("encode", context))
            return original_encode(text, context=context, offset=offset)

        def predict(source):
            events.append(("predict",))
            self.assertEqual(source.shape, (256,))
            self.assertFalse(source.flags.writeable)
            return np.zeros_like(source)

        with patch.object(encoder, "encode_window", side_effect=encode):
            result = evaluate_context_predictor(["abcda", "dcbaa"], encoder, predict)
        self.assertEqual(events, [("encode", 0), ("predict",), ("encode", 1)] * 2)
        self.assertEqual(result["metrics"]["cases"], 2)

    def test_held_out_prediction_does_not_modify_factor_evidence(self) -> None:
        config = self.small_config()
        model = TextFactorModel(
            context_model_config(config, 7, "coactivation"), alphabet=ALPHABET
        )
        dataset = make_context_dataset(config, 7)
        for text in dataset.train * config.epochs:
            model.learn_context_transform(text)

        def snapshot():
            return (
                model.memory.step,
                tuple(
                    (
                        point,
                        cluster.bits.tobytes(),
                        cluster.bit_hits.tobytes(),
                        cluster.status,
                        cluster.partial_hits,
                        cluster.exact_hits,
                        cluster.partial_errors,
                        cluster.complete_errors,
                        tuple(row.tobytes() for row in cluster.activation_history),
                    )
                    for point, cluster in model.memory.iter_clusters()
                ),
            )

        before = snapshot()
        evaluate_context_predictor(
            dataset.test,
            model.encoder,
            lambda source: model.memory.predict(source).output,
        )
        self.assertEqual(before, snapshot())

    def test_raw_traces_recompute_metrics_and_nearest_uses_training_only(self) -> None:
        report = run_context_transfer(self.small_config())
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "complete")
        run = report["runs"][0]
        methods = run["methods"]
        for method in methods.values():
            for split in ("dev", "test"):
                result = method[split]
                self.assertEqual(
                    result["metrics"], sdr_trace_metrics(result["traces"], 256)
                )
        for key in ("receptors_sha256", "output_map_sha256", "codebook_sha256"):
            self.assertEqual(methods["frequency"][key], methods["coactivation"][key])
        train = run["training_pairs"]
        for row in methods["nearest_source_sdr"]["test"]["traces"]:
            source = set(row["source_bits"])
            best = max(
                train,
                key=lambda pair: (
                    len(source & set(pair["source_bits"]))
                    / len(source | set(pair["source_bits"]))
                ),
            )
            self.assertEqual(row["predicted_bits"], best["target_bits"])
            self.assertNotIn(row["text"], run["data"]["train"])

    def test_counts_and_predictions_are_reproducible(self) -> None:
        first = run_context_transfer(self.small_config())
        second = run_context_transfer(self.small_config())
        self.assertEqual(first["aggregate"], second["aggregate"])
        a, b = first["runs"][0], second["runs"][0]
        self.assertEqual(a["data_sha256"], b["data_sha256"])
        for method, values in a["methods"].items():
            for split in ("dev", "test"):
                self.assertEqual(
                    values[split]["traces"], b["methods"][method][split]["traces"]
                )

    def test_timeout_remains_visible_without_invented_metrics(self) -> None:
        config = replace(self.small_config(), seeds=(7, 17), seconds_per_seed=0.5)
        with patch(
            "text_factors.evaluation.context_transfer.perf_counter",
            side_effect=range(1000),
        ):
            report = run_context_transfer(config)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual([run["status"] for run in report["runs"]], ["timed_out"] * 2)
        for method in report["aggregate"].values():
            self.assertFalse(method["all_seeds_complete"])
            self.assertEqual(method["completed_seeds"], [])
            self.assertIsNone(method["bit_f1"]["mean"])

    def test_invalid_configuration_rejected(self) -> None:
        for values in (
            {"seeds": ()},
            {"seeds": (7, 7)},
            {"points": False},
            {"epochs": 0},
            {"train_size": 3},
            {"test_size": 1024},
            {"seconds_per_seed": float("inf")},
        ):
            with self.assertRaises(ValueError):
                ContextTransferConfig(**cast(dict[str, Any], values))


if __name__ == "__main__":
    unittest.main()

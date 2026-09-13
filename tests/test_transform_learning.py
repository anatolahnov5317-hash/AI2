import json
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from text_factors.config import ModelConfig
from text_factors.evaluation.context_transfer import sdr_trace_metrics
from text_factors.evaluation.transform_learning import (
    BIT_COUNT,
    METHODS,
    POSITIONS,
    VALUES,
    PartEncoder,
    TransformLearningConfig,
    evaluate_transform_predictors,
    hidden_mappings,
    make_transform_dataset,
    run_transform_learning,
    transform_memory_config,
)
from text_factors.transforms import LearnedSDRTransform


class TransformLearningTests(unittest.TestCase):
    def small_config(self) -> TransformLearningConfig:
        return TransformLearningConfig(seeds=(7,), points=32, epochs=3)

    def test_given_partition_has_unseen_combinations_and_seen_parts(self) -> None:
        for seed in (7, 17, 42):
            data = make_transform_dataset(seed)
            self.assertEqual(len(data["train"]), 18)
            self.assertEqual(len(data["held_out"]), 9)
            self.assertFalse(set(data["train"]) & set(data["held_out"]))
            self.assertEqual(len(set(data["train"] + data["held_out"])), 27)
            train_parts = {
                (position, value)
                for case in data["train"]
                for position, value in enumerate(case)
            }
            self.assertEqual(
                train_parts,
                {(p, v) for p in range(POSITIONS) for v in range(VALUES)},
            )
            self.assertEqual(data, make_transform_dataset(seed))
            first, second = hidden_mappings(seed)
            self.assertNotEqual(first, second)
            for mapping in (first, second):
                outputs = {
                    mapping.apply(case) for case in data["train"] + data["held_out"]
                }
                self.assertEqual(len(outputs), 27)

    def test_views_are_independent_and_given_to_evaluator_only(self) -> None:
        source, target = PartEncoder(7, 0), PartEncoder(7, 1)
        self.assertFalse(np.array_equal(source.codebook, target.codebook))
        self.assertTrue(np.array_equal(source.codebook, PartEncoder(7, 0).codebook))
        model = LearnedSDRTransform(transform_memory_config(self.small_config(), 7))
        self.assertEqual(set(vars(model)), {"memory"})
        for case in make_transform_dataset(7)["train"]:
            self.assertEqual(int(np.count_nonzero(source.encode(case))), 18)

    def test_all_predictions_precede_target_and_receive_only_readonly_source(
        self,
    ) -> None:
        events: list[str] = []
        encoder = PartEncoder(7, 0)

        def source(case):
            events.append("source")
            return encoder.encode(case)

        def predict_first(bits):
            events.append("first")
            self.assertEqual(bits.shape, (BIT_COUNT,))
            self.assertFalse(bits.flags.writeable)
            return np.zeros(BIT_COUNT, dtype=np.bool_)

        def predict_second(bits):
            events.append("second")
            self.assertFalse(bits.flags.writeable)
            return bits.copy()

        def target(case):
            events.append("target")
            return encoder.encode(tuple(reversed(case)))

        result = evaluate_transform_predictors(
            [(0, 1, 2), (2, 1, 0)],
            source,
            target,
            {"first": predict_first, "second": predict_second},
        )
        self.assertEqual(events, ["source", "first", "second", "target"] * 2)
        self.assertEqual(result["first"]["metrics"]["bit_f1"], 0)

    def test_invalid_pairs_do_not_mutate_memory(self) -> None:
        model = LearnedSDRTransform(transform_memory_config(self.small_config(), 7))
        good = PartEncoder(7, 0).encode((0, 1, 2))
        bad_arrays = [
            np.zeros(BIT_COUNT, dtype=np.bool_),
            np.ones(BIT_COUNT - 1, dtype=np.bool_),
            np.full(BIT_COUNT, 2, dtype=np.int32),
            np.ones(BIT_COUNT, dtype=np.float64),
            np.full(BIT_COUNT, "1"),
        ]
        for bad in bad_arrays:
            with self.assertRaises(ValueError):
                model.observe(good, bad)
            with self.assertRaises(ValueError):
                model.observe(bad, good)
            self.assertEqual(model.memory.step, 0)
        model.observe(good.astype(np.int32), good.astype(np.int64))
        self.assertEqual(model.memory.step, 1)

    def test_prediction_does_not_change_factor_support(self) -> None:
        model = LearnedSDRTransform(transform_memory_config(self.small_config(), 7))
        source_encoder, target_encoder = PartEncoder(7, 0), PartEncoder(7, 1)
        data = make_transform_dataset(7)
        mapping = hidden_mappings(7)[0]
        for case in data["train"] * 3:
            model.observe(
                source_encoder.encode(case), target_encoder.encode(mapping.apply(case))
            )

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
                    )
                    for point, cluster in model.memory.iter_clusters()
                ),
            )

        before = snapshot()
        evaluate_transform_predictors(
            data["held_out"],
            source_encoder.encode,
            lambda case: target_encoder.encode(mapping.apply(case)),
            {"factor": lambda bits: model.predict(bits).output},
        )
        self.assertEqual(before, snapshot())

    def test_pair_learning_produces_target_from_stable_factors(self) -> None:
        config = ModelConfig(
            input_bits=32,
            output_bits=8,
            receptive_bits=16,
            point_count=64,
            activation_threshold=2,
            create_threshold=3,
            seed=7,
        )
        model = LearnedSDRTransform(config)
        source = np.zeros(32, dtype=np.bool_)
        source[:8] = True
        target = np.zeros(8, dtype=np.bool_)
        target[[1, 3]] = True
        self.assertFalse(np.any(model.predict(source).output))
        for _ in range(8):
            model.observe(source, target)
        np.testing.assert_array_equal(model.predict(source).output, target)

    def test_controls_traces_and_counts_are_auditable_and_reproducible(self) -> None:
        first = run_transform_learning(self.small_config())
        second = run_transform_learning(self.small_config())
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["completed_tasks"], 2)
        self.assertEqual(first["aggregate"], second["aggregate"])
        for run, repeat in zip(first["runs"], second["runs"], strict=True):
            self.assertEqual(run["data_sha256"], repeat["data_sha256"])
            pairs = run["training_pairs"]
            self.assertEqual(len(pairs), 18)
            self.assertEqual(run["training_calls_completed"], 108)
            coverage = run["coverage_diagnostics"]
            self.assertEqual(coverage["held_out_target_bits_absent_from_training"], [])
            self.assertEqual(coverage["missing_unseen_target_bit_occurrences"], 0)
            self.assertEqual(
                coverage["missing_known_target_bit_occurrences"],
                run["methods"]["factor"]["metrics"]["false_negative_bits"],
            )
            self.assertEqual(
                sorted(tuple(pair["target_bits"]) for pair in pairs),
                sorted(tuple(pair["shuffled_target_bits"]) for pair in pairs),
            )
            for index, pair in enumerate(pairs):
                self.assertNotEqual(pair["shuffled_target_from_train_index"], index)
                self.assertNotEqual(pair["target_bits"], pair["shuffled_target_bits"])
            for name in METHODS:
                method = run["methods"][name]
                self.assertEqual(
                    method["metrics"], sdr_trace_metrics(method["traces"], BIT_COUNT)
                )
                self.assertEqual(method["traces"], repeat["methods"][name]["traces"])
                state = run["memory_state"][name]
                expected_calls = 0 if name == "untrained" else 54
                self.assertEqual(state["observation_calls"], expected_calls)
                for key in ("receptors_sha256", "output_map_sha256"):
                    self.assertEqual(state[key], run["memory_state"]["factor"][key])
            self.assertEqual(run["methods"]["untrained"]["metrics"]["bit_f1"], 0)
            train_cases = {tuple(pair["case"]) for pair in pairs}
            self.assertTrue(
                all(
                    tuple(row["case"]) not in train_cases
                    for row in run["methods"]["factor"]["traces"]
                )
            )

    def test_timeout_emits_status_without_inventing_missing_metrics(self) -> None:
        events: list[dict[str, Any]] = []
        with patch(
            "text_factors.evaluation.transform_learning.perf_counter",
            side_effect=range(1000),
        ):
            report = run_transform_learning(
                replace(self.small_config(), seconds=0.5), progress=events.append
            )
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["completed_tasks"], 0)
        self.assertEqual(report["runs"][0]["status"], "timed_out")
        self.assertEqual(report["runs"][0]["methods"], {})
        self.assertTrue(any(event["stage"] == "timed_out" for event in events))
        for method in report["aggregate"].values():
            self.assertFalse(method["all_tasks_complete"])
            self.assertIsNone(method["bit_f1"]["mean"])

    def test_partial_evaluation_keeps_raw_traces_without_complete_metrics(self) -> None:
        def interrupt_after_first_case(event):
            if event["stage"] == "evaluation" and event["completed"] == 1:
                raise TimeoutError("simulated timeout between cases")

        report = run_transform_learning(
            self.small_config(), progress=interrupt_after_first_case
        )
        run = report["runs"][0]
        self.assertEqual(run["status"], "timed_out")
        self.assertEqual(run["stage"], "evaluation")
        self.assertEqual(run["methods"], {})
        self.assertEqual(set(run["partial_traces"]), set(METHODS))
        for rows in run["partial_traces"].values():
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["target_bits"]), 18)

    def test_invalid_configuration_is_rejected(self) -> None:
        for values in (
            {"seeds": ()},
            {"seeds": (7, 7)},
            {"points": False},
            {"epochs": 0},
            {"seconds": float("inf")},
            {"seconds": 0},
        ):
            with self.assertRaises(ValueError):
                TransformLearningConfig(**cast(dict[str, Any], values))


if __name__ == "__main__":
    unittest.main()

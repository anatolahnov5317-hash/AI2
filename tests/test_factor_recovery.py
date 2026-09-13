import unittest
from unittest.mock import patch

import numpy as np

from text_factors import ClusterStatus, ModelConfig
from text_factors.evaluation.factor_recovery import (
    evaluate_factor_recovery,
    fit_factor_memory,
    make_factor_dataset,
    matched_history_control,
    run_factor_recovery,
)
from text_factors.memory import Cluster, CombinatorialMemory


class FactorRecoveryTests(unittest.TestCase):
    def test_generator_separates_observations_and_evaluator_truth(self) -> None:
        data = make_factor_dataset(
            7, "multiple_clean", train_samples=32, test_samples=16
        )
        self.assertEqual(data.training_inputs.shape, (32, 128))
        self.assertEqual(data.training_presence.shape, (32, 4))
        expected = np.zeros_like(data.training_inputs)
        for index, factor in enumerate(data.factors):
            expected[:, factor] = data.training_presence[:, index, None]
        np.testing.assert_array_equal(data.training_inputs, expected)

        config = ModelConfig(point_count=1, min_active_points=1, input_bits=128)
        with patch.object(CombinatorialMemory, "observe", autospec=True) as observe:
            fit_factor_memory(data.training_inputs, config)
        self.assertEqual(observe.call_count, 32)
        for call, row in zip(observe.call_args_list, data.training_inputs, strict=True):
            # An accidental supervised target or latent-label argument fails.
            self.assertEqual(len(call.args), 2)
            self.assertEqual(call.kwargs, {})
            np.testing.assert_array_equal(call.args[1], row)

    def test_control_preserves_marginals_without_truth_labels(self) -> None:
        noisy = make_factor_dataset(
            17, "multiple_nuisance", train_samples=64, test_samples=32
        )
        control = make_factor_dataset(
            17, "marginal_shuffle_control", train_samples=64, test_samples=32
        )
        self.assertEqual(control.factors, ())
        self.assertEqual(control.testing_presence.shape, (32, 0))
        for original, shuffled in (
            (noisy.training_inputs, control.training_inputs),
            (noisy.testing_inputs, control.testing_inputs),
        ):
            np.testing.assert_array_equal(original.sum(axis=0), shuffled.sum(axis=0))
            self.assertFalse(np.array_equal(original, shuffled))

    def test_isolated_control_preserves_degrees_without_union(self) -> None:
        original, control = matched_history_control(42)
        np.testing.assert_array_equal(original.sum(axis=0), control.sum(axis=0))
        np.testing.assert_array_equal(original.sum(axis=1), control.sum(axis=1))
        self.assertFalse(
            np.array_equal(
                original.T @ original.astype(int), control.T @ control.astype(int)
            )
        )
        self.assertFalse(bool(np.any(np.all(original, axis=1))))

    def test_local_projection_and_global_coverage_are_distinct(self) -> None:
        config = ModelConfig(
            input_bits=8,
            active_bits_per_symbol=1,
            positions=2,
            frame_size=1,
            context_count=2,
            receptive_bits=4,
            point_count=1,
            output_bits=1,
            create_threshold=2,
            activation_threshold=2,
            min_active_points=1,
            probation_after=2,
            stable_after=3,
        )
        memory = CombinatorialMemory(
            config,
            receptors=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
            output_map=np.asarray([0], dtype=np.int32),
        )
        memory.clusters[0] = [
            Cluster(
                bits=np.asarray([0, 1], dtype=np.int32),
                bit_hits=np.asarray([3, 3], dtype=np.int64),
                created_at=0,
                last_seen=2,
                partial_hits=3,
                exact_hits=3,
                status=ClusterStatus.STABLE,
            )
        ]
        factors = (
            np.asarray([0, 1, 4, 5], dtype=np.int32),
            np.asarray([2, 6, 7], dtype=np.int32),
        )
        inputs = np.zeros((2, 8), dtype=np.bool_)
        inputs[0, factors[0]] = True
        inputs[1, factors[1]] = True
        presence = np.asarray([[True, False], [False, True]], dtype=np.bool_)
        result = evaluate_factor_recovery(memory, factors, inputs, presence)
        self.assertEqual(result["exact_local_fraction"], 1.0)
        self.assertEqual(result["best_local_f1_mean"], 1.0)
        self.assertEqual(result["global_factor_bit_recall_mean"], 0.5)
        self.assertEqual(result["observable_factors"], 1)
        self.assertEqual(result["heldout_selectivity_f1_mean"], 1.0)
        self.assertEqual(result["per_factor"][1]["observable_points"], 0)

        null_result = evaluate_factor_recovery(
            memory, (), inputs, np.zeros((2, 0), dtype=np.bool_)
        )
        self.assertEqual(null_result["unmatched_local_clusters"], 1)
        self.assertIsNone(null_result["global_factor_bit_recall_mean"])

    def test_suite_is_deterministic_and_resource_bounds_are_enforced(self) -> None:
        first = run_factor_recovery(
            (7,), train_samples=32, test_samples=16, point_count=4
        )
        second = run_factor_recovery(
            (7,), train_samples=32, test_samples=16, point_count=4
        )
        for result in (first, second):
            for run in result["runs"]:
                del run["fit_seconds"]
                del run["evaluation_seconds"]
            for seed in result["seed_status"]:
                del seed["elapsed_seconds"]
        self.assertEqual(first, second)
        self.assertEqual(len(first["runs"]), 8)
        self.assertEqual(len(first["datasets"]), 4)
        self.assertEqual(first["config"]["training_passes"], 1)
        with self.assertRaisesRegex(ValueError, "point_count"):
            run_factor_recovery((7,), point_count=129)
        with self.assertRaisesRegex(ValueError, "train_samples"):
            run_factor_recovery((7,), train_samples=257)
        with self.assertRaisesRegex(ValueError, "distinct"):
            run_factor_recovery((7, 7))

    def test_timeout_preserves_missingness_instead_of_inventing_results(self) -> None:
        result = run_factor_recovery((7,), point_count=1, seconds_per_seed=1e-12)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["seed_status"][0]["status"], "timed_out")
        self.assertEqual(result["runs"], [])
        for row in result["aggregate"]:
            self.assertEqual(row["completed_seeds"], [])
            self.assertFalse(row["all_seeds_complete"])
            self.assertIsNone(row["stable_clusters"])


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import replace
from typing import Any, cast

import numpy as np

from text_factors import ClusterStatus, ModelConfig
from text_factors.consolidation import coactivation_weights
from text_factors.memory import Cluster, CombinatorialMemory


def config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 8,
        "output_bits": 1,
        "point_count": 1,
        "receptive_bits": 5,
        "active_bits_per_symbol": 1,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 6,
        "stable_after": 10,
        "min_error_observations": 20,
        "consolidation_method": "coactivation",
    }
    values.update(overrides)
    return ModelConfig(**values)


def memory(settings: ModelConfig) -> CombinatorialMemory:
    return CombinatorialMemory(
        settings,
        receptors=np.arange(5, dtype=np.int32).reshape(1, 5),
        output_map=np.zeros(1, dtype=np.int32),
    )


def active(*bits: int) -> np.ndarray:
    result = np.zeros(8, dtype=np.bool_)
    result[list(bits)] = True
    return result


class ConsolidationTests(unittest.TestCase):
    def test_joint_filter_sees_information_missing_from_marginal_counts(self) -> None:
        # Isolated injected candidate, deliberately NOT an end-to-end discovery
        # fixture: a three-bit conjunction and a two-bit conjunction each occur
        # four times. All marginal frequencies are equal, but joint sizes differ.
        rows = [
            np.array(row, dtype=np.bool_)
            for row in ([[1, 1, 1, 0, 0]] * 4 + [[0, 0, 0, 1, 1]] * 4)
        ]
        candidate = Cluster(
            bits=np.arange(5, dtype=np.int32),
            bit_hits=np.full(5, 4, dtype=np.int64),
            created_at=0,
            last_seen=7,
            partial_hits=8,
            activation_history=rows,
        )
        marginal = replace(candidate, activation_history=[])
        self.assertFalse(
            memory(config(consolidation_method="frequency"))._prune(marginal)
        )
        self.assertTrue(memory(config())._prune(candidate))
        np.testing.assert_array_equal(candidate.bits, [0, 1, 2])
        self.assertTrue(all(row.shape == (3,) for row in candidate.activation_history))
        self.assertEqual(candidate.partial_hits, 8)

    def test_equal_strength_components_remain_a_mixture(self) -> None:
        rows = [
            np.array(row, dtype=np.bool_)
            for row in ([[1, 1, 0, 0]] * 8 + [[0, 0, 1, 1]] * 8)
        ]
        for ordered in (rows, list(reversed(rows)), rows[::2] + rows[1::2]):
            np.testing.assert_allclose(coactivation_weights(ordered), np.ones(4))

    def test_history_is_bounded_and_includes_negative_partial_matches(self) -> None:
        model = memory(config(coactivation_history_size=3))
        observations = [active(0, 1, 2, 3, 4), active(0, 1), active(2, 3), active(1, 4)]
        model.observe(observations[0], target=np.array([True]))
        for observation in observations[1:]:
            model.observe(observation, target=np.array([False]))
        cluster = model.clusters[0][0]
        self.assertEqual(cluster.partial_hits, 4)
        self.assertEqual(cluster.partial_errors, 3)
        self.assertEqual(len(cluster.activation_history), 3)
        for row, observation in zip(
            cluster.activation_history, observations[1:], strict=True
        ):
            np.testing.assert_array_equal(row, observation[cluster.bits])
        observations[-1][:] = False
        self.assertEqual(int(np.sum(cluster.activation_history[-1])), 2)

    def test_replay_never_promotes_one_observation_into_many_confirmations(
        self,
    ) -> None:
        model = memory(config())
        model.observe(active(0, 1, 2, 3, 4))
        cluster = model.clusters[0][0]
        for _ in range(12):
            model.replay_consolidation()
        self.assertEqual(model.step, 1)
        self.assertEqual(cluster.partial_hits, 1)
        self.assertEqual(cluster.exact_hits, 1)
        self.assertEqual(cluster.status, ClusterStatus.TEMPORARY)
        self.assertEqual(len(cluster.activation_history), 1)
        self.assertEqual(model.stats()["clusters"], 1)

    def test_frequency_mode_keeps_no_history(self) -> None:
        model = memory(config(consolidation_method="frequency"))
        for _ in range(12):
            model.observe(active(0, 1, 2, 3, 4))
        self.assertTrue(all(not c.activation_history for _, c in model.iter_clusters()))
        with self.assertRaisesRegex(ValueError, "coactivation"):
            model.replay_consolidation()

    def test_history_work_budget_is_validated(self) -> None:
        for changes in (
            {"coactivation_history_size": 0},
            {"coactivation_history_size": 257},
            {"coactivation_passes": 0},
            {"coactivation_passes": 17},
            {"coactivation_passes": True},
            {"consolidation_method": "unknown"},
            {"prune_keep_ratio": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)
        with self.assertRaises(ValueError):
            coactivation_weights([])
        for malformed in ([1], [np.array(1)], np.ones((2, 2), dtype=np.bool_)):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                coactivation_weights(cast(Any, malformed))


if __name__ == "__main__":
    unittest.main()

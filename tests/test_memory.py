import unittest
from typing import Any

import numpy as np

from text_factors import ClusterStatus, ModelConfig
from text_factors.memory import CombinatorialMemory


def memory_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 16,
        "active_bits_per_symbol": 2,
        "positions": 4,
        "frame_size": 2,
        "context_count": 4,
        "receptive_bits": 4,
        "point_count": 4,
        "output_bits": 2,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
        "prune_keep_ratio": 0.75,
        "max_clusters_per_point": 2,
        "min_error_observations": 2,
    }
    values.update(overrides)
    return ModelConfig(**values)


def controlled_memory(config: ModelConfig) -> CombinatorialMemory:
    receptors = np.asarray(
        [
            [0, 1, 2, 3],
            [0, 1, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12, 13],
        ],
        dtype=np.int32,
    )
    output_map = np.asarray([0, 1, 0, 1], dtype=np.int32)
    return CombinatorialMemory(
        config,
        receptors=receptors,
        output_map=output_map,
    )


class CombinatorialMemoryTests(unittest.TestCase):
    def test_cluster_requires_repeated_evidence_before_stabilizing(self) -> None:
        memory = controlled_memory(memory_config())
        active = np.zeros(16, dtype=np.bool_)
        active[[0, 1, 2, 3]] = True

        self.assertEqual(memory.observe(active), 1)
        cluster = memory.clusters[0][0]
        self.assertEqual(cluster.status, ClusterStatus.TEMPORARY)

        memory.observe(active)
        self.assertEqual(cluster.status, ClusterStatus.PROBATION)

        memory.observe(active)
        self.assertEqual(cluster.status, ClusterStatus.STABLE)
        readout = memory.read(active)
        self.assertEqual(readout.active_output_bits, [0])
        self.assertEqual(readout.quality, 4)

    def test_cluster_capacity_has_no_off_by_one(self) -> None:
        config = memory_config(max_clusters_per_point=1)
        memory = controlled_memory(config)
        first = np.zeros(16, dtype=np.bool_)
        first[[0, 1, 2]] = True
        second = np.zeros(16, dtype=np.bool_)
        second[[0, 1, 3]] = True

        memory.observe(first)
        memory.observe(second)
        self.assertEqual(len(memory.clusters[0]), 1)

    def test_full_receptive_overlap_is_safe(self) -> None:
        config = ModelConfig(
            input_bits=32,
            active_bits_per_symbol=1,
            positions=2,
            frame_size=1,
            context_count=2,
            receptive_bits=32,
            point_count=1,
            output_bits=1,
            create_threshold=32,
            activation_threshold=32,
            min_active_points=1,
            probation_after=2,
            stable_after=3,
        )
        memory = CombinatorialMemory(
            config,
            receptors=np.arange(32, dtype=np.int32).reshape(1, 32),
            output_map=np.asarray([0], dtype=np.int32),
        )
        active = np.ones(32, dtype=np.bool_)
        self.assertEqual(int(memory.overlap_counts(active)[0]), 32)
        self.assertEqual(memory.observe(active), 1)

    def test_error_rate_removes_bad_supervised_cluster(self) -> None:
        config = memory_config(
            max_complete_error_rate=0.0,
            max_partial_error_rate=0.0,
        )
        memory = controlled_memory(config)
        active = np.zeros(16, dtype=np.bool_)
        active[[0, 1, 2, 3]] = True
        positive = np.asarray([True, False], dtype=np.bool_)
        negative = np.asarray([False, True], dtype=np.bool_)

        memory.observe(active, target=positive)
        self.assertEqual(len(memory.clusters[0]), 1)
        memory.observe(active, target=negative)
        self.assertEqual(len(memory.clusters[0]), 0)


if __name__ == "__main__":
    unittest.main()

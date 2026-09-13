import unittest
from typing import Any, cast

import numpy as np

from text_factors import ClusterStatus, ModelConfig
from text_factors.memory import Cluster, CombinatorialMemory


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
    def test_supplied_index_arrays_must_be_integral_and_receptors_unique(self) -> None:
        config = memory_config()
        receptors = controlled_memory(config).receptors
        output_map = np.asarray([0, 1, 0, 1], dtype=np.int32)

        with self.assertRaisesRegex(ValueError, "integer"):
            CombinatorialMemory(
                config,
                receptors=cast(Any, receptors.astype(np.float64)),
                output_map=output_map,
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            CombinatorialMemory(
                config,
                receptors=receptors,
                output_map=cast(Any, output_map.astype(np.float64)),
            )
        duplicate = receptors.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            CombinatorialMemory(
                config,
                receptors=duplicate,
                output_map=output_map,
            )

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

    def test_loaded_cluster_requires_valid_structure_and_counters(self) -> None:
        def cluster(**overrides: Any) -> Cluster:
            values: dict[str, Any] = {
                "bits": np.asarray([0, 1], dtype=np.int32),
                "bit_hits": np.asarray([2, 1], dtype=np.int64),
                "created_at": 0,
                "last_seen": 1,
                "status": ClusterStatus.TEMPORARY,
                "partial_hits": 2,
                "exact_hits": 1,
                "partial_errors": 1,
                "complete_errors": 0,
            }
            values.update(overrides)
            return Cluster(**values)

        invalid_clusters = [
            cluster(bits=np.asarray([1, 0], dtype=np.int32)),
            cluster(bits=np.asarray([0, 0], dtype=np.int32)),
            cluster(bits=np.asarray([0, 6], dtype=np.int32)),
            cluster(bits=np.asarray([0.0, 1.0])),
            cluster(bit_hits=np.asarray([1.0, 1.0])),
            cluster(partial_hits=0),
            cluster(exact_hits=3),
            cluster(partial_errors=-1),
            cluster(partial_errors=3),
            cluster(complete_errors=2),
            cluster(bit_hits=np.asarray([3, 1], dtype=np.int64)),
            cluster(created_at=2),
            cluster(last_seen=2),
            cluster(status=3),
        ]
        for invalid in invalid_clusters:
            with self.subTest(cluster=invalid):
                memory = controlled_memory(memory_config())
                memory.step = 1
                with self.assertRaises(ValueError):
                    memory.add_loaded_cluster(0, invalid)

        memory = controlled_memory(memory_config())
        memory.step = 1
        memory.add_loaded_cluster(0, cluster())
        self.assertEqual(len(memory.clusters[0]), 1)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from text_factors.config import ModelConfig
from text_factors.evaluation.transform_diagnostics import diagnose_transform_bits
from text_factors.memory import Cluster, ClusterStatus
from text_factors.transforms import LearnedSDRTransform


class TransformDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            input_bits=8,
            active_bits_per_symbol=1,
            positions=2,
            frame_size=1,
            context_count=2,
            receptive_bits=4,
            point_count=4,
            output_bits=4,
            create_threshold=2,
            activation_threshold=2,
            min_active_points=1,
            probation_after=2,
            stable_after=3,
            prediction_vote_threshold=3,
        )
        self.model = LearnedSDRTransform(self.config)
        self.model.memory.receptors[:] = np.asarray([[0, 1, 2, 7]] * 4, dtype=np.int32)
        self.model.memory.output_map[:] = np.arange(4, dtype=np.int32)
        self.model.memory.step = 3
        self.source = self.bits([0, 1, 2], 8)

    @staticmethod
    def bits(active: list[int], size: int) -> np.ndarray:
        result = np.zeros(size, dtype=np.bool_)
        result[active] = True
        return result

    def add_cluster(
        self, point: int, bits: list[int], status: ClusterStatus, *, exact_hits: int = 3
    ) -> None:
        cluster = Cluster(
            bits=np.asarray(bits, dtype=np.int32),
            bit_hits=np.full(len(bits), exact_hits, dtype=np.int64),
            created_at=0,
            last_seen=2,
            status=status,
            partial_hits=exact_hits,
            exact_hits=exact_hits,
        )
        self.model.memory.add_loaded_cluster(point, cluster)

    def snapshot(self) -> tuple[object, ...]:
        return (
            self.model.memory.step,
            self.model.memory.output_map.tobytes(),
            tuple(
                (
                    point,
                    cluster.bits.tobytes(),
                    cluster.bit_hits.tobytes(),
                    cluster.status,
                    cluster.partial_hits,
                    cluster.exact_hits,
                )
                for point, cluster in self.model.memory.iter_clusters()
            ),
        )

    def test_each_missing_bit_has_the_nearest_hand_built_gate(self) -> None:
        # bit 0 has no cluster; bit 1 only a temporary cluster; bit 2 has a
        # stable non-match; bit 3 has an exact stable vote of 2 below threshold 3.
        self.add_cluster(1, [0, 1], ClusterStatus.TEMPORARY)
        self.add_cluster(2, [0, 7], ClusterStatus.STABLE)
        self.add_cluster(3, [0, 1, 2], ClusterStatus.STABLE, exact_hits=5)
        target = np.ones(4, dtype=np.bool_)
        prediction = self.model.predict(self.source).output
        before = self.snapshot()

        result = diagnose_transform_bits(
            self.model, self.source, target, predicted=prediction
        )

        self.assertEqual(before, self.snapshot())
        self.assertEqual(
            [row["gate"] for row in result["missing_bits"]],
            ["no_cluster", "no_stable", "no_exact", "below_vote"],
        )
        self.assertEqual(result["counts"]["false_negative"], 4)
        self.assertEqual(result["counts"]["false_positive"], 0)
        self.assertEqual(result["counts"]["no_cluster"], 1)
        self.assertEqual(result["counts"]["no_stable"], 1)
        self.assertEqual(result["counts"]["no_exact"], 1)
        self.assertEqual(result["counts"]["below_vote"], 1)
        below = result["missing_bits"][3]
        self.assertEqual(below["weighted_score"], 2.0)
        self.assertEqual(below["observed_output_support"], 5)
        self.assertEqual(below["wired_point_count"], 1)

    def test_weighted_scores_reproduce_prediction_and_mask_scopes_counts(self) -> None:
        # Two exact clusters at one point sum to three votes for output bit 0.
        self.add_cluster(0, [0, 1, 2], ClusterStatus.STABLE)
        self.add_cluster(0, [0, 1], ClusterStatus.STABLE)
        prediction = self.model.predict(self.source).output
        self.assertTrue(prediction[0])
        full_target = self.bits([1, 2], 4)

        result = diagnose_transform_bits(
            self.model,
            self.source,
            full_target,
            predicted=prediction,
            evaluated_bits=np.asarray([1, 2], dtype=np.int32),
        )

        # Output 0 can be correct for a neighbouring object and is outside this
        # focal evaluation, so it is not a focal false positive.
        self.assertEqual(result["counts"]["false_positive"], 0)
        self.assertEqual(result["counts"]["false_negative"], 2)
        self.assertEqual(result["counts"]["evaluated_bits"], 2)

    def test_inconsistent_supplied_prediction_is_rejected(self) -> None:
        self.add_cluster(0, [0, 1, 2], ClusterStatus.STABLE)
        self.add_cluster(0, [0, 1], ClusterStatus.STABLE)
        wrong = np.zeros(4, dtype=np.bool_)
        with self.assertRaisesRegex(AssertionError, "unchanged transform readout"):
            diagnose_transform_bits(
                self.model, self.source, self.bits([1], 4), predicted=wrong
            )

    def test_arrays_and_cancellation_are_strictly_validated(self) -> None:
        target = self.bits([0], 4)
        for bad in (
            np.ones(7, dtype=np.bool_),
            np.full(8, 2, dtype=np.int32),
            np.ones(8, dtype=np.float64),
        ):
            with self.assertRaises(ValueError):
                diagnose_transform_bits(self.model, bad, target)
        with self.assertRaises(ValueError):
            diagnose_transform_bits(
                self.model, self.source, target, evaluated_bits=np.asarray([0, 0])
            )
        with self.assertRaises(InterruptedError):
            diagnose_transform_bits(
                self.model, self.source, target, cancelled=lambda: True
            )


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from dataclasses import replace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from text_factors.config import ModelConfig
from text_factors.dialogue import GroundedDialogue
from text_factors.memory import ClusterStatus, CombinatorialMemory
from text_factors.recognition import ContextView, RecognitionLimits
from text_factors.scene_recognition import FactorSceneReader, SceneRecognitionConfig


def bits(*positions: int) -> np.ndarray:
    result = np.zeros(16, dtype=np.bool_)
    result[list(positions)] = True
    return result


def trained_memory() -> CombinatorialMemory:
    config = ModelConfig(
        input_bits=16,
        active_bits_per_symbol=2,
        positions=2,
        frame_size=2,
        context_count=2,
        receptive_bits=4,
        point_count=4,
        output_bits=4,
        create_threshold=2,
        activation_threshold=2,
        min_active_points=1,
        probation_after=2,
        stable_after=3,
        max_clusters_per_point=8,
        seed=17,
    )
    memory = CombinatorialMemory(
        config,
        receptors=np.asarray(
            [[0, 1, 2, 12], [1, 2, 3, 13], [4, 5, 6, 14], [5, 6, 7, 15]],
            dtype=np.int32,
        ),
        output_map=np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    for _ in range(3):
        memory.observe(bits(0, 1, 2, 3))
        memory.observe(bits(4, 5, 7))
    return memory


def reader() -> FactorSceneReader:
    value = FactorSceneReader(trained_memory())
    assert value.observe_portrait("opaque-a", bits(0, 1, 2, 3))
    assert value.observe_portrait("opaque-b", bits(4, 5, 7))
    return value


def snapshot(memory: CombinatorialMemory) -> tuple[Any, ...]:
    return (
        memory.step,
        memory.stats(),
        memory.receptors.tobytes(),
        memory.output_map.tobytes(),
        tuple(
            (
                point,
                cluster.signature,
                cluster.bit_hits.tobytes(),
                cluster.created_at,
                cluster.last_seen,
                cluster.status,
                cluster.partial_hits,
                cluster.exact_hits,
                cluster.partial_errors,
                cluster.complete_errors,
                tuple(row.tobytes() for row in cluster.activation_history),
            )
            for point, cluster in memory.iter_clusters()
        ),
    )


class SceneRecognitionTests(unittest.TestCase):
    def test_whole_scene_keeps_weak_part_beside_strong_part(self) -> None:
        value = reader()
        result = value.recognize_views(
            [ContextView("whole", "ctx", bits(0, 1, 2, 3, 4, 5, 7), ())],
            total_views=1,
        )

        self.assertTrue(result.complete)
        self.assertEqual(
            {item.portrait_id for item in result.proposals}, {"opaque-a", "opaque-b"}
        )
        self.assertEqual([item.coverage for item in result.proposals], [1.0, 1.0])
        self.assertEqual([item.support for item in result.proposals], [6, 4])
        self.assertEqual(result.view_traces[0].unexplained_bits, ())
        self.assertEqual(result.recognition.relations[0].kind, "compatible")
        self.assertTrue(
            all(not item.source_positions for item in result.recognition.candidates)
        )
        self.assertTrue(all(not item.ambiguous for item in result.proposals))

    def test_unexplained_residual_does_not_become_named_part(self) -> None:
        value = reader()
        result = value.recognize_views(
            [ContextView("whole", "ctx", bits(0, 1, 2, 3, 8, 9), ())],
            total_views=1,
        )

        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].portrait_id, "opaque-a")
        self.assertEqual(result.proposals[0].support_bits, (0, 1, 2, 3))
        self.assertEqual(result.view_traces[0].unexplained_bits, (8, 9))
        self.assertEqual(
            {
                b
                for e in result.recognition.candidates[0].evidence
                for b in e.matched_bits
            },
            {0, 1, 2, 3},
        )

    def test_missing_bits_change_content_key_and_do_not_inflate_coverage(self) -> None:
        value = reader()
        full = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2, 3), ())], total_views=1
        )
        partial = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2), ())], total_views=1
        )

        self.assertEqual(partial.proposals[0].support, 5)
        self.assertAlmostEqual(partial.proposals[0].coverage, 5 / 6)
        self.assertNotEqual(
            full.recognition.candidates[0].content_key,
            partial.recognition.candidates[0].content_key,
        )
        self.assertEqual(
            full.recognition.candidates[0].evidence[1].signature,
            partial.recognition.candidates[0].evidence[1].signature,
        )

    def test_exact_part_key_is_separate_from_whole_scene_residual(self) -> None:
        value = reader()
        single = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2, 3), ())], total_views=1
        )
        mixed = value.recognize_views(
            [ContextView("ab", "ctx", bits(0, 1, 2, 3, 4, 5, 7, 9), ())], total_views=1
        )
        self.assertEqual(
            single.recognition.candidates[0].content_key,
            mixed.recognition.candidates[0].content_key,
        )
        dialogue = GroundedDialogue(value.encoding_id, output_width=4)
        remembered = dialogue.remember_recognition(
            "mixture", mixed.recognition, encoding_id=value.encoding_id
        )
        self.assertTrue(remembered)
        self.assertEqual(len(dialogue.to_dict()["references"][0]["candidates"]), 2)
        self.assertEqual(mixed.view_traces[0].unexplained_bits, (9,))

    def test_overlapping_portraits_remain_explicitly_ambiguous(self) -> None:
        value = reader()
        self.assertTrue(value.observe_portrait("opaque-overlap", bits(0, 1, 2)))
        result = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2, 3), ())], total_views=1
        )

        self.assertEqual(len(result.recognition.candidates), 2)
        self.assertTrue(all(item.ambiguous for item in result.proposals))
        self.assertTrue(
            all(len(item.overlapping_candidates) == 1 for item in result.proposals)
        )
        self.assertEqual(result.recognition.relations[0].kind, "undetermined")
        self.assertEqual(result.recognition.suppressed, ())

    def test_shared_raw_bit_at_different_points_is_not_factor_overlap(self) -> None:
        memory = CombinatorialMemory(
            trained_memory().config,
            receptors=np.asarray(
                [[0, 1, 2, 12], [0, 2, 3, 13], [0, 4, 5, 14], [0, 5, 6, 15]],
                dtype=np.int32,
            ),
            output_map=np.asarray([0, 1, 2, 3], dtype=np.int32),
        )
        for _ in range(3):
            memory.observe(bits(0, 1, 2, 3))
            memory.observe(bits(0, 4, 5, 6))
        value = FactorSceneReader(memory)
        self.assertTrue(value.observe_portrait("a", bits(0, 1, 2, 3)))
        self.assertTrue(value.observe_portrait("b", bits(0, 4, 5, 6)))
        result = value.recognize_views(
            [ContextView("whole", "ctx", bits(0, 1, 2, 3, 4, 5, 6), ())],
            total_views=1,
        )
        self.assertTrue(result.complete)
        self.assertEqual(len(result.proposals), 2)
        self.assertTrue(all(0 in item.support_bits for item in result.proposals))
        self.assertTrue(all(not item.ambiguous for item in result.proposals))
        self.assertEqual(result.recognition.relations[0].kind, "compatible")
        self.assertEqual(result.view_traces[0].supported_bits, (0, 1, 2, 3, 4, 5, 6))

    def test_same_raw_bits_do_not_replace_changed_learned_conjunction(self) -> None:
        value = reader()
        for _, cluster in value.memory.iter_clusters():
            if cluster.signature == (0, 1, 2):
                cluster.bits = np.asarray([0, 1, 2, 12], dtype=np.int32)
                break
        result = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2, 3), ())], total_views=1
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.view_traces[0].unexplained_bits, (0, 1, 2, 3))

    def test_shared_conjunction_must_reach_activation_threshold(self) -> None:
        value = reader()
        result = value.recognize_views(
            [ContextView("fragments", "ctx", bits(0, 3), ())], total_views=1
        )
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.view_traces[0].unexplained_bits, (0, 3))

    def test_queries_do_not_learn_or_retain_mutable_input_arrays(self) -> None:
        value = reader()
        memory_before = snapshot(value.memory)
        portraits_before = value.to_dict()
        supplied = bits(0, 1, 2, 3, 4, 5, 7)
        result = value.recognize_views(
            [ContextView("whole", "ctx", supplied, ())],
            total_views=1,
            progress=lambda _: supplied.fill(False),
        )
        self.assertEqual(snapshot(value.memory), memory_before)
        self.assertEqual(value.to_dict(), portraits_before)
        self.assertEqual(result.view_traces[0].active_bits, (0, 1, 2, 3, 4, 5, 7))
        json.dumps(result.to_dict())
        json.dumps(value.to_dict())

    def test_progress_training_aborts_before_mixing_memory_states(self) -> None:
        for total in (1, 2):
            with self.subTest(total_views=total):
                value = reader()
                original_step = value.memory.step
                portraits = value.to_dict()
                views = [
                    ContextView("a", "ctx", bits(0, 1, 2, 3), ()),
                    ContextView("b", "ctx", bits(4, 5, 7), ()),
                ]

                def train(_: str, memory: CombinatorialMemory = value.memory) -> None:
                    memory.observe(bits(0, 1, 2, 3))

                result = value.recognize_views(
                    views[:total],
                    total_views=total,
                    progress=train,
                )
                self.assertFalse(result.complete)
                self.assertEqual(result.stop_reason, "memory_changed")
                self.assertEqual(result.recognition.examined_views, 1)
                self.assertEqual(result.recognition.memory_step, original_step)
                self.assertEqual(value.memory.step, original_step + 1)
                self.assertEqual(len(result.proposals), 1)
                self.assertEqual(result.proposals[0].portrait_id, "opaque-a")
                self.assertEqual(value.to_dict(), portraits)

    def test_registration_is_read_only_idempotent_and_owns_evidence(self) -> None:
        memory = trained_memory()
        value = FactorSceneReader(memory)
        before = snapshot(memory)
        supplied = bits(0, 1, 2, 3)
        self.assertTrue(value.observe_portrait("opaque", supplied))
        registered = value.to_dict()
        supplied.fill(False)
        self.assertTrue(value.observe_portrait("opaque", bits(0, 1, 2, 3)))
        self.assertEqual(value.to_dict(), registered)
        self.assertEqual(snapshot(memory), before)
        with self.assertRaises(ValueError):
            value.observe_portrait("opaque", bits(4, 5, 7))
        self.assertEqual(value.to_dict(), registered)

    def test_insufficient_and_unstable_evidence_is_not_registered(self) -> None:
        memory = trained_memory()
        value = FactorSceneReader(memory)
        self.assertFalse(value.observe_portrait("none", bits(8, 9)))
        for _, cluster in memory.iter_clusters():
            cluster.status = ClusterStatus.PROBATION
        self.assertFalse(value.observe_portrait("unstable", bits(0, 1, 2, 3)))
        self.assertEqual(value.portraits, ())

    def test_coordinate_replacement_is_rejected_with_explicit_encoding_id(self) -> None:
        value = FactorSceneReader(trained_memory(), encoding_id="declared")
        self.assertTrue(value.observe_portrait("a", bits(0, 1, 2, 3)))
        value.memory.output_map[0] = 1
        with self.assertRaisesRegex(ValueError, "coordinate encoding changed"):
            value.recognize_views([], total_views=0)
        with self.assertRaisesRegex(ValueError, "coordinate encoding changed"):
            value.observe_portrait("b", bits(4, 5, 7))

    def test_between_call_learning_does_not_invalidate_coordinates(self) -> None:
        value = reader()
        before = value.to_dict()
        value.memory.observe(bits(0, 1, 2, 3))
        result = value.recognize_views(
            [ContextView("a", "ctx", bits(0, 1, 2, 3), ())], total_views=1
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.proposals[0].portrait_id, "opaque-a")
        self.assertEqual(value.to_dict(), before)

    def test_cancelled_registration_is_atomic(self) -> None:
        value = FactorSceneReader(trained_memory())
        before = value.to_dict()
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            value.observe_portrait("a", bits(0, 1, 2, 3), cancelled=lambda: True)
        self.assertEqual(value.to_dict(), before)

    def test_cancelled_query_and_complete_prefix_are_visible(self) -> None:
        value = reader()
        views = [
            ContextView("a", "ctx", bits(0, 1, 2, 3), ()),
            ContextView("b", "ctx", bits(4, 5, 7), ()),
        ]
        cancelled = value.recognize_views(views, total_views=2, cancelled=lambda: True)
        self.assertFalse(cancelled.complete)
        self.assertEqual(cancelled.stop_reason, "cancelled")
        self.assertEqual(cancelled.recognition.examined_views, 0)
        stop = False

        def progress(_: str) -> None:
            nonlocal stop
            stop = True

        partial = value.recognize_views(
            views, total_views=2, progress=progress, cancelled=lambda: stop
        )
        self.assertFalse(partial.complete)
        self.assertEqual(partial.recognition.examined_views, 1)
        self.assertEqual(len(partial.proposals), 1)
        self.assertTrue(partial.view_traces[0].complete)

    def test_candidate_budget_does_not_present_partial_view_as_complete(self) -> None:
        value = reader()
        result = value.recognize_views(
            [ContextView("whole", "ctx", bits(0, 1, 2, 3, 4, 5, 7), ())],
            total_views=1,
            limits=RecognitionLimits(max_candidates=1),
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.stop_reason, "candidate_limit")
        self.assertEqual(result.recognition.examined_views, 0)
        self.assertEqual(result.proposals, ())
        self.assertFalse(result.view_traces[0].complete)

    def test_work_and_evidence_limits_abort_atomically(self) -> None:
        value = reader()
        view = ContextView("whole", "ctx", bits(0, 1, 2, 3, 4, 5, 7), ())
        evidence_limited = value.recognize_views(
            [view], total_views=1, limits=RecognitionLimits(max_evidence=1)
        )
        self.assertEqual(evidence_limited.stop_reason, "evidence_limit")
        self.assertEqual(evidence_limited.proposals, ())
        value.config = replace(value.config, max_atom_visits=1)
        work_limited = value.recognize_views([view], total_views=1)
        self.assertEqual(work_limited.stop_reason, "atom_visit_limit")
        self.assertEqual(work_limited.proposals, ())

    def test_portrait_capacity_and_time_budget_are_bounded(self) -> None:
        value = FactorSceneReader(
            trained_memory(), config=SceneRecognitionConfig(max_portraits=1)
        )
        self.assertTrue(value.observe_portrait("a", bits(0, 1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "portrait capacity"):
            value.observe_portrait("b", bits(4, 5, 7))
        with patch(
            "text_factors.scene_recognition.perf_counter", side_effect=[0.0, 6.0, 6.0]
        ):
            result = value.recognize_views([], total_views=0)
        self.assertEqual(result.stop_reason, "time_budget")
        self.assertFalse(result.complete)

    def test_two_instances_require_distinct_encoded_support(self) -> None:
        value = reader()
        result = value.recognize_views(
            [ContextView("same-code-twice", "ctx", bits(0, 1, 2, 3), (0, 10))],
            total_views=1,
        )
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.recognition.candidates[0].source_positions, (0, 10))

    def test_input_and_configuration_validation(self) -> None:
        value = reader()
        with self.assertRaises(ValueError):
            value.observe_portrait("a", cast(Any, np.zeros(16, dtype=np.int32)))
        with self.assertRaises(ValueError):
            value.recognize_views([], total_views=1)
        with self.assertRaises(ValueError):
            value.recognize_views([], total_views=True)
        invalid_configs: tuple[dict[str, Any], ...] = (
            {"coverage": 0},
            {"coverage": float("nan")},
            {"seconds": 0},
            {"min_atoms": 1, "min_points": 2},
            {"max_portraits": True},
        )
        for kwargs in invalid_configs:
            with self.assertRaises(ValueError):
                SceneRecognitionConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import numpy as np

from text_factors.config import ModelConfig
from text_factors.context_pipeline import LearnedContextPipeline
from text_factors.memory import CombinatorialMemory
from text_factors.recognition import RecognitionLimits, recognize_views
from text_factors.transforms import LearnedSDRTransform


def bits(*indices: int) -> np.ndarray:
    value = np.zeros(8, dtype=np.bool_)
    value[list(indices)] = True
    return value


def config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = dict(
        input_bits=8,
        output_bits=8,
        active_bits_per_symbol=2,
        positions=2,
        frame_size=2,
        context_count=1,
        receptive_bits=8,
        point_count=64,
        create_threshold=2,
        activation_threshold=2,
        min_active_points=1,
        probation_after=2,
        stable_after=3,
        prediction_vote_threshold=1,
        max_clusters_per_point=4,
        seed=11,
    )
    values.update(overrides)
    return ModelConfig(**values)


def snapshot(memory: CombinatorialMemory) -> tuple[Any, ...]:
    return (
        memory.step,
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


class ContextPipelineTests(unittest.TestCase):
    def fixture(self):
        source, first, second = bits(0, 1), bits(2, 3), bits(4, 5)
        left, right = LearnedSDRTransform(config()), LearnedSDRTransform(config())
        for _ in range(4):
            left.observe(source, first)
            right.observe(source, second)
        common = CombinatorialMemory(
            config(receptive_bits=2, point_count=2, output_bits=2),
            receptors=np.asarray([[2, 3], [4, 5]], dtype=np.int32),
            output_map=np.asarray([0, 1], dtype=np.int32),
        )
        for _ in range(4):
            common.observe(first)
            common.observe(second)
        return source, common, {"left": left, "right": right}

    def test_real_learned_predictions_flow_to_one_common_memory(self) -> None:
        source, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common,
            transforms,
            input_encoding_id="sensor-v1",
            memory_namespace="common-v1",
        )
        with patch(
            "text_factors.context_pipeline.recognize_views", wraps=recognize_views
        ) as read:
            result = pipeline.recognize(
                source, observation_id="scene", source_positions=(5, 9)
            )
        self.assertEqual(read.call_count, 1)
        self.assertIs(read.call_args.args[0], common)
        self.assertTrue(result.complete)
        self.assertEqual(result.recognition.examined_views, 2)
        self.assertEqual(result.recognition.total_views, 2)
        self.assertEqual(len(result.recognition.responses), 2)
        self.assertTrue(all(r.active for r in result.recognition.responses))
        self.assertEqual(
            {r.context_id for r in result.recognition.responses}, {"left", "right"}
        )
        self.assertEqual(
            {trace.predicted_bits for trace in result.transforms}, {(2, 3), (4, 5)}
        )
        self.assertEqual(
            {c.output_bits for c in result.recognition.candidates}, {(0,), (1,)}
        )
        self.assertEqual(
            {c.context_id for c in result.recognition.candidates}, {"left", "right"}
        )
        self.assertTrue(
            all(c.source_positions == (5, 9) for c in result.recognition.candidates)
        )
        self.assertTrue(
            all(
                t.forwarded and t.origin == "learned_transform"
                for t in result.transforms
            )
        )
        self.assertEqual(result.recognition.encoding_id, "common-v1")
        json.dumps(result.to_dict(), allow_nan=False)

    def test_empty_prediction_does_not_fall_back_to_recognizable_source(self) -> None:
        _, common, _ = self.fixture()
        pipeline = LearnedContextPipeline(
            common,
            {"untrained": LearnedSDRTransform(config())},
            input_encoding_id="sensor-v1",
        )
        self.assertTrue(common.read(bits(2, 3)).active_output_bits)
        result = pipeline.recognize(bits(2, 3))
        self.assertTrue(result.complete)
        self.assertEqual(result.transforms[0].predicted_bits, ())
        self.assertEqual(result.recognition.candidates, ())
        self.assertEqual(len(result.recognition.responses), 1)
        self.assertFalse(result.recognition.responses[0].active)
        self.assertEqual(result.recognition.responses[0].score, 0.0)

    def test_predictions_and_readouts_do_not_mutate_any_memory(self) -> None:
        source, common, transforms = self.fixture()
        memories = (common, *(t.memory for t in transforms.values()))
        before = tuple(snapshot(memory) for memory in memories)
        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        pipeline.recognize(source)
        pipeline.recognize(source)
        self.assertEqual(tuple(snapshot(memory) for memory in memories), before)

    def test_response_input_digest_excludes_observation_name_and_scope(self) -> None:
        source, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        first = pipeline.recognize(
            source, observation_id="first", source_positions=(0,)
        )
        renamed = pipeline.recognize(
            source, observation_id="renamed", source_positions=(9,)
        )
        self.assertEqual(
            [r.input_digest for r in first.recognition.responses],
            [r.input_digest for r in renamed.recognition.responses],
        )
        self.assertNotEqual(
            first.recognition.responses[0].observation_id,
            renamed.recognition.responses[0].observation_id,
        )

    def test_source_copy_is_readonly_and_isolated_from_caller_changes(self) -> None:
        source, common, transforms = self.fixture()
        left = transforms["left"]
        original = left.predict
        received: list[np.ndarray] = []

        def predict(value):
            self.assertFalse(value.flags.writeable)
            self.assertFalse(np.shares_memory(value, source))
            received.append(value.copy())
            return original(value)

        def progress(_):
            source[:] = False

        pipeline = LearnedContextPipeline(
            common, {"left": left}, input_encoding_id="sensor-v1"
        )
        with patch.object(left, "predict", side_effect=predict):
            result = pipeline.recognize(source, progress=progress)
        np.testing.assert_array_equal(received[0], bits(0, 1))
        self.assertEqual(result.transforms[0].predicted_bits, (2, 3))

    def test_catalogue_snapshot_and_width_validation(self) -> None:
        source, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        transforms.clear()
        self.assertEqual(pipeline.recognize(source).recognition.total_views, 2)
        for bad in ({}, {"bad": LearnedSDRTransform(config(output_bits=7))}):
            with self.assertRaises(ValueError):
                LearnedContextPipeline(common, bad, input_encoding_id="sensor-v1")
        with self.assertRaises(ValueError):
            LearnedContextPipeline(
                common,
                {
                    "a": LearnedSDRTransform(config()),
                    "b": LearnedSDRTransform(config(input_bits=9)),
                },
                input_encoding_id="sensor-v1",
            )

    def test_invalid_source_and_provenance_rejected_before_prediction(self) -> None:
        _, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        for source in (
            np.ones(7, dtype=bool),
            np.zeros(8, dtype=bool),
            np.ones(8, dtype=float),
            np.full(8, 2, dtype=int),
        ):
            with self.assertRaises(ValueError):
                pipeline.recognize(source)
        for positions in ((1, 1), (2, 1), (-1,), (True,)):
            with self.assertRaises(ValueError):
                pipeline.recognize(bits(0, 1), source_positions=positions)
        self.assertTrue(pipeline.recognize(bits(0, 1).astype(np.int32)).complete)

    def test_view_limit_prevents_unused_transform_prediction(self) -> None:
        source, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common,
            transforms,
            input_encoding_id="sensor-v1",
            limits=RecognitionLimits(max_views=1),
        )
        with patch.object(
            transforms["right"], "predict", side_effect=AssertionError("not needed")
        ):
            result = pipeline.recognize(source)
        self.assertFalse(result.complete)
        self.assertFalse(result.recognition.complete)
        self.assertEqual(result.stop_reason, "view_limit")
        self.assertEqual(result.recognition.examined_views, 1)
        self.assertEqual(len(result.recognition.candidates), 1)
        self.assertEqual(len(result.transforms), 1)
        self.assertEqual(len(result.recognition.responses), 1)
        self.assertEqual(result.recognition.responses[0].context_id, "left")

    def test_cancellation_before_and_after_prediction_is_visible(self) -> None:
        source, common, transforms = self.fixture()
        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        early = pipeline.recognize(source, cancelled=lambda: True)
        self.assertEqual(early.stop_reason, "cancelled")
        self.assertFalse(early.complete)
        self.assertEqual(early.transforms, ())
        flag = [False]
        original = transforms["left"].predict

        def predict(value):
            result = original(value)
            flag[0] = True
            return result

        with patch.object(transforms["left"], "predict", side_effect=predict):
            late = pipeline.recognize(source, cancelled=lambda: flag[0])
        self.assertEqual(late.stop_reason, "cancelled")
        self.assertEqual(late.transforms[0].predicted_bits, (2, 3))
        self.assertFalse(late.transforms[0].forwarded)
        self.assertEqual(late.recognition.examined_views, 0)
        self.assertEqual(late.recognition.responses, ())

    def test_cancellation_during_recognition_preserves_completed_prefix(self) -> None:
        source, common, transforms = self.fixture()
        flag = [False]

        def progress(message):
            if message.startswith("recognition:"):
                flag[0] = True

        pipeline = LearnedContextPipeline(
            common, transforms, input_encoding_id="sensor-v1"
        )
        result = pipeline.recognize(
            source, progress=progress, cancelled=lambda: flag[0]
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(result.recognition.examined_views, 1)
        self.assertEqual(len(result.recognition.candidates), 1)
        self.assertEqual(len(result.recognition.responses), 1)

    def test_prediction_crossing_deadline_does_not_become_complete_recognition(
        self,
    ) -> None:
        source, common, transforms = self.fixture()
        clock = [0.0]
        original = transforms["left"].predict

        def predict(value):
            result = original(value)
            clock[0] = 2.0
            return result

        pipeline = LearnedContextPipeline(
            common,
            transforms,
            input_encoding_id="sensor-v1",
            limits=RecognitionLimits(seconds=1.0),
        )
        with (
            patch(
                "text_factors.context_pipeline.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch.object(transforms["left"], "predict", side_effect=predict),
        ):
            result = pipeline.recognize(source)
        self.assertFalse(result.complete)
        self.assertEqual(result.stop_reason, "time_budget")
        self.assertEqual(len(result.transforms), 1)
        self.assertFalse(result.transforms[0].forwarded)
        self.assertEqual(result.recognition.examined_views, 0)

    def test_invalid_predicted_shape_has_no_fallback(self) -> None:
        source, common, transforms = self.fixture()
        left = transforms["left"]
        invalid = replace(left.predict(source), output=np.zeros(7, dtype=np.bool_))
        pipeline = LearnedContextPipeline(
            common, {"left": left}, input_encoding_id="sensor-v1"
        )
        with (
            patch.object(left, "predict", return_value=invalid),
            self.assertRaises(ValueError),
        ):
            pipeline.recognize(source)


if __name__ == "__main__":
    unittest.main()

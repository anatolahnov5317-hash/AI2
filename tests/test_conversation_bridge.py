import copy
import json
import unittest
from dataclasses import replace
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from text_factors.conversation.bridge import (
    LABELS,
    MAX_PAIRS,
    FactorSemanticBridge,
)
from text_factors.memory import CombinatorialMemory
from text_factors.transforms import LearnedSDRTransform

PAIRS = [
    ("лежит", "locate"),
    ("находится", "locate"),
    ("положил", "move"),
    ("переложил", "move"),
    ("отдал", "give"),
    ("передал", "give"),
    ("владеет", "have"),
    ("держит", "have"),
]


def memory_snapshot(memory: CombinatorialMemory) -> tuple:
    return (
        memory.step,
        memory.receptors.tobytes(),
        memory.output_map.tobytes(),
        tuple(
            (
                point,
                cluster.signature,
                cluster.bit_hits.tobytes(),
                cluster.status,
                cluster.partial_hits,
                cluster.exact_hits,
                cluster.partial_errors,
                cluster.complete_errors,
                cluster.last_seen,
            )
            for point, cluster in memory.iter_clusters()
        ),
    )


class ConversationBridgeTests(unittest.TestCase):
    def fitted(self, **kwargs) -> FactorSemanticBridge:
        bridge = FactorSemanticBridge(**kwargs)
        bridge.fit(PAIRS)
        return bridge

    def test_real_transform_and_common_memory_determine_labels(self) -> None:
        bridge = self.fitted()
        self.assertIsInstance(bridge._transform, LearnedSDRTransform)
        assert bridge._transform is not None and bridge._reader is not None
        with (
            patch.object(
                bridge._transform, "predict", wraps=bridge._transform.predict
            ) as transform,
            patch.object(
                bridge._reader,
                "recognize_views",
                wraps=bridge._reader.recognize_views,
            ) as reader,
        ):
            for cue, label in PAIRS:
                result = bridge.classify(cue)
                self.assertEqual(result.label, label)
                self.assertEqual(result.candidates, (label,))
                self.assertGreater(result.score, 0)
                self.assertTrue(result.evidence["factor_evidence_used"])
                self.assertTrue(result.evidence["predicted_bits"])
                self.assertGreater(result.evidence["portraits"][0]["support_atoms"], 0)
                self.assertFalse(result.evidence["score_is_probability"])
                json.dumps(result.to_dict(), allow_nan=False)
            self.assertEqual(transform.call_count, len(PAIRS))
            self.assertEqual(reader.call_count, len(PAIRS))

    def test_zeroed_transform_cannot_fall_back_to_taught_label(self) -> None:
        bridge = self.fitted()
        assert bridge._transform is not None
        bridge._transform.memory = CombinatorialMemory(bridge._transform.memory.config)
        result = bridge.classify("положил")
        self.assertIsNone(result.label)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.evidence["predicted_bits"], [])

    def test_removed_common_portraits_cannot_fall_back_to_predicted_label(self) -> None:
        bridge = self.fitted()
        assert bridge._reader is not None
        bridge._reader._portraits.clear()
        result = bridge.classify("положил")
        self.assertIsNone(result.label)
        self.assertTrue(result.evidence["predicted_bits"])

    def test_factor_inference_does_not_read_training_labels(self) -> None:
        bridge = self.fitted()
        before = bridge.classify("положил").to_dict()
        bridge._pairs = tuple((cue, "have") for cue, _ in bridge._pairs)
        self.assertEqual(bridge.classify("положил").to_dict(), before)

    def test_untrained_and_shuffled_are_observable_controls(self) -> None:
        untrained, shuffled = (
            self.fitted(mode="untrained"),
            self.fitted(mode="shuffled"),
        )
        self.assertTrue(all(untrained.classify(cue).label is None for cue, _ in PAIRS))
        self.assertTrue(
            any(shuffled.classify(cue).label != label for cue, label in PAIRS)
        )
        self.assertTrue(all(shuffled.classify(cue).label in LABELS for cue, _ in PAIRS))

    def test_nearest_mode_is_disclosed_nonfactor_lookup_baseline(self) -> None:
        bridge = self.fitted(mode="nearest")
        result = bridge.classify("  ПОЛОЖИЛ  ")
        self.assertEqual(result.label, "move")
        self.assertFalse(result.evidence["factor_evidence_used"])
        self.assertEqual(result.evidence["mode"], "nearest")
        self.assertIsNone(bridge.classify("телепортировал").label)

    def test_unknown_normalized_and_conflicted_cues(self) -> None:
        bridge = self.fitted()
        self.assertEqual(bridge.classify("  ПОЛОЖИЛ ").label, "move")
        self.assertIsNone(bridge.classify("телепортировал").label)
        report = bridge.fit([("ПОЛОЖИЛ", "give")])
        self.assertEqual(report["conflicted_cues"], ["положил"])
        result = bridge.classify("положил")
        self.assertIsNone(result.label)
        self.assertEqual(result.candidates, ("give", "move"))
        self.assertEqual(result.evidence["reason"], "conflicting_teaching")
        self.assertEqual(bridge.classify("лежит").label, "locate")

    def test_duplicates_are_not_independent_teaching(self) -> None:
        bridge = FactorSemanticBridge()
        report = bridge.fit([("лежит", "locate"), (" ЛЕЖИТ ", "locate")])
        self.assertEqual(report["unique_teaching_pairs"], 1)
        self.assertEqual(report["new_unique_pairs"], 1)
        self.assertEqual(report["training_presentations"], 8)
        self.assertFalse(report["presentations_are_independent_evidence"])
        before = bridge.classify("лежит").to_dict()
        repeat = bridge.fit([("лежит", "locate")])
        self.assertEqual(repeat["new_unique_pairs"], 0)
        self.assertEqual(bridge.classify("лежит").to_dict(), before)

    def test_roundtrip_replay_preserves_recipe_and_inference_exactly(self) -> None:
        for mode in ("factor", "untrained", "shuffled", "nearest"):
            bridge = self.fitted(mode=mode)
            recipe = json.loads(json.dumps(bridge.to_dict()))
            before = [bridge.classify(cue).to_dict() for cue, _ in PAIRS]
            restored = FactorSemanticBridge.from_dict(recipe)
            self.assertEqual(restored.to_dict(), recipe)
            self.assertEqual(
                [restored.classify(cue).to_dict() for cue, _ in PAIRS], before
            )
        empty = FactorSemanticBridge()
        self.assertEqual(
            FactorSemanticBridge.from_dict(empty.to_dict()).to_dict(), empty.to_dict()
        )

    def test_read_only_inference_preserves_all_memory_evidence(self) -> None:
        bridge = self.fitted()
        assert bridge._transform is not None and bridge._reader is not None
        before = (
            memory_snapshot(bridge._transform.memory),
            memory_snapshot(bridge._reader.memory),
            bridge.to_dict(),
        )
        for cue, _ in PAIRS:
            bridge.classify(cue)
        after = (
            memory_snapshot(bridge._transform.memory),
            memory_snapshot(bridge._reader.memory),
            bridge.to_dict(),
        )
        self.assertEqual(after, before)

    def test_timeout_is_atomic_for_existing_and_empty_bridge(self) -> None:
        bridge = self.fitted()
        before = bridge.to_dict(), bridge.classify("положил").to_dict()
        with self.assertRaises(TimeoutError):
            bridge.fit([("поместил", "move")], seconds=1e-12)
        self.assertEqual(
            (bridge.to_dict(), bridge.classify("положил").to_dict()), before
        )
        empty = FactorSemanticBridge()
        with self.assertRaises(TimeoutError):
            empty.fit(PAIRS, seconds=1e-12)
        self.assertEqual(empty.to_dict()["examples"], [])

    def test_incomplete_common_read_abstains_even_with_supported_portrait(self) -> None:
        bridge = self.fitted()
        assert bridge._reader is not None
        original = bridge._reader.recognize_views

        def interrupted(*args, **kwargs):
            return replace(
                original(*args, **kwargs), complete=False, stop_reason="cancelled"
            )

        with patch.object(bridge._reader, "recognize_views", side_effect=interrupted):
            result = bridge.classify("положил")
        self.assertIsNone(result.label)
        self.assertFalse(result.evidence["recognition_complete"])

    def test_invalid_inputs_and_states_are_rejected_before_mutation(self) -> None:
        bridge = self.fitted()
        before = bridge.to_dict()
        for pairs in (
            [],
            [("x", "unsupported")],
            [["x", "move"]],
            [("x" * 65, "move")],
        ):
            with self.assertRaises(ValueError):
                bridge.fit(cast(Any, pairs))
        for epoch in (True, 0, 17):
            with self.assertRaises(ValueError):
                bridge.fit(PAIRS, epochs=epoch)
        for seconds in (True, 0, float("nan"), float("inf"), 61):
            with self.assertRaises(ValueError):
                bridge.fit(PAIRS, seconds=seconds)
        for cue in ("", "x" * 65, "<code>", "123", 123):
            with self.assertRaises(ValueError):
                bridge.classify(cast(Any, cue))
        variants = []
        for field, value in (
            ("seed", True),
            ("seed", -1),
            ("mode", "llm"),
            ("mode", []),
            ("epochs", 0),
            ("epochs", True),
            ("schema", "unknown"),
            ("examples", [["not normalized", "move"]] * (MAX_PAIRS + 1)),
            ("examples", [[" ЛЕЖИТ ", "locate"]]),
            ("examples", [["лежит", "locate"], ["лежит", "locate"]]),
        ):
            malformed = copy.deepcopy(before)
            malformed[field] = value
            variants.append(malformed)
        for malformed in variants:
            with self.assertRaises(ValueError):
                FactorSemanticBridge.from_dict(malformed)
        self.assertEqual(bridge.to_dict(), before)

    def test_fixed_seed_recipes_are_fast_and_work_across_seeds(self) -> None:
        started = perf_counter()
        for seed in (0, 7, 42, 2**32 - 1):
            bridge = self.fitted(seed=seed)
            self.assertEqual(
                [bridge.classify(cue).label for cue, _ in PAIRS],
                [label for _, label in PAIRS],
            )
        self.assertLess(perf_counter() - started, 15.0)

    def test_prediction_is_source_only_and_uses_an_owned_readonly_sdr(self) -> None:
        bridge = self.fitted()
        assert bridge._transform is not None
        original = bridge._transform.predict
        received = []

        def predict(source):
            self.assertEqual(source.shape, (256,))
            self.assertEqual(source.dtype, np.dtype(np.bool_))
            self.assertFalse(source.flags.writeable)
            received.append(source.copy())
            return original(source)

        with patch.object(bridge._transform, "predict", side_effect=predict):
            bridge.classify("положил")
            bridge.classify("передал")
        self.assertEqual(len(received), 2)
        self.assertFalse(np.array_equal(received[0], received[1]))


if __name__ == "__main__":
    unittest.main()

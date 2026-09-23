"""Open development checks for partial evidence and independent contexts."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

from text_factors.real_data import ContextRegistry, LearningEpisode, SparseTransform

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_p09_p10_open.py"
SPEC = importlib.util.spec_from_file_location("evaluate_p09_p10_open", SCRIPT)
assert SPEC and SPEC.loader
evaluation_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation_script)


class FactorMemoryDevelopmentTests(unittest.TestCase):
    def test_one_repeated_story_cannot_outvote_two_independent_groups(self) -> None:
        transform = SparseTransform(32)
        for _ in range(50):
            transform.fit((1,), (10,), group_id="family-a")
        transform.fit((1,), (11,), group_id="family-b")
        transform.fit((1,), (11,), group_id="family-c")
        self.assertEqual(transform.predict((1,), limit=1), (11,))
        self.assertEqual(transform.support_for((1,), 10), (1, 0))
        self.assertEqual(transform.support_for((1,), 11), (2, 0))
        self.assertEqual(transform.to_dict()["counts"]["1"], {"10": 1, "11": 2})

    def test_replay_after_save_and_load_is_still_capped(self) -> None:
        registry = ContextRegistry(width=32)
        ctx, _, _ = registry.learn(LearningEpisode("one", "family-a", (1,), (10,)))
        restored = ContextRegistry.from_dict(registry.to_dict())
        for index in range(30):
            restored.learn(LearningEpisode(f"repeat-{index}", "family-a", (1,), (10,)))
        self.assertEqual(
            restored._contexts[ctx].transform.support_for((1,), 10), (1, 0)
        )
        self.assertEqual(
            restored._contexts[ctx].transform.to_dict()["counts"]["1"]["10"], 1
        )
        self.assertEqual(
            ContextRegistry.from_dict(restored.to_dict()).to_dict(), restored.to_dict()
        )

    def test_masked_out_unknown_target_does_not_split_context(self) -> None:
        registry = ContextRegistry(width=32, assignment_threshold=0.6)
        ctx, _, _ = registry.learn(LearningEpisode("first", "a", (1,), (10,)))
        result, created, score = registry.learn(
            LearningEpisode("partial", "b", (1,), (10, 11), (10,))
        )
        self.assertEqual((result, created, score), (ctx, False, 1.0))
        self.assertEqual(len(registry.contexts), 1)
        self.assertEqual(registry.predict(ctx, (1,), limit=2), (10,))
        self.assertEqual(registry._contexts[ctx].independent_support, 2)

    def test_negative_and_unknown_bits_are_different(self) -> None:
        transform = SparseTransform(32)
        transform.fit((1,), (10,), observed_target_bits=(10, 11), group_id="a")
        transform.fit((1,), (10, 12), observed_target_bits=(10,), group_id="b")
        self.assertEqual(transform.support_for((1,), 10), (2, 0))
        self.assertEqual(transform.support_for((1,), 11), (0, 1))
        self.assertEqual(transform.support_for((1,), 12), (0, 0))
        self.assertEqual(transform.predict((1,), limit=3), (10,))

    def test_known_absence_counts_against_a_context_and_can_be_repeated(self) -> None:
        registry = ContextRegistry(width=32, assignment_threshold=0.6)
        present, _, _ = registry.learn(LearningEpisode("a", "group-a", (1,), (10,)))
        absent, created, score = registry.learn(
            LearningEpisode("b", "group-b", (1,), (), (10,))
        )
        self.assertTrue(created)
        self.assertNotEqual(present, absent)
        self.assertEqual(score, 0.0)
        again, created, score = registry.learn(
            LearningEpisode("c", "group-c", (1,), (), (10,))
        )
        self.assertEqual((again, created, score), (absent, False, 1.0))
        self.assertEqual(
            registry._contexts[absent].transform.support_for((1,), 10), (0, 2)
        )

    def test_conflicting_edits_in_one_group_remove_prior_vote(self) -> None:
        transform = SparseTransform(32)
        transform.fit((1,), (10,), observed_target_bits=(10,), group_id="a")
        transform.fit((1,), (), observed_target_bits=(10,), group_id="a")
        self.assertEqual(transform.support_for((1,), 10), (0, 0))
        self.assertEqual(transform.predict((1,), limit=1), ())
        transform.fit((1,), (10,), observed_target_bits=(10,), group_id="a")
        self.assertEqual(transform.support_for((1,), 10), (0, 0))
        self.assertEqual(
            SparseTransform.from_dict(transform.to_dict()).to_dict(),
            transform.to_dict(),
        )

    def test_group_votes_capacity_failure_does_not_mutate_model(self) -> None:
        transform = SparseTransform(32)
        transform.MAX_GROUP_VOTES = 2
        transform.fit((1,), (10,), group_id="a")
        before = transform.to_dict()
        with self.assertRaisesRegex(ValueError, "capacity"):
            transform.fit((1,), (10, 11), group_id="b")
        self.assertEqual(transform.to_dict(), before)

    def test_legacy_state_loads_without_inventing_independent_groups(self) -> None:
        old = {"width": 32, "episodes": 5, "counts": {"1": {"10": 5}}}
        transform = SparseTransform.from_dict(old)
        self.assertEqual(transform.predict((1,), limit=1), (10,))
        self.assertEqual(transform.support_for((1,), 10), (0, 0))
        transform.fit((1,), (11,), group_id="new")
        self.assertEqual(
            SparseTransform.from_dict(transform.to_dict()).to_dict(),
            transform.to_dict(),
        )

    def test_inconsistent_persisted_group_votes_are_rejected(self) -> None:
        transform = SparseTransform(32)
        transform.fit((1,), (10,), group_id="a")
        invalid = copy.deepcopy(transform.to_dict())
        invalid["counts"]["1"].pop("10")
        with self.assertRaisesRegex(ValueError, "exceed association counts"):
            SparseTransform.from_dict(invalid)

    def test_open_development_predictions_are_frozen_and_group_separated(self) -> None:
        result = evaluation_script.evaluate()
        self.assertEqual(result["train_independent_groups"], 6)
        self.assertEqual(result["development_independent_groups"], 2)
        self.assertEqual(result["frequency_correct_before_update"], 0)
        self.assertEqual(result["group_capped_correct_before_update"], 2)
        self.assertTrue(result["masked_context_reused"])
        self.assertTrue(result["distinct_context_discovered"])

    def test_empty_target_mask_rejected_without_creating_context(self) -> None:
        registry = ContextRegistry(width=32)
        with self.assertRaisesRegex(ValueError, "observed target bits"):
            registry.learn(LearningEpisode("empty", "a", (1,), (10,), ()))
        self.assertFalse(registry.contexts)


if __name__ == "__main__":
    unittest.main()

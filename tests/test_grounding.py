import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from text_factors.config import ModelConfig
from text_factors.dialogue import (
    GroundedCandidate,
    GroundedDialogue,
    GroundingEvidence,
    GroundingPolicy,
    LabelEvent,
    SceneReference,
)
from text_factors.grounding import MAX_MATCH_ATOMS
from text_factors.memory import CombinatorialMemory
from text_factors.recognition import ContextView, recognize_views


def candidate(
    key: str,
    evidence: tuple[tuple[int, tuple[int, ...]], ...],
) -> GroundedCandidate:
    return GroundedCandidate(
        key,
        "view",
        key,
        "observation",
        (0,),
        (0,),
        tuple(GroundingEvidence(point, bits, bits, 3, 0) for point, bits in evidence),
    )


BASE = ((0, (0, 1)), (1, (2, 3)))


class FactorGroundingTests(unittest.TestCase):
    def dialogue(self, **policy: Any) -> GroundedDialogue:
        return GroundedDialogue(
            "memory",
            output_width=1,
            grounding_policy=GroundingPolicy(mode="factor", **policy),
        )

    def remember(
        self,
        dialogue: GroundedDialogue,
        value: GroundedCandidate,
        *,
        word: str | None = None,
        event_id: int = 0,
    ) -> None:
        dialogue.remember(SceneReference(value.candidate_id, (value,), ()))
        if word is not None:
            dialogue.confirm(
                LabelEvent(event_id, value.candidate_id, value.candidate_id, word)
            )

    def test_exact_default_stays_unknown_on_key_drift_and_preserves_aliases(
        self,
    ) -> None:
        dialogue = GroundedDialogue("memory", output_width=1)
        old = candidate("old", BASE)
        new = candidate("new", BASE)
        self.remember(dialogue, old, word="куб")
        dialogue.confirm(LabelEvent(1, "old", "old", "блок"))
        self.remember(dialogue, new)
        self.assertEqual(dialogue.resolve_candidate(new).method, "unknown")
        exact = dialogue.resolve_candidate(old)
        self.assertEqual(exact.words, ("куб", "блок"))
        self.assertEqual(exact.method, "exact")
        self.assertFalse(exact.ambiguous)

    def test_real_cluster_training_changes_key_but_factor_name_can_survive(
        self,
    ) -> None:
        config = ModelConfig(
            input_bits=8,
            active_bits_per_symbol=2,
            positions=3,
            frame_size=2,
            context_count=2,
            receptive_bits=3,
            point_count=3,
            output_bits=1,
            create_threshold=2,
            activation_threshold=2,
            min_active_points=1,
            probation_after=2,
            stable_after=3,
            max_clusters_per_point=4,
            seed=17,
        )
        memory = CombinatorialMemory(
            config,
            receptors=np.asarray([[0, 1, 2], [3, 4, 5], [5, 6, 7]], dtype=np.int32),
            output_map=np.asarray([0, 0, 0], dtype=np.int32),
        )
        original = np.asarray([1, 1, 0, 1, 1, 0, 0, 0], dtype=np.bool_)
        addition = np.asarray([0, 0, 0, 0, 0, 1, 1, 0], dtype=np.bool_)
        query = original | addition
        for _ in range(3):
            memory.observe(original)
        before = recognize_views(
            memory, (ContextView("old", "ctx", query, (0,)),), total_views=1
        )
        dialogue = GroundedDialogue(
            before.encoding_id,
            output_width=1,
            grounding_policy=GroundingPolicy(mode="factor", threshold=0.6),
        )
        dialogue.remember_recognition("before", before, encoding_id=before.encoding_id)
        dialogue.confirm(LabelEvent(0, "before", "old", "куб"))
        for _ in range(3):
            memory.observe(addition)
        after = recognize_views(
            memory, (ContextView("new", "ctx", query, (0,)),), total_views=1
        )
        self.assertNotEqual(
            before.candidates[0].content_key, after.candidates[0].content_key
        )
        self.assertEqual(
            before.candidates[0].output_bits, after.candidates[0].output_bits
        )
        dialogue.remember_recognition("after", after, encoding_id=after.encoding_id)
        resolution = dialogue.resolve_candidate(
            dialogue._references["after"].candidates[0]
        )
        self.assertEqual(resolution.words, ("куб",))
        self.assertEqual(resolution.method, "factor")
        self.assertAlmostEqual(resolution.score, 4 / 6)
        self.assertEqual(
            resolution.matched_content_keys, (before.candidates[0].content_key,)
        )
        self.assertEqual(dialogue.lookup("куб").candidate_ids, ("new",))
        self.assertEqual(dialogue.stats()["events"], 1)

    def test_same_output_or_input_bits_at_other_points_do_not_supply_names(
        self,
    ) -> None:
        dialogue = self.dialogue()
        self.remember(dialogue, candidate("old", BASE), word="куб")
        for key, atoms in (
            ("other-bits", ((0, (20, 21)), (1, (22, 23)))),
            ("other-points", ((20, (0, 1)), (21, (2, 3)))),
        ):
            with self.subTest(key=key):
                value = candidate(key, atoms)
                self.remember(dialogue, value)
                resolved = dialogue.resolve_candidate(value)
                self.assertEqual(resolved.method, "unknown")
                self.assertEqual(resolved.words, ())

    def test_equal_overlapping_names_and_lookup_require_clarification(self) -> None:
        dialogue = self.dialogue()
        self.remember(dialogue, candidate("cube", BASE), word="куб")
        self.remember(
            dialogue,
            candidate("ball", ((0, (0, 1)), (1, (2, 4)))),
            word="шар",
            event_id=1,
        )
        query = candidate("query", ((0, (0, 1)), (1, (2, 3, 4))))
        self.remember(dialogue, query)
        resolution = dialogue.resolve_candidate(query)
        self.assertTrue(resolution.ambiguous)
        self.assertEqual(set(resolution.words), {"куб", "шар"})
        self.assertEqual(resolution.support_event_ids, (0, 1))
        self.assertAlmostEqual(resolution.score, 0.8)
        self.assertEqual(dialogue.describe().kind, "clarification")
        self.assertEqual(dialogue.lookup("куб").kind, "clarification")

    def test_strong_main_part_plus_a_small_named_remainder_is_not_one_name(
        self,
    ) -> None:
        dialogue = self.dialogue()
        large = tuple((point, (2 * point, 2 * point + 1)) for point in range(8))
        small = ((8, (16, 17)), (9, (18, 19)))
        self.remember(dialogue, candidate("large", large), word="куб")
        self.remember(dialogue, candidate("small", small), word="шар", event_id=1)
        combined = candidate("combined", large + small)
        self.remember(dialogue, combined)
        resolution = dialogue.resolve_candidate(combined)
        self.assertTrue(resolution.ambiguous)
        self.assertEqual(resolution.reason, "multiple_named_parts")
        self.assertEqual(dialogue.describe().kind, "clarification")

    def test_reciprocal_coverage_rejects_large_unconfirmed_additions(self) -> None:
        dialogue = self.dialogue()
        self.remember(dialogue, candidate("old", BASE), word="куб")
        mixture = candidate("mixture", BASE + ((2, (4, 5)), (3, (6, 7))))
        self.remember(dialogue, mixture)
        resolution = dialogue.resolve_candidate(mixture)
        self.assertEqual(resolution.words, ())
        self.assertAlmostEqual(resolution.score, 0.5)
        self.assertEqual(resolution.reason, "below_threshold")

    def test_corrections_remove_old_support_without_inference_self_training(
        self,
    ) -> None:
        dialogue = self.dialogue()
        self.remember(dialogue, candidate("old", BASE), word="куб")
        query = candidate("query", BASE)
        self.remember(dialogue, query)
        before = dialogue.to_dict()
        for _ in range(4):
            self.assertEqual(dialogue.resolve_candidate(query).words, ("куб",))
            dialogue.describe()
            dialogue.lookup("куб")
        self.assertEqual(dialogue.to_dict(), before)
        correction = LabelEvent(1, "old", "old", "шар", 0)
        self.assertTrue(dialogue.confirm(correction))
        self.assertFalse(dialogue.confirm(correction))
        resolved = dialogue.resolve_candidate(query)
        self.assertEqual(resolved.words, ("шар",))
        self.assertEqual(resolved.support_event_ids, (1,))
        self.assertEqual(dialogue.lookup("куб").kind, "clarification")
        self.assertEqual(query.content_key, "query")
        self.assertEqual(dialogue.stats()["events"], 2)

    def test_minimum_points_and_work_limits_do_not_return_partial_confidence(
        self,
    ) -> None:
        dialogue = self.dialogue()
        one_point = ((0, (0, 1, 2, 3)),)
        self.remember(dialogue, candidate("old", one_point), word="куб")
        query = candidate("query", one_point)
        self.remember(dialogue, query)
        self.assertEqual(
            dialogue.resolve_candidate(query).reason, "insufficient_evidence"
        )
        too_many = candidate(
            "large",
            tuple(
                (point, tuple(range(2048)))
                for point in range(MAX_MATCH_ATOMS // 2048 + 1)
            ),
        )
        self.assertEqual(
            dialogue.resolve_candidate(too_many, encoding_id="memory").reason,
            "work_limit",
        )

    def test_unarchived_and_foreign_encoding_cannot_silently_enter_the_matcher(
        self,
    ) -> None:
        dialogue = self.dialogue()
        old = candidate("old", BASE)
        self.remember(dialogue, old, word="куб")
        query = candidate("query", BASE)
        with self.assertRaisesRegex(ValueError, "encoding_id"):
            dialogue.resolve_candidate(query)
        with self.assertRaisesRegex(ValueError, "encoding_id"):
            dialogue.resolve_candidate(query, encoding_id="foreign")
        self.assertEqual(
            dialogue.resolve_candidate(query, encoding_id="memory").words, ("куб",)
        )

    def test_policy_and_complete_exemplars_round_trip_and_legacy_defaults_to_exact(
        self,
    ) -> None:
        dialogue = self.dialogue(threshold=0.6)
        self.remember(dialogue, candidate("old", BASE), word="куб")
        query = candidate("query", BASE + ((2, (4, 5)),))
        self.remember(dialogue, query)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grounded.json"
            dialogue.save(path)
            restored = GroundedDialogue.load(path)
        self.assertEqual(restored.to_dict(), dialogue.to_dict())
        self.assertEqual(
            restored.resolve_candidate(query), dialogue.resolve_candidate(query)
        )
        legacy = dialogue.to_dict()
        legacy["format_version"] = 1
        del legacy["grounding_policy"]
        restored_legacy = GroundedDialogue.from_dict(legacy)
        self.assertEqual(restored_legacy.grounding_policy.mode, "exact")
        self.assertEqual(restored_legacy.resolve_candidate(query).method, "unknown")

    def test_policy_validates_nonfinite_and_unbounded_or_untyped_settings(self) -> None:
        for changes in (
            {"mode": "magic"},
            {"threshold": float("nan")},
            {"threshold": True},
            {"margin": -1},
            {"min_atoms": 100000},
            {"min_points": True},
            {"min_atoms": 1, "min_points": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GroundingPolicy(**cast(Any, changes))
        data = self.dialogue().to_dict()
        data["grounding_policy"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            GroundedDialogue.from_dict(data)


if __name__ == "__main__":
    unittest.main()

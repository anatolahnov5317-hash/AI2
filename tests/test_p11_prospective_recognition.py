"""P11: source-only readout, observed residual, and frozen transfer controls."""

import unittest
from typing import cast
from unittest.mock import patch

from text_factors.real_data.contexts import ContextRegistry
from text_factors.real_data.contracts import LearningEpisode
from text_factors.real_data.recognition import (
    evaluate_frozen_transfer,
    prequential_step,
    prospective_read,
    recognize_observation,
)


def learned() -> tuple[ContextRegistry, tuple[LearningEpisode, ...]]:
    registry = ContextRegistry(width=16, assignment_threshold=0.5)
    episodes = (
        LearningEpisode("a1", "history-a", (1, 2), (8, 9)),
        LearningEpisode("a2", "history-b", (1, 2), (8, 9)),
        LearningEpisode("b1", "history-c", (3, 4), (10, 11)),
        LearningEpisode("b2", "history-d", (3, 4), (10, 11)),
    )
    for episode in episodes:
        registry.learn(episode)
    return registry, episodes


class P11ProspectiveRecognitionTests(unittest.TestCase):
    def test_prediction_is_source_only_and_readout_preserves_unknowns(self) -> None:
        registry, _ = learned()
        before = registry.to_dict()
        prospect = prospective_read(registry, (1, 2, 5), output_limit=2)
        self.assertEqual(prospect.predicted_bits, (8, 9))
        self.assertEqual(prospect.responses[0].independent_support, 2)
        self.assertEqual(prospect.responses[0].source_coverage, 2 / 3)

        observed = recognize_observation(prospect, (8, 12))
        self.assertEqual(observed.explained_bits, (8,))
        self.assertEqual(observed.unexplained_bits, (12,))
        self.assertEqual(observed.selected_context_ids, (prospect.primary_context_id,))
        self.assertEqual(observed.observed_absent, ())
        self.assertEqual(registry.to_dict(), before)

    def test_known_absence_blocks_contradicted_response(self) -> None:
        registry, _ = learned()
        prospect = prospective_read(registry, (1, 2), output_limit=2)
        observed = recognize_observation(prospect, (8, 12), observed_absent=(9,))
        self.assertEqual(observed.selected_context_ids, ())
        self.assertEqual(observed.unexplained_bits, (8, 12))
        self.assertEqual(
            observed.contradicted_context_ids, (prospect.primary_context_id,)
        )

    def test_two_distinct_contexts_cover_different_parts_of_observation(self) -> None:
        registry, _ = learned()
        prospect = prospective_read(registry, (1, 2, 3, 4), output_limit=2)
        self.assertEqual(len(prospect.responses), 2)
        observed = recognize_observation(prospect, (8, 10, 12))
        self.assertEqual(observed.selected_context_ids, ("ctx_0001", "ctx_0002"))
        self.assertEqual(observed.explained_bits, (8, 10))
        self.assertEqual(observed.unexplained_bits, (12,))

        contradicted = recognize_observation(
            prospect, (8, 10, 12), observed_absent=(9,)
        )
        self.assertEqual(contradicted.selected_context_ids, ("ctx_0002",))
        self.assertEqual(contradicted.unexplained_bits, (8, 12))

    def test_repeats_of_one_group_do_not_pass_minimum_support(self) -> None:
        registry = ContextRegistry(width=16)
        registry.learn(LearningEpisode("a1", "same-history", (1,), (8,)))
        for index in range(2, 10):
            registry.learn(LearningEpisode(f"a{index}", "same-history", (1,), (8,)))
        self.assertEqual(prospective_read(registry, (1,)).predicted_bits, ())

    def test_legacy_frequency_without_bit_provenance_abstains_after_reload(
        self,
    ) -> None:
        registry, _ = learned()
        legacy = registry.to_dict()
        for context in legacy["contexts"]:
            context["transform"].pop("negative_counts")
            context["transform"].pop("group_votes")
        restored = ContextRegistry.from_dict(legacy)
        self.assertEqual(restored.reliable_context_ids(), ("ctx_0001", "ctx_0002"))
        self.assertEqual(restored.contexts[0].transform.support_for((1, 2), 8), (0, 0))
        self.assertEqual(prospective_read(restored, (1, 2)).predicted_bits, ())

        # Two newly traced, unrelated groups establish support for each bit.
        restored.learn(LearningEpisode("new-1", "group-new-1", (1, 2), (8, 9)))
        self.assertEqual(prospective_read(restored, (1, 2)).predicted_bits, ())
        restored.learn(LearningEpisode("new-2", "group-new-2", (1, 2), (8, 9)))
        self.assertEqual(prospective_read(restored, (1, 2)).predicted_bits, (8, 9))
        reloaded = ContextRegistry.from_dict(restored.to_dict())
        self.assertEqual(prospective_read(reloaded, (1, 2)).predicted_bits, (8, 9))

    def test_prequential_step_reveals_and_updates_only_after_prediction(self) -> None:
        registry, _ = learned()
        before = registry.to_dict()
        events: list[str] = []
        actual = prospective_read

        def predict(*args, **kwargs):
            self.assertEqual(registry.to_dict(), before)
            events.append("prediction")
            return actual(*args, **kwargs)

        def reveal() -> LearningEpisode:
            self.assertEqual(registry.to_dict(), before)
            events.append("observation")
            return LearningEpisode(
                "future-1", "future-group", (1, 2), (8, 12), (8, 9, 12)
            )

        with patch(
            "text_factors.real_data.recognition.prospective_read", side_effect=predict
        ):
            result = prequential_step(registry, (1, 2), reveal)
        self.assertEqual(events, ["prediction", "observation"])
        self.assertEqual(result.readout.prospect.predicted_bits, (8, 9))
        self.assertEqual(result.readout.unexplained_bits, (8, 12))
        self.assertIsNotNone(result.learned_context_id)
        self.assertNotEqual(registry.to_dict(), before)
        self.assertIn(
            "future-group", {g for c in registry.contexts for g in c.group_ids}
        )

    def test_prequential_readonly_and_mismatched_reveal_never_learn(self) -> None:
        registry, _ = learned()
        before = registry.to_dict()
        episode = LearningEpisode("future", "new-group", (1, 2), (8, 12))
        result = prequential_step(registry, (1, 2), lambda: episode, learn=False)
        self.assertEqual(result.readout.unexplained_bits, (12,))
        self.assertIsNone(result.learned_context_id)
        self.assertEqual(registry.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "does not match"):
            prequential_step(
                registry,
                (1,),
                lambda: LearningEpisode("bad", "new-group", (2,), (8,)),
            )
        self.assertEqual(registry.to_dict(), before)

    def test_transfer_evaluates_new_groups_without_learning(self) -> None:
        registry, training = learned()
        before = registry.to_dict()
        evaluation = (
            LearningEpisode(
                "future-1", "future-history-a", (1, 2, 5), (8, 12), (8, 9, 12)
            ),
            LearningEpisode(
                "future-2",
                "future-history-b",
                (3, 4, 6),
                (10, 13),
                (10, 11, 13),
            ),
        )
        result = evaluate_frozen_transfer(
            registry, training, evaluation, output_limit=2
        )
        self.assertEqual(result["groups"], 2)
        self.assertEqual(registry.to_dict(), before)
        self.assertEqual(result["rows"][0]["predictions"]["context"], (8, 9))
        self.assertEqual(result["rows"][0]["unexplained_bits"], (8, 12))
        self.assertEqual(
            result["comparisons"]["context"]["contradicted_known_absent_bits"],
            2,
        )
        self.assertIn("nearest", result["comparisons"])
        self.assertIn("majority", result["comparisons"])
        self.assertIn("zero", result["comparisons"])

    def test_target_is_not_available_until_after_all_predictors(self) -> None:
        registry, training = learned()
        revealed = [False]

        class GuardedEpisode:
            episode_id = "guarded"
            group_id = "future-history"
            source_code = (1, 2)

            @property
            def target_code(self) -> tuple[int, ...]:
                if not revealed[0]:
                    raise AssertionError("target read before source-only prediction")
                return (8,)

            @property
            def observed_target_bits(self) -> tuple[int, ...]:
                if not revealed[0]:
                    raise AssertionError("mask read before source-only prediction")
                return (8, 9)

        actual = prospective_read

        def predict(*args, **kwargs):
            outcome = actual(*args, **kwargs)
            revealed[0] = True
            return outcome

        with patch(
            "text_factors.real_data.recognition.prospective_read", side_effect=predict
        ):
            result = evaluate_frozen_transfer(
                registry, training, (cast(LearningEpisode, GuardedEpisode()),)
            )
        self.assertEqual(result["groups"], 1)
        self.assertTrue(revealed[0])

    def test_rejects_same_group_in_training_and_evaluation(self) -> None:
        registry, training = learned()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_frozen_transfer(registry, training, training[:1])
        unseen_registry, _ = learned()
        with self.assertRaisesRegex(ValueError, "outside the training split"):
            evaluate_frozen_transfer(
                unseen_registry,
                training[:2],
                (LearningEpisode("new", "new-group", (1,), (8,)),),
            )

    def test_unknown_bits_do_not_become_negative_labels(self) -> None:
        registry, training = learned()
        evaluation = (LearningEpisode("e5", "future-history", (1, 2), (8, 12)),)
        report = evaluate_frozen_transfer(registry, training, evaluation)
        self.assertIsNone(
            report["comparisons"]["context"]["known_prediction_precision"]
        )
        self.assertEqual(
            report["comparisons"]["context"]["contradicted_known_absent_bits"],
            0,
        )


if __name__ == "__main__":
    unittest.main()

import json
import math
import unittest
from typing import Any, cast

from text_factors.microworld import (
    HYPOTHESES,
    Entity,
    ExactEpisodicLearner,
    HiddenMatchWorld,
    Interaction,
    Observation,
    TwoRuleLearner,
    run_microworld,
)


def interaction(
    object_id: str,
    object_color: str,
    object_shape: str,
    lock_id: str,
    lock_color: str,
    lock_shape: str,
) -> Interaction:
    return Interaction(
        Entity(object_id, object_color, object_shape),
        Entity(lock_id, lock_color, lock_shape),
    )


class TwoRuleLearnerTests(unittest.TestCase):
    def test_hypothesis_class_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            cast(Any, HYPOTHESES)["color_match"] = lambda _: False

    def setUp(self) -> None:
        self.both_match = interaction("o0", "red", "round", "l0", "red", "round")
        self.neither = interaction("o1", "red", "round", "l1", "blue", "square")
        self.color_only = interaction("o2", "red", "round", "l2", "red", "square")
        self.shape_only = interaction("o3", "red", "round", "l3", "blue", "round")

    def ambiguous_learner(self) -> TwoRuleLearner:
        learner = TwoRuleLearner()
        learner.observe(Observation(self.both_match, True))
        learner.observe(Observation(self.neither, False))
        return learner

    def test_correlated_observations_leave_protocol_ambiguous(self) -> None:
        learner = self.ambiguous_learner()

        self.assertEqual(learner.candidates, ("color_match", "shape_match"))
        prediction = learner.predict(self.color_only)
        self.assertEqual(prediction.probability, 0.5)
        self.assertTrue(prediction.abstained)
        self.assertEqual(prediction.state, "disagreement")

    def test_one_intervention_discriminates_color_rule(self) -> None:
        learner = self.ambiguous_learner()
        learner.observe(HiddenMatchWorld("color_match").act(self.color_only))

        self.assertEqual(learner.candidates, ("color_match",))
        prediction = learner.predict(self.shape_only)
        self.assertFalse(prediction.abstained)
        self.assertEqual(prediction.probability, 0.0)

    def test_one_intervention_discriminates_shape_rule(self) -> None:
        learner = self.ambiguous_learner()
        learner.observe(HiddenMatchWorld("shape_match").act(self.color_only))

        self.assertEqual(learner.candidates, ("shape_match",))
        prediction = learner.predict(self.shape_only)
        self.assertFalse(prediction.abstained)
        self.assertEqual(prediction.probability, 1.0)

    def test_information_gain_choice_and_tie_are_deterministic(self) -> None:
        learner = self.ambiguous_learner()
        uninformative = interaction("z", "red", "round", "z2", "red", "round")
        later_tie = interaction("y", "red", "round", "y2", "blue", "round")
        expected = interaction("x", "blue", "round", "x2", "blue", "square")

        first = learner.select_query([uninformative, later_tie, expected])
        second = learner.select_query([expected, uninformative, later_tie])

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(first.interaction, expected)
        self.assertEqual(second.interaction, expected)
        self.assertEqual(first.information_gain_bits, 1.0)

    def test_selection_does_not_require_world_or_outcomes(self) -> None:
        learner = self.ambiguous_learner()
        candidates = [self.both_match, self.color_only, self.shape_only]

        choice = learner.select_query(candidates)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertIn(choice.interaction, (self.color_only, self.shape_only))

    def test_contradiction_becomes_explicit_unknown_without_reset(self) -> None:
        learner = self.ambiguous_learner()
        learner.observe(Observation(self.both_match, False))

        self.assertEqual(learner.candidates, ())
        prediction = learner.predict(self.color_only)
        self.assertIsNone(prediction.probability)
        self.assertTrue(prediction.abstained)
        self.assertEqual(prediction.state, "unknown")
        self.assertIsNone(learner.select_query([self.color_only]))

    def test_outcome_and_semantic_choice_are_invariant_to_renaming(self) -> None:
        renamed_color = interaction(
            "completely-new-object", "red", "round", "new-lock", "red", "square"
        )
        world = HiddenMatchWorld("color_match")
        self.assertEqual(
            world.act(self.color_only).outcome, world.act(renamed_color).outcome
        )

        learner = self.ambiguous_learner()
        original = learner.select_query([self.both_match, self.color_only])
        renamed = learner.select_query([self.both_match, renamed_color])
        self.assertIsNotNone(original)
        self.assertIsNotNone(renamed)
        assert original is not None and renamed is not None
        self.assertEqual(
            original.interaction.object.color, renamed.interaction.object.color
        )
        self.assertEqual(
            original.interaction.lock.shape, renamed.interaction.lock.shape
        )

    def test_exact_episodic_lookup_does_not_transfer_to_new_ids(self) -> None:
        learner = ExactEpisodicLearner()
        learner.observe(Observation(self.color_only, True))
        renamed = interaction("new-o", "red", "round", "new-l", "red", "square")

        self.assertFalse(learner.predict(self.color_only).abstained)
        self.assertTrue(learner.predict(renamed).abstained)


class MicroworldRunTests(unittest.TestCase):
    def test_report_is_repeatable_and_json_serializable(self) -> None:
        first = run_microworld(seed=17, episodes=4, action_budget=1)
        second = run_microworld(seed=17, episodes=4, action_budget=1)

        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)

    def test_active_query_resolves_both_hidden_rules_and_transfers(self) -> None:
        report = run_microworld(seed=8, episodes=2, action_budget=1)
        rules = {
            episode["scoring_rule_revealed_after_actions"]
            for episode in report["episodes"]
        }

        self.assertEqual(rules, {"color_match", "shape_match"})
        for episode in report["episodes"]:
            active = episode["strategies"]["active"]
            self.assertEqual(
                active["hypotheses_before_queries"],
                ["color_match", "shape_match"],
            )
            self.assertEqual(len(active["posterior_hypotheses"]), 1)
            self.assertEqual(active["queries"][0]["information_gain_bits"], 1.0)
            self.assertTrue(
                all(record["correct"] for record in active["held_out_predictions"])
            )

    def test_held_out_uses_new_ids_and_recombined_attributes(self) -> None:
        episode = run_microworld(episodes=1)["episodes"][0]
        observed = episode["initial_observations"] + episode["candidate_pool"]
        observed_object_ids = {item["object"]["id"] for item in observed}
        observed_lock_ids = {item["lock"]["id"] for item in observed}

        for held_out in episode["held_out"]:
            self.assertNotIn(held_out["object"]["id"], observed_object_ids)
            self.assertNotIn(held_out["lock"]["id"], observed_lock_ids)
            # Every individual attribute was seen, but the structured entity tuple
            # is a held-out recombination.
            seen_objects = {
                (item["object"]["color"], item["object"]["shape"]) for item in observed
            }
            self.assertNotIn(
                (held_out["object"]["color"], held_out["object"]["shape"]),
                seen_objects,
            )
            # This pilot does NOT claim unseen lock tuples, only fresh lock IDs.
            self.assertIn(
                (held_out["lock"]["color"], held_out["lock"]["shape"]),
                {(item["lock"]["color"], item["lock"]["shape"]) for item in observed},
            )

    def test_zero_budget_preserves_uncertainty_and_finite_metrics(self) -> None:
        report = run_microworld(seed=4, episodes=3, action_budget=0)

        for name in ("active", "random"):
            self.assertEqual(report["metrics"][name]["coverage"], 0.0)
            self.assertEqual(report["metrics"][name]["success"], 0.0)
            for episode in report["episodes"]:
                strategy = episode["strategies"][name]
                self.assertEqual(
                    strategy["posterior_hypotheses"],
                    ["color_match", "shape_match"],
                )
        for metrics in report["metrics"].values():
            for value in metrics.values():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))

    def test_metrics_show_active_transfer_and_episodic_abstention(self) -> None:
        report = run_microworld(seed=42, episodes=32, action_budget=1)

        self.assertEqual(report["metrics"]["active"]["success"], 1.0)
        self.assertEqual(report["metrics"]["active"]["coverage"], 1.0)
        self.assertEqual(report["metrics"]["active"]["query_cost"], 1.0)
        self.assertEqual(report["metrics"]["episodic"]["coverage"], 0.0)
        self.assertLess(report["metrics"]["random"]["coverage"], 1.0)

    def test_inputs_are_validated(self) -> None:
        invalid_calls = [
            {"seed": True},
            {"seed": -1},
            {"episodes": 0},
            {"episodes": 1.5},
            {"action_budget": -1},
            {"action_budget": 7},
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                run_microworld(**arguments)


if __name__ == "__main__":
    unittest.main()

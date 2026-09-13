import json
import unittest
from dataclasses import replace
from typing import Any, cast

import numpy as np

from text_factors.config import ModelConfig
from text_factors.experience import (
    ContextKey,
    ExperienceMemory,
    ObservedTransition,
    SparseCode,
    run_experience_demo,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        input_bits=16,
        output_bits=16,
        active_bits_per_symbol=2,
        positions=4,
        frame_size=2,
        context_count=4,
        receptive_bits=16,
        point_count=128,
        create_threshold=2,
        activation_threshold=2,
        min_active_points=1,
        probation_after=2,
        stable_after=3,
        prediction_vote_threshold=1,
        max_clusters_per_point=2,
        consolidation_method="coactivation",
        seed=42,
    )


def code(*bits: int) -> SparseCode:
    return SparseCode("test-v1", 16, bits)


class SparseCodeTests(unittest.TestCase):
    def test_mutating_input_and_output_arrays_cannot_change_saved_code(self) -> None:
        bits = np.zeros(16, dtype=np.bool_)
        bits[[2, 3]] = True
        sparse = SparseCode.from_array("test-v1", bits)
        bits[:] = False
        output = sparse.to_array()
        output[:] = False

        self.assertEqual(sparse.active_bits, (2, 3))
        self.assertEqual(int(np.count_nonzero(sparse.to_array())), 2)

    def test_invalid_encodings_do_not_silently_coerce(self) -> None:
        for active in ((1, 1), (2, 1), (-1,), (16,), (True,), [1, 2]):
            with self.subTest(active=active), self.assertRaises(ValueError):
                SparseCode("test-v1", 16, cast(Any, active))
        with self.assertRaisesRegex(ValueError, "boolean"):
            SparseCode.from_array("test-v1", cast(Any, np.asarray([0, 1])))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            SparseCode.from_array("test-v1", np.zeros((2, 8), dtype=np.bool_))
        with self.assertRaises(ValueError):
            SparseCode("", 16, ())


class ExperienceMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = ContextKey("view-a", "view-b", "move")
        self.source = code(0, 1)
        self.outcome = code(12, 13)
        self.bank = ExperienceMemory(tiny_config(), encoding_id="test-v1")

    def event(
        self, event_id: int, *, key: ContextKey | None = None
    ) -> ObservedTransition:
        return ObservedTransition(event_id, key or self.key, self.source, self.outcome)

    def test_forecasts_cannot_be_learned_as_confirmed_observations(self) -> None:
        forecast = self.bank.forecast(self.key, self.source)
        before = self.bank.stats()

        with self.assertRaisesRegex(TypeError, "ObservedTransition"):
            self.bank.learn(cast(Any, forecast))
        with self.assertRaisesRegex(TypeError, "SparseCode observations"):
            ObservedTransition(0, self.key, self.source, cast(Any, forecast))
        self.assertEqual(self.bank.stats(), before)
        self.assertEqual(forecast.predicted.active_bits, ())

    def test_retained_duplicate_does_not_inflate_support_or_stabilize(self) -> None:
        observation = self.event(10)
        self.assertTrue(self.bank.learn(observation))
        before = self.bank.stats()
        for _ in range(8):
            self.assertFalse(self.bank.learn(observation))
        self.assertEqual(self.bank.stats(), before)
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted.active_bits, ()
        )

        self.bank.learn(self.event(11))
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted.active_bits, ()
        )
        self.bank.learn(self.event(12))
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted, self.outcome
        )

    def test_conflicting_duplicate_is_rejected_before_learning(self) -> None:
        self.bank.learn(self.event(0))
        before = self.bank.stats()
        with self.assertRaisesRegex(ValueError, "different observed contents"):
            self.bank.learn(ObservedTransition(0, self.key, self.source, code(14, 15)))
        self.assertEqual(self.bank.stats(), before)

    def test_bounded_archive_never_relearns_an_evicted_or_late_event(self) -> None:
        bank = ExperienceMemory(tiny_config(), encoding_id="test-v1", max_events=2)
        for event_id in (10, 20, 30):
            bank.learn(self.event(event_id))
        before = bank.stats()
        self.assertEqual(before["retained_events"], 2)
        self.assertEqual(before["observations"], 3)
        self.assertEqual(
            bank.predecessors(self.key, self.outcome)[0].event_ids, (20, 30)
        )
        for event_id in (10, 15):
            with self.assertRaisesRegex(ValueError, "stale or evicted"):
                bank.learn(self.event(event_id))
        self.assertFalse(bank.learn(self.event(20)))
        self.assertEqual(bank.stats(), before)

    def test_operation_capacity_refusal_keeps_existing_memory_and_archive(self) -> None:
        bank = ExperienceMemory(tiny_config(), encoding_id="test-v1", max_operations=1)
        bank.learn(self.event(0))
        before = bank.stats()
        other = ContextKey("view-a", "view-b", "other")
        self.assertEqual(bank.forecast(other, self.source).training_events, 0)
        self.assertEqual(bank.stats(), before)
        with self.assertRaisesRegex(ValueError, "operation capacity"):
            bank.learn(self.event(1, key=other))
        self.assertEqual(bank.stats(), before)
        self.assertTrue(bank.learn(self.event(1)))

    def test_actions_and_source_and_target_contexts_are_isolated(self) -> None:
        other_keys = (
            ContextKey("view-a", "view-b", "stay"),
            ContextKey("different-source", "view-b", "move"),
            ContextKey("view-a", "different-target", "move"),
        )
        for event_id in range(3):
            self.bank.learn(self.event(event_id))
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted, self.outcome
        )
        before = self.bank.stats()
        for key in other_keys:
            self.assertEqual(
                self.bank.forecast(key, self.source).predicted.active_bits, ()
            )
            self.assertEqual(self.bank.predecessors(key, self.outcome), ())
        self.assertEqual(self.bank.stats(), before)

        for event_id in range(3, 6):
            self.bank.learn(
                ObservedTransition(event_id, other_keys[0], self.source, code(14, 15))
            )
        self.assertEqual(
            self.bank.forecast(other_keys[0], self.source).predicted, code(14, 15)
        )
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted, self.outcome
        )

    def test_historical_inverse_preserves_ambiguity_and_does_not_generalize(
        self,
    ) -> None:
        self.bank.learn(self.event(0))
        self.bank.learn(ObservedTransition(1, self.key, code(2, 3), self.outcome))
        self.bank.learn(self.event(2))
        predecessors = self.bank.predecessors(self.key, self.outcome)
        self.assertEqual(len(predecessors), 2)
        self.assertEqual(predecessors[0].source, self.source)
        self.assertEqual(predecessors[0].event_ids, (0, 2))
        self.assertEqual(predecessors[1].source, code(2, 3))
        self.assertEqual(predecessors[1].event_ids, (1,))
        self.assertEqual(self.bank.predecessors(self.key, code(12, 14)), ())

    def test_forecasts_are_read_only_and_replay_never_adds_evidence(self) -> None:
        self.bank.learn(self.event(0))
        before = self.bank.stats()
        for _ in range(6):
            self.bank.forecast(self.key, self.source)
            self.bank.predecessors(self.key, self.outcome)
        self.assertEqual(self.bank.stats(), before)
        for _ in range(6):
            self.bank.replay_consolidation()
        after = self.bank.stats()
        self.assertEqual(after["observations"], before["observations"])
        self.assertEqual(after["memory_steps"], before["memory_steps"])
        self.assertLessEqual(after["clusters"], before["clusters"])
        self.assertEqual(
            self.bank.forecast(self.key, self.source).predicted.active_bits, ()
        )

    def test_cluster_and_archive_counts_remain_within_declared_capacity(self) -> None:
        bank = ExperienceMemory(
            replace(tiny_config(), max_clusters_per_point=1),
            encoding_id="test-v1",
            max_events=3,
            max_operations=1,
        )
        for event_id in range(50):
            source = code(event_id % 8, event_id % 8 + 8)
            bank.learn(ObservedTransition(event_id, self.key, source, self.outcome))
        stats = bank.stats()
        self.assertEqual(stats["observations"], 50)
        self.assertEqual(stats["retained_events"], 3)
        self.assertLessEqual(stats["clusters"], stats["cluster_capacity"])

    def test_encoding_width_and_version_fail_before_any_memory_is_created(self) -> None:
        for bad_code in (SparseCode("test-v1", 17, ()), SparseCode("test-v2", 16, ())):
            for source, outcome in ((bad_code, self.outcome), (self.source, bad_code)):
                with self.assertRaises(ValueError):
                    self.bank.learn(ObservedTransition(0, self.key, source, outcome))
            with self.assertRaises(ValueError):
                self.bank.forecast(self.key, bad_code)
        self.assertEqual(self.bank.stats()["operations"], 0)
        self.assertEqual(self.bank.stats()["observations"], 0)

    def test_configuration_capacities_and_event_ids_require_integers(self) -> None:
        for name in ("max_events", "max_operations", "max_action_candidates"):
            for value in (0, -1, True, float("inf"), 2.5):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    ExperienceMemory(
                        tiny_config(), encoding_id="test-v1", **{name: cast(Any, value)}
                    )
        with self.assertRaisesRegex(ValueError, "equal input and output"):
            ExperienceMemory(
                replace(tiny_config(), output_bits=8), encoding_id="test-v1"
            )
        for event_id in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.event(cast(Any, event_id))

    def test_action_choice_uses_learned_outcomes_without_observing_candidates(
        self,
    ) -> None:
        useful = ContextKey("view-a", "view-b", "z-useful")
        useless = ContextKey("view-a", "view-b", "a-useless")
        for event_id in range(3):
            self.bank.learn(
                ObservedTransition(event_id, useful, self.source, self.outcome)
            )
        for event_id in range(3, 6):
            self.bank.learn(
                ObservedTransition(event_id, useless, self.source, code(14, 15))
            )
        before = self.bank.stats()
        chosen = self.bank.choose_action(
            self.source,
            [useless, useful],
            lambda forecast: float(12 in forecast.predicted.active_bits),
        )
        self.assertEqual(chosen.selected.forecast.key, useful)
        self.assertFalse(chosen.explored)
        self.assertEqual(self.bank.stats(), before)

    def test_ties_and_explicit_exploration_are_seeded_and_order_independent(
        self,
    ) -> None:
        keys = [self.key, ContextKey("view-a", "view-b", "alternative")]
        banks = [
            ExperienceMemory(tiny_config(), encoding_id="test-v1", policy_seed=7)
            for _ in range(2)
        ]
        tie = banks[0].choose_action(self.source, keys, lambda _: -2.0)
        self.assertEqual(tie.selected.forecast.key, min(keys))
        sequences = []
        for bank, candidates in zip(banks, (keys, list(reversed(keys))), strict=True):
            sequences.append(
                [
                    bank.choose_action(
                        self.source,
                        candidates,
                        lambda _: -2.0,
                        exploration_probability=1.0,
                    ).selected.forecast.key
                    for _ in range(12)
                ]
            )
            self.assertEqual(bank.stats()["observations"], 0)
            self.assertEqual(bank.stats()["operations"], 0)
        self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(set(sequences[0]), set(keys))

    def test_action_candidates_are_bounded_and_utility_must_be_finite(self) -> None:
        bank = ExperienceMemory(
            tiny_config(), encoding_id="test-v1", max_action_candidates=1
        )
        for candidates in ([], [self.key, self.key]):
            with self.assertRaisesRegex(ValueError, "candidate count"):
                bank.choose_action(self.source, candidates, lambda _: 1.0)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                bank.choose_action(
                    self.source, [self.key], lambda _, result=value: result
                )
        for probability in (-1.0, 1.1, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "exploration_probability"):
                bank.choose_action(
                    self.source,
                    [self.key],
                    lambda _: 1.0,
                    exploration_probability=probability,
                )
        self.assertEqual(bank.stats()["observations"], 0)
        self.assertEqual(bank.stats()["operations"], 0)


class ExperienceDemoTests(unittest.TestCase):
    def test_small_actual_outcome_loop_is_reproducible_and_separates_factors(
        self,
    ) -> None:
        first = run_experience_demo(seed=42)
        second = run_experience_demo(seed=42)
        for report in (first, second):
            timings = report.pop("timing")
            self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["training_observations"], 36)
        self.assertEqual(first["held_out_count"], 12)
        self.assertEqual(first["held_out_exact_match"], 1.0)
        self.assertEqual(first["held_out_bit_precision"], 1.0)
        self.assertEqual(first["held_out_bit_recall"], 1.0)
        self.assertTrue(first["cross_view_input_codes_distinct"])
        self.assertTrue(first["same_factor_cross_view_agreement_nonempty"])
        self.assertTrue(first["different_factor_codes_distinct_nonempty"])
        self.assertEqual(first["constant_output_exact_match"], 0.5)
        self.assertFalse(first["constant_output_distinguishes_factors"])
        self.assertEqual(first["retained_inverse_predecessors_for_one_outcome"], 6)
        self.assertTrue(first["scoring_did_not_learn"])
        self.assertEqual(first["selected_action"], "right")
        self.assertEqual(first["selected_forecast_bits"], [24, 25])
        self.assertEqual(first["actual_action_outcome_bits"], [24, 25])
        self.assertEqual(first["replay_added_observations"], 0)
        self.assertEqual(first["replay_added_memory_steps"], 0)
        self.assertEqual(first["final_stats"]["observations"], 37)
        self.assertGreater(
            first["final_stats"]["model_numpy_payload_bytes_lower_bound"], 0
        )
        train_distractors = set(range(4, 10))
        for record in first["held_out"]:
            self.assertNotIn(record["distractor_bit"], train_distractors)
            self.assertIn(record["distractor_bit"], record["source_bits"])
        view_a = {
            (record["scoring_factor_id"], record["distractor_bit"]): record[
                "source_bits"
            ]
            for record in first["held_out"]
            if record["action"] == "left"
        }
        for record in first["held_out"]:
            if record["action"] == "paired-view":
                self.assertNotEqual(
                    record["source_bits"],
                    view_a[record["scoring_factor_id"], record["distractor_bit"]],
                )


if __name__ == "__main__":
    unittest.main()

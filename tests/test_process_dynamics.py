"""Behavioral controls for the open P13 transition forecasting component.

All histories here are invented mechanical checks, never pilot observations.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from text_factors.real_data.contracts import RoleValue
from text_factors.real_data.process_dynamics import (
    ProcessDynamics,
    ProcessEvent,
    ProcessFact,
    TransitionCase,
    TransitionOutcome,
    TransitionPrompt,
    evaluate_open_stream,
)

DAY = timedelta(days=1)
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def case(
    episode: str,
    group: str,
    *,
    split: str = "train",
    subject: str = "task",
    old: str = "open",
    new: str = "done",
    relation: str = "change-status",
    day: int = 0,
    fact_prediction: tuple[ProcessFact, ...] | None = None,
) -> TransitionCase:
    before = (ProcessFact(subject, "status", old),)
    after = (ProcessFact(subject, "status", new),)
    prompt = TransitionPrompt(
        episode_id=episode,
        group_id=group,
        split=split,  # type: ignore[arg-type]
        before_at=START + day * DAY,
        cutoff_at=START + day * DAY,
        outcome_due_at=START + (day + 1) * DAY,
        before=before,
        event=ProcessEvent(
            relation,
            (
                RoleValue("task", subject),
                RoleValue("old", old),
                RoleValue("new", new),
            ),
        ),
        tracked_relations=("status",),
        before_complete=True,
    )
    return TransitionCase(
        prompt,
        TransitionOutcome(episode, prompt.outcome_due_at, after, after_complete=True),
        fact_gold=before if fact_prediction is not None else None,
        fact_prediction=fact_prediction,
    )


class ProcessDynamicsTests(unittest.TestCase):
    def test_role_bound_transition_on_unseen_entities_and_controlled_outcomes(
        self,
    ) -> None:
        model = ProcessDynamics(min_support_groups=2)
        model.fit_train(
            (
                case("a", "family-a", subject="ticket-a"),
                case("b", "family-b", subject="ticket-b"),
            )
        )
        observed = case(
            "c", "family-c", split="development", subject="novel-ticket", day=3
        )
        prediction = model.predict(observed.prompt)
        self.assertTrue(prediction.supported)
        self.assertEqual(prediction.support_groups, 2)
        self.assertEqual(prediction.after, observed.outcome.after)
        self.assertNotEqual(
            model.persistence_control(observed.prompt).after, observed.outcome.after
        )
        unknown = case(
            "d", "family-d", split="development", relation="unknown-change", day=4
        )
        self.assertFalse(model.predict(unknown.prompt).supported)

    def test_repeat_family_cannot_increase_independent_support(self) -> None:
        model = ProcessDynamics(min_support_groups=2)
        model.learn(case("a", "same-family"))
        for index in range(1, 20):
            model.learn(case(f"a-{index}", "same-family", day=index))
        probe = case("probe", "new-family", split="development", day=25)
        prediction = model.predict(probe.prompt)
        self.assertFalse(prediction.supported)
        self.assertEqual(prediction.support_groups, 1)
        model.learn(case("b", "second-family", day=20))
        self.assertTrue(model.predict(probe.prompt).supported)
        self.assertEqual(model.predict(probe.prompt).support_groups, 2)

    def test_dev_evaluation_is_frozen_and_fact_quality_requires_independent_gold(
        self,
    ) -> None:
        model = ProcessDynamics()
        model.fit_train((case("a", "a"), case("b", "b")))
        wrong_fact = (ProcessFact("task", "status", "wrong"),)
        development = (
            case("c", "c", split="development", day=3),
            case("d", "d", split="development", day=5, fact_prediction=wrong_fact),
        )
        result = evaluate_open_stream(model, development, split="development")
        self.assertEqual(result["updates_after_reveal"], 0)
        self.assertEqual(result["forecast"]["learned"]["independent_groups"], 2)
        self.assertEqual(result["forecast"]["learned"]["exact_per_group"], 1.0)
        self.assertEqual(result["forecast"]["persistence"]["exact_per_group"], 0.0)
        self.assertEqual(result["current_fact"]["episodes"], 1)
        self.assertEqual(result["current_fact"]["independent_groups"], 1)
        self.assertEqual(result["current_fact"]["group_exact_numerator"], 0.0)
        self.assertEqual(result["current_fact"]["exact_per_group"], 0.0)
        self.assertEqual(len(model.trained_groups), 2)
        with self.assertRaisesRegex(ValueError, "frozen"):
            model.learn(development[0])

    def test_future_stream_predicts_before_reveal_then_updates(self) -> None:
        model = ProcessDynamics(min_support_groups=2)
        first = case("f1", "future-1", split="future_stream", day=0)
        second = case("f2", "future-2", split="future_stream", day=2)
        third = case("f3", "future-3", split="future_stream", day=4)
        result = evaluate_open_stream(
            model, (first, second, third), split="future_stream"
        )
        self.assertEqual(result["updates_after_reveal"], 3)
        self.assertEqual(result["forecast"]["learned"]["coverage"], 1 / 3)
        self.assertEqual(result["forecast"]["learned"]["supported_episodes"], 1)
        self.assertEqual(result["forecast"]["learned"]["abstained_episodes"], 2)
        self.assertEqual(result["forecast"]["learned"]["group_coverage_numerator"], 1)
        self.assertEqual(result["forecast"]["persistence"]["coverage"], 1.0)
        self.assertEqual(result["forecast"]["learned"]["independent_groups"], 3)

    def test_outcome_is_not_accessed_until_a_forecast_is_recorded(self) -> None:
        order: list[str] = []
        unlocked = [True]

        class GuardedCase(TransitionCase):
            def __getattribute__(self, name: str):
                if name == "outcome" and unlocked[0] is False:
                    raise AssertionError("future label accessed before prediction")
                return super().__getattribute__(name)

        class SpyModel(ProcessDynamics):
            def predict(self, prompt: TransitionPrompt):
                order.append("predict")
                result = super().predict(prompt)
                unlocked[0] = True
                return result

            def learn(self, item: TransitionCase) -> None:
                order.append("learn")
                super().learn(item)

        model = SpyModel(min_support_groups=1)
        source = case("f", "future", split="future_stream")
        guarded = GuardedCase(source.prompt, source.outcome)
        unlocked[0] = False
        evaluate_open_stream(model, (guarded,), split="future_stream")
        self.assertEqual(order, ["predict", "learn"])
        with self.assertRaisesRegex(ValueError, "before the first future"):
            evaluate_open_stream(
                model,
                (case("d", "dev", split="development"),),
                split="development",
            )
        with self.assertRaisesRegex(ValueError, "train is frozen"):
            model.learn(case("t", "new-train-group"))

    def test_incomplete_state_does_not_turn_missing_into_negative_fact(self) -> None:
        with self.assertRaisesRegex(ValueError, "before must be complete"):
            original = case("p", "g").prompt
            TransitionPrompt(
                episode_id="partial",
                group_id="other",
                split="train",
                before_at=original.before_at,
                cutoff_at=original.cutoff_at,
                outcome_due_at=original.outcome_due_at,
                before=original.before,
                event=original.event,
                tracked_relations=("status",),
                before_complete=False,
            )
        with self.assertRaisesRegex(ValueError, "after must be complete"):
            TransitionOutcome("partial", START + DAY, ())

    def test_no_train_dev_overlap_or_sealed_and_no_backdated_outcome(self) -> None:
        model = ProcessDynamics()
        model.learn(case("a", "family"))
        with self.assertRaisesRegex(ValueError, "overlaps train"):
            evaluate_open_stream(
                model,
                (case("b", "family", split="development"),),
                split="development",
            )
        with self.assertRaisesRegex(ValueError, "sealed"):
            case("s", "s", split="sealed_test")
        prompt = case("c", "c", split="development").prompt
        with self.assertRaisesRegex(ValueError, "horizon"):
            TransitionCase(
                prompt,
                TransitionOutcome(
                    prompt.episode_id, prompt.cutoff_at, (), after_complete=True
                ),
            )
        later_model = ProcessDynamics()
        later_model.learn(case("later", "later-family", day=10))
        with self.assertRaisesRegex(ValueError, "predates a training outcome"):
            evaluate_open_stream(
                later_model,
                (case("early", "early-family", split="development", day=3),),
                split="development",
            )


if __name__ == "__main__":
    unittest.main()

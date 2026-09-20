"""Deterministic controls for revision transactions, not language quality."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace

from text_factors.conversation.schema import Budget
from text_factors.learning.attention import AttentionState, ReviewCue
from text_factors.learning.candidate_selection import select
from text_factors.learning.dynamics import TransitionPrediction
from text_factors.learning.hypotheses import (
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.revision import (
    Reinterpretation,
    commit_revision,
    prepare_archive_revision,
)
from text_factors.learning.schema import (
    DialogueContext,
    Event,
    Interpretation,
    Meaning,
    Query,
)
from text_factors.learning.world import ExperienceWorld


def location(subject="book", place="drawer"):
    return Meaning(
        "inform", Event("locate", object=subject, place=place, time="present")
    )


class Dynamics:
    """A learned-component test double whose licensed predictions are observable."""

    _experience = None

    def __init__(self):
        self.calls = []

    def score_interpretations(self, before, events, **kwargs):
        return [
            {
                "candidate": i,
                "score": 0.9 - i * 0.1,
                "supported": True,
                "contexts": [],
                "reason": "",
            }
            for i in range(len(events))
        ]

    def predict(self, before, event, **kwargs):
        self.calls.append((deepcopy(before), event))
        if not event.actual or (event.negated and event.predicate in {"move", "give"}):
            return TransitionPrediction((), True)
        relation = "location" if event.predicate in {"move", "locate"} else "holder"
        value = (
            event.place
            if relation == "location"
            else event.recipient
            if event.predicate == "give"
            else event.actor
        )
        effect = {
            "op": "exclude" if event.negated else "set",
            "subject": event.object,
            "relation": relation,
            "value": value,
            "spatial": event.spatial if relation == "location" else "in",
        }
        unchanged = any(
            f["subject"] == event.object
            and f["relation"] == relation
            and f["value"] == value
            and f["spatial"] == effect["spatial"]
            and f["negated"] == event.negated
            for f in before
        )
        return TransitionPrediction(() if unchanged else (effect,), True)


class HistoryRevisionTests(unittest.TestCase):
    def setUp(self):
        self.world = ExperienceWorld()
        self.archive = AttentionState("revision-test-model")
        self.dynamics = Dynamics()
        self.context = DialogueContext()

    def add(self, turn, *meanings, unresolved=False):
        observation = Observation(
            f"turn:{turn}", f"observed text {turn}", turn, f"сообщение {turn}"
        )
        candidates = tuple(
            Hypothesis.create(
                observation.observation_id, m, 0.0 if unresolved else i * 0.08
            )
            for i, m in enumerate(meanings)
        )
        batch = CandidateSet(observation, digest(self.context.to_dict()), candidates)
        before = [
            {k: f[k] for k in ("subject", "relation", "value", "negated", "spatial")}
            for f in self.world.facts()
        ]
        if unresolved:
            old_score = self.dynamics.score_interpretations
            self.dynamics.score_interpretations = lambda before, events, **kwargs: [
                {
                    "candidate": i,
                    "score": 0.9,
                    "supported": True,
                    "contexts": [],
                    "reason": "",
                }
                for i in range(len(events))
            ]
        result = select(
            batch,
            Interpretation(None),
            self.dynamics,
            before,
            model_fingerprint="revision-test-model",
            seconds=1,
            dependency_event_ids=tuple(
                sorted({f["event_id"] for f in self.world.facts()})
            ),
        )
        if unresolved:
            self.dynamics.score_interpretations = old_score
        assert result.diagnostics is not None
        self.archive.remember(result.diagnostics["hypotheses"], self.context, before)
        if result.meaning is not None:
            prediction = self.dynamics.predict(before, result.meaning.event)
            self.world.apply(
                result.meaning,
                prediction.effects,
                turn_id=turn,
                source=observation.source,
            )
        return observation

    def prepare(self, target, turn, meaning, **kwargs):
        after = self.archive.clone()
        cue = ReviewCue(
            f"turn:{target}",
            Observation(f"turn:{turn}", "clarification", turn, f"сообщение {turn}"),
            replace(meaning, act="correct"),
        )
        proposal = after.prepare_review(cue, self.dynamics, None)
        self.assertTrue(proposal["complete"], proposal)
        after.commit_reviews([proposal], self.dynamics)
        plan = prepare_archive_revision(
            self.world,
            self.archive,
            after,
            self.dynamics,
            turn_id=turn,
            source=cue.observation.source,
            **kwargs,
        )
        return plan, after

    def setup_branch(self, *, later_noop=False):
        self.add(1, location())
        self.add(2, location("toy", "box"))
        self.add(3, location("toy", "table"), location("book", "table"))
        if later_noop:
            self.add(4, location("toy", "table"))
        self.add(5, location("key", "pocket"))

    def test_review_preserves_original_events_and_independent_fact(
        self,
    ):
        self.setup_branch()
        original = self.world.events
        old_answer = self.world.query(Query("where", "toy"))
        plan, _ = self.prepare(3, 6, location("book", "table"))
        self.assertTrue(plan.complete, plan.reason)
        self.assertTrue(plan.applicable)
        self.assertEqual(self.world.events, original)
        outcome = commit_revision(self.world, plan)
        assert outcome is not None
        self.assertEqual(outcome["action"], "corrected")
        self.assertEqual(self.world.events, original)
        places = {f["subject"]: f["value"] for f in self.world.facts()}
        self.assertEqual(places, {"book": "table", "toy": "box", "key": "pocket"})
        self.assertEqual(self.world.at_turn(5).query(Query("where", "toy")), old_answer)
        self.assertNotIn(4, plan.affected_event_ids)
        self.assertIn(3, plan.affected_event_ids)

    def test_dependent_original_noop_gets_repredicted(self):
        self.setup_branch(later_noop=True)
        self.assertEqual(self.world.events[3]["effects"], [])
        plan, _ = self.prepare(3, 6, location("book", "table"))
        self.assertTrue(plan.complete, plan.reason)
        commit_revision(self.world, plan)
        toy = self.world.query(Query("where", "toy"))["assertions"][0]
        self.assertEqual((toy["value"], toy["event_id"]), ("table", 4))
        self.assertEqual(self.world.events[3]["effects"], [])
        assert plan.trace is not None
        self.assertEqual(plan.trace["replayed_event_ids"], [4])

    def test_previously_unresolved_turn_gets_stable_virtual_event_id(self):
        self.add(1, location())
        self.add(2, location("toy", "box"))
        self.add(
            3, location("toy", "table"), location("book", "table"), unresolved=True
        )
        self.add(4, location("key", "pocket"))
        plan, after = self.prepare(3, 5, location("book", "table"))
        self.assertTrue(plan.complete, plan.reason)
        commit_revision(self.world, plan)
        self.assertEqual([e["id"] for e in self.world.events], [1, 2, 3])
        resolved = self.world.query(Query("where", "book"))["assertions"][0]
        self.assertEqual(resolved["event_id"], 4)
        self.world.apply(Meaning("retract"), (), turn_id=6)
        self.assertEqual(
            self.world.query(Query("where", "book"))["assertions"][0]["value"], "drawer"
        )
        self.archive = after
        # Restore the archived version as the session does on revision undo.
        undone = self.world.retracted_revision(6)
        assert undone is not None
        self.archive.records[2] = undone["reviews"][0]["archive_before"]
        self.archive.generation += 1
        again, _ = self.prepare(3, 7, location("book", "table"))
        self.assertTrue(again.complete, again.reason)
        commit_revision(self.world, again)
        self.assertEqual(
            self.world.query(Query("where", "book"))["assertions"][0]["event_id"], 4
        )

    def test_roundtrip_replays_revisions_and_historical_answers(self):
        self.setup_branch(later_noop=True)
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        restored = ExperienceWorld.from_dict(
            json.loads(json.dumps(self.world.to_dict()))
        )
        self.assertEqual(restored.to_dict(), self.world.to_dict())
        self.assertEqual(restored.revision_outcome(6), plan.outcome)
        self.assertEqual(restored.at_turn(5).events, self.world.events)

    def test_confirmation_is_a_revision_and_does_not_overwrite_later_explicit_location(
        self,
    ):
        self.add(1, location("book", "table"))
        self.add(2, location("book", "shelf"))
        plan, _ = self.prepare(1, 3, location("book", "table"))
        self.assertTrue(plan.applicable)
        commit_revision(self.world, plan)
        self.assertEqual(
            self.world.query(Query("where", "book"))["assertions"][0]["value"], "shelf"
        )

    def test_revision_retraction_restores_versions_without_changing_its_stable_target(
        self,
    ):
        self.setup_branch()
        before = self.world.facts()
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        result = self.world.apply(Meaning("retract"), (), turn_id=7)
        self.assertEqual(result["reason"], "historical_revision_retracted")
        self.assertEqual(self.world.facts(), before)
        self.assertEqual(self.world.events[-1]["revision_target"], 1)
        self.assertEqual(
            ExperienceWorld.from_dict(self.world.to_dict()).facts(), before
        )

    def test_conflicting_reviews_suppress_assertions_and_guard_current_queries(self):
        self.setup_branch()
        first, after = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, first)
        self.archive = after
        conflict, _ = self.prepare(3, 7, location("toy", "table"))
        self.assertTrue(conflict.complete, conflict.reason)
        outcome = commit_revision(self.world, conflict)
        assert outcome is not None
        self.assertEqual(outcome["reason"], "historical_interpretation_uncertain")
        self.assertFalse(outcome["assertions"])
        self.assertEqual(self.world.uncertain_subjects, ("book", "toy"))
        for subject in self.world.uncertain_subjects:
            answer = self.world.query(Query("where", subject))
            self.assertEqual(answer["action"], "clarify")
            self.assertFalse(answer["assertions"])
        self.assertEqual(self.world.query(Query("where", "key"))["action"], "answer")
        restored = ExperienceWorld.from_dict(self.world.to_dict())
        self.assertEqual(restored.uncertain_subjects, ("book", "toy"))
        self.world.apply(Meaning("retract"), (), turn_id=8)
        self.assertFalse(self.world.uncertain_subjects)

    def test_later_explicit_evidence_limits_historical_uncertainty(self):
        self.setup_branch(later_noop=True)
        first, after = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, first)
        self.archive = after
        conflict, _ = self.prepare(3, 7, location("toy", "table"))
        commit_revision(self.world, conflict)
        self.assertEqual(self.world.uncertain_subjects, ("book",))
        self.assertEqual(self.world.query(Query("where", "toy"))["action"], "answer")

    def test_later_normal_retraction_keeps_target_then_undoes_revision(self):
        self.setup_branch()
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        updated = location("key", "desk")
        self.world.apply(
            updated, self.dynamics.predict([], updated.event).effects, turn_id=7
        )
        event_id = self.world.events[-1]["id"]
        self.world.apply(Meaning("retract"), (), turn_id=8)
        self.assertEqual(self.world.events[-1]["target"], event_id)
        self.assertNotIn("revision_target", self.world.events[-1])
        self.world.apply(Meaning("retract"), (), turn_id=9)
        self.assertEqual(self.world.events[-1]["revision_target"], 1)
        self.assertEqual(
            ExperienceWorld.from_dict(self.world.to_dict()).to_dict(),
            self.world.to_dict(),
        )

    def test_stale_plan_and_replay_budget_change_neither_world_nor_archive(self):
        self.setup_branch(later_noop=True)
        original, archive = self.world.to_dict(), self.archive.to_dict()
        limited, _ = self.prepare(3, 6, location("book", "table"), max_replayed=1)
        self.assertFalse(limited.complete)
        self.assertEqual(self.world.to_dict(), original)
        self.assertEqual(self.archive.to_dict(), archive)
        plan, _ = self.prepare(3, 6, location("book", "table"))
        self.world.apply(
            location("coin", "pocket"),
            self.dynamics.predict([], location("coin", "pocket").event).effects,
            turn_id=6,
        )
        modified = self.world.to_dict()
        with self.assertRaisesRegex(ValueError, "stale"):
            commit_revision(self.world, plan)
        self.assertEqual(self.world.to_dict(), modified)

    def test_expired_budget_leaves_state_unchanged(self):
        self.setup_branch()
        budget = Budget(1)
        budget.deadline = 0
        original = self.world.to_dict()
        plan, _ = self.prepare(3, 6, location("book", "table"), budget=budget)
        self.assertFalse(plan.complete)
        self.assertEqual(self.world.to_dict(), original)

    def test_serialized_effect_origin_and_dependency_forgery_rejected(self):
        self.setup_branch(later_noop=True)
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        for change in (
            lambda u: u["effects"][0].update(value="forged"),
            lambda u: u["observation"].update(text="rewritten observation"),
            lambda u: u.update(dependencies=[5]),
            lambda u: u.update(previous_digest="forged"),
        ):
            saved = self.world.to_dict()
            change(saved["revisions"][0]["updates"][0])
            with self.assertRaises((ValueError, KeyError)):
                ExperienceWorld.from_dict(saved)

    def test_explicit_dependent_reparse_hook_preserves_independent_events(self):
        self.setup_branch()
        self.add(
            6,
            Meaning(
                "inform", Event("give", actor="alice", object="toy", recipient="bob")
            ),
        )

        def reparse(observation, meaning, before, dependencies):
            if observation.turn_id == 6:
                return Reinterpretation(
                    replace(meaning, event=replace(meaning.event, object="book")), (3,)
                )
            return None

        plan, _ = self.prepare(3, 7, location("book", "table"), reparse=reparse)
        self.assertTrue(plan.complete, plan.reason)
        commit_revision(self.world, plan)
        self.assertEqual(
            self.world.query(Query("who_has", "book"))["assertions"][0]["value"], "bob"
        )
        self.assertEqual(
            self.world.query(Query("where", "toy"))["assertions"][0]["value"], "box"
        )
        self.assertEqual(
            self.world.query(Query("where", "key"))["assertions"][0]["value"], "pocket"
        )

    def test_consistently_forged_delta_requires_the_saved_review_proof(self):
        self.setup_branch()
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        saved = self.world.to_dict()
        update = saved["revisions"][0]["updates"][0]
        update["meaning"]["event"]["place"] = "forged"
        update["effects"][0]["value"] = "forged"
        saved["revisions"][0]["reviews"][0]["selected_id"] = Hypothesis.identity(
            "turn:3", Meaning.from_dict(update["meaning"]), ()
        )
        with self.assertRaises(ValueError):
            ExperienceWorld.from_dict(saved)

    def test_review_proof_rejects_another_cue_source_or_turn(self):
        self.setup_branch()
        plan, _ = self.prepare(3, 6, location("book", "table"))
        commit_revision(self.world, plan)
        for field, value in (("source", "another source"), ("turn_id", 100)):
            saved = self.world.to_dict()
            after = saved["revisions"][0]["reviews"][0]["archive_after"]
            after["cues"][0]["observation"][field] = value
            with self.assertRaises(ValueError):
                ExperienceWorld.from_dict(saved)


if __name__ == "__main__":
    unittest.main()

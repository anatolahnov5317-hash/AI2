"""Dependent-reread controls using public inputs, not a language benchmark."""

from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

from text_factors.conversation.schema import Budget, BudgetExceeded
from text_factors.learning.attention import AttentionState, ReviewCue
from text_factors.learning.candidate_selection import select
from text_factors.learning.dependent_reparse import make_reparser
from text_factors.learning.dynamics import TransitionPrediction
from text_factors.learning.hypotheses import (
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.language_data import ENTITY_BY_NAME
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact
from text_factors.learning.revision import commit_revision, prepare_archive_revision
from text_factors.learning.schema import (
    DialogueContext,
    Event,
    Interpretation,
    Meaning,
    Query,
)
from text_factors.learning.session import LearnedSession
from text_factors.learning.world import ExperienceWorld

MODEL = Path(__file__).resolve().parents[1] / "docs/results/v05_model_42.json"
FINGERPRINT = "dependent-reparse-test"
SEMANTIC = ("subject", "relation", "value", "negated", "spatial")


def meaning(event):
    names = tuple(
        dict.fromkeys(
            n for n in (event.object, event.actor, event.recipient, event.place) if n
        )
    )
    return Meaning(
        "inform",
        event,
        entities=tuple(ENTITY_BY_NAME[name].entity for name in names),
    )


def location(subject="игрушка", place="стол"):
    return meaning(Event("locate", object=subject, place=place, time="present"))


def semantic(world):
    return [{key: fact[key] for key in SEMANTIC} for fact in world.facts()]


class Dynamics:
    _experience = None

    def __init__(self):
        self.comparisons = []

    def score_interpretations(self, before, events, **kwargs):
        self.comparisons.append((deepcopy(before), tuple(events), kwargs))
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
        relation = "location" if event.predicate in {"locate", "move"} else "holder"
        value = (
            event.place
            if relation == "location"
            else event.recipient
            if event.predicate == "give"
            else event.actor
        )
        return TransitionPrediction(
            (
                {
                    "op": "set",
                    "subject": event.object,
                    "relation": relation,
                    "value": value,
                    "spatial": event.spatial,
                },
            ),
            True,
        )


class Understanding:
    def __init__(self, choose=None):
        self.calls = []
        self.choose = choose or (lambda ctx: (location(ctx.focus[0], "ящик"),))

    def propose(self, text, context, *, observation, limits):
        self.calls.append((text, context, observation, limits))
        return CandidateSet(
            observation,
            digest(context.to_dict()),
            tuple(
                Hypothesis.create(observation.observation_id, m, 0.0)
                for m in self.choose(context)
            ),
        )


class DependentReparseTests(unittest.TestCase):
    def setUp(self):
        self.world = ExperienceWorld()
        self.before = AttentionState(FINGERPRINT)
        self.dynamics = Dynamics()
        self.understanding = Understanding()
        self.context = DialogueContext()

    def add(self, turn, text, *meanings, unresolved=False):
        observation = Observation(f"turn:{turn}", text, turn, f"сообщение {turn}")
        batch = CandidateSet(
            observation,
            digest(self.context.to_dict()),
            tuple(
                Hypothesis.create(observation.observation_id, m, i * 0.08)
                for i, m in enumerate(meanings)
            ),
        )
        scorer = self.dynamics.score_interpretations
        if unresolved:

            def scorer(before, events, **kwargs):
                return [
                    {
                        "candidate": i,
                        "score": 0.9,
                        "supported": True,
                        "contexts": [],
                        "reason": "",
                    }
                    for i in range(len(events))
                ]

            batch = replace(
                batch,
                candidates=tuple(
                    replace(c, language_regret=0.0) for c in batch.candidates
                ),
            )
        facts = semantic(self.world)
        with patch.object(self.dynamics, "score_interpretations", side_effect=scorer):
            selected = select(
                batch,
                Interpretation(None),
                self.dynamics,
                facts,
                model_fingerprint=FINGERPRINT,
                seconds=1,
                dependency_event_ids=tuple(
                    sorted({f["event_id"] for f in self.world.facts()})
                ),
            )
        assert selected.diagnostics is not None
        self.before.remember(selected.diagnostics["hypotheses"], self.context, facts)
        if selected.meaning is not None and selected.meaning.event is not None:
            prediction = self.dynamics.predict(facts, selected.meaning.event)
            self.world.apply(
                selected.meaning,
                prediction.effects,
                turn_id=turn,
                source=observation.source,
            )
        self.context = LearnedSession._advance_context(
            self.context,
            text,
            selected.meaning,
            "clarify" if selected.meaning is None else "ack",
        )
        return observation

    def story(self, *, unresolved=False):
        self.add(1, "Книга в ящике", location("книга", "ящик"))
        self.add(2, "Игрушка в коробке", location("игрушка", "коробка"))
        self.add(
            3, "Она на столе", location(), location("книга"), unresolved=unresolved
        )

    def reviewed(self, turn, target=3, chosen=None):
        after = self.before.clone()
        chosen = chosen or location("книга")
        cue = ReviewCue(
            f"turn:{target}",
            Observation(
                f"turn:{turn}", "Нет, книга на столе", turn, f"сообщение {turn}"
            ),
            replace(chosen, act="correct"),
        )
        proposal = after.prepare_review(cue, self.dynamics, None)
        self.assertTrue(proposal["complete"], proposal)
        after.commit_reviews([proposal], self.dynamics)
        return after

    def prepare(self, after, turn, *, max_reparses=2, understanding=None):
        budget = Budget(5)
        hook = make_reparser(
            self.world,
            self.before,
            after,
            understanding or self.understanding,
            self.dynamics,
            model_fingerprint=FINGERPRINT,
            budget=budget,
            max_reparses=max_reparses,
        )
        plan = prepare_archive_revision(
            self.world,
            self.before,
            after,
            self.dynamics,
            turn_id=turn,
            source=f"сообщение {turn}",
            budget=budget,
            reparse=hook,
        )
        return plan, hook

    def test_chained_rereads_preserve_observations_archives_and_literal_names(self):
        self.story()
        self.add(4, "Она в ящике", location(place="ящик"))
        self.add(5, "Маша положила ее в ящик", location(place="ящик"))
        self.add(6, "Игрушка в коробке", location(place="коробка"))
        after = self.reviewed(7)
        original = self.world.to_dict()
        archives = (self.before.to_dict(), after.to_dict())
        plan, hook = self.prepare(after, 7)
        self.assertTrue(plan.complete, plan.reason)
        self.assertEqual(hook.trace["attempts"], 2)
        self.assertEqual(
            [r["dependency_event_ids"] for r in hook.trace["rereads"]], [[3], [3, 4]]
        )
        self.assertEqual(
            hook.trace["rereads"][1]["context"]["turns"],
            [r["observation"]["text"] for r in self.before.records[:4]],
        )
        self.assertTrue(
            all(
                "Нет, книга на столе" not in call[1].turns
                for call in self.understanding.calls
            )
        )
        self.assertEqual(self.world.to_dict(), original)
        self.assertEqual((self.before.to_dict(), after.to_dict()), archives)
        commit_revision(self.world, plan)
        self.assertEqual(self.world.events, original["events"])
        revised = {e["turn_id"]: e for e in self.world.effective_events}
        self.assertEqual(revised[4]["meaning"]["event"]["object"], "книга")
        self.assertEqual(revised[5]["meaning"]["event"]["object"], "книга")
        self.assertEqual(revised[6]["meaning"], original["events"][5]["meaning"])
        self.assertEqual(
            ExperienceWorld.from_dict(self.world.to_dict()).to_dict(),
            self.world.to_dict(),
        )

    def test_all_candidates_use_one_current_prefix_and_selection_is_learned(self):
        self.story()
        self.add(4, "Она в ящике", location(place="ящик"))
        # The model may choose a third entity. The reparser must not replace
        # the old object with the root's corrected object by a string rule.
        parser = Understanding(
            lambda ctx: (location("ключ", "ящик"), location("книга", "ящик"))
        )
        plan, hook = self.prepare(self.reviewed(5), 5, understanding=parser)
        self.assertTrue(plan.complete, plan.reason)
        read = hook.trace["rereads"][0]
        comparison = read["comparison"]
        calls = [
            c
            for c in self.dynamics.comparisons
            if c[2].get("observation_id") == "turn:4" and len(c[1]) == 2
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(comparison["snapshot"]["before_digest"], digest(calls[0][0]))
        self.assertTrue(
            any(f["subject"] == "книга" and f["value"] == "стол" for f in calls[0][0])
        )
        assert plan.world is not None
        self.assertEqual(
            plan.world.effective_events[-1]["meaning"]["event"]["object"], "ключ"
        )

    def test_missing_intervening_nonworld_observation_fails_closed(self):
        self.story()
        self.add(4, "Где игрушка?", Meaning("ask", query=Query("where", "игрушка")))
        self.add(5, "Она в ящике", location(place="ящик"))
        self.before.records.pop(3)
        after = self.reviewed(6)
        snapshot = self.world.to_dict()
        plan, hook = self.prepare(after, 6)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.reason, "revision_source_not_retained")
        self.assertEqual(hook.trace["attempts"], 0)
        self.assertEqual(self.world.to_dict(), snapshot)

    def test_reread_capacity_aborts_whole_revision_without_partial_commit(self):
        self.story()
        for turn in (4, 5, 6):
            self.add(turn, "Она в ящике", location(place="ящик"))
        after = self.reviewed(7)
        snapshots = (self.world.to_dict(), self.before.to_dict(), after.to_dict())
        plan, hook = self.prepare(after, 7)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.reason, "revision_reparse_capacity")
        self.assertEqual(hook.trace["attempts"], 2)
        self.assertEqual(
            (self.world.to_dict(), self.before.to_dict(), after.to_dict()), snapshots
        )

    def test_unresolved_root_uses_new_event_id_with_original_turn_provenance(self):
        self.story(unresolved=True)
        self.add(4, "Она в ящике", location(place="ящик"))
        virtual_id = self.world._next_id()
        plan, hook = self.prepare(self.reviewed(5), 5)
        self.assertTrue(plan.complete, plan.reason)
        self.assertEqual(hook.trace["rereads"][0]["dependency_event_ids"], [virtual_id])
        commit_revision(self.world, plan)
        updates = self.world.revisions[-1]["updates"]
        self.assertEqual(updates[0]["event_id"], virtual_id)
        self.assertEqual(updates[0]["observation"]["turn_id"], 3)
        self.assertEqual(updates[1]["interpretation_dependencies"], [virtual_id])

    def test_no_archive_change_or_unchanged_context_does_not_reread(self):
        self.story()
        self.add(4, "Она в ящике", location(place="ящик"))
        plan, hook = self.prepare(self.before.clone(), 5)
        self.assertTrue(plan.complete)
        self.assertFalse(plan.applicable)
        self.assertEqual(hook.trace["attempts"], 0)
        plan, hook = self.prepare(self.reviewed(5, chosen=location()), 5)
        self.assertTrue(plan.complete, plan.reason)
        self.assertEqual(hook.trace["attempts"], 0)
        self.assertEqual(self.understanding.calls, [])

    def test_incomplete_candidate_search_and_shared_deadline_fail_closed(self):
        self.story()
        observation = self.add(4, "Она в ящике", location(place="ящик"))
        after = self.reviewed(5)
        original = self.understanding.propose

        def incomplete(*args, **kwargs):
            return replace(
                original(*args, **kwargs),
                complete=False,
                stop_reason="candidate_expansion_budget",
            )

        with patch.object(self.understanding, "propose", side_effect=incomplete):
            plan, _ = self.prepare(after, 5)
        self.assertFalse(plan.complete)
        self.assertEqual(plan.reason, "revision_incomplete_candidate_search")
        budget = Budget(5)
        hook = make_reparser(
            self.world,
            self.before,
            after,
            self.understanding,
            self.dynamics,
            model_fingerprint=FINGERPRINT,
            budget=budget,
        )
        budget.deadline = 0
        with self.assertRaisesRegex(BudgetExceeded, "turn_time_budget"):
            hook(observation, location(place="ящик"), [], (3,))

    def test_public_fitted_parser_rereads_actual_text_under_revised_context(self):
        bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        names = ("книга", "петя", "миша", "маша", "стол")
        self.context = DialogueContext(
            entities=tuple(ENTITY_BY_NAME[n].entity for n in names),
            focus=names,
        )

        def root(actor):
            return meaning(Event("give", actor=actor, object="книга", recipient="маша"))

        self.add(1, "Он передал книгу Маше", root("петя"), root("миша"))
        text = "Он положил книгу в ящик"
        batch = bundle.understanding.propose(text, self.context)
        selected = select(
            batch,
            Interpretation(None),
            bundle.dynamics,
            [],
            model_fingerprint=FINGERPRINT,
            seconds=1,
        )
        self.assertIsNotNone(selected.meaning)
        self.add(2, text, selected.meaning)
        after = self.reviewed(3, target=1, chosen=root("миша"))
        self.dynamics = bundle.dynamics
        plan, hook = self.prepare(after, 3, understanding=bundle.understanding)
        self.assertTrue(plan.complete, plan.reason)
        self.assertEqual(hook.trace["attempts"], 1)
        read = hook.trace["rereads"][0]
        self.assertEqual(read["context"]["turns"], ["Он передал книгу Маше"])
        self.assertEqual(read["comparison"]["batch"]["observation"]["text"], text)
        self.assertTrue(read["changed"])
        assert plan.world is not None
        self.assertIsNotNone(plan.world.effective_events[-1]["meaning"])
        self.assertNotEqual(
            plan.world.effective_events[-1]["meaning"],
            self.world.effective_events[-1]["meaning"],
        )

    def test_session_roundtrip_with_controlled_cue_and_real_dependent_parser(self):
        bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        fingerprint = bundle.fingerprint
        cue = "Нет, Миша передал книгу Маше"
        propose = bundle.understanding.propose

        def controlled_cue(text, context, **kwargs):
            batch = propose(text, context, **kwargs)
            if text != cue:
                return batch
            # The public parser learns correction tense from present-state
            # examples. Supply this test's past-tense cue semantics explicitly;
            # every historical utterance and dependent reread uses fitted scores.
            selected = replace(
                meaning(Event("give", actor="миша", object="книга", recipient="маша")),
                act="correct",
            )
            return replace(
                batch,
                candidates=(
                    Hypothesis.create(
                        batch.observation.observation_id,
                        selected,
                        0.0,
                    ),
                ),
                complete=True,
                stop_reason="",
            )

        with patch.object(
            ModelBundle,
            "fingerprint",
            new_callable=PropertyMock,
            return_value=fingerprint,
        ):
            session = LearnedSession(bundle)
            for text in (
                "Петя положил игрушку в коробку",
                "Миша положил ключ в ящик",
                "Он передал книгу Маше",
                "Он положил книгу на стол",
            ):
                response = session.respond(text)
                self.assertEqual(response["action"], "ack", response)
            archive = deepcopy(session.attention.records)
            events = session.world.events
            with patch.object(
                bundle.understanding, "propose", side_effect=controlled_cue
            ):
                response = session.respond(cue)
            self.assertEqual(response["action"], "corrected", response)
            reads = response["diagnostics"]["understanding"]["details"][
                "dependent_reparse"
            ]["rereads"]
            self.assertEqual(len(reads), 1)
            self.assertTrue(reads[0]["changed"])
            self.assertEqual(reads[0]["dependency_event_ids"], [3])
            self.assertEqual(session.world.events, events)
            self.assertEqual(session.attention.records[3], archive[3])
            self.assertNotEqual(
                session.world.effective_events[3]["meaning"], events[3]["meaning"]
            )
            saved = session.to_dict()
            restored = LearnedSession.from_dict(saved, bundle)
            self.assertEqual(restored.to_dict(), saved)
            forged = deepcopy(saved)
            forged["history"][-1]["response"]["diagnostics"]["understanding"][
                "details"
            ]["dependent_reparse"]["rereads"][0]["dependency_event_ids"] = []
            with self.assertRaises(ValueError):
                LearnedSession.from_dict(forged, bundle)


if __name__ == "__main__":
    unittest.main()

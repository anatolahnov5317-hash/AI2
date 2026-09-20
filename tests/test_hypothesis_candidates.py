"""v6 block 1 contracts and public development regressions, not held-out scores."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_scene_recognition import bits, reader, trained_memory

from text_factors.context_affinity import AdaptiveContextAffinity
from text_factors.learning.candidate_search import SearchLimits
from text_factors.learning.candidate_selection import select
from text_factors.learning.hypotheses import (
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.language_data import ENTITY_BY_NAME, development_examples
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact
from text_factors.learning.schema import DialogueContext, Event, Interpretation, Meaning
from text_factors.learning.session import LearnedSession
from text_factors.recognition import ContextView, RecognitionLimits
from text_factors.scene_recognition import FactorSceneReader

MODEL = Path(__file__).resolve().parents[1] / "docs/results/v05_model_42.json"


def context(*names):
    return DialogueContext(
        entities=tuple(ENTITY_BY_NAME[n].entity for n in names), focus=names
    )


def fact(owner, subject="книга"):
    return {
        "subject": subject,
        "relation": "holder",
        "value": owner,
        "negated": False,
        "spatial": "in",
    }


class CandidateDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        cls.checkpoint = cls.bundle.to_dict()

    def tearDown(self):
        self.assertEqual(self.bundle.to_dict(), self.checkpoint)

    def test_public_development_targets_are_in_proposed_structures(self):
        for example in development_examples():
            with self.subTest(text=example.text):
                batch = self.bundle.understanding.propose(example.text, example.context)
                self.assertTrue(batch.complete)
                self.assertIn(
                    example.meaning, [h.meaning for h in batch.candidates if h.complete]
                )

    def test_ambiguous_reference_retains_both_candidates_before_selection(self):
        ctx = context("книга", "петя", "миша", "стол")
        text = "Он передал книгу Пете"
        self.assertIsNone(self.bundle.understanding.interpret(text, ctx).meaning)
        batch = self.bundle.understanding.propose(text, ctx)
        events = [h.meaning.event for h in batch.candidates if h.complete and h.meaning]
        self.assertIn(
            Event("give", actor="петя", object="книга", recipient="петя"), events
        )
        self.assertIn(
            Event("give", actor="миша", object="книга", recipient="петя"), events
        )
        for h in batch.candidates:
            if h.complete:
                self.assertIn(("object", 2, "книга"), h.bindings)
        self.assertEqual(batch.observation.text, text)

    def test_real_language_candidates_are_disambiguated_by_factor_experience(self):
        ctx = context("книга", "петя", "миша", "стол")
        text = "Он передал книгу Пете"
        initial = self.bundle.understanding.interpret(text, ctx)
        batch = self.bundle.understanding.propose(text, ctx, initial=initial)
        result = select(
            batch,
            initial,
            self.bundle.dynamics,
            [],
            model_fingerprint=self.bundle.fingerprint,
            seconds=1,
        )
        self.assertIsNotNone(result.meaning)
        assert result.meaning and result.meaning.event
        self.assertEqual(result.meaning.event.actor, "миша")
        assert result.diagnostics is not None
        self.assertEqual(
            result.diagnostics["hypotheses"]["reason"],
            "common_experience_disambiguation",
        )
        self.assertTrue(result.alternatives)

    def test_equal_memory_support_keeps_uncertainty(self):
        ctx = context("книга", "петя", "миша", "маша", "стол")
        text = "Он передал книгу Маше"
        initial = self.bundle.understanding.interpret(text, ctx)
        batch = self.bundle.understanding.propose(text, ctx, initial=initial)
        result = select(
            batch,
            initial,
            self.bundle.dynamics,
            [],
            model_fingerprint=self.bundle.fingerprint,
            seconds=1,
        )
        self.assertIsNone(result.meaning)
        self.assertGreaterEqual(len(result.alternatives), 2)

    def test_memory_changes_choice_between_same_structured_candidates(self):
        observation = Observation("sample", "неоднозначное сообщение")
        first = Meaning(
            "inform", Event("give", actor="петя", object="книга", recipient="миша")
        )
        second = Meaning(
            "inform", Event("give", actor="миша", object="книга", recipient="петя")
        )
        batch = CandidateSet(
            observation,
            digest(DialogueContext().to_dict()),
            tuple(Hypothesis.create("sample", m, 0.0) for m in (first, second)),
        )
        for owner, expected in [("петя", first), ("миша", second)]:
            result = select(
                batch,
                Interpretation(None),
                self.bundle.dynamics,
                [fact(owner)],
                model_fingerprint=self.bundle.fingerprint,
                seconds=1,
            )
            self.assertEqual(result.meaning, expected)
        # A different object's fact must not decide the unresolved book transfer.
        irrelevant = select(
            batch,
            Interpretation(None),
            self.bundle.dynamics,
            [fact("петя", "ключ")],
            model_fingerprint=self.bundle.fingerprint,
            seconds=1,
        )
        self.assertIsNone(irrelevant.meaning)

    def test_swapping_candidate_order_does_not_change_winner(self):
        observation = Observation("sample", "неоднозначное сообщение")
        meanings = [
            Meaning("inform", Event("give", actor=a, object="книга", recipient=b))
            for a, b in [("петя", "миша"), ("миша", "петя")]
        ]
        candidates = tuple(Hypothesis.create("sample", m, 0.0) for m in meanings)
        results = []
        for ordered in [candidates, tuple(reversed(candidates))]:
            batch = CandidateSet(
                observation, digest(DialogueContext().to_dict()), ordered
            )
            results.append(
                select(
                    batch,
                    Interpretation(None),
                    self.bundle.dynamics,
                    [fact("петя")],
                    model_fingerprint=self.bundle.fingerprint,
                    seconds=1,
                ).meaning
            )
        self.assertEqual(results, [meanings[0], meanings[0]])

    def test_prefix_keeps_observed_mentions_without_inventing_a_fact(self):
        batch = self.bundle.understanding.propose("Маша передала")
        self.assertTrue(batch.candidates)
        self.assertTrue(all(not c.complete for c in batch.candidates))
        self.assertIn(("mention", 0, "маша"), batch.candidates[0].bindings)
        session = LearnedSession(self.bundle)
        result = session.respond("Маша передала")
        self.assertIn(result["action"], ("clarify", "unknown"))
        self.assertFalse(session.world.events)
        self.assertFalse(result["assertions"])

    def test_unknown_words_are_not_salvaged_into_facts(self):
        batch = self.bundle.understanding.propose("Петя телепортировал книгу Маше")
        self.assertFalse(batch.candidates)

    def test_expansion_limit_is_visible_and_prevents_selection(self):
        initial = self.bundle.understanding.interpret("Петя передал книгу Маше")
        batch = self.bundle.understanding.propose(
            "Петя передал книгу Маше",
            initial=initial,
            limits=SearchLimits(max_expansions=1),
        )
        self.assertFalse(batch.complete)
        self.assertEqual(batch.expansions, 1)
        self.assertEqual(batch.stop_reason, "candidate_expansion_budget")
        with patch.object(
            self.bundle.dynamics,
            "score_interpretations",
            side_effect=AssertionError("incomplete search must not choose"),
        ):
            result = select(
                batch,
                initial,
                self.bundle.dynamics,
                [],
                model_fingerprint=self.bundle.fingerprint,
                seconds=1,
            )
        self.assertIsNone(result.meaning)

    def test_time_limit_marks_unexamined_instead_of_disproved(self):
        batch = self.bundle.understanding.propose(
            "Петя передал книгу Маше", limits=SearchLimits(seconds=1e-12)
        )
        self.assertFalse(batch.complete)
        self.assertEqual(batch.stop_reason, "candidate_time_budget")
        self.assertFalse(batch.candidates)

    def test_candidate_capacity_does_not_silently_collapse_ambiguity(self):
        batch = self.bundle.understanding.propose(
            "Он передал книгу Пете",
            context("книга", "петя", "миша", "стол"),
            limits=SearchLimits(max_candidates=1),
        )
        self.assertLessEqual(len(batch.candidates), 1)
        self.assertFalse(batch.complete)

    def test_session_calls_candidate_comparison_before_projection(self):
        session = LearnedSession(self.bundle)
        compare = self.bundle.dynamics.score_interpretations
        seen = []

        def checked(*args, **kwargs):
            seen.append(session.world.facts())
            return compare(*args, **kwargs)

        with patch.object(
            self.bundle.dynamics, "score_interpretations", side_effect=checked
        ):
            response = session.respond("Петя передал книгу Маше")
        self.assertEqual(seen, [()])
        self.assertEqual(response["action"], "ack")
        self.assertEqual(session.world.facts()[0]["value"], "маша")

    def test_timeout_during_candidate_comparison_rolls_back_everything(self):
        session = LearnedSession(self.bundle)
        snapshot = session.to_dict()
        with patch.object(
            self.bundle.dynamics, "score_interpretations", side_effect=TimeoutError
        ):
            response = session.respond("Петя передал книгу Маше")
        self.assertEqual(response["action"], "limit")
        self.assertEqual(session.to_dict(), snapshot)

    def test_truncated_candidate_comparison_rolls_back_everything(self):
        session = LearnedSession(self.bundle)
        snapshot = session.to_dict()
        with patch.object(
            self.bundle.dynamics, "score_interpretations", return_value=()
        ):
            response = session.respond("Петя передал книгу Маше")
        self.assertEqual(response["action"], "error")
        self.assertEqual(session.to_dict(), snapshot)

    def test_same_words_on_different_turns_keep_distinct_observations(self):
        a = self.bundle.understanding.propose(
            "Книга в ящике", observation=Observation("turn:1", "Книга в ящике", 1)
        )
        b = self.bundle.understanding.propose(
            "Книга в ящике", observation=Observation("turn:2", "Книга в ящике", 2)
        )
        self.assertNotEqual(
            a.candidates[0].hypothesis_id, b.candidates[0].hypothesis_id
        )
        self.assertEqual(a.candidates[0].meaning, b.candidates[0].meaning)

    def test_resume_retains_candidate_provenance_and_matches_uninterrupted_run(self):
        first = LearnedSession(self.bundle)
        for text in ["Маша положила книгу в ящик", "Маша передала книгу Пете"]:
            self.assertEqual(first.respond(text)["action"], "ack")
        snapshot = json.loads(json.dumps(first.to_dict(), ensure_ascii=False))
        resumed = LearnedSession.from_dict(snapshot, self.bundle)
        for text in ["У кого книга?", "Петя обещал передать книгу Маше", "Где книга?"]:
            left = first.respond(text)
            right = resumed.respond(text)
            left.pop("elapsed_seconds")
            right.pop("elapsed_seconds")
            self.assertEqual(left, right)
        trace = snapshot["history"][1]["response"]["diagnostics"]["understanding"][
            "details"
        ]["hypotheses"]
        self.assertEqual(trace["snapshot"]["dependency_event_ids"], [1])
        self.assertEqual(
            trace["batch"]["observation"]["text"], "Маша передала книгу Пете"
        )

    def test_changed_observation_is_rejected_on_restore(self):
        session = LearnedSession(self.bundle)
        session.respond("Петя передал книгу Маше")
        snapshot = session.to_dict()
        trace = snapshot["history"][0]["response"]["diagnostics"]["understanding"][
            "details"
        ]["hypotheses"]
        trace["batch"]["observation"]["text"] = "Маша передала книгу Пете"
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(snapshot, self.bundle)

    def test_mixed_memory_versions_are_rejected_on_restore(self):
        session = LearnedSession(self.bundle)
        session.respond("Петя передал книгу Маше")
        snapshot = session.to_dict()
        trace = snapshot["history"][0]["response"]["diagnostics"]["understanding"][
            "details"
        ]["hypotheses"]
        trace["reads"][0]["contexts"][1]["memory_step"] += 1
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(snapshot, self.bundle)

    def test_saved_choice_cannot_claim_another_context_even_with_new_digest(self):
        session = LearnedSession(self.bundle)
        session.respond("Петя передал книгу Маше")
        snapshot = session.to_dict()
        trace = snapshot["history"][0]["response"]["diagnostics"]["understanding"][
            "details"
        ]["hypotheses"]
        other = digest(context("миша").to_dict())
        trace["batch"]["context_digest"] = other
        trace["snapshot"]["context_digest"] = other
        trace["snapshot_id"] = digest(trace["snapshot"])
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(snapshot, self.bundle)

    def test_comparison_budget_is_shared_across_the_whole_batch(self):
        with self.assertRaises(TimeoutError):
            self.bundle.dynamics.score_interpretations(
                [],
                [Event("give", actor="петя", object="книга", recipient="маша")] * 8,
                seconds=1e-12,
            )

    def test_memory_change_mid_batch_is_rejected(self):
        dynamics = ModelBundle.from_dict(self.checkpoint).dynamics
        original = dynamics._compatibility

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            dynamics._experience.memory.step += 1
            return result

        with (
            patch.object(dynamics, "_compatibility", side_effect=changed),
            self.assertRaises(ValueError),
        ):
            dynamics.score_interpretations(
                [], [Event("give", actor="петя", object="книга", recipient="маша")]
            )

    def test_candidate_contract_roundtrip_and_rejects_forged_identity(self):
        batch = self.bundle.understanding.propose("Петя передал книгу Маше")
        data = json.loads(json.dumps(batch.to_dict(), ensure_ascii=False))
        self.assertEqual(CandidateSet.from_dict(data), batch)
        changed = deepcopy(data)
        changed["candidates"][0]["meaning"]["event"]["actor"] = "миша"
        with self.assertRaises(ValueError):
            CandidateSet.from_dict(changed)
        with self.assertRaises(ValueError):
            replace(batch, candidates=batch.candidates * 9)
        with self.assertRaises(ValueError):
            SearchLimits(max_expansions=True)
        with self.assertRaises(ValueError):
            SearchLimits(seconds=float("nan"))


class SceneResponseContractTests(unittest.TestCase):
    def test_portrait_activation_uses_scene_gate_when_raw_gate_is_stricter(self):
        memory = trained_memory()
        memory.config = replace(memory.config, min_active_points=4)
        scene = FactorSceneReader(memory)
        self.assertTrue(scene.observe_portrait("portrait", bits(0, 1, 2, 3)))
        result = scene.recognize_views(
            [ContextView("a", "a", bits(0, 1, 2, 3), (0,), "observation")],
            total_views=1,
        )
        self.assertTrue(result.proposals)
        self.assertTrue(result.recognition.responses[0].active)
        self.assertTrue(
            AdaptiveContextAffinity(scene.encoding_id).observe(
                "one", result.recognition
            )
        )

    def test_scene_responses_include_active_and_silent_reads_for_affinity(self):
        scene = reader()
        result = scene.recognize_views(
            [
                ContextView("a", "a", bits(0, 1, 2, 3), (0, 1), "observation"),
                ContextView("b", "b", bits(9, 10), (0, 1), "observation"),
            ],
            total_views=2,
        )
        self.assertEqual(len(result.recognition.responses), 2)
        self.assertEqual(
            [r.active for r in result.recognition.responses], [True, False]
        )
        affinity = AdaptiveContextAffinity(scene.encoding_id)
        self.assertTrue(affinity.observe("test", result.recognition))
        stats = affinity.pair_statistics()[0]
        self.assertEqual(stats["n10"], 1)

    def test_copied_view_keeps_same_input_digest_and_origin(self):
        scene = reader()
        result = scene.recognize_views(
            [
                ContextView(name, name, bits(0, 1, 2, 3), (0,), "observation")
                for name in ["a", "b"]
            ],
            total_views=2,
        )
        self.assertEqual(
            result.recognition.responses[0].input_digest,
            result.recognition.responses[1].input_digest,
        )
        self.assertEqual(
            {r.observation_id for r in result.recognition.responses}, {"observation"}
        )

    def test_interrupted_view_does_not_emit_a_completed_response(self):
        scene = reader()
        result = scene.recognize_views(
            [
                ContextView(name, name, bits(0, 1, 2, 3), (0,), "observation")
                for name in ["a", "b"]
            ],
            total_views=2,
            limits=RecognitionLimits(max_views=1),
        )
        self.assertFalse(result.complete)
        self.assertEqual(len(result.recognition.responses), 1)
        self.assertEqual(result.recognition.examined_views, 1)

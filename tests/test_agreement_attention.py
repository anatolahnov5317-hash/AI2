"""Block 2 causal controls and persistence checks; not a language benchmark."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from text_factors.conversation.schema import ConversationLimits
from text_factors.learning.agreement import (
    Claim,
    EvidenceLedger,
    Scope,
    exchange,
    relation,
)
from text_factors.learning.attention import (
    AttentionLimits,
    AttentionState,
    ReviewCue,
    Work,
)
from text_factors.learning.attention_ranker import AttentionRanker, fit
from text_factors.learning.candidate_selection import select
from text_factors.learning.hypotheses import (
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from text_factors.learning.language_data import ENTITY_BY_NAME
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact
from text_factors.learning.schema import DialogueContext, Event, Interpretation, Meaning
from text_factors.learning.session import LearnedSession

MODEL = Path(__file__).resolve().parents[1] / "docs/results/v05_model_42.json"


def context():
    names = ("книга", "петя", "миша", "маша", "стол")
    return DialogueContext(
        entities=tuple(ENTITY_BY_NAME[n].entity for n in names), focus=names
    )


def transfer(actor="петя", obj="книга"):
    return Meaning(
        "inform", Event("give", actor=actor, object=obj, recipient="маша", time="past")
    )


def cue(target=1, turn=100, actor="миша", obj="книга"):
    return ReviewCue(
        f"turn:{target}",
        Observation(f"turn:{turn}", "Уточнение участника", turn),
        transfer(actor, obj),
    )


class AgreementTests(unittest.TestCase):
    def test_all_five_relations_are_distinct(self):
        scope = Scope("one")
        a = Claim("a", scope, "actor", "петя", frozenset({"source"}))
        self.assertEqual(relation(a, replace(a, claim_id="b")), "duplicate")
        self.assertEqual(
            relation(a, replace(a, claim_id="b", roots=frozenset({"other"}))), "support"
        )
        self.assertEqual(
            relation(a, replace(a, claim_id="b", value="миша")), "conflict"
        )
        self.assertEqual(
            relation(a, replace(a, claim_id="b", slot="recipient")), "coexistence"
        )
        self.assertEqual(
            relation(a, replace(a, claim_id="b", depends_on=frozenset({"a"}))),
            "dependency",
        )

    def test_time_speaker_modality_and_event_identity_bound_conflicts(self):
        a = Claim("a", Scope("one"), "place", "ящик")
        for scope in [
            Scope("two"),
            Scope("one", time="past"),
            Scope("one", speaker="petya"),
            Scope("one", mode="reported"),
        ]:
            self.assertEqual(
                relation(a, Claim("b", scope, "place", "стол")), "coexistence"
            )

    def test_cloned_heads_and_cross_channel_rereads_do_not_add_roots(self):
        ledger = EvidenceLedger("snapshot")
        for _ in range(20):
            for channel in ["language", "memory", "history"]:
                ledger.observe(
                    "one episode", channel, 0.8, snapshot_id="snapshot", complete=True
                )
        self.assertEqual(
            ledger.to_dict(),
            {"one episode": {"language": 0.8, "memory": 0.8, "history": 0.8}},
        )

    def test_incomplete_and_stale_reads_do_not_enter_ledger(self):
        ledger = EvidenceLedger("one")
        ledger.observe("root", "memory", 1, snapshot_id="one", complete=False)
        self.assertFalse(ledger.to_dict())
        with self.assertRaises(ValueError):
            ledger.observe("root", "memory", 1, snapshot_id="old", complete=True)

    def test_support_cycle_transmits_one_root_only(self):
        nodes = tuple(
            Claim(
                n,
                Scope("one"),
                "actor",
                "петя",
                frozenset({"root"}) if n == "a" else frozenset(),
            )
            for n in "abc"
        )
        links = (("a", "b"), ("b", "c"), ("c", "a"))
        result = exchange(nodes, links)
        self.assertTrue(result["complete"])
        self.assertEqual(result["roots"], {n: ["root"] for n in "abc"})
        self.assertEqual(
            exchange(tuple(reversed(nodes)), tuple(reversed(links))), result
        )

    def test_cycle_without_observation_does_not_create_support(self):
        nodes = tuple(Claim(n, Scope("one"), "actor", "петя") for n in "ab")
        self.assertEqual(
            exchange(nodes, (("a", "b"), ("b", "a")))["roots"], {"a": [], "b": []}
        )

    def test_round_budget_marks_propagation_incomplete(self):
        nodes = (
            Claim("a", Scope("one"), "x", "v", frozenset({"root"})),
            Claim("b", Scope("one"), "x", "v"),
        )
        result = exchange(nodes, (("a", "b"),), max_rounds=1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["reason"], "agreement_round_budget")

    def test_conflict_cannot_be_relabelled_as_positive_support(self):
        nodes = (
            Claim("a", Scope("one"), "actor", "петя"),
            Claim("b", Scope("one"), "actor", "миша"),
        )
        with self.assertRaises(ValueError):
            exchange(nodes, (("a", "b"),))


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        cls.checkpoint = cls.bundle.to_dict()
        cls.fingerprint = cls.bundle.fingerprint

    def tearDown(self):
        self.assertEqual(self.bundle.to_dict(), self.checkpoint)

    def state(self, **limits):
        return AttentionState(self.fingerprint, AttentionLimits(**limits))

    def add(self, state, number=1, obj="книга", *, real=False):
        ctx = context()
        observation = Observation(f"turn:{number}", "Он передал книгу Маше", number)
        if real:
            batch = self.bundle.understanding.propose(
                observation.text, ctx, observation=observation
            )
        else:
            batch = CandidateSet(
                observation,
                digest(ctx.to_dict()),
                tuple(
                    Hypothesis.create(
                        observation.observation_id, transfer(actor, obj), regret
                    )
                    for actor, regret in [("петя", 0.0), ("миша", 0.08)]
                ),
            )
        result = select(
            batch,
            Interpretation(None),
            self.bundle.dynamics,
            [],
            model_fingerprint=self.fingerprint,
            seconds=1,
        )
        assert result.diagnostics is not None
        state.remember(result.diagnostics["hypotheses"], ctx, [])
        return batch

    def review(self, state, value=None, *, work=None):
        return state.prepare_review(
            value or cue(), self.bundle.dynamics, self.bundle.understanding, work=work
        )

    def commit(self, state, proposal):
        state.commit_reviews([proposal], self.bundle.dynamics)

    def test_late_cue_returns_previously_suppressed_candidate(self):
        state = self.state()
        batch = self.add(state)
        before = deepcopy(state.records[0])
        self.assertEqual(before["selected_meaning"]["event"]["actor"], "петя")
        proposal = self.review(state)
        self.assertTrue(proposal["complete"])
        self.assertTrue(proposal["summary"]["changed"])
        self.assertEqual(state.records[0], before)
        self.commit(state, proposal)
        r = state.records[0]
        self.assertEqual(r["selected_meaning"]["event"]["actor"], "миша")
        self.assertEqual(r["observation"], batch.observation.to_dict())
        self.assertEqual(r["initial_suppressed"], before["suppressed"])
        self.assertEqual(r["before"], before["before"])

    def test_real_language_alternatives_are_returned_after_clarification(self):
        state = self.state()
        self.add(state, real=True)
        self.assertIsNone(state.records[0]["selected_id"])
        proposal = self.review(state)
        self.commit(state, proposal)
        self.assertEqual(state.records[0]["selected_meaning"]["event"]["actor"], "миша")

    def test_paired_distractor_does_not_change_archived_choice(self):
        state = self.state()
        self.add(state)
        snapshot = state.to_dict()
        proposal = self.review(state, cue(obj="ключ"))
        self.assertEqual(proposal["reason"], "inapplicable_scope")
        self.commit(state, proposal)
        self.assertEqual(state.to_dict(), snapshot)

    def test_relevant_confirmation_preserves_existing_winner(self):
        state = self.state()
        self.add(state)
        proposal = self.review(state, cue(actor="петя"))
        self.assertFalse(proposal["summary"]["changed"])
        self.commit(state, proposal)
        self.assertEqual(state.records[0]["selected_meaning"]["event"]["actor"], "петя")

    def test_conflicting_clarifications_remain_uncertain_not_latest_wins(self):
        state = self.state()
        self.add(state)
        self.commit(state, self.review(state, cue(actor="петя", turn=10)))
        proposal = self.review(state, cue(actor="миша", turn=11))
        self.assertEqual(proposal["reason"], "ambiguous_conflicting_clarifications")
        self.commit(state, proposal)
        self.assertIsNone(state.records[0]["selected_id"])
        self.assertEqual(len(state.records[0]["cues"]), 2)

    def test_repeated_read_of_same_cue_does_not_add_evidence(self):
        state = self.state()
        self.add(state)
        self.commit(state, self.review(state))
        evidence = deepcopy(state.records[0]["review"]["evidence"])
        proposal = self.review(state)
        self.assertEqual(proposal["summary"]["evidence"], evidence)
        self.assertFalse(proposal["summary"]["changed"])

    def test_later_time_and_other_speaker_do_not_revise_earlier_event(self):
        state = self.state()
        self.add(state)
        original = cue()
        assert original.meaning.event is not None
        altered = replace(
            original,
            meaning=replace(
                original.meaning, event=replace(original.meaning.event, time="future")
            ),
        )
        for value in [altered, replace(original, speaker="someone_else")]:
            proposal = self.review(state, value)
            self.assertEqual(proposal["reason"], "inapplicable_scope")
            self.assertEqual(proposal["work"]["memory_calls"], 0)

    def test_all_candidates_use_same_original_world_and_current_memory(self):
        state = self.state()
        batch = self.add(state)
        compare = self.bundle.dynamics.score_interpretations
        seen = []

        def checked(before, events, **kwargs):
            seen.append((deepcopy(before), len(events)))
            return compare(before, events, **kwargs)

        with patch.object(
            self.bundle.dynamics, "score_interpretations", side_effect=checked
        ):
            proposal = self.review(state)
        self.assertEqual(seen, [([], len(batch.candidates))])
        rows = proposal["summary"]["comparison"]["reads"]
        self.assertEqual(
            len(
                {
                    (c["memory_namespace"], c["memory_step"])
                    for row in rows
                    for c in row["contexts"]
                }
            ),
            1,
        )

    def test_old_relevant_episode_found_among_sixty_distractors(self):
        state = self.state(max_archives=4)
        self.add(state)
        for n in range(2, 62):
            self.add(state, n, obj="ключ")
        work = Work(state.limits)
        selected = state.retrieve(transfer("миша"), work)
        self.assertEqual(
            [r["observation"]["observation_id"] for r in selected], ["turn:1"]
        )
        self.assertLessEqual(work.counts["scanned"], state.limits.max_scanned)
        self.assertTrue(self.review(state)["summary"]["regenerated"])

    def test_rebuild_after_candidate_eviction_uses_saved_source(self):
        state = self.state(max_archives=1)
        self.add(state, real=True)
        self.add(state, 2, obj="ключ")
        self.assertIsNone(state.records[0]["batch"])
        proposal = self.review(state)
        self.assertTrue(proposal["complete"])
        self.assertTrue(proposal["summary"]["regenerated"])
        self.assertGreater(proposal["work"]["expansions"], 0)
        self.commit(state, proposal)
        self.assertEqual(state.records[0]["selected_meaning"]["event"]["actor"], "миша")

    def test_missing_source_is_not_reconstructed_from_winning_summary(self):
        state = self.state(max_sources=2, max_archives=1)
        for n in range(1, 4):
            self.add(state, n)
        proposal = self.review(state)
        self.assertEqual(proposal["reason"], "source_not_retained")
        self.assertIsNone(proposal["record"])
        self.assertEqual(state.evicted_sources, 1)

    def test_missing_alternative_rebuilt_from_source_with_new_cue_context(self):
        state = self.state()
        names = ("книга", "ключ", "петя", "миша", "маша", "стол")
        ctx = DialogueContext(
            entities=tuple(ENTITY_BY_NAME[n].entity for n in names), focus=names
        )
        observed = Observation("turn:1", "Он передал книгу Маше", 1)
        batch = self.bundle.understanding.propose(
            observed.text, ctx, observation=observed
        )
        self.assertFalse(
            any(
                c.meaning and c.meaning.event and c.meaning.event.actor == "петя"
                for c in batch.candidates
            )
        )
        result = select(
            batch,
            Interpretation(None),
            self.bundle.dynamics,
            [],
            model_fingerprint=self.fingerprint,
            seconds=1,
        )
        assert result.diagnostics is not None
        state.remember(result.diagnostics["hypotheses"], ctx, [])
        proposal = self.review(state, cue(actor="петя"))
        self.assertTrue(proposal["complete"])
        self.assertTrue(proposal["summary"]["regenerated"])
        self.commit(state, proposal)
        self.assertEqual(state.records[0]["selected_meaning"]["event"]["actor"], "петя")
        AttentionState.from_dict(state.to_dict(), self.fingerprint)

    def test_explicit_reference_hint_changes_snapshot_but_not_observation(self):
        observed = Observation("turn:1", "Он передал книгу Маше", 1)
        first = self.bundle.understanding.propose(
            observed.text, context(), observation=observed
        )
        second = self.bundle.understanding.propose(
            observed.text,
            context(),
            observation=observed,
            reference_hints={"actor": "петя"},
        )
        self.assertEqual(first.observation, second.observation)
        self.assertNotEqual(first.context_digest, second.context_digest)
        self.assertTrue(
            any(
                c.meaning and c.meaning.event and c.meaning.event.actor == "петя"
                for c in second.candidates
            )
        )

    def test_reference_hint_cannot_overwrite_an_explicit_name(self):
        batch = self.bundle.understanding.propose(
            "Миша передал книгу Маше", context(), reference_hints={"actor": "петя"}
        )
        self.assertFalse(
            any(
                c.meaning and c.meaning.event and c.meaning.event.actor == "петя"
                for c in batch.candidates
            )
        )
        with self.assertRaises(ValueError):
            self.bundle.understanding.propose(
                "Он передал книгу Маше",
                context(),
                reference_hints={"invented_role": "петя"},
            )

    def test_stale_archive_result_rejected_atomically(self):
        state = self.state()
        self.add(state)
        proposal = self.review(state)
        self.add(state, 2, obj="ключ")
        saved = state.to_dict()
        with self.assertRaises(ValueError):
            self.commit(state, proposal)
        self.assertEqual(state.to_dict(), saved)

    def test_memory_change_between_review_and_commit_rejected(self):
        state = self.state()
        self.add(state)
        model = ModelBundle.from_dict(self.checkpoint)
        proposal = state.prepare_review(cue(), model.dynamics, model.understanding)
        snapshot = state.to_dict()
        model.dynamics._experience.memory.step += 1
        with self.assertRaises(ValueError):
            state.commit_reviews([proposal], model.dynamics)
        self.assertEqual(state.to_dict(), snapshot)

    def test_completion_order_does_not_change_atomic_batch(self):
        state = self.state()
        self.add(state)
        self.add(state, 2, obj="ключ")
        a = self.review(state, cue(1, 10))
        b = self.review(state, cue(2, 11, obj="ключ"))
        other = state.clone()
        state.commit_reviews([a, b], self.bundle.dynamics)
        other.commit_reviews([b, a], self.bundle.dynamics)
        self.assertEqual(state.to_dict(), other.to_dict())

    def test_deadline_and_memory_call_budget_leave_state_unchanged(self):
        state = self.state()
        self.add(state)
        snapshot = state.to_dict()
        with patch.object(
            self.bundle.dynamics, "score_interpretations", side_effect=TimeoutError
        ):
            proposal = self.review(state)
        self.assertFalse(proposal["complete"])
        self.assertEqual(state.to_dict(), snapshot)
        work = Work(state.limits)
        work.counts["memory_calls"] = state.limits.max_memory_calls
        self.assertFalse(self.review(state, work=work)["complete"])
        with self.assertRaises(ValueError):
            self.commit(state, proposal)
        self.assertEqual(state.to_dict(), snapshot)

    def test_round_limit_does_not_commit_partial_exchange(self):
        state = self.state(max_rounds=1)
        self.add(state)
        self.commit(state, self.review(state, cue(turn=10)))
        saved = state.to_dict()
        proposal = self.review(state, cue(turn=11))
        self.assertFalse(proposal["complete"])
        self.assertEqual(proposal["reason"], "agreement_round_budget")
        self.assertEqual(state.to_dict(), saved)

    def test_reconstruction_expansion_limit_keeps_archive_unchanged(self):
        state = self.state(max_archives=1, max_expansions=1)
        self.add(state, real=True)
        self.add(state, 2, obj="ключ")
        saved = state.to_dict()
        proposal = self.review(state)
        self.assertFalse(proposal["complete"])
        self.assertEqual(state.to_dict(), saved)

    def test_retrieval_scan_limit_is_enforced(self):
        state = self.state(max_scanned=1)
        self.add(state)
        self.add(state, 2, obj="ключ")
        with self.assertRaises(TimeoutError):
            state.retrieve(transfer(), Work(state.limits))

    def test_ambiguous_target_is_not_resolved_by_recency_or_top_k(self):
        state = self.state(max_retrieved=1)
        self.add(state)
        self.add(state, 2)
        saved = state.to_dict()
        observation = Observation("turn:3", "Уточнение", 3)
        meaning = replace(transfer("миша"), act="correct")
        batch = CandidateSet(
            observation,
            digest(context().to_dict()),
            (Hypothesis.create("turn:3", meaning, 0),),
        )
        result = state.process(
            batch, meaning, self.bundle.dynamics, self.bundle.understanding, seconds=1
        )
        self.assertEqual(result["reason"], "ambiguous_review_target")
        self.assertFalse(result["reviews"])
        self.assertEqual(state.to_dict(), saved)

    def test_partial_memory_batch_cannot_replace_archive(self):
        state = self.state()
        self.add(state)
        saved = state.to_dict()
        with (
            patch.object(
                self.bundle.dynamics, "score_interpretations", return_value=[]
            ),
            self.assertRaises(ValueError),
        ):
            self.review(state)
        self.assertEqual(state.to_dict(), saved)

    def test_roundtrip_then_review_matches_uninterrupted_archive(self):
        state = self.state()
        self.add(state)
        self.commit(state, self.review(state, cue(actor="петя", turn=10)))
        restored = AttentionState.from_dict(
            json.loads(json.dumps(state.to_dict())), self.fingerprint
        )
        self.commit(state, self.review(state, cue(turn=11)))
        self.commit(restored, self.review(restored, cue(turn=11)))
        self.assertEqual(state.to_dict(), restored.to_dict())

    def test_forged_origin_and_changed_ranker_rejected(self):
        state = self.state()
        self.add(state)
        for change in ["origin", "ranker"]:
            saved = state.to_dict()
            if change == "origin":
                saved["records"][0]["observation"]["text"] = "Другой текст"
            else:
                saved["ranker_fingerprint"] = "other"
            saved["digest"] = digest({k: v for k, v in saved.items() if k != "digest"})
            with self.assertRaises(ValueError):
                AttentionState.from_dict(saved, self.fingerprint)

    def test_forged_review_choice_rejected_even_with_updated_checksum(self):
        state = self.state()
        batch = self.add(state)
        self.commit(state, self.review(state))
        saved = state.to_dict()
        r = saved["records"][0]
        wrong = batch.candidates[0]
        assert wrong.meaning is not None
        r["selected_id"] = r["review"]["selected_id"] = wrong.hypothesis_id
        r["selected_meaning"] = wrong.meaning.to_dict()
        saved["digest"] = digest({k: v for k, v in saved.items() if k != "digest"})
        with self.assertRaises(ValueError):
            AttentionState.from_dict(saved, self.fingerprint)

    def test_ranker_prefers_target_over_recent_same_object_wrong_event(self):
        state = self.state()
        self.add(state)
        good = state.records[0]
        other = deepcopy(good)
        other["index_event"] = Event("locate", object="книга", place="стол").to_dict()
        expected_event = transfer().event
        assert expected_event is not None
        good_view = {**good, "index_event": expected_event.to_dict()}
        self.assertGreater(
            state.ranker.score("", transfer(), good_view),
            state.ranker.score("", transfer(), other),
        )
        self.assertTrue(any(state.ranker.weights))
        with self.assertRaises(ValueError):
            AttentionRanker([float("nan")] * 7)
        with self.assertRaises(ValueError):
            fit([[1.0] * 7], [2])


class AttentionSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))

    def start(self, **limits):
        s = LearnedSession(self.bundle, ConversationLimits(**limits))
        for text in [
            "Маша положила книгу в ящик",
            "Петя положил игрушку в коробку",
            "Она на столе",
        ]:
            r = s.respond(text)
            self.assertNotIn(r["action"], ("limit", "error"), r)
        return s

    def test_working_chat_correction_invokes_archive_review(self):
        session = self.start()
        r = session.respond("Нет, книга на столе")
        self.assertNotIn(r["action"], ("limit", "error"), r)
        attention = r["diagnostics"]["understanding"]["details"]["attention"]
        self.assertTrue(attention["reviews"])
        self.assertEqual(attention["reviews"][0]["target"], "turn:3")
        record = session.attention.get("turn:3")
        assert record is not None
        self.assertEqual(record["selected_meaning"]["event"]["object"], "книга")
        LearnedSession.from_dict(session.to_dict(), self.bundle)

    def test_late_ordinary_event_does_not_revise_history(self):
        session = self.start()
        original = deepcopy(session.attention.get("turn:3"))
        r = session.respond("Книга в ящике")
        self.assertNotIn(r["action"], ("limit", "error"), r)
        self.assertEqual(session.attention.get("turn:3"), original)

    def test_archive_outlives_short_receipt_window(self):
        session = self.start(max_history=2)
        for _ in range(18):
            self.assertNotIn(session.respond("Привет")["action"], ("limit", "error"))
        self.assertEqual(len(session.to_dict()["history"]), 2)
        self.assertIsNotNone(session.attention.get("turn:3"))
        restored = LearnedSession.from_dict(session.to_dict(), self.bundle)
        r = restored.respond("Нет, книга на столе")
        self.assertNotIn(r["action"], ("limit", "error"), r)
        self.assertEqual(
            r["diagnostics"]["understanding"]["details"]["attention"]["reviews"][0][
                "target"
            ],
            "turn:3",
        )

    def test_session_budget_failure_rolls_back_world_archive_and_history(self):
        session = self.start()
        snapshot = session.to_dict()
        with patch.object(
            AttentionState,
            "process",
            return_value={"complete": False, "reason": "attention_rounds_budget"},
        ):
            r = session.respond("Нет, книга на столе")
        self.assertEqual(r["action"], "limit")
        self.assertFalse(r["complete"])
        self.assertEqual(session.to_dict(), snapshot)

    def test_resume_with_archive_equals_continuous_chat(self):
        session = self.start()
        resumed = LearnedSession.from_dict(
            json.loads(json.dumps(session.to_dict())), self.bundle
        )
        for text in ["Привет", "Нет, книга на столе", "Где книга?"]:
            left, right = session.respond(text), resumed.respond(text)
            left.pop("elapsed_seconds")
            right.pop("elapsed_seconds")
            self.assertEqual(left, right)

    def test_legacy_session_without_attention_remains_loadable(self):
        session = LearnedSession(self.bundle)
        session.respond("Книга в ящике")
        saved = session.to_dict()
        saved.pop("attention")
        for r in saved["history"]:
            r["response"]["diagnostics"]["understanding"]["details"].pop("attention")
        restored = LearnedSession.from_dict(saved, self.bundle)
        self.assertEqual(restored.world.to_dict(), session.world.to_dict())
        self.assertFalse(restored.attention.records)

    def test_new_receipts_require_archive_and_matching_origin(self):
        session = self.start()
        saved = session.to_dict()
        saved.pop("attention")
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(saved, self.bundle)

    def test_saved_attention_receipt_rejects_another_cue(self):
        session = self.start()
        session.respond("Нет, книга на столе")
        saved = session.to_dict()
        trace = saved["history"][-1]["response"]["diagnostics"]["understanding"][
            "details"
        ]["attention"]
        trace["reviews"][0]["cue_id"] = "turn:500"
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(saved, self.bundle)

    def test_saved_review_rejects_other_memory_version_after_rehash(self):
        session = self.start()
        session.respond("Нет, книга на столе")
        saved = session.to_dict()
        saved["attention"]["records"][2]["review"]["memory_version"][1] += 1
        saved["attention"]["digest"] = digest(
            {k: v for k, v in saved["attention"].items() if k != "digest"}
        )
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(saved, self.bundle)
        saved = session.to_dict()
        r = saved["attention"]["records"][0]
        r["context"] = context().to_dict()
        r["original_snapshot"]["context_digest"] = digest(r["context"])
        saved["attention"]["digest"] = digest(
            {k: v for k, v in saved["attention"].items() if k != "digest"}
        )
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(saved, self.bundle)

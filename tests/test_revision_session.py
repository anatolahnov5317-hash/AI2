"""Block 3 session contracts on public development examples, not a benchmark.

The fitted public model is shared without training or inspecting sealed cases.
Checks concern versioned evidence, immutable receipts, rollback and local guards.
"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import PropertyMock, patch

from text_factors.conversation.persistence import encode_json
from text_factors.conversation.schema import ConversationLimits
from text_factors.learning import session as session_module
from text_factors.learning.dialogue_learning import GeneratedReply, PolicyDecision
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact
from text_factors.learning.session import LearnedSession

MODEL = Path(__file__).resolve().parents[1] / "docs/results/v05_model_42.json"
BOOK = "Маша положила книгу в ящик"
TOY = "Петя положил игрушку в коробку"
AMBIGUOUS = "Она на столе"
CUE = "Нет, книга на столе"
UNKNOWN_UPDATE = "Книгу перепрятали"


class RevisionSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = ModelBundle.from_dict(read_artifact(MODEL, kind="model"))
        cls.checkpoint = cls.bundle.to_dict()
        # These tests never change parameters. Avoid repeatedly encoding the
        # whole fitted model when constructing/restoring many small sessions.
        fingerprint = cls.bundle.fingerprint
        cache = patch.object(
            ModelBundle,
            "fingerprint",
            new_callable=PropertyMock,
            return_value=fingerprint,
        )
        cache.start()
        cls.addClassCleanup(cache.stop)

    def tearDown(self) -> None:
        self.assertEqual(self.bundle.to_dict(), self.checkpoint)

    def turn(
        self,
        session: LearnedSession,
        text: str,
        action: str | tuple[str, ...],
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        response = session.respond(text, request_id=request_id)
        expected = (action,) if isinstance(action, str) else action
        self.assertIn(response["action"], expected, response)
        self.assertEqual(
            response["complete"], response["action"] not in {"clarify", "unknown"}
        )
        self.assertTrue(response["diagnostics"]["generation"]["grounded"])
        return response

    def start(self, *, ambiguous: bool = True, **limits: Any) -> LearnedSession:
        session = LearnedSession(self.bundle, ConversationLimits(**limits))
        self.turn(session, BOOK, "ack")
        self.turn(session, TOY, "ack")
        if ambiguous:
            response = self.turn(session, AMBIGUOUS, "clarify")
            self.assertEqual(response["assertions"], [])
        return session

    def test_literal_correction_does_not_rewrite_incompatible_archive(self) -> None:
        session = LearnedSession(self.bundle)
        self.turn(session, "Книга в ящике", "ack")
        original = deepcopy(session.attention.get("turn:1"))
        self.turn(session, "Нет, книга на столе", "corrected")
        self.assertEqual(session.attention.get("turn:1"), original)
        self.assertEqual(session.world.revisions, [])
        self.assertEqual(session.world.facts()[0]["value"], "стол")
        self.assertEqual(
            LearnedSession.from_dict(session.to_dict(), self.bundle).to_dict(),
            session.to_dict(),
        )

    def location(
        self,
        session: LearnedSession,
        subject: str,
        place: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        response = self.turn(
            session, f"Где {subject}?", "answer", request_id=request_id
        )
        self.assertEqual(
            [
                (fact["subject"], fact["relation"], fact["value"], fact["negated"])
                for fact in response["assertions"]
            ],
            [(subject, "location", place, False)],
        )
        evidence = {event["id"]: event for event in response["evidence"]}
        for fact in response["assertions"]:
            event = evidence[fact["event_id"]]
            self.assertEqual(fact["source"], event["source"])
            self.assertIn(
                {
                    "op": "set",
                    "subject": subject,
                    "relation": "location",
                    "value": place,
                    "spatial": fact["spatial"],
                },
                event["effects"],
            )
        return response

    def revise(
        self, session: LearnedSession, *, request_id: str | None = None
    ) -> dict[str, Any]:
        response = self.turn(session, CUE, "corrected", request_id=request_id)
        revision = response["diagnostics"]["revision"]
        self.assertEqual(revision["plan"]["target_observation_ids"], ["turn:3"])
        self.assertEqual(revision["plan"]["turn_id"], response["turn_id"])
        self.assertIn("Пересмотрены", response["text"])
        self.assertNotIn(
            "Пересмотрены",
            " ".join(
                item["text"]
                for item in response["diagnostics"]["generation"]["segments"]
            ),
        )
        return response

    def restored(self, session: LearnedSession) -> LearnedSession:
        saved = json.loads(json.dumps(session.to_dict(), allow_nan=False))
        result = LearnedSession.from_dict(saved, self.bundle)
        self.assertEqual(result.to_dict(), saved)
        return result

    def staged_revision(self, prepared: list[Any]):
        original = session_module.prepare_archive_revision

        def prepare(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            prepared.append(result)
            return result

        return patch.object(
            session_module, "prepare_archive_revision", side_effect=prepare
        )

    def test_revision_changes_old_interpretation_and_preserves_original_receipts(self):
        session = self.start()
        observation = deepcopy(session.attention.get("turn:3"))
        original_events = session.world.events
        answer = self.location(session, "книга", "ящик")
        old_receipts = session.to_dict()["history"]

        response = self.revise(session)

        self.assertEqual(session.world.events, original_events)
        self.assertEqual(session.to_dict()["history"][:-1], old_receipts)
        revised = session.attention.get("turn:3")
        assert observation is not None and revised is not None
        self.assertEqual(revised["observation"], observation["observation"])
        self.assertEqual(revised["before"], observation["before"])
        self.assertEqual(revised["original_snapshot"], observation["original_snapshot"])
        self.assertEqual(revised["selected_meaning"]["event"]["object"], "книга")
        self.assertEqual(
            response["diagnostics"]["revision"]["affected_answers"],
            [
                {
                    "turn_id": answer["turn_id"],
                    "event_ids": [answer["assertions"][0]["event_id"]],
                }
            ],
        )
        effective = session.world.effective_events
        target = [event for event in effective if event["turn_id"] == 3]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]["meaning"]["event"]["object"], "книга")
        self.assertFalse(any(event["turn_id"] == 5 for event in effective))
        current = self.location(session, "книга", "стол")
        self.assertEqual(current["assertions"][0]["source"], "сообщение 3")
        self.assertEqual(current["assertions"][0]["event_id"], target[0]["id"])
        self.location(session, "игрушка", "коробка")

    def test_revision_receipt_does_not_link_an_unrelated_answer(self):
        session = self.start()
        toy_answer = self.location(session, "игрушка", "коробка")
        book_answer = self.location(session, "книга", "ящик")
        response = self.revise(session)
        links = response["diagnostics"]["revision"]["affected_answers"]
        self.assertEqual([link["turn_id"] for link in links], [book_answer["turn_id"]])
        self.assertNotIn(toy_answer["turn_id"], [link["turn_id"] for link in links])

    def test_conflicting_late_cues_block_current_answers_until_new_explicit_evidence(
        self,
    ):
        session = self.start()
        self.revise(session)
        conflict = self.turn(session, "Нет, игрушка на столе", "corrected")
        self.assertEqual(
            conflict["diagnostics"]["revision"]["plan"]["target_observation_ids"],
            ["turn:3"],
        )
        archived = session.attention.get("turn:3")
        assert archived is not None
        self.assertIsNone(archived["selected_meaning"])
        for subject in ("книга", "игрушка"):
            blocked = self.turn(session, f"Где {subject}?", "clarify")
            self.assertEqual(blocked["reason"], "historical_interpretation_uncertain")
            self.assertEqual(blocked["assertions"], [])
            generated = " ".join(
                segment["text"]
                for segment in blocked["diagnostics"]["generation"]["segments"]
            )
            self.assertTrue(blocked["text"].startswith(generated))
            self.assertGreater(len(blocked["text"]), len(generated))
        restored = self.restored(session)
        self.turn(restored, "Маша положила книгу в шкаф", "ack")
        self.location(restored, "книга", "шкаф")
        self.turn(restored, "Где игрушка?", "clarify")
        self.restored(restored)

    def test_replayed_old_observation_cannot_overwrite_a_later_explicit_move(self):
        session = self.start()
        self.turn(session, "Маша положила книгу в шкаф", "ack")
        before = self.location(session, "книга", "шкаф")
        self.revise(session)
        answer = self.location(session, "книга", "шкаф")
        self.assertEqual(answer["assertions"], before["assertions"])
        self.location(session, "игрушка", "коробка")
        restored = self.restored(session)
        self.location(restored, "книга", "шкаф")

    def test_restore_keeps_historical_answer_and_uses_revised_current_evidence(self):
        session = self.start()
        old_answer = self.location(session, "книга", "ящик")
        revision = self.revise(session)
        restored = self.restored(session)
        history = restored.to_dict()["history"]
        self.assertEqual(history[3]["response"], old_answer)
        self.assertEqual(history[4]["response"], revision)
        self.location(restored, "книга", "стол")
        self.location(restored, "игрушка", "коробка")

    def test_request_retry_returns_detached_old_and_revision_receipts_without_work(
        self,
    ):
        session = self.start()
        answer = self.location(session, "книга", "ящик", request_id="old-answer")
        revision = self.revise(session, request_id="revision")
        expected = deepcopy(revision)
        snapshot = session.to_dict()
        revision["diagnostics"]["revision"]["plan"]["target_observation_ids"].clear()
        revision["text"] = "caller changed its copy"
        for candidate in (session, self.restored(session)):
            with patch.object(
                self.bundle.understanding, "interpret", side_effect=AssertionError
            ):
                retry = candidate.respond(CUE, request_id="revision")
                self.assertEqual(retry, expected)
                self.assertEqual(
                    candidate.respond("Где книга?", request_id="old-answer"), answer
                )
            retry["diagnostics"]["revision"]["affected_answers"].clear()
            self.assertEqual(candidate.to_dict(), snapshot)
            with self.assertRaises(ValueError):
                candidate.respond("другое сообщение", request_id="revision")
            self.assertEqual(candidate.to_dict(), snapshot)

    def test_two_receipt_window_retains_valid_revision_link_to_evicted_answer(self):
        session = self.start(max_history=2)
        answer = self.location(session, "книга", "ящик")
        self.revise(session)
        self.turn(session, "Привет", "greet")
        saved = session.to_dict()
        self.assertEqual(len(saved["history"]), 2)
        self.assertEqual(
            [receipt["response"]["turn_id"] for receipt in saved["history"]], [5, 6]
        )
        self.assertEqual(
            saved["history"][0]["response"]["diagnostics"]["revision"][
                "affected_answers"
            ][0]["turn_id"],
            answer["turn_id"],
        )
        restored = self.restored(session)
        self.location(restored, "книга", "стол")
        self.location(self.restored(restored), "игрушка", "коробка")

    def test_retracting_revision_restores_prior_facts_without_erasing_history(self):
        session = self.start()
        self.location(session, "книга", "ящик")
        revision = self.revise(session)
        revision_history = session.world.revisions
        self.turn(session, "Отмени последнее сообщение", "retracted")
        self.assertEqual(session.world.revisions, revision_history)
        self.assertEqual(session.to_dict()["history"][4]["response"], revision)
        restored = self.restored(session)
        self.location(restored, "книга", "ящик")
        self.location(restored, "игрушка", "коробка")

    def test_generation_failure_rolls_back_staged_world_and_archive_revisions(self):
        session = self.start()
        snapshot = session.to_dict()
        prepared: list[Any] = []
        with (
            self.staged_revision(prepared),
            patch.object(
                self.bundle.generator,
                "generate",
                return_value=GeneratedReply("", (), False),
            ),
        ):
            response = session.respond(CUE)
        self.assertEqual(response["action"], "error", response)
        self.assertEqual(response["reason"], "generation_grounding_failed")
        self.assertEqual(len(prepared), 1)
        self.assertTrue(prepared[0].applicable)
        self.assertTrue(prepared[0].complete)
        self.assertEqual(session.to_dict(), snapshot)
        self.revise(session)

    def test_state_budget_failure_rolls_back_completed_revision_transaction(self):
        session = self.start()
        session.limits = replace(
            session.limits,
            max_state_bytes=len(encode_json(session.to_dict())) + 128,
        )
        snapshot = session.to_dict()
        prepared: list[Any] = []
        with self.staged_revision(prepared):
            response = session.respond(CUE)
        self.assertEqual(response["action"], "limit", response)
        self.assertEqual(response["reason"], "state_capacity", response)
        self.assertEqual(len(prepared), 1)
        self.assertTrue(prepared[0].applicable)
        self.assertTrue(prepared[0].complete)
        self.assertEqual(session.to_dict(), snapshot)

    def test_policy_clarification_discards_revision_but_keeps_new_observation(self):
        session = self.start()
        world = session.world.to_dict()
        archived = session.attention.records
        prepared: list[Any] = []

        def clarify(_features: Any, eligible: tuple[str, ...]) -> PolicyDecision:
            return PolicyDecision(
                "clarify", {action: float(action == "clarify") for action in eligible}
            )

        with (
            self.staged_revision(prepared),
            patch.object(self.bundle.policy, "choose", side_effect=clarify),
        ):
            response = self.turn(session, CUE, "clarify")
        self.assertTrue(prepared[0].applicable)
        self.assertEqual(response["assertions"], [])
        self.assertNotIn("revision", response["diagnostics"])
        self.assertNotIn("Пересмотрены", response["text"])
        self.assertEqual(session.world.to_dict(), world)
        self.assertEqual(session.attention.records[: len(archived)], archived)
        self.assertIsNotNone(session.attention.get("turn:4"))
        self.location(self.restored(session), "книга", "ящик")

    def test_unknown_update_blocks_only_its_explicit_subject_and_preserves_facts(self):
        session = self.start(ambiguous=False)
        world = session.world.to_dict()
        response = self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
        self.assertIsNone(response["meaning"])
        self.assertEqual(session.world.to_dict(), world)
        response = self.turn(session, "Где книга?", "clarify")
        self.assertEqual(response["reason"], "local_actuality_uncertain")
        self.assertEqual(response["assertions"], [])
        self.assertEqual({item["subject"] for item in response["evidence"]}, {"книга"})
        self.assertEqual(
            response["diagnostics"]["uncertainty"]["evidence"], response["evidence"]
        )
        self.location(session, "игрушка", "коробка")
        restored = self.restored(session)
        self.turn(restored, "Где книга?", "clarify")
        self.assertEqual(restored.world.to_dict(), world)

    def test_accepted_explicit_noop_confirmation_clears_only_its_subject_guard(self):
        session = self.start(ambiguous=False)
        self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
        self.turn(session, "Игрушку перепрятали", ("unknown", "clarify"))
        self.assertEqual(
            {marker["subject"] for marker in session.uncertainty.markers},
            {"книга", "игрушка"},
        )
        # Whether the fitted policy accepts this wording is not this contract:
        # a guard can clear only after an eligible, committed confirmation.
        guards = session.uncertainty.to_dict()
        with patch.object(
            self.bundle.policy,
            "choose",
            return_value=PolicyDecision("clarify", {"ack": 0.0, "clarify": 1.0}),
        ):
            self.turn(session, "Книга в ящике", "clarify")
        self.assertEqual(session.uncertainty.to_dict(), guards)
        with patch.object(
            self.bundle.policy,
            "choose",
            return_value=PolicyDecision("ack", {"ack": 1.0, "clarify": 0.0}),
        ):
            response = self.turn(session, "Книга в ящике", "ack")
        self.assertEqual(response["diagnostics"]["dynamics"]["effects"], [])
        self.assertEqual(
            {marker["subject"] for marker in session.uncertainty.markers}, {"игрушка"}
        )
        self.location(session, "книга", "ящик")
        self.turn(session, "Где игрушка?", "clarify")
        restored = self.restored(session)
        self.location(restored, "книга", "ящик")
        self.turn(restored, "Где игрушка?", "clarify")

    def test_fitted_policy_accepts_explicit_noop_after_freshness_clarification(self):
        session = self.start(ambiguous=False)
        facts = session.world.facts()
        self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
        self.turn(session, "Игрушку перепрятали", ("unknown", "clarify"))
        blocked = self.turn(session, "Где книга?", "clarify")
        self.assertEqual(blocked["reason"], "local_actuality_uncertain")

        confirmation = self.turn(session, "Книга в ящике", "ack")

        self.assertEqual(confirmation["diagnostics"]["dynamics"]["effects"], [])
        self.assertEqual(session.world.facts(), facts)
        self.assertEqual(
            {marker["subject"] for marker in session.uncertainty.markers}, {"игрушка"}
        )
        self.location(session, "книга", "ящик")
        restored = self.restored(session)
        self.location(restored, "книга", "ящик")
        self.turn(restored, "Где игрушка?", "clarify")

    def test_failed_generation_neither_adds_nor_clears_uncertainty(self):
        session = self.start(ambiguous=False)
        for text in (UNKNOWN_UPDATE, "Книга в ящике"):
            with self.subTest(text=text):
                if text != UNKNOWN_UPDATE:
                    self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
                    self.assertTrue(session.uncertainty.markers)
                saved = session.to_dict()
                with patch.object(
                    self.bundle.generator,
                    "generate",
                    return_value=GeneratedReply("", (), False),
                ):
                    response = session.respond(text)
                self.assertEqual(response["action"], "error", response)
                self.assertEqual(session.to_dict(), saved)

    def test_unparsed_question_or_greeting_does_not_poison_known_book(self):
        for text in (
            "Книгу перепрятали?",
            "Где книгу перепрятали",
            "Привет, книгу перепрятали",
        ):
            with self.subTest(text=text):
                session = self.start(ambiguous=False)
                response = self.turn(session, text, ("unknown", "clarify"))
                self.assertIsNone(response["meaning"])
                self.assertEqual(session.uncertainty.markers, [])
                self.location(session, "книга", "ящик")
                self.location(session, "игрушка", "коробка")

    def test_restore_rejects_forged_revision_plan_links_or_fixed_notice(self):
        session = self.start()
        self.location(session, "книга", "ящик")
        self.revise(session)
        snapshot = session.to_dict()
        for field in ("plan", "links", "notice", "missing_receipt", "world_version"):
            with self.subTest(field=field):
                saved = deepcopy(snapshot)
                response = saved["history"][-1]["response"]
                revision = response["diagnostics"]["revision"]
                if field == "plan":
                    revision["plan"]["target_observation_ids"] = ["turn:2"]
                elif field == "links":
                    revision["affected_answers"] = []
                elif field == "notice":
                    response["text"] = " ".join(
                        item["text"]
                        for item in response["diagnostics"]["generation"]["segments"]
                    )
                elif field == "missing_receipt":
                    response["diagnostics"].pop("revision")
                else:
                    saved["world"]["revisions"] = []
                with self.assertRaises(ValueError):
                    LearnedSession.from_dict(saved, self.bundle)

    def test_restore_rejects_missing_or_forged_uncertainty_and_notice(self):
        session = self.start(ambiguous=False)
        self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
        self.turn(session, "Где книга?", "clarify")
        snapshot = session.to_dict()
        for field in ("missing", "marker", "notice", "evidence"):
            with self.subTest(field=field):
                saved = deepcopy(snapshot)
                response = saved["history"][-1]["response"]
                if field == "missing":
                    saved.pop("uncertainty")
                elif field == "marker":
                    saved["uncertainty"]["markers"][0]["subject"] = "игрушка"
                elif field == "notice":
                    response["text"] = " ".join(
                        item["text"]
                        for item in response["diagnostics"]["generation"]["segments"]
                    )
                else:
                    response["diagnostics"]["uncertainty"]["evidence"] = []
                with self.assertRaises(ValueError):
                    LearnedSession.from_dict(saved, self.bundle)

    def test_internally_valid_guard_from_another_observation_cannot_be_substituted(
        self,
    ):
        session = self.start(ambiguous=False)
        other = self.start(ambiguous=False)
        self.turn(session, UNKNOWN_UPDATE, ("unknown", "clarify"))
        self.turn(other, "Книгу переложили заново", ("unknown", "clarify"))
        self.restored(other)
        saved = session.to_dict()
        saved["uncertainty"] = other.uncertainty.to_dict()
        with self.assertRaises(ValueError):
            LearnedSession.from_dict(saved, self.bundle)


if __name__ == "__main__":
    unittest.main()

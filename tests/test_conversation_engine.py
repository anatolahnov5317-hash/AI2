"""Independent engineering regressions, separate from the held-out experiment."""

import json
import unittest
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

from text_factors.conversation.bridge import LABELS, FactorSemanticBridge
from text_factors.conversation.engine import ConversationSession
from text_factors.conversation.language import DEFAULT_TEACHING_PAIRS
from text_factors.conversation.schema import ConversationLimits


class ConversationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = FactorSemanticBridge()
        cls.bridge.fit(list(DEFAULT_TEACHING_PAIRS))

    def session(self, **kwargs: Any) -> ConversationSession:
        return ConversationSession(
            bridge=kwargs.pop("bridge", self.bridge), train_defaults=False, **kwargs
        )

    def test_default_teaching_and_real_factor_evidence_are_used(self) -> None:
        self.assertEqual(len(DEFAULT_TEACHING_PAIRS), 33)
        for cue, label in DEFAULT_TEACHING_PAIRS:
            self.assertEqual(self.bridge.classify(cue).label, label)
        session = self.session()
        response = session.respond("Ключ в ящике.")
        self.assertTrue(response.complete)
        self.assertEqual(response.frames[0].predicate, "locate")
        self.assertTrue(response.evidence[0]["evidence"]["factor_evidence_used"])
        self.assertEqual(session.state.facts()[0].value, "ящик")

    def test_assertion_move_query_correction_and_unknown_are_grounded(self) -> None:
        session = self.session()
        for message in ("Ключ в ящике.", "Я переложил его в сумку."):
            self.assertTrue(session.respond(message).complete)
        answer = session.respond("Где ключ?")
        self.assertEqual(answer.assertions[0].value, "сумка")
        self.assertTrue(session.respond("Нет, ключ на столе.").complete)
        self.assertEqual(session.respond("Где ключ?").assertions[0].value, "стол")
        unknown = session.respond("У кого паспорт?")
        self.assertTrue(unknown.complete)
        self.assertEqual(unknown.action, "unknown")
        self.assertEqual(unknown.assertions, ())

    def test_question_shifts_reference_before_next_movement(self) -> None:
        session = self.session()
        for message in ("Ключ в ящике.", "Паспорт в сумке.", "Где ключ?"):
            self.assertTrue(session.respond(message).complete)
        moved = session.respond("Я переложил его на стол.")
        self.assertTrue(moved.complete)
        self.assertEqual(moved.assertions[0].subject, "ключ")
        self.assertEqual(session.respond("Где ключ?").assertions[0].value, "стол")
        self.assertEqual(session.respond("Где паспорт?").assertions[0].value, "сумка")

    def test_final_clause_failure_rolls_back_prior_assertion_and_focus(self) -> None:
        session = self.session()
        session.respond("Ключ в ящике.")
        before = session.state.to_dict(), session.bridge.to_dict()
        failed = session.respond("Книга в сумке. Я телепортировал ключ на стол.")
        self.assertFalse(failed.complete)
        self.assertEqual(failed.reason, "untaught_cue")
        self.assertEqual((session.state.to_dict(), session.bridge.to_dict()), before)

    def test_rejected_syntax_does_not_commit_valid_prefix(self) -> None:
        session = self.session()
        before = session.state.to_dict()
        response = session.respond("Книга в сумке. А теперь объясни устройство звезды.")
        self.assertFalse(response.complete)
        self.assertEqual(session.state.to_dict(), before)

    def test_model_exception_is_contained_and_does_not_leak_details(self) -> None:
        session = self.session()
        before = session.state.to_dict(), session.bridge.to_dict()
        with patch.object(
            session.bridge, "classify", side_effect=RuntimeError("secret")
        ):
            result = session.respond("Ключ в ящике.")
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "internal_error")
        self.assertNotIn("secret", result.text)
        self.assertEqual((session.state.to_dict(), session.bridge.to_dict()), before)

    def test_model_deadline_after_first_clause_discards_staged_changes(self) -> None:
        session = self.session()
        before = session.state.to_dict()
        original = session.bridge.classify
        calls = 0

        def classify(cue: str):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("injected model deadline")
            return original(cue)

        with patch.object(session.bridge, "classify", side_effect=classify):
            result = session.respond("Ключ в ящике. Книга в сумке.")
        self.assertEqual(calls, 2)
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "turn_time_budget")
        self.assertEqual(session.state.to_dict(), before)

    def test_cooperative_deadline_fails_without_model_work(self) -> None:
        session = self.session(limits=ConversationLimits(turn_seconds=1e-12))
        before = session.state.to_dict()
        with patch.object(
            session.bridge, "classify", wraps=session.bridge.classify
        ) as run:
            response = session.respond("Ключ в ящике.")
        self.assertFalse(response.complete)
        self.assertEqual(response.reason, "turn_time_budget")
        self.assertEqual(run.call_count, 0)
        self.assertEqual(session.state.to_dict(), before)

    def test_teaching_deadline_preserves_existing_model_and_world(self) -> None:
        session = self.session()
        before = session.state.to_dict(), session.bridge.to_dict()
        with patch(
            "text_factors.conversation.engine.FactorSemanticBridge.fit",
            side_effect=TimeoutError("injected training deadline"),
        ):
            result = session.respond("/teach спрятал move")
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "turn_time_budget")
        self.assertEqual((session.state.to_dict(), session.bridge.to_dict()), before)

    def test_snapshot_serialization_deadline_cannot_commit_completed_answer(
        self,
    ) -> None:
        from text_factors.conversation.engine import _canonical

        session = self.session()
        before = session.to_dict()
        clock = [0.0]

        def slow_serialization(value):
            result = _canonical(value)
            if value.get("world", {}).get("events"):
                clock[0] += 3.0
            return result

        with (
            patch(
                "text_factors.conversation.engine.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch(
                "text_factors.conversation.schema.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch(
                "text_factors.conversation.engine._canonical",
                side_effect=slow_serialization,
            ),
        ):
            result = session.respond("Ключ в ящике.")
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "turn_time_budget")
        self.assertEqual(session.to_dict(), before)

    def test_whole_snapshot_byte_cap_prevents_partial_commit(self) -> None:
        original = self.session().to_dict()
        size = len(
            json.dumps(
                original,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        session = self.session(limits=ConversationLimits(max_state_bytes=size + 32))
        before = session.to_dict()
        result = session.respond("Ключ в ящике.")
        self.assertFalse(result.complete)
        self.assertEqual(session.to_dict(), before)

    def test_repeated_queries_do_not_train_the_factor_memories(self) -> None:
        session = self.session()
        assert self.bridge._transform is not None and self.bridge._reader is not None
        before = (
            self.bridge.to_dict(),
            self.bridge._transform.memory.stats(),
            self.bridge._reader.memory.stats(),
        )
        for message in (
            "Привет",
            "Ключ в ящике.",
            "Где ключ?",
            "Где паспорт?",
            "Спасибо",
        ):
            session.respond(message)
        self.assertEqual(
            (
                self.bridge.to_dict(),
                self.bridge._transform.memory.stats(),
                self.bridge._reader.memory.stats(),
            ),
            before,
        )

    def test_untrained_control_has_no_hidden_canonical_predicate_fallback(self) -> None:
        untrained = FactorSemanticBridge(mode="untrained")
        untrained.fit(list(DEFAULT_TEACHING_PAIRS))
        session = self.session(bridge=untrained)
        response = session.respond("Ключ в ящике.")
        self.assertFalse(response.complete)
        self.assertEqual(session.state.facts(), ())
        self.assertIsNone(response.evidence[0]["label"])

    def test_lookup_baseline_remains_explicit_and_readonly(self) -> None:
        baseline = FactorSemanticBridge(mode="nearest")
        baseline.fit(list(DEFAULT_TEACHING_PAIRS))
        session = self.session(bridge=baseline)
        before = baseline.to_dict()
        result = session.respond("Ключ в ящике.")
        self.assertTrue(result.complete)
        self.assertFalse(result.evidence[0]["evidence"]["factor_evidence_used"])
        self.assertEqual(result.evidence[0]["evidence"]["mode"], "nearest")
        self.assertEqual(baseline.to_dict(), before)

    def test_request_id_retries_return_identical_receipt_without_new_events(
        self,
    ) -> None:
        session = self.session()
        first = session.respond("Ключ в ящике.", request_id="request-1")
        before = session.to_dict()
        repeated = session.respond("Ключ в ящике.", request_id="request-1")
        self.assertEqual(repeated.to_dict(), first.to_dict())
        self.assertEqual(session.to_dict(), before)
        conflict = session.respond("Ключ в сумке.", request_id="request-1")
        self.assertFalse(conflict.complete)
        self.assertEqual(conflict.reason, "request_id_conflict")
        self.assertEqual(session.to_dict(), before)

    def test_busy_session_rejects_work_and_snapshot_without_mutation(self) -> None:
        session = self.session()
        before = session.to_dict()
        session._lock.acquire()
        try:
            self.assertEqual(session.respond("Ключ в ящике.").reason, "busy")
            with self.assertRaises(ValueError):
                session.to_dict()
        finally:
            session._lock.release()
        self.assertEqual(session.to_dict(), before)

    def test_surrogate_and_oversized_inputs_never_enter_history(self) -> None:
        session = self.session(limits=ConversationLimits(max_chars=64))
        before = session.to_dict()
        for text in ("\ud800", "\udfff", "x" * 65, "Ключ\x00в ящике"):
            with self.subTest(text=repr(text)):
                response = session.respond(text)
                self.assertFalse(response.complete)
                self.assertEqual(session.to_dict(), before)
        for request_id in ("", "has space", "\n", "x" * 129, "\ud800", True):
            with self.subTest(request_id=repr(request_id)):
                with self.assertRaises(ValueError):
                    session.respond("Привет", request_id=cast(Any, request_id))
                self.assertEqual(session.to_dict(), before)

    def test_entity_event_token_clause_and_history_caps(self) -> None:
        for limits, message in (
            (ConversationLimits(max_tokens=2), "Ключ в ящике."),
            (ConversationLimits(max_clauses=1), "Ключ в ящике. Книга в сумке."),
            (ConversationLimits(max_entities=1), "Ключ в ящике."),
        ):
            session = self.session(limits=limits)
            self.assertFalse(session.respond(message).complete)
            self.assertEqual(session.state.facts(), ())
        session = self.session(limits=ConversationLimits(max_events=1))
        session.respond("Ключ в ящике.")
        before = session.state.to_dict()
        self.assertFalse(session.respond("Где ключ?").complete)
        self.assertEqual(session.state.to_dict(), before)
        limited_history = self.session(limits=ConversationLimits(max_history=2))
        for i in range(4):
            limited_history.respond("Привет", request_id=f"greeting-{i}")
        self.assertEqual(len(limited_history.to_dict()["history"]), 2)

    def test_explicit_teaching_stages_a_new_bridge_and_preserves_original(self) -> None:
        session = self.session(limits=ConversationLimits(turn_seconds=15))
        before = self.bridge.to_dict()
        self.assertFalse(session.respond("Я спрятал ключ в ящике.").complete)
        taught = session.respond("/teach спрятал move", request_id="teaching")
        self.assertTrue(taught.complete)
        self.assertEqual(taught.action, "taught")
        self.assertIsNot(session.bridge, self.bridge)
        self.assertEqual(self.bridge.to_dict(), before)
        self.assertTrue(session.respond("Я спрятал ключ в ящике.").complete)
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(
            restored.respond("/teach спрятал move", request_id="teaching").to_dict(),
            taught.to_dict(),
        )

    def test_duplicate_teaching_at_full_capacity_is_accepted_after_deduplication(
        self,
    ) -> None:
        pairs = list(DEFAULT_TEACHING_PAIRS)
        pairs += [
            ("cue" + chr(97 + i // 26) + chr(97 + i % 26), LABELS[i % 4])
            for i in range(64 - len(pairs))
        ]
        # Capacity handling is independent of the numerical algorithm; use the
        # explicit lookup control to keep this boundary regression inexpensive.
        baseline = FactorSemanticBridge(mode="nearest")
        baseline.fit(pairs)
        session = self.session(bridge=baseline)
        duplicate = session.respond("/teach лежит locate")
        self.assertTrue(duplicate.complete)
        self.assertEqual(len(session.bridge.to_dict()["examples"]), 64)
        before = session.bridge.to_dict()
        rejected = session.respond("/teach скрыт locate")
        self.assertFalse(rejected.complete)
        self.assertEqual(rejected.reason, "teaching_capacity")
        self.assertEqual(session.bridge.to_dict(), before)

    def test_prefit_teaching_must_fit_the_configured_capacity(self) -> None:
        with self.assertRaises(ValueError):
            self.session(limits=ConversationLimits(max_training_examples=1))

    def test_snapshot_roundtrip_is_exact_and_detached(self) -> None:
        session = self.session()
        session.respond("Ключ в ящике.", request_id="one")
        session.respond("Где ключ?", request_id="two")
        snapshot = json.loads(json.dumps(session.to_dict()))
        restored = ConversationSession.from_dict(snapshot)
        self.assertEqual(restored.to_dict(), snapshot)
        snapshot["history"].clear()
        self.assertEqual(len(restored.to_dict()["history"]), 2)

    def test_historical_receipt_survives_later_correction(self) -> None:
        session = self.session()
        session.respond("Ключ в ящике.")
        original = session.respond("Где ключ?", request_id="old-question")
        session.respond("Нет, ключ в сумке.")
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(
            restored.respond("Где ключ?", request_id="old-question").to_dict(),
            original.to_dict(),
        )
        self.assertEqual(restored.respond("Где ключ?").assertions[0].value, "сумка")

    def test_nonexistent_mismatched_future_and_text_only_receipts_are_rejected(
        self,
    ) -> None:
        session = self.session()
        session.respond("Ключ в ящике.")
        session.respond("Где ключ?", request_id="cached")
        session.respond("Паспорт в сумке.")
        original = session.to_dict()
        for field, value in (("event_id", 999), ("value", "сейф")):
            snapshot = deepcopy(original)
            snapshot["history"][1]["response"]["assertions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ConversationSession.from_dict(snapshot)
        text_only = deepcopy(original)
        text_only["history"][1]["response"]["text"] = "Ключ в сейфе."
        with self.assertRaises(ValueError):
            ConversationSession.from_dict(text_only)
        future = deepcopy(original)
        future["history"][1]["response"]["assertions"] = deepcopy(
            future["history"][2]["response"]["assertions"]
        )
        with self.assertRaises(ValueError):
            ConversationSession.from_dict(future)

    def test_valid_failure_receipts_roundtrip_and_remain_idempotent(self) -> None:
        session = self.session()
        session.respond("Ключ в ящике. Книга в сумке.")
        failed = session.respond("Я переложил его на стол.", request_id="ambiguous")
        self.assertFalse(failed.complete)
        self.assertTrue(failed.alternatives)
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(
            restored.respond(
                "Я переложил его на стол.", request_id="ambiguous"
            ).to_dict(),
            failed.to_dict(),
        )

    def test_snapshot_rejects_identity_limits_and_future_turn_corruption(self) -> None:
        session = self.session()
        session.respond("Ключ в ящике.", request_id="one")
        session.respond("Где ключ?", request_id="two")
        original = session.to_dict()
        malformed: list[dict[str, Any]] = []
        for key, value in (("turn_count", True), ("schema", "bad"), ("turn_count", 0)):
            snapshot = deepcopy(original)
            snapshot[key] = value
            malformed.append(snapshot)
        duplicate = deepcopy(original)
        duplicate["history"][1]["request_id"] = "one"
        malformed.append(duplicate)
        nonmonotonic = deepcopy(original)
        nonmonotonic["history"][1]["response"]["turn_id"] = 1
        malformed.append(nonmonotonic)
        undersized = deepcopy(original)
        undersized["limits"]["max_state_bytes"] = 1
        malformed.append(undersized)
        for snapshot in malformed:
            with self.assertRaises(ValueError):
                ConversationSession.from_dict(snapshot)

    def test_query_fact_display_cap_preserves_full_grounded_receipt(self) -> None:
        session = self.session(limits=ConversationLimits(max_candidates=1))
        session.respond("У Маши есть ключ.")
        session.respond("У Маши есть книга.")
        result = session.respond("Что у Маши?", request_id="many")
        self.assertTrue(result.complete)
        self.assertEqual(len(result.assertions), 2)
        self.assertIn("Показаны первые 1 из 2", result.text)
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(
            restored.respond("Что у Маши?", request_id="many").to_dict(),
            result.to_dict(),
        )

    def test_topic_and_nonfactual_receipts_roundtrip(self) -> None:
        session = self.session()
        for message in ("Привет", "Спасибо", "Помощь", "Тема: работа", "Где ключ?"):
            self.assertTrue(session.respond(message).complete)
        restored = ConversationSession.from_dict(session.to_dict())
        self.assertEqual(restored.to_dict(), session.to_dict())


if __name__ == "__main__":
    unittest.main()

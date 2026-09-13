"""Strict public serialization contracts, not task-quality evaluation."""

import json
import unittest
from typing import Any, cast
from unittest.mock import patch

from text_factors.conversation.schema import (
    Assertion,
    Budget,
    BudgetExceeded,
    ConversationLimits,
    ParseResult,
    SemanticFrame,
    StateOutcome,
    TurnResponse,
)


class ConversationSchemaTests(unittest.TestCase):
    def frame(self) -> SemanticFrame:
        return SemanticFrame(
            "inform",
            cue="положил",
            predicate="move",
            actor="я",
            object="ключ",
            place="ящик",
            raw="Я положил ключ в ящик.",
        )

    def assertion(self) -> Assertion:
        return Assertion("ключ", "location", "ящик", 1)

    def response(self) -> TurnResponse:
        return TurnResponse(
            1,
            "Запомнил: Ключ в ящике.",
            "ack",
            frames=(self.frame(),),
            assertions=(self.assertion(),),
            evidence=({"score": 1.0, "bits": [1, 4], "calibrated": False},),
            elapsed_seconds=0.1,
        )

    def test_limits_exact_roundtrip_and_all_integer_caps(self) -> None:
        limits = ConversationLimits()
        self.assertEqual(ConversationLimits.from_dict(limits.to_dict()), limits)
        for name in limits.to_dict():
            if name == "turn_seconds":
                continue
            for invalid in (False, 0, -1, 100_000_000, 1.5, "1"):
                data = limits.to_dict()
                data[name] = invalid
                with (
                    self.subTest(name=name, value=invalid),
                    self.assertRaises(ValueError),
                ):
                    ConversationLimits.from_dict(data)
        for invalid in (False, 0, -1, 31, float("nan"), float("inf"), "1"):
            data = limits.to_dict()
            data["turn_seconds"] = invalid
            with self.assertRaises(ValueError):
                ConversationLimits.from_dict(data)

    def test_limit_fields_are_exact(self) -> None:
        for value in ({}, [], {**ConversationLimits().to_dict(), "extra": 1}):
            with self.assertRaises(ValueError):
                ConversationLimits.from_dict(value)

    def test_semantic_frame_preserves_roles_negation_time_and_reference(self) -> None:
        frame = SemanticFrame(
            "correct",
            cue="отдал",
            predicate="give",
            actor="петя",
            object="ключ",
            recipient="маша",
            negated=True,
            modality="reported",
            tense="past",
            reference=5,
            topic="работа",
        )
        self.assertEqual(SemanticFrame.from_dict(frame.to_dict()), frame)
        self.assertEqual(frame.actor, "петя")
        self.assertEqual(frame.recipient, "маша")
        self.assertTrue(frame.negated)

    def test_semantic_frame_rejects_invalid_categories_scalars_and_lengths(
        self,
    ) -> None:
        cases = {
            "act": ["invent", True, None],
            "predicate": ["delete", False],
            "query": ["execute"],
            "modality": ["certain"],
            "tense": ["yesterday"],
            "spatial": ["under"],
            "negated": [0, 1, "false"],
            "reference": [0, -1, True, 1.0],
            "actor": ["x" * 129, "x\x00y", 1],
            "raw": ["x" * 2049, "\x00"],
        }
        for field, values in cases.items():
            for value in values:
                data = self.frame().to_dict()
                data[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    SemanticFrame.from_dict(data)
        data = self.frame().to_dict()
        data.pop("raw")
        with self.assertRaises(ValueError):
            SemanticFrame.from_dict(data)

    def test_assertion_strict_roundtrip_and_scalar_validation(self) -> None:
        self.assertEqual(
            Assertion.from_dict(self.assertion().to_dict()), self.assertion()
        )
        for field, values in {
            "subject": ["", 1, "x" * 129, "x\ny", "x\x7fy"],
            "relation": ["is_a", 1],
            "qualifier": ["under", ""],
            "source": ["", False],
            "topic": ["", "\x00"],
            "event_id": [0, -1, True, 1.0, 2**53],
            "negated": [0, "true"],
        }.items():
            for value in values:
                data = self.assertion().to_dict()
                data[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    Assertion.from_dict(data)

    def test_parse_and_state_exports_are_json_arrays_and_detached(self) -> None:
        parse = ParseResult((self.frame(),), alternatives=("first", "second"))
        data = parse.to_dict()
        self.assertIsInstance(data["frames"], list)
        self.assertIsInstance(data["alternatives"], list)
        data["frames"][0]["actor"] = "someone else"
        self.assertEqual(parse.frames[0].actor, "я")
        outcome = StateOutcome(
            "answer",
            assertions=(self.assertion(),),
            event_ids=(1,),
            alternatives=("first",),
            resolved_frame=self.frame(),
        )
        exported = outcome.to_dict()
        json.dumps(exported, allow_nan=False)
        self.assertIsInstance(exported["event_ids"], list)
        self.assertEqual(exported["resolved_frame"]["actor"], "я")

    def test_turn_response_exact_roundtrip_detaches_nested_evidence(self) -> None:
        response = self.response()
        encoded = response.to_dict()
        restored = TurnResponse.from_dict(encoded)
        self.assertEqual(restored, response)
        self.assertEqual(restored.to_dict(), encoded)
        encoded["evidence"][0]["bits"].append(99)
        self.assertEqual(restored.evidence[0]["bits"], [1, 4])
        self.assertEqual(response.evidence[0]["bits"], [1, 4])

    def test_response_invalid_actions_completeness_and_nonfinite_numbers(self) -> None:
        fields = {
            "turn_id": [-1, True, 1.0, 2**53],
            "action": ["execute", 123],
            "complete": [False, 1],
            "text": [123, "x" * 16385, "\x00"],
            "reason": ["x" * 129, None],
            "elapsed_seconds": [True, -1, float("nan"), float("inf"), "1"],
            "alternatives": [("one",), [False], ["x" * 129], ["a"] * 33],
        }
        for field, values in fields.items():
            for value in values:
                data = self.response().to_dict()
                data[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    TurnResponse.from_dict(data)
        data = self.response().to_dict()
        data["action"] = "clarify"
        with self.assertRaises(ValueError):
            TurnResponse.from_dict(data)
        data["complete"] = False
        self.assertFalse(TurnResponse.from_dict(data).complete)

    def test_response_rejects_invalid_nested_evidence_and_container_shapes(
        self,
    ) -> None:
        for field, value in (
            ("frames", {}),
            ("frames", [{}]),
            ("frames", [self.frame().to_dict()] * 33),
            ("assertions", [{}]),
            ("evidence", ["text"]),
            ("evidence", [{"score": float("nan")}]),
            ("evidence", [{"score": float("inf")}]),
            ("evidence", [{"object": object()}]),
            ("evidence", [{"long": "x" * 1_000_001}]),
        ):
            data = self.response().to_dict()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                TurnResponse.from_dict(data)
        recursive: dict[str, Any] = {}
        recursive["self"] = recursive
        data = self.response().to_dict()
        data["evidence"] = [recursive]
        with self.assertRaises(ValueError):
            TurnResponse.from_dict(data)

    def test_response_requires_exact_fields(self) -> None:
        data = self.response().to_dict()
        variants = [
            {**data, "extra": 1},
            {k: v for k, v in data.items() if k != "reason"},
        ]
        for invalid in variants:
            with self.assertRaises(ValueError):
                TurnResponse.from_dict(invalid)

    def test_budget_checks_deterministic_deadline_and_input_types(self) -> None:
        with patch(
            "text_factors.conversation.schema.perf_counter", side_effect=[1, 1.5, 2]
        ):
            budget = Budget(1.0)
            budget.check()
            with self.assertRaises(BudgetExceeded):
                budget.check()
        for seconds in (True, 0, -1, float("inf"), float("nan"), "1"):
            with self.assertRaises(ValueError):
                Budget(cast(Any, seconds))


if __name__ == "__main__":
    unittest.main()

"""Public acceptance/development checks, not independent frozen evaluation."""

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from text_factors.learning.language_data import (
    ENTITY_BY_NAME,
    TrainingUtterance,
    development_examples,
    training_examples,
)
from text_factors.learning.schema import DialogueContext, Event, Meaning, Query
from text_factors.learning.understanding import LearnedUnderstanding

_EMPTY_CONTEXT = DialogueContext()


def context(*names: str, focus: tuple[str, ...] | None = None) -> DialogueContext:
    return DialogueContext(
        entities=tuple(ENTITY_BY_NAME[name].entity for name in names),
        focus=names if focus is None else focus,
    )


class LearnedUnderstandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = LearnedUnderstanding.fit(seed=42, seconds=30)

    def meaning(self, text: str, previous: DialogueContext = _EMPTY_CONTEXT) -> Meaning:
        result = self.model.interpret(text, previous)
        self.assertIsNotNone(result.meaning, (text, result.reason))
        assert result.meaning is not None
        return result.meaning

    def test_public_development_role_compositions_are_exact(self) -> None:
        for example in development_examples():
            with self.subTest(text=example.text):
                self.assertEqual(
                    self.meaning(example.text, example.context), example.meaning
                )

    def test_user_transfer_paraphrases_have_identical_role_bound_events(self) -> None:
        received = self.meaning("Маша получила книгу от Пети")
        transferred = self.meaning("Петя передал книгу Маше")
        self.assertEqual(received, transferred)
        self.assertEqual(
            received.event,
            Event("give", actor="петя", object="книга", recipient="маша"),
        )

    def test_swapping_people_changes_roles_without_changing_predicate(self) -> None:
        for text, actor, recipient in (
            ("Иван передал паспорт Оле.", "иван", "оля"),
            ("Оля передала паспорт Ивану.", "оля", "иван"),
            ("Иван получил паспорт от Оли.", "оля", "иван"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    self.meaning(text).event,
                    Event("give", actor=actor, object="паспорт", recipient=recipient),
                )

    def test_actual_negative_transfer_and_nested_negation_remain_distinct(self) -> None:
        actual = Event("give", actor="петя", object="книга", recipient="маша")
        intended = replace(actual, modality="intended", time="future")
        cases = (
            ("Петя передал книгу Маше", actual),
            ("Петя не передал книгу Маше", replace(actual, negated=True)),
            (
                "Петя обещал передать книгу Маше",
                Event("promise", actor="петя", content=intended),
            ),
            (
                "Петя не обещал передать книгу Маше",
                Event("promise", actor="петя", content=intended, negated=True),
            ),
            (
                "Петя обещал не передать книгу Маше",
                Event("promise", actor="петя", content=replace(intended, negated=True)),
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.meaning(text).event, expected)

    def test_query_spatial_relation_and_negation_survive_prediction(self) -> None:
        for text, expected in (
            (
                "Книга не на столе?",
                Query(
                    "verify",
                    subject="книга",
                    value="стол",
                    relation="location",
                    spatial="on",
                    negated=True,
                ),
            ),
            (
                "Книга в столе?",
                Query("verify", subject="книга", value="стол", relation="location"),
            ),
            (
                "У Пети нет книги?",
                Query(
                    "verify",
                    subject="книга",
                    value="петя",
                    relation="holder",
                    negated=True,
                ),
            ),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.meaning(text).query, expected)

    def test_context_roles_resolve_pronouns_to_different_entities(self) -> None:
        previous = context("книга", "анна", "шкаф")
        self.assertEqual(
            self.meaning("Она положила книгу на стол.", previous).event,
            Event("move", actor="анна", object="книга", place="стол", spatial="on"),
        )
        self.assertEqual(
            self.meaning("Петя положил ее на стол.", previous).event,
            Event("move", actor="петя", object="книга", place="стол", spatial="on"),
        )
        self.assertEqual(
            self.meaning("Где она?", previous).query,
            Query("where", subject="книга"),
        )

    def test_missing_ambiguous_and_wrong_gender_references_abstain(self) -> None:
        cases = (
            ("Он положил книгу на стол.", DialogueContext()),
            ("Он положил книгу на стол.", context("книга", "петя", "миша", "стол")),
            ("Он положил книгу на стол.", context("книга", "анна", "шкаф")),
            ("Она положила книгу на стол.", context("книга", "петя", "стол")),
        )
        for text, previous in cases:
            with self.subTest(text=text, context=previous.focus):
                result = self.model.interpret(text, previous)
                self.assertIsNone(result.meaning)
                self.assertEqual(result.reason, "ambiguous_or_missing_reference")

    def test_provenance_followup_uses_most_recent_focus_in_taught_contexts(
        self,
    ) -> None:
        for names in (("книга", "маша"), ("маша", "книга")):
            with self.subTest(focus=names):
                self.assertEqual(
                    self.meaning("почему?", context(*names)).query,
                    Query("why", subject=names[0]),
                )
        self.assertIsNone(self.model.interpret("почему?").meaning)

    def test_unsupported_lexemes_and_incomplete_clauses_do_not_become_facts(
        self,
    ) -> None:
        cases = (
            "Маша запустила книгу от Пети",
            "Маша получила книгу от неизвестного",
            "Петя передал книгу",
            "Петя положил книгу",
            "Книга на столе. Петя передал ключ Маше.",
            "Маша получила книгу от Пети и уничтожила ее",
            "\ud800",
            "Книга\nна столе",
            "а" * 2049,
        )
        for text in cases:
            with self.subTest(text=repr(text)):
                result = self.model.interpret(text)
                self.assertIsNone(result.meaning)
                self.assertTrue(result.reason)

    def test_numeric_checkpoint_round_trip_never_calls_a_trainer(self) -> None:
        payload = json.loads(json.dumps(self.model.to_dict(), allow_nan=False))
        self.assertTrue(np.any(np.asarray(payload["role_weights"]) != 0))
        self.assertNotIn("examples", payload)
        with (
            patch(
                "text_factors.learning.understanding.training_examples",
                side_effect=AssertionError("inference must not consult examples"),
            ),
            patch(
                "text_factors.learning.understanding.reference_examples",
                side_effect=AssertionError(
                    "inference must not consult reference targets"
                ),
            ),
        ):
            restored = LearnedUnderstanding.from_dict(payload)
            for example in development_examples():
                self.assertEqual(
                    restored.interpret(example.text, example.context),
                    self.model.interpret(example.text, example.context),
                )
        self.assertEqual(restored.to_dict(), payload)

    def test_bad_checkpoint_shapes_and_values_are_rejected_before_array_conversion(
        self,
    ) -> None:
        for bad in (float("nan"), float("inf"), True, 10**1000):
            with self.subTest(value_type=type(bad).__name__):
                payload = self.model.to_dict()
                payload["role_weights"][0][0] = bad
                with (
                    patch(
                        "numpy.asarray", side_effect=AssertionError("early allocation")
                    ),
                    self.assertRaises(ValueError),
                ):
                    LearnedUnderstanding.from_dict(payload)
        for field in ("role_weights", "reference_weights"):
            with self.subTest(field=field):
                payload = self.model.to_dict()
                payload[field].append([])
                with self.assertRaises(ValueError):
                    LearnedUnderstanding.from_dict(payload)
        payload = self.model.to_dict()
        payload["config"]["reference_dim"] = 2**40
        with self.assertRaises(ValueError):
            LearnedUnderstanding.from_dict(payload)

    def test_bad_checkpoint_metadata_is_rejected(self) -> None:
        for field, value in (
            ("seed", True),
            ("fingerprint", "invalid"),
            ("trained", 1),
            ("vocabulary", ["word", "word"]),
            ("heads", {}),
        ):
            with self.subTest(field=field):
                payload = self.model.to_dict()
                payload[field] = value
                with self.assertRaises(ValueError):
                    LearnedUnderstanding.from_dict(payload)
        payload = self.model.to_dict()
        payload["training_metrics"]["seconds"] = 10**1000
        with self.assertRaises(ValueError):
            LearnedUnderstanding.from_dict(payload)

    def test_untrained_and_shuffled_controls_do_not_pass_development(self) -> None:
        examples = development_examples()
        untrained = LearnedUnderstanding()
        self.assertTrue(
            all(untrained.interpret(ex.text).meaning is None for ex in examples)
        )
        restored = LearnedUnderstanding.from_dict(untrained.to_dict())
        self.assertFalse(restored.trained)
        shuffled = LearnedUnderstanding.fit(seconds=30, seed=42, shuffle_targets=True)
        exact = sum(
            shuffled.interpret(ex.text, ex.context).meaning == ex.meaning
            for ex in examples
        )
        self.assertLess(exact, len(examples) // 2)
        self.assertNotEqual(shuffled.fingerprint, self.model.fingerprint)
        self.assertTrue(shuffled.training_metrics["shuffle_targets"])

    def test_inference_does_not_mutate_or_train_the_model(self) -> None:
        before = deepcopy(self.model.to_dict())
        self.model.interpret("Петя передал книгу Маше")
        self.model.interpret("неподдерживаемое слово")
        self.assertEqual(self.model.to_dict(), before)

    def test_training_deadlines_and_annotation_validation_fail_cleanly(self) -> None:
        for seconds in (0, -1, float("nan"), float("inf"), True, 10**1000):
            with (
                self.subTest(seconds_type=type(seconds).__name__),
                self.assertRaises(ValueError),
            ):
                LearnedUnderstanding.fit(seconds=seconds)
        with self.assertRaises(TimeoutError):
            LearnedUnderstanding.fit(seconds=1e-12)
        with self.assertRaises(ValueError):
            LearnedUnderstanding.fit([])
        with self.assertRaises(ValueError):
            LearnedUnderstanding.fit(shuffle_targets=1)  # type: ignore[arg-type]
        example = training_examples()[0]
        self.assertEqual(TrainingUtterance.from_dict(example.to_dict()), example)
        with self.assertRaises(ValueError):
            TrainingUtterance(
                "Петя передал книгу Маше",
                example.meaning,
                context=context("петя"),
                links=((0, "петя"),),
            )


if __name__ == "__main__":
    unittest.main()

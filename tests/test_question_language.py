"""Open synthetic checks for cautious language-to-question inference."""

from __future__ import annotations

import hashlib
import unittest

from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledText,
    Span,
)
from text_factors.real_data.question_language import (
    LabeledQuestion,
    QuestionLanguageModel,
    ReviewedIdentity,
)
from text_factors.real_data.raw_language import RawSemanticModel


def _span(text: str, surface: str) -> Span:
    start = text.index(surface)
    return Span(start, start + len(surface))


def _mention(
    text: str, surface: str, instance: str, kind: str, morphology: str | None
) -> IdentifiedMention:
    return IdentifiedMention(
        f"mention:{surface}", instance, _span(text, surface), kind, morphology
    )


def _raw_model() -> RawSemanticModel:
    first = "Анна передала ключ Борису."
    second = "Олег передал папку Еве."
    third = "Мария передала ключ Олегу."
    return RawSemanticModel().fit(
        (
            LabeledText(
                first,
                (
                    _mention(first, "Анна", "anna", "person", "nom"),
                    _mention(first, "ключ", "key", "thing", None),
                    _mention(first, "Борису", "boris", "person", "dat"),
                ),
                (
                    LabeledEvent(
                        "transfer",
                        _span(first, "передала"),
                        (
                            ("actor", "mention:Анна"),
                            ("object", "mention:ключ"),
                            ("recipient", "mention:Борису"),
                        ),
                    ),
                ),
            ),
            LabeledText(
                second,
                (
                    _mention(second, "Олег", "oleg", "person", "nom"),
                    _mention(second, "папку", "folder", "thing", None),
                    _mention(second, "Еве", "eve", "person", "dat"),
                ),
                (
                    LabeledEvent(
                        "transfer",
                        _span(second, "передал"),
                        (
                            ("actor", "mention:Олег"),
                            ("object", "mention:папку"),
                            ("recipient", "mention:Еве"),
                        ),
                    ),
                ),
            ),
            LabeledText(
                third,
                (
                    _mention(third, "Мария", "maria", "person", "nom"),
                    _mention(third, "ключ", "key", "thing", None),
                    _mention(third, "Олегу", "oleg", "person", "dat"),
                ),
                (
                    LabeledEvent(
                        "transfer",
                        _span(third, "передала"),
                        (
                            ("actor", "mention:Мария"),
                            ("object", "mention:ключ"),
                            ("recipient", "mention:Олегу"),
                        ),
                    ),
                ),
            ),
        )
    )


def _question(text: str, subject: str, asked_role: str, family: str) -> LabeledQuestion:
    return LabeledQuestion(
        text,
        family,
        _mention(
            text,
            subject,
            "only-in-annotation",
            "thing" if subject == "ключ" else "person",
            None,
        ),
        "transfer",
        asked_role,
    )


class QuestionLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = _raw_model()
        self.question = QuestionLanguageModel().fit(
            (
                _question("Что передала Анна?", "Анна", "object", "train_1"),
                _question("Кому передала Анна?", "Анна", "recipient", "train_2"),
                _question("Кто передал ключ?", "ключ", "actor", "train_3"),
            ),
            held_out_families=frozenset({"evaluation_new_people"}),
        )
        self.identities = (
            ReviewedIdentity("София", "person", "person:sophia"),
            ReviewedIdentity("Анна", "person", "person:anna"),
            ReviewedIdentity("блокнот", "thing", "thing:notebook"),
        )

    def parse(self, text: str, identities=None):  # type: ignore[no-untyped-def]
        return self.question.parse(
            text,
            raw_model=self.raw,
            reviewed_identities=self.identities if identities is None else identities,
        )

    def test_new_name_and_asked_role_transfer_without_inference_id_leak(self) -> None:
        first = self.parse("Что передала София?")
        second = self.parse("Кому передала София?")
        self.assertTrue(first.resolved)
        self.assertTrue(second.resolved)
        self.assertEqual(first.subject_id, "person:sophia")
        self.assertEqual(first.relation_id, "transfer")
        self.assertEqual(first.asked_role, "object")
        self.assertEqual(second.asked_role, "recipient")
        self.assertEqual(first.ambiguity, ())
        self.assertEqual(first.residual, ())
        self.assertEqual(
            first.text_sha256,
            hashlib.sha256("Что передала София?".encode()).hexdigest(),
        )
        self.assertNotIn("question:", first.subject_id or "")

    def test_swap_subject_role_and_new_object_surface(self) -> None:
        result = self.parse("Кто передал блокнот?")
        self.assertTrue(result.resolved)
        self.assertEqual(result.subject_id, "thing:notebook")
        self.assertEqual(result.asked_role, "actor")

    def test_unknown_words_order_and_negation_abstain(self) -> None:
        extra = self.parse("Что передала София сейчас?")
        self.assertFalse(extra.resolved)
        self.assertFalse(extra.fully_covered)
        self.assertEqual(extra.ambiguity, ("unknown_language",))
        self.assertIn("сейчас", extra.residual)
        reversed_order = self.parse("Что София передала?")
        self.assertFalse(reversed_order.resolved)
        negative = self.parse("Что не передала София?")
        self.assertEqual(negative.ambiguity, ("negative_question",))
        self.assertFalse(negative.fully_covered)

    def test_ambiguous_or_unreviewed_identity_abstains(self) -> None:
        unseen = self.parse("Что передала София?", ())
        self.assertEqual(unseen.ambiguity, ("unresolved_identity",))
        self.assertEqual(unseen.residual, ("София",))
        duplicated = self.parse(
            "Что передала София?",
            self.identities
            + (ReviewedIdentity("София", "person", "person:sophia_other"),),
        )
        self.assertEqual(duplicated.ambiguity, ("ambiguous_identity",))
        self.assertFalse(duplicated.resolved)
        multiple = self.parse("Что передала София и Анна?")
        self.assertEqual(multiple.ambiguity, ("multiple_subjects",))

    def test_holdout_overlap_and_contradictory_training_are_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            QuestionLanguageModel().fit(
                (_question("Что передала Анна?", "Анна", "object", "eval"),),
                held_out_families=frozenset({"eval"}),
            )
        ambiguous = QuestionLanguageModel().fit(
            (
                _question("Что передала Анна?", "Анна", "object", "train_1"),
                _question("Что передала Анна?", "Анна", "recipient", "train_2"),
            )
        )
        query = ambiguous.parse(
            "Что передала София?",
            raw_model=self.raw,
            reviewed_identities=self.identities,
        )
        self.assertEqual(query.ambiguity, ("ambiguous_template",))
        self.assertFalse(query.resolved)


if __name__ == "__main__":
    unittest.main()

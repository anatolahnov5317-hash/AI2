"""Checks for the explicit language scaffold, not evidence of learned grammar."""

import time
import unittest

from text_factors.conversation.language import (
    DEFAULT_TEACHING_PAIRS,
    RussianParser,
    surface_entity,
)
from text_factors.conversation.schema import ConversationLimits, SemanticFrame


class RussianParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RussianParser()

    def frame(self, text: str) -> SemanticFrame:
        result = self.parser.parse(text)
        self.assertTrue(result.complete, (text, result.reason))
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].predicate, "")
        return result.frames[0]

    def rejected(self, text: str) -> str:
        result = self.parser.parse(text)
        self.assertFalse(result.complete, (text, result.frames))
        self.assertEqual(result.frames, ())
        self.assertTrue(result.reason)
        return result.reason

    def test_verbless_location_has_raw_teacher_cue_not_predicate(self) -> None:
        frame = self.frame("КЛЮЧ в ЯЩИКЕ.")
        self.assertEqual((frame.act, frame.cue), ("inform", "местонахождение"))
        self.assertEqual(
            (frame.object, frame.place, frame.spatial), ("ключ", "ящик", "in")
        )

    def test_on_and_in_are_not_collapsed(self) -> None:
        inside = self.frame("ключ в столе")
        on_top = self.frame("ключ на столе")
        self.assertEqual(inside.place, on_top.place)
        self.assertEqual((inside.spatial, on_top.spatial), ("in", "on"))

    def test_location_cues_stay_uninterpreted(self) -> None:
        for cue, tense in (
            ("лежит", "current"),
            ("находится", "current"),
            ("остался", "past"),
            ("лежал", "past"),
        ):
            with self.subTest(cue=cue):
                frame = self.frame(f"ключ {cue} в ящике")
                self.assertEqual((frame.cue, frame.tense), (cue, tense))

    def test_noun_accusative_and_location_cases_are_explicit(self) -> None:
        frame = self.frame("Маша положила книгу на полку.")
        self.assertEqual(
            (frame.actor, frame.object, frame.place, frame.spatial),
            ("маша", "книга", "полка", "on"),
        )
        self.assertEqual(frame.tense, "past")

    def test_move_syntax_preserves_actor_and_raw_cue(self) -> None:
        for cue in ("положил", "переложил", "поместил"):
            with self.subTest(cue=cue):
                frame = self.frame(f"я {cue} ключ в сумку")
                self.assertEqual(frame.actor, "@speaker")
                self.assertEqual(
                    (frame.cue, frame.object, frame.place), (cue, "ключ", "сумка")
                )

    def test_transfer_role_reversal_is_preserved(self) -> None:
        forward = self.frame("Миша передал ключ Маше.")
        backward = self.frame("Маша передала ключ Мише.")
        self.assertEqual((forward.actor, forward.recipient), ("миша", "маша"))
        self.assertEqual((backward.actor, backward.recipient), ("маша", "миша"))
        self.assertEqual(forward.object, backward.object)

    def test_declared_dative_can_precede_object(self) -> None:
        frame = self.frame("Миша передал Маше книгу")
        self.assertEqual(
            (frame.actor, frame.object, frame.recipient), ("миша", "книга", "маша")
        )

    def test_wrong_known_actor_or_recipient_case_does_not_swap_roles(self) -> None:
        for text in (
            "Мише передал ключ Маша",
            "Мишу передал ключ Маше",
            "Миша передал ключ Маша",
            "Миша передал ключ Маши",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_possession_uses_nominative_not_accusative(self) -> None:
        for person, obj, expected in (
            ("Маши", "книга", "книга"),
            ("Анны", "игрушка", "игрушка"),
            ("Миши", "ключ", "ключ"),
        ):
            with self.subTest(obj=obj):
                frame = self.frame(f"У {person} есть {obj}.")
                self.assertEqual((frame.cue, frame.object), ("есть", expected))
                self.assertFalse(frame.negated)
        self.rejected("у миши есть книгу")

    def test_possession_absence_preserves_negation_and_genitive(self) -> None:
        frame = self.frame("У Миши нет книги.")
        self.assertEqual(
            (frame.cue, frame.actor, frame.object), ("нет", "миша", "книга")
        )
        self.assertTrue(frame.negated)
        self.rejected("у миши нет книгу")

    def test_past_and_verbless_possession(self) -> None:
        frame = self.frame("у миши не было ключа")
        self.assertEqual(
            (frame.cue, frame.tense, frame.negated), ("было", "past", True)
        )
        self.assertEqual(self.frame("у миши ключ").cue, "обладание")
        self.assertEqual(self.frame("миша имеет книгу").object, "книга")

    def test_questions_are_explicit_scaffold(self) -> None:
        cases = (
            ("Где ключ?", "where", "ключ", ""),
            ("Где находится книга?", "where", "книга", ""),
            ("У кого ключ?", "who_has", "ключ", ""),
            ("У кого есть книга?", "who_has", "книга", ""),
            ("Что у Миши?", "what_has", "", "миша"),
            ("Что у Маши есть?", "what_has", "", "маша"),
        )
        for text, query, obj, actor in cases:
            with self.subTest(text=text):
                frame = self.frame(text)
                self.assertEqual(
                    (frame.act, frame.query, frame.object, frame.actor),
                    ("ask", query, obj, actor),
                )

    def test_question_word_does_not_require_question_mark(self) -> None:
        self.assertEqual(self.frame("где ключ").act, "ask")

    def test_provenance_queries_use_explicit_scaffold_and_object_reference(
        self,
    ) -> None:
        for text in ("почему?", "почему ты так считаешь?", "откуда ты знаешь?"):
            with self.subTest(text=text):
                frame = self.frame(text)
                self.assertEqual(
                    (frame.act, frame.query, frame.object), ("ask", "why", "@object")
                )
                self.assertEqual(frame.cue, "")
        frame = self.frame("почему ключ?")
        self.assertEqual((frame.act, frame.query, frame.object), ("ask", "why", "ключ"))

    def test_provenance_scaffold_does_not_consume_partial_causal_questions(
        self,
    ) -> None:
        for text in (
            "почему ключ в ящике?",
            "почему миша передал ключ маше?",
            "почему ты так считаешь ключ в ящике?",
            "ключ в ящике. почему ключ не в сумке?",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_location_yes_no_retains_raw_cue_and_relation(self) -> None:
        frame = self.frame("Книга на столе?")
        self.assertEqual(
            (frame.act, frame.query, frame.cue, frame.spatial),
            ("ask", "verify", "местонахождение", "on"),
        )
        self.assertTrue(self.frame("ключ не в ящике?").negated)
        self.assertEqual(self.frame("правда ли, что ключ в ящике").act, "ask")

    def test_possession_yes_no_is_not_assertion(self) -> None:
        frame = self.frame("У Маши есть книга?")
        self.assertEqual(
            (frame.act, frame.query, frame.actor, frame.object),
            ("ask", "verify", "маша", "книга"),
        )

    def test_negation_is_never_dropped_from_supported_roles(self) -> None:
        for text in (
            "ключ не в ящике",
            "ключ не лежит в ящике",
            "я не положил ключ в сумку",
            "Миша не передал ключ Маше",
            "миша не имеет книги",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.frame(text).negated)

    def test_misplaced_double_or_limited_negation_is_rejected(self) -> None:
        for text in (
            "не ключ в ящике",
            "ключ не не в ящике",
            "ключ не только в ящике",
            "миша передал не ключ маше",
            "миша никогда не положил ключ в сумку",
            "ключ нигде не лежит",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_explicit_correction_retains_negation_and_scope(self) -> None:
        frame = self.frame("Нет, ключ остался в ящике.")
        self.assertEqual(
            (frame.act, frame.object, frame.place), ("correct", "ключ", "ящик")
        )
        self.assertEqual(self.frame("нет, ключ не в ящике").act, "correct")
        self.assertTrue(self.frame("нет, ключ не в ящике").negated)
        self.rejected("нет")

    def test_retraction_commands_and_reported_error(self) -> None:
        for text in (
            "последнее сообщение было ошибкой",
            "отмени последнее утверждение",
            "нет, последнее сообщение было ошибкой",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.frame(text).act, "retract")
        self.assertEqual(self.frame("отмени сообщение 3").reference, 3)
        self.rejected("отмени сообщение 0")
        self.rejected("отмени последнее утверждение?")

    def test_conditional_statement_never_asserts(self) -> None:
        for text in (
            "если ключ в ящике",
            "если я положил ключ в сумку",
            "я бы положил ключ в сумку",
            "если бы я положил ключ в сумку",
            "возможно, ключ в ящике",
            "предположим, ключ в ящике",
        ):
            with self.subTest(text=text):
                frame = self.frame(text)
                self.assertEqual(
                    (frame.act, frame.modality), ("hypothesis", "hypothetical")
                )

    def test_future_is_hypothetical_and_not_completed_move(self) -> None:
        for text in (
            "завтра я положил ключ в сумку",
            "я положу ключ в сумку",
            "ключ будет в ящике",
            "ключ будет лежать в ящике",
            "я буду класть ключ в сумку",
            "миша передаст ключ маше",
        ):
            with self.subTest(text=text):
                frame = self.frame(text)
                self.assertEqual(
                    (frame.act, frame.modality, frame.tense),
                    ("hypothesis", "hypothetical", "future"),
                )

    def test_unknown_future_form_cannot_silently_become_assertion(self) -> None:
        self.rejected("миша перенесет ключ в сумку")
        self.rejected("миша переместит ключ в коробку")
        self.rejected("ключ окажется в сумку")

    def test_hypothetical_question_or_correction_is_not_mutation(self) -> None:
        for text in (
            "если ключ в ящике?",
            "нет, если ключ в ящике",
            "нет, завтра ключ в ящике",
            "нет, где ключ?",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_conditional_branches_are_not_split_into_facts(self) -> None:
        self.rejected("если ключ в ящике, то паспорт в сумке")
        self.rejected("ключ в ящике или ключ в сумке")

    def test_quoted_and_reported_speech_is_not_assertion(self) -> None:
        for text in (
            '"ключ в ящике"',
            "«ключ в ящике»",
            "'ключ в ящике'",
            "Миша сказал, что ключ в ящике",
            "Миша говорит ключ в ящике",
            "думаю ключ в ящике",
            "ключ якобы в ящике",
            "ключ в ящике. Маша сообщила ключ в сумке",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_greetings_thanks_help_are_explicit_scaffold(self) -> None:
        for text, act in (
            ("Привет!", "greet"),
            ("Добрый день", "greet"),
            ("Спасибо", "thanks"),
            ("что ты умеешь?", "help"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.frame(text).act, act)

    def test_reporting_like_past_cues_cannot_be_learned_as_asserted_actions(
        self,
    ) -> None:
        # In particular, предположил contains положил but is not a move event.
        for cue in ("предположил", "вообразил", "придумал", "обещал", "планировал"):
            with self.subTest(cue=cue):
                reason = self.rejected(f"миша {cue} ключ в сумку")
                self.assertEqual(reason, "reported_speech_unsupported")

    def test_topics_preserve_normalized_name_and_do_not_resolve_entities(self) -> None:
        frame = self.frame("Тема: рабочий Дом-2")
        self.assertEqual((frame.act, frame.topic), ("topic", "рабочий дом-2"))
        self.rejected("тема:")
        self.rejected("тема: работа, и ключ в ящике")

    def test_open_entities_are_preserved_without_guessed_stemming(self) -> None:
        frame = self.frame("Зарина положила флешку-7 в контейнере-42")
        self.assertEqual(
            (frame.actor, frame.object, frame.place),
            ("зарина", "флешку-7", "контейнере-42"),
        )
        frame2 = self.frame("Флешка-7 в контейнере-42")
        self.assertNotEqual(frame.object, frame2.object)

    def test_object_kind_comes_only_from_declared_entity_tables(self) -> None:
        cases = (
            ("Петя в комнате", "person"),
            ("Где Маша?", "person"),
            ("Миша передал Машу Пете", "person"),
            ("ключ в ящике", "thing"),
            ("Маша положила книгу на стол", "thing"),
            ("У Миши нет книги", "thing"),
            ("зарина в комнате", "unknown"),
            ("флешка в ящике", "unknown"),
            ("он в комнате", "unknown"),
            ("я положил его в сумку", "unknown"),
            ("привет", "unknown"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.frame(text).object_kind, expected)

    def test_unknown_past_verb_is_only_raw_cue_and_not_teacher_label(self) -> None:
        frame = self.frame("миша блимил ключ в сумку")
        self.assertEqual(frame.cue, "блимил")
        self.assertNotIn(frame.cue, dict(DEFAULT_TEACHING_PAIRS))
        self.assertEqual(frame.predicate, "")

    def test_pronouns_have_explicit_unresolved_markers(self) -> None:
        frame = self.frame("я положил его в сумку")
        self.assertEqual((frame.actor, frame.object), ("@speaker", "@object"))
        self.assertEqual(self.frame("он положил ее в сумку").actor, "@person")
        self.assertEqual(self.frame("я передал ключ ей").recipient, "@person")
        self.assertEqual(self.frame("миша передал ключ мне").recipient, "@speaker")
        self.assertEqual(self.frame("ключ в ней").place, "@place")
        self.assertEqual(self.frame("ключ на нем").spatial, "on")
        self.assertEqual(self.frame("это в ящике").object, "@ambiguous")
        self.assertEqual(self.frame("он в комнате").object, "@ambiguous")
        self.assertEqual(self.frame("у меня есть ключ").actor, "@speaker")

    def test_yo_normalizes_without_guessing_other_morphology(self) -> None:
        self.assertEqual(self.frame("я положил ЕЁ в сумку").object, "@object")

    def test_unsupported_adjectives_and_omitted_roles_abstain(self) -> None:
        for text in (
            "красная книга в ящике",
            "миша положил ключ",
            "потом переложил его в сумку",
            "ключ там",
            "миша положил ключ туда",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_multiclause_success_consumes_all_clauses(self) -> None:
        result = self.parser.parse(
            "Ключ в ящике. Маша положила книгу на стол; где ключ?"
        )
        self.assertTrue(result.complete, result.reason)
        self.assertEqual(
            [frame.act for frame in result.frames], ["inform", "inform", "ask"]
        )
        self.assertTrue(all(frame.predicate == "" for frame in result.frames))

    def test_failed_suffix_drops_all_successful_prefix_frames(self) -> None:
        for text in (
            "ключ в ящике. совершенно непонятно",
            "ключ в ящике спасибо",
            "ключ в ящике и паспорт в сумке",
            "ключ в ящике. не ключ в сумке",
        ):
            with self.subTest(text=text):
                self.rejected(text)

    def test_newline_separates_clauses(self) -> None:
        result = self.parser.parse("ключ в ящике\nгде ключ?")
        self.assertTrue(result.complete)
        self.assertEqual(len(result.frames), 2)

    def test_character_limit_is_checked_before_work(self) -> None:
        parser = RussianParser(ConversationLimits(max_chars=8))
        result = parser.parse("а" * 9)
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "character_limit")

    def test_token_limit_is_total_not_per_clause(self) -> None:
        parser = RussianParser(ConversationLimits(max_tokens=5))
        result = parser.parse("ключ в ящике. где ключ?")
        self.assertFalse(result.complete)
        self.assertEqual(result.reason, "token_limit")

    def test_clause_limit_rolls_back_prefix(self) -> None:
        parser = RussianParser(ConversationLimits(max_clauses=1))
        result = parser.parse("ключ в ящике. где ключ?")
        self.assertFalse(result.complete)
        self.assertEqual((result.reason, result.frames), ("clause_limit", ()))

    def test_bad_types_controls_and_symbols_are_rejected(self) -> None:
        self.assertFalse(self.parser.parse(None).complete)  # type: ignore[arg-type]
        self.rejected("ключ\x00 в ящике")
        self.rejected("ключ = в ящике")
        self.rejected("@object в ящике")
        self.rejected("")
        self.rejected("...?")

    def test_oversized_individual_field_does_not_raise_schema_error(self) -> None:
        self.rejected(f"{'я' * 129} в ящике")

    def test_bounded_adversarial_inputs_finish_promptly(self) -> None:
        started = time.perf_counter()
        for _ in range(100):
            result = self.parser.parse("а-" * 1000)
            self.assertFalse(result.complete)
        self.assertLess(time.perf_counter() - started, 2.0)

    def test_teacher_corpus_is_explicit_unique_and_bounded(self) -> None:
        self.assertLessEqual(len(DEFAULT_TEACHING_PAIRS), 64)
        self.assertEqual(len(DEFAULT_TEACHING_PAIRS), len(dict(DEFAULT_TEACHING_PAIRS)))
        self.assertEqual(
            {label for _, label in DEFAULT_TEACHING_PAIRS},
            {"locate", "move", "give", "have"},
        )
        self.assertTrue(all("<" not in cue for cue, _ in DEFAULT_TEACHING_PAIRS))


class SurfaceScaffoldTests(unittest.TestCase):
    def test_finite_declared_cases(self) -> None:
        self.assertEqual(surface_entity("ключ"), "ключ")
        self.assertEqual(surface_entity("книга", "accusative"), "книгу")
        self.assertEqual(surface_entity("маша", "genitive"), "Маши")
        self.assertEqual(surface_entity("маша", "dative"), "Маше")
        self.assertEqual(surface_entity("сумка", "locative"), "сумке")

    def test_unknown_surface_is_quoted_without_fake_inflection(self) -> None:
        self.assertEqual(surface_entity("контейнер-42", "locative"), "«контейнер-42»")
        self.assertEqual(surface_entity("зарина", "dative"), "«зарина»")

    def test_bad_case_is_explicit_error(self) -> None:
        with self.assertRaises(ValueError):
            surface_entity("ключ", "made-up")


if __name__ == "__main__":
    unittest.main()

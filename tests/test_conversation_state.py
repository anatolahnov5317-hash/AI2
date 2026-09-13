"""World-state correctness tests; these are not learned-intelligence claims."""

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from typing import Any

from text_factors.conversation.schema import ConversationLimits, SemanticFrame
from text_factors.conversation.state import WorldState


def located(object: str = "ключ", place: str = "ящик", **kwargs) -> SemanticFrame:
    kwargs.setdefault("object_kind", "thing")
    return SemanticFrame(
        "inform", predicate="locate", object=object, place=place, **kwargs
    )


def ask(query: str = "where", object: str = "ключ", **kwargs) -> SemanticFrame:
    return SemanticFrame("ask", query=query, object=object, **kwargs)


def values(state: WorldState) -> set[tuple[str, str, str, bool, str]]:
    return {
        (f.subject, f.relation, f.value, f.negated, f.qualifier) for f in state.facts()
    }


class ConversationStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = WorldState()

    def test_assertion_requires_learned_decoded_predicate(self) -> None:
        outcome = self.state.apply(
            SemanticFrame("inform", cue="лежит", object="ключ", place="ящик"),
            turn_id=1,
        )
        self.assertEqual(
            (outcome.action, outcome.reason), ("clarify", "predicate_required")
        )
        self.assertEqual(self.state.events, [])
        self.assertEqual(self.state.facts(), ())

    def test_decoded_predicates_cannot_silently_discard_supplied_roles(self) -> None:
        frames = (
            located(actor="миша"),
            located(recipient="маша"),
            SemanticFrame(
                "inform", predicate="have", actor="миша", object="ключ", place="ящик"
            ),
            SemanticFrame(
                "inform",
                predicate="give",
                actor="миша",
                object="ключ",
                recipient="маша",
                place="ящик",
            ),
            SemanticFrame(
                "inform",
                predicate="move",
                actor="миша",
                object="ключ",
                recipient="маша",
                place="ящик",
            ),
        )
        for frame in frames:
            with self.subTest(frame=frame):
                self.assertEqual(
                    self.state.apply(frame, turn_id=1).reason, "incompatible_roles"
                )
                self.assertEqual(self.state.events, [])
        incomplete = SemanticFrame(
            "inform", predicate="move", object="ключ", place="ящик"
        )
        self.assertEqual(
            self.state.apply(incomplete, turn_id=1).reason, "missing_assertion_roles"
        )

    def test_incompatible_persisted_roles_are_rejected_even_when_both_frames_match(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        snapshot = self.state.to_dict()
        snapshot["events"][0]["input_frame"]["actor"] = "миша"
        snapshot["events"][0]["frame"]["actor"] = "миша"
        with self.assertRaises(ValueError):
            WorldState.from_dict(snapshot)

    def test_invalid_unicode_returns_clarification_without_partial_mutation(
        self,
    ) -> None:
        for frame in (
            located(raw="\ud800"),
            located(cue="\udfff"),
            located(object="\ud800"),
        ):
            with self.subTest(frame=frame):
                self.assertEqual(self.state.apply(frame, turn_id=1).action, "clarify")
                self.assertEqual(self.state.events, [])

    def test_location_provenance_and_spatial_qualifier(self) -> None:
        outcome = self.state.apply(located(spatial="on"), turn_id=1, source="alice")
        self.assertEqual(outcome.action, "ack")
        fact = outcome.assertions[0]
        self.assertEqual(
            (fact.subject, fact.relation, fact.value, fact.qualifier),
            ("ключ", "location", "ящик", "on"),
        )
        self.assertEqual(
            (fact.event_id, fact.source, fact.topic), (1, "alice", "default")
        )
        answer = self.state.apply(ask(), turn_id=2)
        self.assertEqual(answer.assertions, (fact,))
        self.assertEqual(answer.event_ids, (1,))

    def test_move_replaces_location_instead_of_remembering_both(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(
            SemanticFrame(
                "inform", predicate="move", actor="я", object="ключ", place="сумка"
            ),
            turn_id=2,
        )
        self.assertEqual(
            values(self.state), {("ключ", "location", "сумка", False, "in")}
        )
        self.assertEqual(len(self.state.events), 2)

    def test_give_preserves_roles_and_does_not_invent_place(self) -> None:
        self.state.apply(located(), turn_id=1)
        frame = SemanticFrame(
            "inform",
            predicate="give",
            actor="петя",
            object="ключ",
            recipient="маша",
        )
        self.state.apply(frame, turn_id=2)
        self.assertEqual(values(self.state), {("ключ", "holder", "маша", False, "in")})
        event = self.state.events[-1]
        self.assertEqual(event["frame"]["actor"], "петя")
        self.assertEqual(event["frame"]["recipient"], "маша")
        self.assertEqual(
            self.state.apply(ask("who_has"), turn_id=3).assertions[0].value, "маша"
        )
        where = self.state.apply(ask(), turn_id=4)
        self.assertEqual(where.assertions[0].relation, "holder")
        self.assertEqual(
            self.state.apply(ask(object="маша"), turn_id=5).action, "unknown"
        )

    def test_transfer_role_swap_changes_holder(self) -> None:
        frame = SemanticFrame(
            "inform",
            predicate="give",
            actor="петя",
            object="ключ",
            recipient="маша",
        )
        self.state.apply(frame, turn_id=1)
        self.state.apply(replace(frame, actor="маша", recipient="петя"), turn_id=2)
        self.assertEqual(self.state.facts()[0].value, "петя")

    def test_have_and_what_has_return_object_holder_assertions(self) -> None:
        for turn_id, object in enumerate(("ключ", "книга"), start=1):
            self.state.apply(
                SemanticFrame("inform", predicate="have", object=object, actor="маша"),
                turn_id=turn_id,
            )
        answer = self.state.apply(ask("what_has", object="", actor="маша"), turn_id=3)
        self.assertEqual(answer.action, "answer")
        self.assertEqual({f.subject for f in answer.assertions}, {"ключ", "книга"})
        self.assertTrue(
            all(f.relation == "holder" and f.value == "маша" for f in answer.assertions)
        )
        self.assertEqual(
            self.state.apply(
                ask("what_has", object="", actor="петя"), turn_id=4
            ).action,
            "unknown",
        )

    def test_negative_location_excludes_only_stated_place(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(negated=True), turn_id=2)
        self.assertEqual(values(self.state), {("ключ", "location", "ящик", True, "in")})
        where = self.state.apply(ask(), turn_id=3)
        self.assertEqual((where.action, where.assertions), ("unknown", ()))
        answer = self.state.apply(
            ask("verify", predicate="locate", place="ящик"), turn_id=4
        )
        self.assertEqual((answer.action, answer.reason), ("answer", "false"))
        self.assertTrue(answer.assertions[0].negated)
        alternative = self.state.apply(
            ask("verify", predicate="locate", place="сумка"), turn_id=5
        )
        self.assertEqual(
            (alternative.action, alternative.reason), ("unknown", "unknown")
        )

    def test_negative_holder_never_invents_another_holder(self) -> None:
        self.state.apply(
            SemanticFrame(
                "inform", predicate="have", object="ключ", actor="маша", negated=True
            ),
            turn_id=1,
        )
        self.assertEqual(self.state.apply(ask("who_has"), turn_id=2).action, "unknown")
        answer = self.state.apply(
            ask("verify", predicate="have", actor="маша"), turn_id=3
        )
        self.assertEqual(answer.reason, "false")

    def test_negative_action_does_not_mean_absent_from_destination(self) -> None:
        self.state.apply(located(), turn_id=1)
        before = self.state.facts()
        outcome = self.state.apply(
            SemanticFrame(
                "inform",
                predicate="move",
                actor="я",
                object="ключ",
                place="ящик",
                negated=True,
            ),
            turn_id=2,
        )
        self.assertEqual(outcome.reason, "negated_action_not_state")
        self.assertEqual(outcome.assertions, ())
        self.assertEqual(self.state.facts(), before)
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="give",
                object="ключ",
                actor="петя",
                recipient="маша",
                negated=True,
            ),
            turn_id=3,
        )
        self.assertEqual(self.state.facts(), before)
        answer = self.state.apply(
            ask("verify", predicate="move", actor="я", place="ящик"), turn_id=4
        )
        self.assertEqual(answer.action, "unknown")

    def test_in_and_on_are_distinct_in_assertions_and_queries(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(negated=True, spatial="on"), turn_id=2)
        self.assertEqual(len(self.state.facts()), 2)
        positive = self.state.apply(
            ask("verify", predicate="locate", place="ящик"), turn_id=3
        )
        negative = self.state.apply(
            ask("verify", predicate="locate", place="ящик", spatial="on"), turn_id=4
        )
        self.assertEqual((positive.reason, negative.reason), ("true", "false"))

    def test_verification_distinguishes_false_unknown_and_negated_question(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        absent = self.state.apply(
            ask("verify", predicate="locate", place="сумка"), turn_id=2
        )
        unknown = self.state.apply(
            ask("verify", object="паспорт", predicate="locate", place="сумка"),
            turn_id=3,
        )
        negated = self.state.apply(
            ask("verify", predicate="locate", place="ящик", negated=True), turn_id=4
        )
        self.assertEqual((absent.action, absent.reason), ("answer", "false"))
        self.assertEqual((unknown.action, unknown.reason), ("unknown", "unknown"))
        self.assertEqual(negated.reason, "false")

    def test_questions_never_become_facts(self) -> None:
        for turn, query in enumerate(("where", "who_has", "why"), start=1):
            self.assertEqual(
                self.state.apply(ask(query), turn_id=turn).action, "unknown"
            )
        self.assertEqual(self.state.facts(), ())
        self.assertEqual(len(self.state.events), 3)
        self.assertTrue(all(event["kind"] == "query" for event in self.state.events))

    def test_successful_question_shifts_object_focus_without_changing_facts(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="паспорт", place="сумка"), turn_id=2)
        before = self.state.facts()
        self.state.apply(ask(object="ключ"), turn_id=3)
        self.assertEqual(self.state.facts(), before)
        outcome = self.state.apply(
            SemanticFrame(
                "inform",
                predicate="move",
                actor="я",
                object="@object",
                place="стол",
                spatial="on",
            ),
            turn_id=4,
        )
        self.assertEqual(outcome.assertions[0].subject, "ключ")
        self.assertEqual(
            self.state.apply(ask(object="паспорт"), turn_id=5).assertions[0].value,
            "сумка",
        )

    def test_unknown_question_explicit_entity_becomes_a_referent_not_a_fact(
        self,
    ) -> None:
        outcome = self.state.apply(ask(object="паспорт"), turn_id=1)
        self.assertEqual(outcome.action, "unknown")
        self.assertEqual(self.state.facts(), ())
        outcome = self.state.apply(located(object="@object", place="сумка"), turn_id=2)
        self.assertEqual(outcome.assertions[0].subject, "паспорт")

    def test_latest_question_focus_overrides_multi_clause_assertion_ambiguity(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="паспорт", place="сумка"), turn_id=1)
        self.state.apply(ask(object="ключ"), turn_id=1)
        outcome = self.state.apply(located(object="@object", place="стол"), turn_id=2)
        self.assertEqual(outcome.assertions[0].subject, "ключ")

    def test_person_question_sets_separate_person_focus(self) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="give",
                actor="миша",
                object="ключ",
                recipient="маша",
            ),
            turn_id=1,
        )
        self.state.apply(ask("what_has", object="", actor="миша"), turn_id=2)
        outcome = self.state.apply(
            SemanticFrame("inform", predicate="have", actor="@person", object="книга"),
            turn_id=3,
        )
        self.assertEqual(outcome.assertions[0].value, "миша")

    def test_query_focus_survives_snapshot_and_query_is_not_a_retraction_target(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(ask(object="паспорт"), turn_id=2)
        restored = WorldState.from_dict(self.state.to_dict())
        self.assertEqual(
            restored.apply(SemanticFrame("retract", reference=2), turn_id=3).reason,
            "reference_inactive",
        )
        restored.apply(SemanticFrame("retract"), turn_id=3)
        self.assertEqual(restored.facts(), ())
        outcome = restored.apply(located(object="@object", place="сумка"), turn_id=4)
        self.assertEqual(outcome.assertions[0].subject, "паспорт")

    def test_duplicate_question_clauses_reapply_focus(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(ask(), turn_id=1)
        self.state.apply(located(object="паспорт", place="сумка"), turn_id=1)
        self.state.apply(ask(), turn_id=1)
        outcome = self.state.apply(located(object="@object", place="стол"), turn_id=2)
        self.assertEqual(outcome.assertions[0].subject, "ключ")
        self.assertEqual(
            WorldState.from_dict(self.state.to_dict()).to_dict(), self.state.to_dict()
        )

    def test_query_focus_updates_are_bounded_and_atomic(self) -> None:
        state = WorldState(ConversationLimits(max_events=1))
        state.apply(ask(), turn_id=1)
        before = state.to_dict()
        outcome = state.apply(ask(object="паспорт"), turn_id=2)
        self.assertEqual(outcome.reason, "state_capacity")
        self.assertEqual(state.to_dict(), before)

    def test_why_returns_source_and_event_provenance(self) -> None:
        self.state.apply(located(), turn_id=1, source="alice")
        answer = self.state.apply(ask("why"), turn_id=2)
        self.assertEqual((answer.action, answer.reason), ("answer", "provenance"))
        self.assertEqual(answer.event_ids, (1,))
        self.assertEqual(answer.assertions[0].source, "alice")

    def test_hypothetical_reported_and_future_do_not_change_state(self) -> None:
        frames = (
            located(modality="hypothetical"),
            located(modality="reported"),
            located(tense="future"),
            replace(located(), act="hypothesis"),
        )
        for frame in frames:
            with self.subTest(frame=frame):
                self.assertEqual(
                    self.state.apply(frame, turn_id=1).action, "hypothetical"
                )
                self.assertEqual(self.state.events, [])
                self.assertEqual(self.state.facts(), ())

    def test_nonasserted_topic_switch_never_changes_visible_facts(self) -> None:
        self.state.apply(located(), turn_id=1)
        before = self.state.to_dict()
        for frame in (
            SemanticFrame("topic", topic="работа", modality="reported"),
            SemanticFrame("topic", topic="работа", modality="hypothetical"),
            SemanticFrame("topic", topic="работа", tense="future"),
        ):
            with self.subTest(frame=frame):
                self.assertEqual(
                    self.state.apply(frame, turn_id=2).action, "hypothetical"
                )
                self.assertEqual(self.state.to_dict(), before)

    def test_object_and_place_coreferences_use_unambiguous_prior_roles(self) -> None:
        self.state.apply(located(), turn_id=1)
        outcome = self.state.apply(
            SemanticFrame(
                "inform", predicate="move", actor="я", object="@object", place="сумка"
            ),
            turn_id=2,
        )
        assert outcome.resolved_frame is not None
        self.assertEqual(outcome.resolved_frame.object, "ключ")
        outcome = self.state.apply(located(object="книга", place="@place"), turn_id=3)
        assert outcome.resolved_frame is not None
        self.assertEqual(outcome.resolved_frame.place, "сумка")

    def test_ambiguous_multi_clause_objects_do_not_silently_pick_last(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="книга", place="полка"), turn_id=1)
        before = self.state.to_dict()
        outcome = self.state.apply(located(object="@object", place="сумка"), turn_id=2)
        self.assertEqual(
            (outcome.action, outcome.reason), ("clarify", "ambiguous_reference")
        )
        self.assertEqual(set(outcome.alternatives), {"ключ", "книга"})
        self.assertEqual(self.state.to_dict(), before)

    def test_person_and_object_salience_are_separate(self) -> None:
        self.state.apply(
            SemanticFrame("inform", predicate="have", object="ключ", actor="маша"),
            turn_id=1,
        )
        self.state.apply(located(object="книга", place="полка"), turn_id=2)
        outcome = self.state.apply(
            SemanticFrame(
                "inform", predicate="have", object="@object", actor="@person"
            ),
            turn_id=3,
        )
        assert outcome.resolved_frame is not None
        self.assertEqual(
            (outcome.resolved_frame.object, outcome.resolved_frame.actor),
            ("книга", "маша"),
        )

    def test_person_location_subject_sets_person_focus_not_object_focus(self) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="have",
                actor="миша",
                object="ключ",
                object_kind="thing",
            ),
            turn_id=1,
        )
        self.state.apply(
            located(object="петя", place="комната", object_kind="person"), turn_id=2
        )
        outcome = self.state.apply(
            SemanticFrame(
                "inform",
                predicate="move",
                actor="@person",
                object="ключ",
                object_kind="thing",
                place="ящик",
            ),
            turn_id=3,
        )
        assert outcome.resolved_frame is not None
        self.assertEqual(outcome.resolved_frame.actor, "петя")
        self.assertEqual(
            self.state.apply(ask(object="@object"), turn_id=4).assertions[0].subject,
            "ключ",
        )

    def test_unknown_later_subject_blocks_stale_person_reference(self) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="have",
                actor="миша",
                object="ключ",
                object_kind="thing",
            ),
            turn_id=1,
        )
        self.state.apply(
            located(object="ксен", place="комната", object_kind="unknown"), turn_id=2
        )
        before = self.state.to_dict()
        outcome = self.state.apply(
            SemanticFrame(
                "inform", predicate="move", actor="@person", object="ключ", place="ящик"
            ),
            turn_id=3,
        )
        self.assertEqual(outcome.action, "clarify")
        self.assertEqual(self.state.to_dict(), before)

    def test_unknown_object_does_not_erase_explicit_actor_in_same_turn(self) -> None:
        self.state.apply(
            SemanticFrame("inform", predicate="have", actor="миша", object="жетон"),
            turn_id=1,
        )
        self.state.apply(
            located(object="ксен", place="комната", object_kind="unknown"), turn_id=1
        )
        outcome = self.state.apply(
            SemanticFrame("inform", predicate="have", actor="@person", object="ключ"),
            turn_id=2,
        )
        self.assertEqual(outcome.assertions[0].value, "миша")

    def test_explicit_unknown_person_question_blocks_stale_person_focus(self) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="have",
                actor="миша",
                object="ключ",
                object_kind="thing",
            ),
            turn_id=1,
        )
        self.state.apply(ask(object="ксен", object_kind="unknown"), turn_id=2)
        outcome = self.state.apply(
            SemanticFrame("inform", predicate="have", actor="@person", object="книга"),
            turn_id=3,
        )
        self.assertEqual(outcome.action, "clarify")

    def test_typed_person_question_sets_person_focus_even_when_answer_unknown(
        self,
    ) -> None:
        self.state.apply(ask(object="петя", object_kind="person"), turn_id=1)
        outcome = self.state.apply(
            SemanticFrame("inform", predicate="have", actor="@person", object="ключ"),
            turn_id=2,
        )
        self.assertEqual(outcome.assertions[0].value, "петя")

    def test_pronoun_resolution_preserves_known_object_type_through_restore(
        self,
    ) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="have",
                actor="миша",
                object="ключ",
                object_kind="thing",
            ),
            turn_id=1,
        )
        self.state.apply(
            located(object="@object", place="стол", object_kind="unknown"), turn_id=2
        )
        self.assertEqual(self.state.events[-1]["frame"]["object_kind"], "thing")
        restored = WorldState.from_dict(self.state.to_dict())
        outcome = restored.apply(
            SemanticFrame("inform", predicate="have", actor="@person", object="книга"),
            turn_id=3,
        )
        self.assertEqual(outcome.assertions[0].value, "миша")

    def test_person_reference_with_two_recent_people_requires_clarification(
        self,
    ) -> None:
        self.state.apply(
            SemanticFrame(
                "inform",
                predicate="give",
                object="ключ",
                actor="петя",
                recipient="маша",
            ),
            turn_id=1,
        )
        before = self.state.to_dict()
        outcome = self.state.apply(
            ask("what_has", object="", actor="@person"), turn_id=2
        )
        self.assertEqual(outcome.reason, "ambiguous_reference")
        self.assertEqual(set(outcome.alternatives), {"петя", "маша"})
        self.assertEqual(self.state.to_dict(), before)

    def test_unknown_and_untyped_references_never_create_entity_names(self) -> None:
        for marker in ("@object", "@person", "@place", "@ambiguous", "@nonsense"):
            with self.subTest(marker=marker):
                outcome = self.state.apply(located(object=marker), turn_id=1)
                self.assertEqual(outcome.action, "clarify")
                self.assertEqual(self.state.events, [])

    def test_speaker_resolution_does_not_merge_distinct_sources(self) -> None:
        frame = SemanticFrame(
            "inform", predicate="have", object="ключ", actor="@speaker"
        )
        self.state.apply(frame, turn_id=1)
        self.assertEqual(self.state.facts()[0].value, "я")
        self.state.apply(replace(frame, object="книга"), turn_id=2, source="alice")
        self.assertEqual({f.value for f in self.state.facts()}, {"я", "alice"})

    def test_retraction_restores_previous_whereabouts(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        outcome = self.state.apply(SemanticFrame("retract"), turn_id=3)
        self.assertEqual((outcome.action, outcome.event_ids), ("retracted", (2,)))
        self.assertEqual(self.state.facts()[0].value, "ящик")
        self.assertEqual(len(self.state.events), 3)
        self.assertEqual(self.state.events[-1]["retracts"], [2])

    def test_retracting_correction_restores_the_state_before_correction(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        corrected = self.state.apply(
            replace(located(place="стол", spatial="on"), act="correct"), turn_id=3
        )
        self.assertEqual(corrected.reason, "corrected")
        self.assertEqual(self.state.facts()[0].value, "стол")
        self.state.apply(SemanticFrame("retract"), turn_id=4)
        self.assertEqual(self.state.facts()[0].value, "сумка")
        self.state.apply(SemanticFrame("retract"), turn_id=5)
        self.assertEqual(self.state.facts()[0].value, "ящик")

    def test_implicit_retraction_removes_whole_latest_assertion_turn(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        self.state.apply(located(object="книга", place="полка"), turn_id=2)
        outcome = self.state.apply(SemanticFrame("retract"), turn_id=3)
        self.assertEqual(outcome.event_ids, (2, 3))
        self.assertEqual(
            values(self.state), {("ключ", "location", "ящик", False, "in")}
        )

    def test_two_implicit_corrections_in_same_turn_do_not_reach_an_older_turn(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        self.state.apply(
            replace(located(place="стол", spatial="on"), act="correct"), turn_id=3
        )
        before = self.state.to_dict()
        outcome = self.state.apply(
            replace(located(object="книга", place="полка"), act="correct"), turn_id=3
        )
        self.assertEqual(outcome.reason, "multiple_implicit_retractions")
        self.assertEqual(self.state.to_dict(), before)

    def test_explicit_reference_retracts_only_named_event(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="книга", place="полка"), turn_id=1)
        self.state.apply(
            replace(located(place="сумка"), act="correct", reference=1), turn_id=2
        )
        self.assertEqual(
            values(self.state),
            {
                ("ключ", "location", "сумка", False, "in"),
                ("книга", "location", "полка", False, "in"),
            },
        )

    def test_correction_with_unresolved_reference_is_atomic(self) -> None:
        self.state.apply(located(), turn_id=1)
        before = self.state.to_dict()
        outcome = self.state.apply(
            replace(located(object="@ambiguous"), act="correct"), turn_id=2
        )
        self.assertEqual(outcome.action, "clarify")
        self.assertEqual(self.state.to_dict(), before)

    def test_retractions_respect_source_topic_and_activity(self) -> None:
        self.state.apply(located(), turn_id=1, source="alice")
        self.assertEqual(
            self.state.apply(SemanticFrame("retract", reference=1), turn_id=2).reason,
            "reference_source_mismatch",
        )
        self.state.apply(
            SemanticFrame("topic", topic="поездка"), turn_id=2, source="alice"
        )
        self.assertEqual(
            self.state.apply(
                SemanticFrame("retract", reference=1), turn_id=3, source="alice"
            ).reason,
            "reference_topic_mismatch",
        )
        self.state.apply(
            SemanticFrame("topic", topic="default"), turn_id=3, source="alice"
        )
        self.state.apply(
            SemanticFrame("retract", reference=1), turn_id=4, source="alice"
        )
        self.assertEqual(
            self.state.apply(
                SemanticFrame("retract", reference=1), turn_id=5, source="alice"
            ).reason,
            "reference_inactive",
        )

    def test_empty_or_invalid_correction_never_invents_a_target(self) -> None:
        for frame in (
            SemanticFrame("retract"),
            replace(located(), act="correct"),
            SemanticFrame("retract", reference=999),
        ):
            with self.subTest(frame=frame):
                self.assertEqual(self.state.apply(frame, turn_id=1).action, "clarify")
                self.assertEqual(self.state.events, [])

    def test_topics_retain_facts_and_keep_reference_resolution_local(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(SemanticFrame("topic", topic="поездка"), turn_id=2)
        self.assertEqual(self.state.facts(), ())
        self.assertEqual(
            self.state.apply(ask(object="@object"), turn_id=3).action, "clarify"
        )
        self.state.apply(located(place="сумка"), turn_id=3)
        self.state.apply(SemanticFrame("topic", topic="default"), turn_id=4)
        self.assertEqual(self.state.facts()[0].value, "ящик")
        self.state.apply(SemanticFrame("topic", topic="поездка"), turn_id=5)
        self.assertEqual(self.state.facts()[0].value, "сумка")
        self.assertEqual(
            self.state.apply(located(topic="default"), turn_id=6).reason,
            "topic_mismatch",
        )

    def test_clone_does_not_share_mutable_events(self) -> None:
        self.state.apply(located(), turn_id=1)
        clone = self.state.clone()
        clone.apply(located(place="сумка"), turn_id=2)
        clone.events[0]["frame"]["object"] = "другой"
        self.assertEqual(len(self.state.events), 1)
        self.assertEqual(self.state.events[0]["frame"]["object"], "ключ")

    def test_event_entity_byte_and_clause_saturation_are_explicit_atomic(self) -> None:
        for limits in (
            ConversationLimits(max_events=1),
            ConversationLimits(max_entities=2),
            ConversationLimits(max_clauses=1),
        ):
            with self.subTest(limits=limits):
                state = WorldState(limits)
                self.assertEqual(state.apply(located(), turn_id=1).action, "ack")
                before = state.to_dict()
                outcome = state.apply(located(object="книга", place="полка"), turn_id=1)
                self.assertEqual(
                    (outcome.action, outcome.reason), ("clarify", "state_capacity")
                )
                self.assertEqual(state.to_dict(), before)
        state = WorldState(ConversationLimits(max_state_bytes=100))
        self.assertEqual(state.apply(located(), turn_id=1).reason, "state_capacity")
        self.assertEqual(state.events, [])

    def test_retained_entities_still_count_after_retraction_and_topic_switch(
        self,
    ) -> None:
        state = WorldState(ConversationLimits(max_entities=2))
        state.apply(located(), turn_id=1)
        state.apply(SemanticFrame("retract"), turn_id=2)
        state.apply(SemanticFrame("topic", topic="другая"), turn_id=3)
        self.assertEqual(
            state.apply(located(object="книга", place="полка"), turn_id=4).reason,
            "state_capacity",
        )

    def test_input_capacity_and_invalid_identifiers(self) -> None:
        state = WorldState(ConversationLimits(max_chars=4))
        self.assertEqual(
            state.apply(located(raw="12345"), turn_id=1).reason, "input_capacity"
        )
        invalid_ids: tuple[Any, ...] = (False, 0, -1, 1.0, (1 << 53))
        for turn_id in invalid_ids:
            with self.subTest(turn_id=turn_id), self.assertRaises(ValueError):
                state.apply(located(), turn_id=turn_id)
        for source in ("", " bad", "line\nfeed", "@speaker"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                state.apply(located(), turn_id=1, source=source)

    def test_duplicate_identity_and_old_turn_rejection(self) -> None:
        first = located()
        self.state.apply(first, turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        duplicate = self.state.apply(first, turn_id=1)
        self.assertEqual(
            (duplicate.action, duplicate.reason), ("ack", "duplicate_event")
        )
        self.assertEqual(duplicate.event_ids, (1,))
        self.assertEqual(len(self.state.events), 2)
        self.assertEqual(self.state.facts()[0].value, "сумка")
        self.assertEqual(
            self.state.apply(located(place="полка"), turn_id=1).reason,
            "nonmonotonic_turn",
        )
        self.assertEqual(
            self.state.apply(located(place="полка"), turn_id=2, source="alice").reason,
            "turn_source_mismatch",
        )

    def test_snapshot_round_trip_preserves_retraction_replay_and_pronouns(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="@object", place="сумка"), turn_id=2)
        self.state.apply(
            replace(located(place="стол", spatial="on"), act="correct"), turn_id=3
        )
        self.state.apply(SemanticFrame("topic", topic="поездка"), turn_id=4)
        self.state.apply(located(object="паспорт", place="чемодан"), turn_id=5)
        restored = WorldState.from_dict(json.loads(json.dumps(self.state.to_dict())))
        self.assertEqual(restored.to_dict(), self.state.to_dict())
        self.assertEqual(restored.facts(), self.state.facts())
        for state in (self.state, restored):
            state.apply(SemanticFrame("topic", topic="default"), turn_id=6)
            state.apply(SemanticFrame("retract"), turn_id=7)
        self.assertEqual(restored.facts(), self.state.facts())
        self.assertEqual(restored.facts()[0].value, "сумка")

    def test_snapshot_does_not_share_mutable_storage(self) -> None:
        self.state.apply(located(), turn_id=1)
        snapshot = self.state.to_dict()
        restored = WorldState.from_dict(snapshot)
        snapshot["events"][0]["frame"]["place"] = "подмена"
        self.assertEqual(restored.facts()[0].value, "ящик")
        self.assertEqual(self.state.facts()[0].value, "ящик")

    def test_snapshot_rejects_malformed_event_identity_and_resolved_content(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(object="@object", place="сумка"), turn_id=2)
        self.state.apply(SemanticFrame("retract"), turn_id=3)
        original = self.state.to_dict()
        mutations = (
            lambda d: d.update(version=True),
            lambda d: d.update(topic="не тот"),
            lambda d: d.update(unexpected=1),
            lambda d: d["events"][0].update(event_id=True),
            lambda d: d["events"][1].update(event_id=1),
            lambda d: d["events"][1].update(turn_id=0),
            lambda d: d["events"][1].update(turn_id=1),
            lambda d: d["events"][1].update(kind="retract"),
            lambda d: d["events"][1].update(topic="не тот"),
            lambda d: d["events"][1].update(extra=[]),
            lambda d: d["events"][1]["frame"].update(object="паспорт"),
            lambda d: d["events"][1]["frame"].update(spatial="near"),
            lambda d: d["events"][2].update(retracts=[2, 2]),
            lambda d: d["events"][2].update(retracts=[1]),
            lambda d: d["events"][2].update(retracts=[3]),
            lambda d: d["events"][2].update(source="alice"),
        )
        for mutation in mutations:
            damaged = deepcopy(original)
            mutation(damaged)
            with self.subTest(snapshot=damaged), self.assertRaises(ValueError):
                WorldState.from_dict(damaged)
        self.assertEqual(self.state.to_dict(), original)

    def test_snapshot_rejects_malformed_suffix_instead_of_returning_prefix(
        self,
    ) -> None:
        self.state.apply(located(), turn_id=1)
        snapshot = self.state.to_dict()
        snapshot["events"].append({"event_id": 2})
        with self.assertRaises(ValueError):
            WorldState.from_dict(snapshot)
        with self.assertRaises(ValueError):
            WorldState.from_dict({"version": 1, "topic": "default", "events": [None]})

    def test_snapshot_rejects_tighter_limits_and_recursive_payload(self) -> None:
        self.state.apply(located(), turn_id=1)
        self.state.apply(located(place="сумка"), turn_id=2)
        for limits in (
            ConversationLimits(max_events=1),
            ConversationLimits(max_entities=2),
            ConversationLimits(max_state_bytes=100),
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                WorldState.from_dict(self.state.to_dict(), limits=limits)
        recursive = {"version": 1, "topic": "default", "events": []}
        recursive["events"].append(recursive)
        with self.assertRaises(ValueError):
            WorldState.from_dict(recursive)


if __name__ == "__main__":
    unittest.main()

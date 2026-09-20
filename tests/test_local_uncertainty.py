"""Local guard contracts; these fabricated examples are not language evaluation."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import patch

from text_factors.conversation.schema import BudgetExceeded
from text_factors.learning.hypotheses import Observation, digest
from text_factors.learning.schema import (
    DialogueContext,
    Entity,
    Event,
    Interpretation,
    Meaning,
    Query,
)
from text_factors.learning.uncertainty import REASON, UncertaintyState
from text_factors.learning.world import ExperienceWorld


def location(subject="книга", place="стол", *, negated=False, time="past"):
    return Meaning(
        "inform",
        Event(
            "locate",
            object=subject,
            place=place,
            spatial="on",
            negated=negated,
            time=time,
        ),
    )


def effect(subject="книга", place="стол", *, negated=False):
    return {
        "op": "exclude" if negated else "set",
        "subject": subject,
        "relation": "location",
        "value": place,
        "spatial": "on",
    }


def world():
    result = ExperienceWorld()
    for turn, subject in enumerate(("книга", "ключ"), start=1):
        result.apply(
            location(subject),
            [effect(subject)],
            turn_id=turn,
            source=f"сообщение {turn}",
        )
    return result


class LocalUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.world = world()
        self.facts = self.world.facts()
        self.context = DialogueContext(
            entities=(
                Entity("книга", "thing"),
                Entity("ключ", "thing"),
                Entity("маша", "person"),
            ),
            focus=("книга",),
        )
        self.state = UncertaintyState()

    def mark(self, *, text="Книгу унесли.", turn=3, state=None):
        return (state or self.state).observe(
            text, None, turn_id=turn, known_facts=self.facts, context=self.context
        )

    def test_marks_only_explicit_known_subject_and_preserves_original(self):
        before = self.world.to_dict()
        marked = self.mark()
        self.assertEqual([marker["subject"] for marker in marked], ["книга"])
        self.assertEqual(marked[0]["source_positions"], [0])
        self.assertEqual(
            marked[0]["observation"],
            {
                "observation_id": "turn:3",
                "text": "Книгу унесли.",
                "turn_id": 3,
                "source": "сообщение 3",
            },
        )
        self.assertEqual(marked[0]["known_event_ids"], [1])
        self.assertEqual(self.world.to_dict(), before)
        self.assertNotIn("value", marked[0])
        self.assertNotIn("effects", marked[0])
        self.assertIsNone(
            self.state.query(Query("where", "ключ"), known_facts=self.facts)
        )
        outcome = self.state.query(Query("where", "книга"), known_facts=self.facts)
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome["action"], "clarify")
        self.assertEqual(outcome["reason"], REASON)
        self.assertEqual(outcome["assertions"], [])
        self.assertEqual(outcome["evidence"], list(marked))

    def test_unknown_meaning_is_treated_as_uninterpreted(self):
        marked = self.state.observe(
            "Книга исчезла.",
            Interpretation(Meaning("unknown")),
            turn_id=3,
            known_facts=self.facts,
        )
        self.assertEqual([marker["subject"] for marker in marked], ["книга"])

    def test_unrelated_questions_greetings_requests_and_names_are_ignored(self):
        texts = (
            "Снег необычный.",
            "Паспорт исчез.",
            "Книга?",
            "Где книга",
            "Книга куда подевалась",
            "Расскажи про книгу",
            "Спасибо за книгу",
            "Привет, книга",
            "Добрый день, книга",
            "Книга",
            "Книга и ключ",
            "Она пропала.",
            "Всё переменилось.",
            "Книгу, пожалуйста",
        )
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(self.mark(text=text), ())
        self.assertEqual(self.state.markers, [])
        self.assertEqual(self.state.records, [])

    def test_explicit_nonactual_cues_veto_uninterpreted_gate(self):
        for text in (
            "Если книга исчезнет.",
            "Книга завтра исчезнет.",
            "Книга может исчезнуть.",
            "Книга, как сказал кто-то, исчезла.",
            "Книга исчезла бы.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.mark(text=text), ())

    def test_understood_nonupdates_and_candidate_alternatives_do_not_mark(self):
        for meaning in (
            Meaning("greet"),
            Meaning("thanks"),
            Meaning("help"),
            Meaning("retract"),
            Meaning("ask", query=Query("where", "книга")),
            location(),
        ):
            with self.subTest(meaning=meaning.act):
                self.assertEqual(
                    self.state.observe(
                        "Книга исчезла.", meaning, turn_id=3, known_facts=self.facts
                    ),
                    (),
                )
        uncertain = Interpretation(None, alternatives=(location(),))
        self.assertEqual(
            self.state.observe(
                "Она исчезла.",
                uncertain,
                turn_id=3,
                known_facts=self.facts,
                context=self.context,
            ),
            (),
        )

    def test_only_known_aliases_or_exact_new_names_bind(self):
        context = DialogueContext(
            entities=(Entity("жетон", "thing"), Entity("маша", "person"))
        )
        self.assertEqual(
            self.state.observe("Маша исчезла.", None, turn_id=3, context=context), ()
        )
        self.assertEqual(
            self.state.observe(
                "Жетона больше не видели.", None, turn_id=3, context=context
            ),
            (),
        )
        marked = self.state.observe("Жетон исчез.", None, turn_id=3, context=context)
        self.assertEqual(marked[0]["subject"], "жетон")
        self.assertEqual(marked[0]["basis"], "known_context_thing")
        self.assertEqual(marked[0]["known_event_ids"], [])

    def test_name_tokens_are_exact_and_multiword_positions_are_preserved(self):
        context = DialogueContext(entities=(Entity("синий жетон", "thing"),))
        self.assertEqual(
            self.state.observe("Жетончик исчез.", None, turn_id=3, context=context), ()
        )
        marked = self.state.observe(
            "Синий жетон исчез.", None, turn_id=3, context=context
        )
        self.assertEqual(marked[0]["source_positions"], [0, 1])
        self.assertEqual(marked[0]["subject"], "синий жетон")

    def test_all_explicit_subjects_can_be_guarded_without_global_invalidation(self):
        marked = self.mark(text="Книга и ключ исчезли.")
        self.assertEqual({marker["subject"] for marker in marked}, {"книга", "ключ"})
        self.assertIsNone(
            self.state.query(Query("where", "телефон"), known_facts=self.facts)
        )

    def test_unfamiliar_description_can_conservatively_overtrigger(self):
        # This limitation is intentional and disclosed, not a semantic color rule.
        self.assertEqual(self.mark(text="Книга красивая.")[0]["subject"], "книга")

    def test_accepted_current_effect_clears_only_its_subject(self):
        self.mark(text="Книга и ключ исчезли.")
        meaning, effects = location(place="полка"), [effect(place="полка")]
        self.world.apply(meaning, effects, turn_id=4, source="сообщение 4")
        self.assertEqual(
            self.state.observe(
                "Книга на полке.",
                meaning,
                turn_id=4,
                known_facts=self.world.facts(),
                effects=effects,
                accepted=True,
            ),
            (),
        )
        self.assertEqual([marker["subject"] for marker in self.state.markers], ["ключ"])
        self.assertIsNone(self.state.query(Query("where", "книга")))
        self.assertEqual(
            self.state.records[-1]["cleared"],
            [self.state.records[0]["marked"][1]["marker_id"]],
        )
        self.assertEqual(
            self.state.records[0]["observation"]["text"], "Книга и ключ исчезли."
        )

    def test_accepted_actual_matching_noop_is_new_confirmation(self):
        self.mark()
        self.state.observe(
            "Книга на столе.",
            location(),
            turn_id=4,
            known_facts=self.facts,
            effects=(),
            accepted=True,
        )
        self.assertEqual(self.state.markers, [])
        self.assertEqual(len(self.state.records), 2)
        self.assertEqual(self.state.records[-1]["known_facts"], [self.facts[0]])

    def test_unaccepted_interpretation_and_other_actual_subject_do_not_clear(self):
        self.mark()
        self.state.observe(
            "Книга на столе.", location(), turn_id=4, known_facts=self.facts
        )
        self.state.observe(
            "Ключ на столе.",
            location("ключ"),
            turn_id=4,
            known_facts=self.facts,
            accepted=True,
        )
        self.assertEqual(len(self.state.markers), 1)
        self.assertEqual(len(self.state.records), 1)

    def test_future_scoped_and_negated_actions_do_not_clear(self):
        self.mark()
        before = self.state.to_dict()
        meanings = (
            location(time="future"),
            Meaning(
                "inform", Event("move", object="книга", place="стол", negated=True)
            ),
            Meaning(
                "inform",
                Event(
                    "give", actor="маша", object="книга", recipient="петя", negated=True
                ),
            ),
            Meaning("inform", Event("promise", content=location().event)),
            Meaning("inform", Event("report", content=location().event)),
            Meaning("inform", Event("conditional", content=location().event)),
        )
        for meaning in meanings:
            with self.subTest(meaning=meaning.event):
                self.state.observe(
                    "Вложенное или неактуальное утверждение.",
                    meaning,
                    turn_id=4,
                    known_facts=self.facts,
                    accepted=True,
                )
                self.assertEqual(self.state.to_dict(), before)

    def test_negative_constraint_does_not_refresh_surviving_positive_fact(self):
        self.mark()
        negative = location(place="полка", negated=True)
        effects = [effect(place="полка", negated=True)]
        self.world.apply(negative, effects, turn_id=4, source="сообщение 4")
        self.state.observe(
            "Книга не на полке.",
            negative,
            turn_id=4,
            known_facts=self.world.facts(),
            effects=effects,
            accepted=True,
        )
        self.assertIsNotNone(
            self.state.query(Query("where", "книга"), known_facts=self.world.facts())
        )

    def test_invalid_effect_or_missing_committed_support_cannot_clear(self):
        self.mark()
        before = self.state.to_dict()
        cases = (
            {"effects": [effect()], "accepted": False, "known_facts": self.facts},
            {"effects": [effect("ключ")], "accepted": True, "known_facts": self.facts},
            {"effects": [effect()], "accepted": True, "known_facts": ()},
            {"effects": [], "accepted": True, "known_facts": ()},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.state.observe("Книга на столе.", location(), turn_id=4, **kwargs)
            self.assertEqual(self.state.to_dict(), before)

    def test_stale_verification_inventory_and_previous_explanation_are_blocked(self):
        possession = Meaning("inform", Event("have", actor="маша", object="книга"))
        held = {
            "op": "set",
            "subject": "книга",
            "relation": "holder",
            "value": "маша",
            "spatial": "in",
        }
        self.world.apply(possession, [held], turn_id=3, source="сообщение 3")
        facts = self.world.facts()
        self.state.observe("Книга исчезла.", None, turn_id=4, known_facts=facts)
        queries = (
            Query("where", "книга"),
            Query("who_has", "книга"),
            Query("verify", "книга", value="маша", relation="holder"),
            Query("what_has", "маша"),
            Query("why"),
            Query("why", "книга"),
        )
        for query in queries:
            with self.subTest(query=query):
                self.assertIsNotNone(
                    self.state.query(query, known_facts=facts, assertions=facts)
                )
        self.assertIsNone(
            self.state.query(Query("what_has", "петя"), known_facts=facts)
        )
        self.assertIsNone(self.state.query(Query("why", "ключ"), assertions=facts))
        self.assertIsNone(
            self.state.query(Query("where", "книга", time="past"), known_facts=facts)
        )

    def test_clone_return_values_and_query_evidence_are_detached(self):
        marked = self.mark()
        checkpoint = self.state.to_dict()
        marked[0]["observation"]["text"] = "changed"
        self.state.markers[0]["subject"] = "changed"
        self.state.records[0]["observation"]["text"] = "changed"
        query = self.state.query(Query("where", "книга"))
        assert query is not None
        query["evidence"].clear()
        clone = self.state.clone()
        self.mark(text="Ключ исчез.", turn=4, state=clone)
        self.assertEqual(self.state.to_dict(), checkpoint)
        self.assertEqual(len(clone.markers), 2)

    def test_multiple_updates_keep_originals_and_historical_freshness(self):
        self.mark()
        self.mark(text="Книга исчезла снова.", turn=4)
        self.state.observe(
            "Книга на столе.",
            location(),
            turn_id=5,
            known_facts=self.facts,
            accepted=True,
        )
        self.assertEqual(self.state.at_turn(2).markers, [])
        self.assertEqual(
            self.state.at_turn(3).markers[0]["observation"]["text"], "Книгу унесли."
        )
        self.assertEqual(
            self.state.at_turn(4).markers[0]["observation"]["text"],
            "Книга исчезла снова.",
        )
        self.assertEqual(self.state.at_turn(5).markers, [])
        self.assertEqual(len(self.state.records), 3)
        self.assertEqual(self.state.at_turn(2).records, [])

    def test_marker_record_and_token_budgets_fail_atomically_without_eviction(self):
        state = UncertaintyState(max_markers=1, max_records=2)
        self.mark(state=state)
        before = state.to_dict()
        with self.assertRaisesRegex(BudgetExceeded, "marker_capacity"):
            self.mark(text="Ключ исчез.", turn=4, state=state)
        self.assertEqual(state.to_dict(), before)
        self.mark(text="Книга исчезла снова.", turn=4, state=state)
        before = state.to_dict()
        with self.assertRaisesRegex(BudgetExceeded, "record_capacity"):
            state.observe(
                "Книга на столе.",
                location(),
                turn_id=5,
                known_facts=self.facts,
                accepted=True,
            )
        self.assertEqual(state.to_dict(), before)
        self.assertIsNotNone(state.at_turn(10**9).query(Query("where", "книга")))
        with self.assertRaisesRegex(BudgetExceeded, "token_capacity"):
            self.mark(text="Книга " + "слово " * 97, turn=5, state=state)
        self.assertEqual(state.to_dict(), before)

    def test_multisubject_capacity_does_not_partially_mark(self):
        state = UncertaintyState(max_markers=1)
        with self.assertRaisesRegex(BudgetExceeded, "marker_capacity"):
            self.mark(text="Книга и ключ исчезли.", state=state)
        self.assertEqual(state.markers, [])
        self.assertEqual(state.records, [])

    def test_encoded_state_budget_fails_before_committing_any_marker(self):
        before = self.state.to_dict()
        with (
            patch("text_factors.learning.uncertainty._MAX_BYTES", 100),
            self.assertRaisesRegex(BudgetExceeded, "state_capacity"),
        ):
            self.mark()
        self.assertEqual(self.state.to_dict(), before)

    def test_turn_order_and_strict_types_are_validated(self):
        self.mark()
        before = self.state.to_dict()
        invalid_turns: tuple[Any, ...] = (True, 0, -1, 1.0, 2**53, 2, 3)
        for turn in invalid_turns:
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                self.mark(turn=turn)
        invalid_capacities: tuple[Any, ...] = (False, 0, 1025, 1.0)
        for value in invalid_capacities:
            with self.subTest(value=value), self.assertRaises(ValueError):
                UncertaintyState(max_markers=value)
        self.assertEqual(self.state.to_dict(), before)


class UncertaintyPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.world = world()
        self.facts = self.world.facts()
        self.state = UncertaintyState()
        self.state.observe("Книгу унесли.", None, turn_id=3, known_facts=self.facts)

    def observations(self, state=None):
        return {
            record["observation"]["observation_id"]: Observation.from_dict(
                record["observation"]
            )
            for record in (state or self.state).records
        }

    def test_json_roundtrip_replays_active_and_resolved_markers(self):
        for cleared in (False, True):
            if cleared:
                self.state.observe(
                    "Книга на столе.",
                    location(),
                    turn_id=4,
                    known_facts=self.facts,
                    accepted=True,
                )
            snapshot = json.loads(json.dumps(self.state.to_dict(), ensure_ascii=False))
            restored = UncertaintyState.from_dict(
                snapshot, observations=self.observations(), events=self.world.events
            )
            self.assertEqual(restored.to_dict(), snapshot)
            snapshot["records"][0]["observation"]["text"] = "changed"
            self.assertNotEqual(restored.to_dict(), snapshot)

    def test_forged_marker_refs_subjects_positions_and_output_are_rejected(self):
        for key, value in (
            ("subject", "ключ"),
            ("source_positions", [1]),
            ("source_positions", [False]),
            ("known_event_ids", [2]),
            ("known_event_ids", [True]),
            ("reason", "safe"),
            ("marker_id", "u-forged"),
        ):
            with self.subTest(key=key, value=value):
                forged = self.state.to_dict()
                forged["records"][0]["marked"][0][key] = value
                forged["markers"][0][key] = value
                with self.assertRaises(ValueError):
                    UncertaintyState.from_dict(forged)
        forged = self.state.to_dict()
        forged["markers"] = []
        with self.assertRaises(ValueError):
            UncertaintyState.from_dict(forged)

    def test_forged_resolution_without_marker_or_support_is_rejected(self):
        self.state.observe(
            "Книга на столе.",
            location(),
            turn_id=4,
            known_facts=self.facts,
            accepted=True,
        )
        for mutate in (
            lambda state: state["records"][1]["cleared"].append("u-forged"),
            lambda state: state["records"][1].update(accepted=False),
            lambda state: state["records"][1].update(known_facts=[]),
            lambda state: state["records"].pop(0),
        ):
            forged = self.state.to_dict()
            mutate(forged)
            with self.assertRaises(ValueError):
                UncertaintyState.from_dict(forged)

    def test_forged_original_source_text_and_turn_are_rejected(self):
        for key, value in (
            ("source", "invented source"),
            ("observation_id", "turn:55"),
            ("turn_id", 0),
            ("text", "Ключ исчез."),
        ):
            with self.subTest(key=key):
                forged = self.state.to_dict()
                forged["records"][0]["observation"][key] = value
                with self.assertRaises(ValueError):
                    UncertaintyState.from_dict(forged)

    def test_record_and_state_extra_fields_or_budget_forgery_is_rejected(self):
        for mutate in (
            lambda state: state.update(extra=1),
            lambda state: state["records"][0].update(extra=1),
            lambda state: state.update(max_records=True),
            lambda state: state.update(max_markers=0),
            lambda state: state["records"].append(deepcopy(state["records"][0])),
            lambda state: state.update(schema="other"),
        ):
            forged = self.state.to_dict()
            mutate(forged)
            with self.assertRaises(ValueError):
                UncertaintyState.from_dict(forged)

    def test_independent_ledgers_reject_self_consistent_forged_event_reference(self):
        forged_facts = deepcopy(list(self.facts))
        forged_facts[0]["event_id"] = 900
        forged = UncertaintyState()
        forged.observe("Книгу унесли.", None, turn_id=3, known_facts=forged_facts)
        # Internal replay checks consistency; an independent ledger binds origins.
        UncertaintyState.from_dict(forged.to_dict())
        with self.assertRaisesRegex(ValueError, "forged event reference"):
            UncertaintyState.from_dict(
                forged.to_dict(),
                observations=self.observations(),
                events=self.world.events,
            )

    def test_independent_ledgers_reject_forged_fact_content_and_future_anchor(self):
        for change in (
            {"value": "полка"},
            {"source": "invented source"},
            {"event_id": 2},
        ):
            with self.subTest(change=change):
                forged_facts = deepcopy(list(self.facts))
                forged_facts[0].update(change)
                state = UncertaintyState()
                state.observe(
                    "Книгу унесли.", None, turn_id=3, known_facts=forged_facts
                )
                with self.assertRaises(ValueError):
                    state.validate_references(
                        observations=self.observations(), events=self.world.events
                    )
        events = self.world.events
        events[0]["turn_id"] = 3
        with self.assertRaises(ValueError):
            self.state.validate_references(
                observations=self.observations(), events=events
            )

    def test_independent_original_observation_and_context_bind_marker(self):
        with self.assertRaisesRegex(ValueError, "not original evidence"):
            self.state.validate_references(observations={}, events=self.world.events)
        state = UncertaintyState()
        context = DialogueContext(entities=(Entity("жетон", "thing"),))
        state.observe("Жетон исчез.", None, turn_id=3, context=context)
        state.validate_references(
            observations=self.observations(state), events=[], contexts={3: context}
        )
        with self.assertRaisesRegex(ValueError, "original context"):
            state.validate_references(
                observations=self.observations(state),
                events=[],
                contexts={3: DialogueContext()},
            )
        with self.assertRaises(ValueError):
            UncertaintyState.from_dict(
                state.to_dict(), observations=self.observations(state)
            )

    def test_rehashing_does_not_license_an_unobserved_subject(self):
        forged = self.state.to_dict()
        record = forged["records"][0]
        record["marked"][0]["subject"] = "ключ"
        marker = record["marked"][0]
        marker["marker_id"] = "u-" + digest(
            {key: value for key, value in marker.items() if key != "marker_id"}
        )
        record["record_id"] = "ur-" + digest(
            {key: value for key, value in record.items() if key != "record_id"}
        )
        forged["markers"] = deepcopy(record["marked"])
        with self.assertRaises(ValueError):
            UncertaintyState.from_dict(forged)


if __name__ == "__main__":
    unittest.main()

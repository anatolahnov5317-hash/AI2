"""Deterministic safety regressions, separate from sealed capability evaluation.

These tests exercise input validation and resource/integrity boundaries, not
held-out wording or capability scores. They do not import evaluation fixtures.
"""

from __future__ import annotations

import sys
import types
import unittest
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

from text_factors.conversation.persistence import decode_json, encode_json
from text_factors.conversation.schema import Budget, BudgetExceeded, ConversationLimits
from text_factors.learning import dynamics
from text_factors.learning.dialogue_learning import (
    GeneratedReply,
    LearnedDialoguePolicy,
    LearnedTokenGenerator,
    PolicyDecision,
)
from text_factors.learning.dynamics import (
    LearnedDynamics,
    TransitionEpisode,
    TransitionPrediction,
    checked_facts,
)
from text_factors.learning.model import ModelBundle
from text_factors.learning.schema import (
    DialogueContext,
    Entity,
    Event,
    Interpretation,
    Meaning,
    Query,
    bounded_text,
)
from text_factors.learning.session import LearnedSession
from text_factors.learning.understanding import LearnedUnderstanding
from text_factors.learning.world import ExperienceWorld, validate_effects


def atomic_event() -> Event:
    return Event("give", actor="Анна", object="ключ", recipient="Мария")


def nested_event_dict(levels: int) -> dict[str, Any]:
    result = atomic_event().to_dict()
    for _ in range(levels - 1):
        result = {
            **atomic_event().to_dict(),
            "predicate": "report",
            "modality": "reported",
            "content": result,
        }
    return result


class LearnedSchemaSafetyTests(unittest.TestCase):
    def test_four_level_meaning_round_trip_and_fifth_level_rejection(self) -> None:
        event = Event.from_dict(nested_event_dict(4))
        self.assertEqual(event.depth(), 4)
        meaning = Meaning(
            "inform", event=event, entities=(Entity("Анна", "person", "female"),)
        )
        restored = Meaning.from_dict(decode_json(encode_json(meaning.to_dict())))
        self.assertEqual(restored, meaning)
        with self.assertRaisesRegex(ValueError, "depth"):
            Event.from_dict(nested_event_dict(5))
        with self.assertRaisesRegex(ValueError, "depth"):
            Event("report", content=event)

    def test_cyclic_or_extremely_deep_input_is_bounded_before_recursion(self) -> None:
        cyclic = nested_event_dict(2)
        cyclic["content"] = cyclic
        with self.assertRaisesRegex(ValueError, "depth"):
            Event.from_dict(cyclic)
        with self.assertRaisesRegex(ValueError, "depth"):
            Event.from_dict(nested_event_dict(2000))
        for depth in (-1, True, 4, 10**10):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                Event.from_dict(atomic_event().to_dict(), _depth=depth)

    def test_scoped_events_and_nonactual_modalities_are_not_actual_events(self) -> None:
        for predicate in ("promise", "report", "conditional"):
            event = Event(predicate, content=atomic_event())
            self.assertFalse(event.actual)
            self.assertTrue(event.content and event.content.actual)
        for modality in ("possible", "intended", "reported", "conditional"):
            self.assertFalse(Event("give", modality=modality).actual)
        self.assertFalse(Event("give", time="future").actual)
        self.assertTrue(atomic_event().actual)

    def test_invalid_scope_shapes_are_rejected(self) -> None:
        for predicate in ("promise", "report", "conditional"):
            with self.subTest(predicate=predicate), self.assertRaises(ValueError):
                Event(predicate)
        with self.assertRaises(ValueError):
            Event("give", content=atomic_event())
        with self.assertRaises(ValueError):
            Event("report", content=atomic_event(), condition=atomic_event())
        with self.assertRaises(ValueError):
            Event("conditional", content=atomic_event(), condition=cast(Any, {}))

    def test_categories_fail_cleanly_for_nonstring_values(self) -> None:
        constructors = (
            lambda bad: Entity("Анна", kind=bad),
            lambda bad: Entity("Анна", gender=bad),
            lambda bad: Event(bad),
            lambda bad: Event("give", modality=bad),
            lambda bad: Event("give", time=bad),
            lambda bad: Event("give", spatial=bad),
            lambda bad: Query(bad),
            lambda bad: Query("where", time=bad),
            lambda bad: Query("where", relation=bad),
            lambda bad: Query("where", spatial=bad),
            lambda bad: Meaning(bad),
        )
        for constructor in constructors:
            for value in (None, True, 1, [], {}, ["actual"]):
                with (
                    self.subTest(constructor=constructor, value=value),
                    self.assertRaises(ValueError),
                ):
                    constructor(value)

    def test_control_characters_invalid_unicode_and_long_names_are_rejected(
        self,
    ) -> None:
        for value in ("a\x00b", "a\nb", "a\tb", "a\x7fb", "\ud800", "x" * 129):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                Entity(value)
        for value in (None, True, 0, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bounded_text(value, "test")
        self.assertEqual(Entity("ключ").name, "ключ")

    def test_query_retains_in_and_on_distinction(self) -> None:
        inside = Query("verify", "ключ", "стол", "location", spatial="in")
        surface = Query("verify", "ключ", "стол", "location", spatial="on")
        self.assertNotEqual(inside, surface)
        self.assertEqual(Query.from_dict(surface.to_dict()), surface)
        self.assertEqual(surface.to_dict()["spatial"], "on")
        with self.assertRaises(ValueError):
            Query("verify", spatial="under")

    def test_decoders_reject_missing_unknown_and_wrong_root_fields(self) -> None:
        samples = (
            (Entity.from_dict, Entity("ключ").to_dict()),
            (Event.from_dict, atomic_event().to_dict()),
            (Query.from_dict, Query("where", subject="ключ").to_dict()),
            (Meaning.from_dict, Meaning("greet").to_dict()),
            (DialogueContext.from_dict, DialogueContext().to_dict()),
        )
        for decode, value in samples:
            invalid = (
                None,
                [],
                {**value, "unexpected": 1},
                dict(list(value.items())[1:]),
            )
            for sample in invalid:
                with (
                    self.subTest(decode=decode, sample=sample),
                    self.assertRaises(ValueError),
                ):
                    decode(sample)

    def test_acts_cannot_smuggle_unexpected_semantic_content(self) -> None:
        with self.assertRaises(ValueError):
            Meaning("ask", event=atomic_event(), query=Query("where"))
        with self.assertRaises(ValueError):
            Meaning("inform", event=atomic_event(), query=Query("where"))
        for act in ("greet", "thanks", "help", "unknown", "retract"):
            with self.subTest(act=act), self.assertRaises(ValueError):
                Meaning(act, event=atomic_event())

    def test_context_capacities_are_checked_before_entity_construction(self) -> None:
        original = DialogueContext().to_dict()
        cases = (
            {**original, "turns": ["text"] * 17},
            {**original, "entities": [Entity("ключ").to_dict()] * 65},
            {**original, "focus": ["ключ"] * 17},
        )
        for value in cases:
            with (
                self.subTest(value=value),
                patch.object(
                    Entity,
                    "from_dict",
                    side_effect=AssertionError("allocated before capacity check"),
                ),
                self.assertRaises(ValueError),
            ):
                DialogueContext.from_dict(value)

    def test_context_valid_boundary_and_independent_serialized_lists(self) -> None:
        context = DialogueContext(
            turns=("текст",) * 16,
            entities=tuple(Entity(f"предмет-{index}") for index in range(64)),
            focus=("ключ",) * 16,
        )
        saved = context.to_dict()
        restored = DialogueContext.from_dict(saved)
        self.assertEqual(restored, context)
        saved["turns"].append("later")
        self.assertEqual(len(restored.turns), 16)
        with self.assertRaises(ValueError):
            DialogueContext(turns=("x" * 2049,))

    def test_meaning_entities_capacity_checked_before_construction(self) -> None:
        value = Meaning("greet").to_dict()
        value["entities"] = [Entity("ключ").to_dict()] * 33
        with (
            patch.object(Entity, "from_dict", side_effect=AssertionError("allocated")),
            self.assertRaises(ValueError),
        ):
            Meaning.from_dict(value)

    def test_interpretation_scores_alternatives_and_diagnostics_are_bounded(
        self,
    ) -> None:
        for score in (float("nan"), float("inf"), -float("inf"), True, 10**10000):
            with self.subTest(score_type=type(score)), self.assertRaises(ValueError):
                Interpretation(None, score=score)
        with self.assertRaises(ValueError):
            Interpretation(None, alternatives=(Meaning("greet"),) * 9)
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle
        invalid_diagnostics = (
            {"score": float("nan")},
            {"score": float("inf")},
            {"score": object()},
            {"trace": "x" * 65537},
            {"values": (1, 2)},
            cycle,
            [],
        )
        for diagnostics in invalid_diagnostics:
            with self.subTest(kind=type(diagnostics)), self.assertRaises(ValueError):
                Interpretation(None, diagnostics=cast(Any, diagnostics))

    def test_interpretation_diagnostics_do_not_alias_caller_metadata(self) -> None:
        diagnostics = {"scores": [0.1, 0.2], "details": {"mode": "test"}}
        original = deepcopy(diagnostics)
        interpretation = Interpretation(Meaning("greet"), diagnostics=diagnostics)
        diagnostics["scores"].append(float("nan"))
        self.assertEqual(interpretation.diagnostics, original)


class LearnedWorldSafetyTests(unittest.TestCase):
    @staticmethod
    def location(
        place: str = "стол", *, spatial: str = "in", negated: bool = False
    ) -> tuple[Meaning, list[dict[str, Any]]]:
        return (
            Meaning(
                "inform",
                event=Event(
                    "locate",
                    object="ключ",
                    place=place,
                    spatial=spatial,
                    negated=negated,
                ),
            ),
            [
                {
                    "op": "exclude" if negated else "set",
                    "subject": "ключ",
                    "relation": "location",
                    "value": place,
                    "spatial": spatial,
                }
            ],
        )

    def test_nonactual_scopes_cannot_leak_positive_predicted_effects(self) -> None:
        meaning, effects = self.location()
        for predicate in ("promise", "report", "conditional"):
            world = ExperienceWorld()
            scoped = Meaning("inform", event=Event(predicate, content=meaning.event))
            before = world.to_dict()
            with self.subTest(predicate=predicate):
                with self.assertRaises(ValueError):
                    world.apply(scoped, effects, turn_id=1)
                self.assertEqual(world.to_dict(), before)
                outcome = world.apply(scoped, [], turn_id=1)
                self.assertEqual(outcome["action"], "nonactual")
                self.assertEqual(world.facts(), ())

    def test_negated_action_does_not_erase_prior_known_location(self) -> None:
        world = ExperienceWorld()
        meaning, effects = self.location("ящик")
        world.apply(meaning, effects, turn_id=1)
        before_facts = world.facts()
        negated_move = Meaning(
            "inform",
            event=Event(
                "move", actor="Анна", object="ключ", place="сумка", negated=True
            ),
        )
        world.apply(negated_move, [], turn_id=2)
        self.assertEqual(world.facts(), before_facts)

    def test_unlicensed_roles_and_malformed_effect_categories_never_mutate_world(
        self,
    ) -> None:
        meaning, effects = self.location()
        for malformed in (
            [{**effects[0], "subject": "паспорт"}],
            [{**effects[0], "value": "другой предмет"}],
            [{**effects[0], "op": []}],
            [{**effects[0], "relation": {}}],
            [{**effects[0], "spatial": []}],
        ):
            world = ExperienceWorld()
            before = world.to_dict()
            with self.subTest(effect=malformed):
                with self.assertRaises(ValueError):
                    world.apply(meaning, malformed, turn_id=1)
                self.assertEqual(world.to_dict(), before)
        self.assertEqual(
            validate_effects(cast(Event, meaning.event), effects), tuple(effects)
        )

    def test_world_time_and_capacity_failures_are_atomic(self) -> None:
        meaning, effects = self.location()
        world = ExperienceWorld(max_events=1)
        budget = Budget(1)
        budget.deadline = 0
        before = world.to_dict()
        with self.assertRaises(BudgetExceeded):
            world.apply(meaning, effects, turn_id=1, budget=budget)
        self.assertEqual(world.to_dict(), before)
        world.apply(meaning, effects, turn_id=1)
        before = world.to_dict()
        with self.assertRaises(BudgetExceeded):
            world.apply(*self.location("сумка"), turn_id=2)
        self.assertEqual(world.to_dict(), before)

    def test_spatial_exclusion_preserves_distinct_positive_relation(self) -> None:
        world = ExperienceWorld()
        world.apply(*self.location(spatial="in"), turn_id=1)
        world.apply(*self.location(spatial="on", negated=True), turn_id=2)
        facts = world.facts()
        self.assertTrue(
            any(not fact["negated"] and fact["spatial"] == "in" for fact in facts)
        )
        self.assertTrue(
            any(fact["negated"] and fact["spatial"] == "on" for fact in facts)
        )
        self.assertEqual(
            world.query(Query("verify", "ключ", "стол", "location", spatial="in"))[
                "truth"
            ],
            "yes",
        )
        self.assertEqual(
            world.query(Query("verify", "ключ", "стол", "location", spatial="on"))[
                "truth"
            ],
            "no",
        )
        world.apply(*self.location(spatial="in"), turn_id=3)
        self.assertTrue(
            any(fact["negated"] and fact["spatial"] == "on" for fact in world.facts())
        )

    def test_checkpoint_retraction_target_must_be_typed_integer(self) -> None:
        world = ExperienceWorld()
        world.apply(*self.location(), turn_id=1)
        world.apply(Meaning("retract"), [], turn_id=2)
        checkpoint = world.to_dict()
        self.assertEqual(ExperienceWorld.from_dict(checkpoint).to_dict(), checkpoint)
        for target in (True, 1.0, "1", -1, 3):
            corrupted = deepcopy(checkpoint)
            corrupted["events"][-1]["target"] = target
            with self.subTest(target=target), self.assertRaises(ValueError):
                ExperienceWorld.from_dict(corrupted)

    def test_returned_facts_events_and_snapshot_do_not_alias_internal_state(
        self,
    ) -> None:
        world = ExperienceWorld()
        world.apply(*self.location(), turn_id=1)
        original = world.to_dict()
        events = world.events
        events[0]["effects"][0]["value"] = "подмена"
        facts = world.facts()
        facts[0]["value"] = "подмена"
        saved = world.to_dict()
        saved["events"].clear()
        self.assertEqual(world.to_dict(), original)


class _JSONComponent:
    """Only an aggregate-checkpoint fixture; not a substitute learned model."""

    def __init__(self, value: dict[str, Any] | None = None) -> None:
        self.value = value if value is not None else {"weights": [[0.1, 0.2]]}

    def to_dict(self) -> dict[str, Any]:
        return self.value


def metadata_bundle() -> ModelBundle:
    return ModelBundle(
        _JSONComponent(),
        _JSONComponent(),
        _JSONComponent(),
        _JSONComponent(),
        {
            "version": "0.5.0a1",
            "seed": 42,
            "dataset_kind": "explicit_annotated_dataset",
            "training_counts": {"utterances": 1, "transitions": 1, "dialogues": 1},
            "pretrained_model": False,
            "implicit_online_learning": False,
            "scope": "safety-test fixture only",
        },
    )


class LearnedBundleSafetyTests(unittest.TestCase):
    def test_bundle_refuses_nonfinite_or_oversized_payload_before_serialization(
        self,
    ) -> None:
        for invalid in (float("nan"), float("inf"), -float("inf")):
            bundle = metadata_bundle()
            bundle.understanding.value["weights"][0][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bundle.to_dict()
        bundle = metadata_bundle()
        bundle.understanding.value["padding"] = "x" * 12_000_001
        with self.assertRaises(ValueError):
            bundle.to_dict()

    def test_bundle_snapshot_is_detached_and_fingerprint_tracks_parameters(
        self,
    ) -> None:
        bundle = metadata_bundle()
        fingerprint = bundle.fingerprint
        saved = bundle.to_dict()
        saved["understanding"]["weights"][0][0] = 99
        saved["metadata"]["seed"] = 12
        self.assertEqual(bundle.fingerprint, fingerprint)
        bundle.understanding.value["weights"][0][0] = 0.4
        self.assertNotEqual(bundle.fingerprint, fingerprint)

    def test_invalid_bundle_metadata_rejected_before_any_numeric_allocation(
        self,
    ) -> None:
        class RefuseNumericalLoad:
            @classmethod
            def from_dict(cls, value: Any) -> Any:
                raise AssertionError("numeric component loaded before metadata checks")

        modules = {}
        for name, classes in (
            ("understanding", ("LearnedUnderstanding",)),
            ("dynamics", ("LearnedDynamics",)),
            ("dialogue_learning", ("LearnedDialoguePolicy", "LearnedTokenGenerator")),
        ):
            module = types.ModuleType(f"text_factors.learning.{name}")
            for class_name in classes:
                setattr(module, class_name, RefuseNumericalLoad)
            modules[module.__name__] = module
        checkpoint = metadata_bundle().to_dict()
        bad_metadata = (
            {**checkpoint["metadata"], "seed": True},
            {**checkpoint["metadata"], "pretrained_model": True},
            {**checkpoint["metadata"], "implicit_online_learning": True},
            {**checkpoint["metadata"], "training_counts": {"utterances": -1}},
            {**checkpoint["metadata"], "dataset_kind": "unknown"},
            {**checkpoint["metadata"], "dataset_kind": []},
            {**checkpoint["metadata"], "scope": "x" * 257},
        )
        invalid = [
            {**checkpoint, "schema": "unsupported"},
            {**checkpoint, "extra": True},
            {**checkpoint, "metadata": {"invalid": "schema"}},
            {**checkpoint, "understanding": {"weights": [[float("nan")]]}},
        ]
        invalid.extend({**checkpoint, "metadata": bad} for bad in bad_metadata)
        with patch.dict(sys.modules, modules):
            for value in invalid:
                with (
                    self.subTest(metadata=value.get("metadata")),
                    self.assertRaises(ValueError),
                ):
                    ModelBundle.from_dict(value)


class LearnedDynamicsSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = Event("move", actor="a", object="x", place="p")
        cls.episode = TransitionEpisode(
            [],
            cls.event,
            [
                {
                    "subject": "x",
                    "relation": "location",
                    "value": "p",
                    "negated": False,
                    "spatial": "in",
                }
            ],
        )
        cls.model = LearnedDynamics()
        cls.model.fit([cls.episode], seconds=5)
        cls.checkpoint = cls.model.to_dict()

    def test_real_numeric_checkpoint_round_trip_does_not_retrain(self) -> None:
        with patch.object(
            LearnedDynamics, "fit", side_effect=AssertionError("checkpoint retrained")
        ):
            restored = LearnedDynamics.from_dict(self.checkpoint)
        self.assertEqual(restored.to_dict(), self.checkpoint)
        self.assertEqual(
            restored.predict([], self.event).effects,
            self.model.predict([], self.event).effects,
        )

    def test_invalid_dimensions_indices_codes_and_counters_are_rejected(self) -> None:
        bad_values = (
            (("transform", "config", "input_bits"), 10**10),
            (("transform", "receptors"), []),
            (("transform", "receptors", 0), []),
            (("transform", "receptors", 0, 0), -1),
            (("transform", "output_map"), []),
            (("transform", "output_map", 0), True),
            (("transform", "clusters", 0, "bit_hits"), []),
            (("transform", "clusters", 0, "point"), True),
            (("templates", 0, "op"), []),
            (("templates", 0, "relation"), {}),
            (("codes", 0), [0, 0, 1, 2]),
            (("codes", 0, 0), True),
            (("training", "epochs"), 17),
            (("training", "episodes"), True),
        )
        for path, value in bad_values:
            checkpoint = deepcopy(self.checkpoint)
            target: Any = checkpoint
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                LearnedDynamics.from_dict(checkpoint)

    def test_nonfinite_checkpoint_fails_before_numpy_array_allocation(self) -> None:
        checkpoint = deepcopy(self.checkpoint)
        checkpoint["transform"]["receptors"][0][0] = float("nan")
        with (
            patch.object(
                dynamics.np, "asarray", side_effect=AssertionError("allocated")
            ),
            self.assertRaises(ValueError),
        ):
            LearnedDynamics.from_dict(checkpoint)

    def test_memory_dimension_check_precedes_array_allocation(self) -> None:
        memory = deepcopy(self.checkpoint["transform"])
        memory["receptors"].pop()
        config = dynamics._config(self.model.seed, "transform")
        with (
            patch.object(
                dynamics.np, "asarray", side_effect=AssertionError("allocated")
            ),
            self.assertRaises(ValueError),
        ):
            dynamics._memory_load(memory, config)

    def test_timeout_or_invalid_retraining_preserves_previous_parameters(self) -> None:
        model = LearnedDynamics.from_dict(self.checkpoint)
        before = model.to_dict()
        with self.assertRaises(TimeoutError):
            model.fit([self.episode], seconds=1e-12)
        self.assertEqual(model.to_dict(), before)
        with self.assertRaises(ValueError):
            model.fit([self.episode] * 513, seconds=1)
        self.assertEqual(model.to_dict(), before)

    def test_bad_fact_types_and_fact_capacity_fail_cleanly(self) -> None:
        fact = {
            "subject": "x",
            "relation": "location",
            "value": "p",
            "negated": False,
            "spatial": "in",
        }
        for key, bad in (("relation", []), ("spatial", {}), ("negated", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_facts([{**fact, key: bad}])
        with self.assertRaises(ValueError):
            checked_facts([fact] * 129)


class LearnedSessionSafetyTests(unittest.TestCase):
    """Integration and fault injection; these are not capability benchmarks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = ModelBundle.fit(seconds=60)

    def setUp(self) -> None:
        self.session = LearnedSession(self.bundle)

    def turn(
        self, text: str, action: str, *, session: LearnedSession | None = None
    ) -> dict[str, Any]:
        response = (session or self.session).respond(text)
        self.assertEqual(response["action"], action, response)
        return response

    def scripted_turn(self, meaning: Meaning, action: str) -> dict[str, Any]:
        with patch.object(
            self.bundle.understanding,
            "interpret",
            return_value=Interpretation(meaning, score=1.0),
        ):
            return self.turn("аннотированная проверка контракта", action)

    def restored(self, session: LearnedSession | None = None) -> LearnedSession:
        original = (session or self.session).to_dict()
        restored = LearnedSession.from_dict(original, self.bundle)
        self.assertEqual(restored.to_dict(), original)
        return restored

    def test_real_model_and_session_restore_parameters_without_retraining(self) -> None:
        for text, action in (
            ("ключ в ящике", "ack"),
            ("где ключ?", "answer"),
            ("почему ключ?", "explain"),
            ("нет, ключ на столе", "corrected"),
            ("ключ на столе?", "answer"),
            ("отмени последнее сообщение", "retracted"),
        ):
            self.turn(text, action)
            self.restored()
        model = self.bundle.to_dict()
        snapshot = self.session.to_dict()
        with (
            patch.object(LearnedUnderstanding, "fit", side_effect=AssertionError),
            patch.object(LearnedDynamics, "fit", side_effect=AssertionError),
            patch.object(LearnedDialoguePolicy, "fit", side_effect=AssertionError),
            patch.object(LearnedTokenGenerator, "fit", side_effect=AssertionError),
        ):
            loaded_bundle = ModelBundle.from_dict(model)
            loaded = LearnedSession.from_dict(snapshot, loaded_bundle)
        self.assertEqual(loaded.to_dict(), snapshot)
        self.assertEqual(loaded_bundle.to_dict(), model)
        self.assertEqual(snapshot["model_fingerprint"], self.bundle.fingerprint)
        self.assertNotIn("understanding", snapshot)
        self.assertNotIn("generator", snapshot)

    def test_learned_noop_restores_against_pre_turn_evidence(self) -> None:
        self.turn("ключ в ящике", "ack")
        facts = self.session.world.facts()
        response = self.turn("ключ в ящике", "ack")
        self.assertEqual(response["diagnostics"]["dynamics"]["effects"], [])
        self.assertEqual(self.session.world.facts(), facts)
        self.restored()
        self.turn("где ключ?", "answer")
        self.restored()

    def test_dynamics_receives_semantic_facts_without_ledger_metadata(self) -> None:
        self.turn("ключ в ящике", "ack")
        original = self.bundle.dynamics.predict
        with patch.object(self.bundle.dynamics, "predict", wraps=original) as predict:
            self.turn("нет, ключ на столе", "corrected")
        before, _ = predict.call_args.args
        self.assertEqual(len(before), 1)
        self.assertEqual(
            set(before[0]), {"subject", "relation", "value", "negated", "spatial"}
        )

    def test_request_id_retry_is_detached_and_does_not_recompute(self) -> None:
        response = self.session.respond("ключ в ящике", "statement-1")
        self.assertEqual(response["action"], "ack")
        expected = deepcopy(response)
        snapshot = self.session.to_dict()
        response["assertions"][0]["value"] = "подмена"
        with patch.object(
            self.bundle.understanding, "interpret", side_effect=AssertionError
        ):
            self.assertEqual(
                self.session.respond("ключ в ящике", "statement-1"), expected
            )
        self.assertEqual(self.session.to_dict(), snapshot)
        loaded = self.restored()
        self.assertEqual(loaded.respond("ключ в ящике", "statement-1"), expected)
        with self.assertRaises(ValueError):
            self.session.respond("другое сообщение", "statement-1")
        self.assertEqual(self.session.to_dict(), snapshot)

    def test_invalid_input_and_capacity_rejections_do_not_mutate_state(self) -> None:
        snapshot = self.session.to_dict()
        for text in (None, 4, [], "text\nnext", "bad\ud800"):
            with self.subTest(text_type=type(text)), self.assertRaises(ValueError):
                self.session.respond(cast(Any, text))
            self.assertEqual(self.session.to_dict(), snapshot)
        for request_id in (True, [], "two words", "x" * 129, "bad\x7f"):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                self.session.respond("привет", cast(Any, request_id))
            self.assertEqual(self.session.to_dict(), snapshot)
        for text in ("x" * 2049, "word " * 97):
            response = self.session.respond(text)
            self.assertEqual(response["action"], "limit")
            self.assertEqual(self.session.to_dict(), snapshot)

    def test_component_faults_and_timeouts_roll_back_candidate_changes(self) -> None:
        self.turn("ключ в ящике", "ack")
        for component, method in (
            (self.bundle.understanding, "interpret"),
            (self.bundle.dynamics, "predict"),
            (self.bundle.policy, "choose"),
            (self.bundle.generator, "generate"),
        ):
            for error, action in (
                (ValueError("injected"), "error"),
                (TimeoutError(), "limit"),
            ):
                snapshot = self.session.to_dict()
                with (
                    self.subTest(component=type(component).__name__, error=type(error)),
                    patch.object(component, method, side_effect=error),
                ):
                    response = self.session.respond("нет, ключ на столе")
                self.assertEqual(response["action"], action, response)
                self.assertEqual(self.session.to_dict(), snapshot)

    def test_reported_internal_deadlines_are_limits_with_no_commit(self) -> None:
        self.turn("ключ в ящике", "ack")
        snapshot = self.session.to_dict()
        with patch.object(
            self.bundle.dynamics,
            "predict",
            return_value=TransitionPrediction(reason="time_budget"),
        ):
            self.turn("нет, ключ на столе", "limit")
        self.assertEqual(self.session.to_dict(), snapshot)
        with patch.object(
            self.bundle.generator,
            "generate",
            return_value=GeneratedReply("", (), False, "generation_deadline"),
        ):
            self.turn("нет, ключ на столе", "limit")
        self.assertEqual(self.session.to_dict(), snapshot)

    def test_generation_and_policy_contract_failures_are_atomic(self) -> None:
        self.turn("ключ в ящике", "ack")
        snapshot = self.session.to_dict()
        for reply in (
            GeneratedReply("Ключ в ящике.", ("неизвестный-токен",), True),
            GeneratedReply("Ключ в ящике.", ("<object>", "в", "<place>", "."), False),
        ):
            with patch.object(self.bundle.generator, "generate", return_value=reply):
                self.turn("нет, ключ на столе", "error")
            self.assertEqual(self.session.to_dict(), snapshot)
        for decision in (
            PolicyDecision("answer", {"answer": 1.0}),
            PolicyDecision("corrected", {"corrected": 0.0, "clarify": 1.0}),
            PolicyDecision("corrected", {"corrected": float("nan"), "clarify": 0.0}),
        ):
            with patch.object(self.bundle.policy, "choose", return_value=decision):
                self.turn("нет, ключ на столе", "error")
            self.assertEqual(self.session.to_dict(), snapshot)

    def test_clarification_discards_an_otherwise_valid_candidate_mutation(self) -> None:
        self.turn("ключ в ящике", "ack")
        world = self.session.world.to_dict()
        with patch.object(
            self.bundle.policy,
            "choose",
            return_value=PolicyDecision("clarify", {"corrected": 0.0, "clarify": 1.0}),
        ):
            response = self.turn("нет, ключ на столе", "clarify")
        self.assertEqual(response["assertions"], [])
        self.assertEqual(self.session.world.to_dict(), world)
        self.assertEqual(self.session.context.pending, "clarify")
        self.restored()

    def test_unsupported_text_and_multiple_clauses_never_add_partial_facts(
        self,
    ) -> None:
        for text in (
            "неизвестное-слово",
            "ключ в ящике. паспорт в сумке",
            "ключ в ящике; паспорт в сумке",
            "ключ в ящике и паспорт в сумке",
        ):
            with self.subTest(text=text):
                response = self.session.respond(text)
                self.assertIsNone(response["meaning"], response)
                self.assertIn(response["action"], {"unknown", "clarify"})
                self.assertEqual(response["assertions"], [])
                self.assertEqual(self.session.world.facts(), ())
                self.restored()
        with patch.object(
            self.bundle.understanding,
            "interpret",
            return_value=Interpretation(None, reason="ambiguous_or_missing_reference"),
        ):
            self.turn("где он?", "clarify")
        self.assertEqual(self.session.world.facts(), ())
        self.restored()

    def test_world_state_and_turn_budgets_preserve_state(self) -> None:
        session = LearnedSession(self.bundle, ConversationLimits(max_events=1))
        self.turn("ключ в ящике", "ack", session=session)
        snapshot = session.to_dict()
        self.turn("нет, ключ на столе", "limit", session=session)
        self.assertEqual(session.to_dict(), snapshot)
        session = LearnedSession(self.bundle, ConversationLimits(turn_seconds=1e-12))
        snapshot = session.to_dict()
        self.turn("ключ в ящике", "limit", session=session)
        self.assertEqual(session.to_dict(), snapshot)
        session = LearnedSession(self.bundle, ConversationLimits(max_state_bytes=1000))
        snapshot = session.to_dict()
        response = self.turn("ключ в ящике", "limit", session=session)
        self.assertEqual(response["reason"], "state_capacity")
        self.assertEqual(session.to_dict(), snapshot)

    def test_restore_rejects_altered_receipts_context_evidence_and_tokens(self) -> None:
        self.turn("ключ в ящике", "ack")
        self.turn("где ключ?", "answer")
        snapshot = self.session.to_dict()
        corruptions = (
            (("model_fingerprint",), "0" * 64),
            (("turn_count",), True),
            (("context", "focus"), ["ящик", "ключ"]),
            (("context", "entities", 0, "gender"), "female"),
            (("history", 1, "input"), "подмена"),
            (("history", 1, "request_id"), "contains whitespace"),
            (("history", 1, "response", "complete"), False),
            (("history", 1, "response", "reason"), "подмена"),
            (("history", 1, "response", "elapsed_seconds"), float("nan")),
            (("history", 1, "response", "assertions", 0, "event_id"), 2),
            (("history", 1, "response", "evidence", 0, "source"), "подмена"),
            (("last_assertions", 0, "value"), "подмена"),
            (("world", "events", 0, "source"), "подмена"),
            (
                (
                    "history",
                    1,
                    "response",
                    "diagnostics",
                    "generation",
                    "segments",
                    0,
                    "tokens",
                ),
                ["<object>", "."],
            ),
            (
                (
                    "history",
                    1,
                    "response",
                    "diagnostics",
                    "generation",
                    "segments",
                    0,
                    "slots",
                    "source",
                ),
                "сообщение 99",
            ),
            (
                ("history", 1, "response", "diagnostics", "policy", "scores", "answer"),
                -1e9,
            ),
            (
                ("history", 0, "response", "diagnostics", "dynamics", "effects"),
                [],
            ),
        )
        for path, replacement in corruptions:
            corrupted = deepcopy(snapshot)
            target: Any = corrupted
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(ValueError):
                LearnedSession.from_dict(corrupted, self.bundle)
        self.assertEqual(self.session.to_dict(), snapshot)

    def test_spatial_and_negated_verification_restore_checks_truth_slots(self) -> None:
        self.turn("ключ на столе", "ack")
        for spatial, negated, truth in (
            ("on", False, "yes"),
            ("in", False, "no"),
            ("on", True, "no"),
            ("in", True, "yes"),
        ):
            query = Query(
                "verify", "ключ", "стол", "location", spatial=spatial, negated=negated
            )
            response = self.scripted_turn(Meaning("ask", query=query), "answer")
            segment = response["diagnostics"]["generation"]["segments"][0]
            self.assertEqual(segment["slots"]["truth"], truth)
            self.restored()
            corrupted = self.session.to_dict()
            last = corrupted["history"][-1]["response"]
            segment = last["diagnostics"]["generation"]["segments"][0]
            segment["slots"]["truth"] = "no" if truth == "yes" else "yes"
            generated = self.bundle.generator.generate(
                "answer", segment["slots"], segment["assertions"]
            )
            self.assertTrue(generated.grounded)
            last["text"] = segment["text"] = generated.text
            segment["tokens"] = list(generated.tokens)
            with self.assertRaisesRegex(ValueError, "slots"):
                LearnedSession.from_dict(corrupted, self.bundle)

    def test_truncated_history_restores_current_evidence_and_reference_context(
        self,
    ) -> None:
        session = LearnedSession(self.bundle, ConversationLimits(max_history=2))
        for text, action in (
            ("ключ в ящике", "ack"),
            ("где ключ?", "answer"),
            ("почему ключ?", "explain"),
            ("нет, ключ на столе", "corrected"),
            ("где ключ?", "answer"),
            ("отмени последнее сообщение", "retracted"),
        ):
            self.turn(text, action, session=session)
            self.restored(session)
        self.assertEqual(len(session.to_dict()["history"]), 2)
        self.assertEqual(len(session.context.turns), 6)
        # Entity annotations are optional in the ontology. A retained reference
        # may use type information from an earlier, now-evicted receipt.
        with patch.object(
            self.bundle.understanding,
            "interpret",
            return_value=Interpretation(
                Meaning("ask", query=Query("where", subject="ключ"))
            ),
        ):
            for _ in range(2):
                self.turn("проверка ссылки", "answer", session=session)
                self.restored(session)


if __name__ == "__main__":
    unittest.main()

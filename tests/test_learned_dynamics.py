"""Public unit/development checks, not the independent frozen evaluation."""

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import patch

from text_factors.learning.dynamics import (
    LearnedDynamics,
    TransitionEpisode,
    checked_facts,
    default_dynamics,
)
from text_factors.learning.schema import Event
from text_factors.learning.transition_data import (
    development_episodes,
    training_episodes,
)
from text_factors.memory import CombinatorialMemory


def fact(
    subject: str = "item",
    relation: str = "holder",
    value: str = "alice",
    *,
    negated: bool = False,
    spatial: str = "in",
) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "value": value,
        "negated": negated,
        "spatial": spatial,
    }


def project(before, effects):
    result = deepcopy(before)
    for effect in effects:

        def same(row, target=effect):
            return all(
                row[key] == target[key]
                for key in ("subject", "relation", "value", "spatial")
            )

        if effect["op"] == "set":
            result = [
                row
                for row in result
                if not (
                    row["subject"] == effect["subject"]
                    and (not row["negated"] or same(row))
                )
            ]
        else:
            result = [row for row in result if not same(row)]
        result.append(
            {key: effect[key] for key in ("subject", "relation", "value", "spatial")}
            | {"negated": effect["op"] == "exclude"}
        )
    return checked_facts(result)


class LearnedDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = default_dynamics(seconds=20)

    def test_public_renamed_development_episodes_project_exactly(self) -> None:
        episodes = development_episodes()
        for episode in episodes:
            with self.subTest(event=episode.event.to_dict(), before=episode.before):
                predicted = self.model.predict(episode.before, episode.event)
                self.assertTrue(predicted.supported, predicted.reason)
                self.assertEqual(
                    project(episode.before, predicted.effects),
                    checked_facts(episode.after),
                )

    def test_role_renamings_and_swapped_participants_copy_the_correct_recipient(
        self,
    ) -> None:
        for actor, recipient in (
            ("alex", "bea"),
            ("bea", "alex"),
            ("newperson", "newother"),
        ):
            event = Event(
                "give", actor=actor, object="unseenobject", recipient=recipient
            )
            prediction = self.model.predict([fact("unseenobject", value=actor)], event)
            self.assertTrue(prediction.supported)
            self.assertEqual(prediction.effects[0]["value"], recipient)
            self.assertEqual(prediction.effects[0]["subject"], "unseenobject")

    def test_nonactual_and_negative_actions_are_learned_supported_noops(self) -> None:
        give = Event("give", actor="alice", object="item", recipient="bob")
        events = [replace(give, negated=True), replace(give, time="future")]
        events += [
            replace(give, modality=mode)
            for mode in ("possible", "intended", "reported", "conditional")
        ]
        events += [
            Event(predicate, actor="alice", content=give)
            for predicate in ("promise", "report", "conditional")
        ]
        events.append(
            Event("conditional", content=Event("promise", actor="alice", content=give))
        )
        for event in events:
            with self.subTest(event=event):
                prediction = self.model.predict([fact()], event)
                self.assertTrue(prediction.supported, prediction.reason)
                self.assertEqual(prediction.effects, ())
                self.assertEqual(prediction.reason, "learned_no_effect")

    def test_negative_states_learn_explicit_exclusions(self) -> None:
        for event, relation, value in (
            (
                Event("locate", object="item", place="room", negated=True),
                "location",
                "room",
            ),
            (
                Event("have", actor="alice", object="item", negated=True),
                "holder",
                "alice",
            ),
        ):
            result = self.model.predict([], event)
            self.assertTrue(result.supported)
            self.assertEqual(result.effects[0]["op"], "exclude")
            self.assertEqual(result.effects[0]["relation"], relation)
            self.assertEqual(result.effects[0]["value"], value)

    def test_missing_before_facts_do_not_invent_a_previous_holder(self) -> None:
        event = Event("give", actor="alice", object="item", recipient="bob")
        prediction = self.model.predict([], event)
        self.assertTrue(prediction.supported)
        self.assertEqual(
            prediction.effects,
            (
                {
                    "op": "set",
                    "subject": "item",
                    "relation": "holder",
                    "value": "bob",
                    "spatial": "in",
                },
            ),
        )

    def test_prior_surface_and_already_known_location_are_supported(self) -> None:
        event = Event("move", actor="alice", object="item", place="room")
        on_surface = [fact("item", "location", "desk", spatial="on")]
        changed = self.model.predict(on_surface, event)
        self.assertTrue(changed.supported)
        self.assertEqual(changed.effects[0]["value"], "room")
        repeated = self.model.predict([fact("item", "location", "room")], event)
        self.assertTrue(repeated.supported)
        self.assertEqual(repeated.effects, ())
        self.assertEqual(repeated.reason, "learned_no_effect")

    def test_common_experience_distinguishes_competing_role_interpretations(
        self,
    ) -> None:
        before = [fact()]
        correct = Event("give", actor="alice", object="item", recipient="bob")
        reversed_roles = replace(correct, actor="bob", recipient="alice")
        scored = self.model.score_interpretations(before, [correct, reversed_roles])
        self.assertGreater(scored[0]["score"], scored[1]["score"])
        self.assertEqual(len(scored[0]["contexts"]), 2)
        self.assertEqual(
            len({row["memory_namespace"] for row in scored[0]["contexts"]}), 1
        )
        self.assertEqual(scored[0]["contexts"][0]["context"], "event_roles")
        self.assertEqual(
            scored[0]["contexts"][1]["context"], "actor_recipient_exchange"
        )

    def test_unfamiliar_scope_or_missing_roles_abstain(self) -> None:
        incomplete = Event("give", object="item")
        result = self.model.predict([], incomplete)
        self.assertFalse(result.supported)
        self.assertEqual(result.effects, ())
        aliased = Event("give", actor="alice", object="alice", recipient="alice")
        self.assertFalse(self.model.predict([], aliased).supported)

    def test_inference_does_not_mutate_either_learned_bank(self) -> None:
        before = self.model.to_dict()
        self.model.predict([], Event("move", actor="a", object="b", place="c"))
        self.model.compatibility([], Event("have", actor="a", object="b"))
        self.assertEqual(self.model.to_dict(), before)

    def test_checkpoint_restores_numeric_memory_without_training_replay(self) -> None:
        checkpoint = self.model.to_dict()
        self.assertTrue(checkpoint["transform"]["clusters"])
        self.assertTrue(checkpoint["experience"]["memory"]["clusters"])
        self.assertNotIn("episodes", checkpoint)
        with patch.object(
            LearnedDynamics, "fit", side_effect=AssertionError("no replay")
        ):
            restored = LearnedDynamics.from_dict(json.loads(json.dumps(checkpoint)))
        self.assertEqual(restored.to_dict(), checkpoint)
        event = Event("give", actor="a", object="b", recipient="c")
        self.assertEqual(
            restored.predict([], event).to_dict(),
            self.model.predict([], event).to_dict(),
        )

    def test_removing_transform_or_experience_prevents_predictions(self) -> None:
        event = Event("give", actor="a", object="b", recipient="c")
        for bank in ("transform", "experience"):
            model = LearnedDynamics.from_dict(self.model.to_dict())
            assert model._transform is not None and model._experience is not None
            if bank == "transform":
                model._transform.memory = CombinatorialMemory(
                    model._transform.memory.config
                )
            else:
                model._experience.memory = CombinatorialMemory(
                    model._experience.memory.config
                )
            result = model.predict([], event)
            self.assertFalse(result.supported)
            self.assertEqual(result.effects, ())

    def test_small_control_models_have_observable_ablations(self) -> None:
        examples = [
            TransitionEpisode(
                [],
                Event("give", actor="a", object="b", recipient="c"),
                [fact("b", value="c")],
            ),
            TransitionEpisode(
                [],
                Event("move", actor="a", object="b", place="c"),
                [fact("b", "location", "c")],
            ),
        ]
        untrained, shuffled = (
            LearnedDynamics(mode="untrained"),
            LearnedDynamics(mode="shuffled"),
        )
        untrained.fit(examples)
        shuffled.fit(examples)
        event = examples[0].event
        self.assertFalse(untrained.predict([], event).supported)
        wrong = shuffled.predict([], event)
        self.assertTrue(
            not wrong.supported
            or project([], wrong.effects) != checked_facts(examples[0].after)
        )

    def test_timeout_invalid_episode_and_conflict_are_atomic(self) -> None:
        model = LearnedDynamics.from_dict(self.model.to_dict())
        before = model.to_dict()
        with self.assertRaises(TimeoutError):
            model.fit(training_episodes(), seconds=1e-12)
        bad = TransitionEpisode(
            [],
            Event("give", actor="a", object="b", recipient="c"),
            [fact("foreign", value="c")],
        )
        with self.assertRaises(ValueError):
            model.fit([bad])
        event = Event("give", actor="a", object="b", recipient="c")
        with self.assertRaises(ValueError):
            model.fit(
                [
                    TransitionEpisode([], event, []),
                    TransitionEpisode([], event, [fact("b", value="c")]),
                ]
            )
        self.assertEqual(model.to_dict(), before)
        self.assertFalse(model.predict([], event, seconds=1e-12).supported)

    def test_checkpoint_dimensions_nonfinite_and_codebook_mismatch_rejected(
        self,
    ) -> None:
        baseline = self.model.to_dict()
        for mutate in (
            lambda d: d["transform"]["config"].update(point_count=100_000_000),
            lambda d: d["transform"]["clusters"][0]["bit_hits"].__setitem__(
                0, float("nan")
            ),
            lambda d: d["transform"]["output_map"].__setitem__(0, True),
            lambda d: d["codes"].reverse(),
        ):
            value = deepcopy(baseline)
            mutate(value)
            with self.assertRaises(ValueError):
                LearnedDynamics.from_dict(value)

    def test_unrelated_negative_knowledge_can_be_preserved_by_training(self) -> None:
        before = [fact("item", "location", "desk", negated=True, spatial="on")]
        after = before + [fact("item", "location", "room")]
        event = Event("move", actor="alice", object="item", place="room")
        model = LearnedDynamics()
        model.fit([TransitionEpisode(before, event, after)], seconds=5)
        result = model.predict(before, event)
        self.assertTrue(result.supported)
        self.assertEqual(project(before, result.effects), checked_facts(after))

    def test_fact_and_episode_validation_is_strict_and_detached(self) -> None:
        example = training_episodes()[0]
        copied = TransitionEpisode.from_dict(example.to_dict())
        self.assertEqual(copied.to_dict(), example.to_dict())
        for wrong in ([], {}, None, True, float("nan")):
            for field in ("relation", "spatial"):
                bad = fact()
                bad[field] = wrong
                with self.assertRaises(ValueError):
                    checked_facts([bad])


if __name__ == "__main__":
    unittest.main()

"""Training/development and generic safety tests, not sealed evaluation cases."""

import json
import math
import unittest
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest import mock

from text_factors.learning.dialogue_data import (
    development_dialogues,
    training_dialogues,
)
from text_factors.learning.dialogue_learning import (
    ACTIONS,
    GeneratedReply,
    LearnedDialoguePolicy,
    LearnedTokenGenerator,
    LearningTimeout,
    default_generator,
    default_policy,
    dialogues_from_data,
)


def fact(**changes):
    value = {
        "subject": "ключ",
        "relation": "location",
        "value": "ящик",
        "negated": False,
        "spatial": "in",
        "source": "сообщение 3",
        "event_id": 1,
    }
    value.update(changes)
    return value


def demonstration(action, response, features=None, evidence=None, slots=None):
    return {
        "user": "Объявленный обучающий пример",
        "features": features or {},
        "action": action,
        "response": response,
        "evidence": evidence or [],
        "slots": slots or {},
    }


class LearnedDialoguePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = default_policy()

    def test_declared_full_dialogues_are_detached_and_bounded(self):
        data = training_dialogues(seed=42)
        self.assertTrue(all(len(sequence) > 1 for sequence in data))
        self.assertTrue(
            any(
                any(turn["action"] == "corrected" for turn in sequence)
                for sequence in data
            )
        )
        copied = dialogues_from_data(data)
        copied[0][0]["response"] = "changed"
        self.assertNotEqual(copied, data)
        self.assertEqual(training_dialogues(), data)
        self.assertNotEqual(development_dialogues(), data)

    def test_policy_learns_demonstrated_actions_with_sequence_context(self):
        correct = total = 0
        for sequence in training_dialogues():
            previous = ""
            for turn in sequence:
                decision = self.policy.choose(
                    {**turn["features"], "prev_action": previous}, ACTIONS
                )
                correct += decision.action == turn["action"]
                total += 1
                self.assertTrue(
                    all(math.isfinite(score) for score in decision.scores.values())
                )
                previous = turn["action"]
        self.assertEqual(correct, total)
        self.assertEqual(self.policy.training_turns, total)

    def test_previous_action_is_learned_not_ignored(self):
        shared = {"act": "unknown", "task": "context_probe"}
        sequences = [
            [
                demonstration("greet", "Привет!", {"act": "greet"}),
                demonstration("help", "Сообщите, где предмет.", shared),
            ],
            [
                demonstration("unknown", "Этого я не знаю.", {"act": "ask"}),
                demonstration("clarify", "Уточните запрос.", shared),
            ],
        ]
        policy = LearnedDialoguePolicy().fit(sequences)
        self.assertEqual(
            policy.choose(
                {**shared, "prev_action": "greet"}, ("help", "clarify")
            ).action,
            "help",
        )
        self.assertEqual(
            policy.choose(
                {**shared, "prev_action": "unknown"}, ("help", "clarify")
            ).action,
            "clarify",
        )

    def test_action_eligibility_does_not_allow_unsupported_actions(self):
        features = {"act": "ask", "has_answer": True, "task": "where"}
        self.assertEqual(self.policy.choose(features, ("unknown",)).action, "unknown")
        self.assertEqual(
            set(self.policy.choose(features, ("unknown", "clarify")).scores),
            {"unknown", "clarify"},
        )
        for actions in ((), ("invent_fact",), "answer"):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.policy.choose(features, actions)

    def test_untrained_and_shuffled_controls_are_real_different_parameters(self):
        fresh = LearnedDialoguePolicy()
        shuffled = LearnedDialoguePolicy().fit(
            training_dialogues(), shuffle_targets=True
        )
        self.assertEqual(fresh.training_turns, 0)
        self.assertEqual(fresh.choose({}, ACTIONS).reason, "untrained_tie_or_prior")
        self.assertNotEqual(
            shuffled.training_fingerprint, self.policy.training_fingerprint
        )
        self.assertNotEqual(
            shuffled.to_dict()["tables"], self.policy.to_dict()["tables"]
        )

    def test_fitting_timeout_or_bad_data_does_not_modify_existing_model(self):
        policy = LearnedDialoguePolicy.from_dict(self.policy.to_dict())
        before = policy.to_dict()
        with (
            mock.patch(
                "text_factors.learning.dialogue_learning._Deadline.check",
                side_effect=LearningTimeout("test"),
            ),
            self.assertRaises(LearningTimeout),
        ):
            policy.fit(training_dialogues())
        self.assertEqual(policy.to_dict(), before)
        with self.assertRaises(ValueError):
            policy.fit([[]])
        self.assertEqual(policy.to_dict(), before)

    def test_numeric_checkpoint_roundtrip_preserves_predictions_and_is_detached(self):
        snapshot = json.loads(json.dumps(self.policy.to_dict()))
        restored = LearnedDialoguePolicy.from_dict(snapshot)
        self.assertEqual(restored.to_dict(), self.policy.to_dict())
        features = {"act": "ask", "has_answer": True, "task": "why"}
        self.assertEqual(
            restored.choose(features, ACTIONS), self.policy.choose(features, ACTIONS)
        )
        snapshot["action_counts"][0] = 999
        self.assertNotEqual(restored.to_dict()["action_counts"][0], 999)
        self.assertEqual(
            LearnedDialoguePolicy.from_dict(
                LearnedDialoguePolicy().to_dict()
            ).training_turns,
            0,
        )

    def test_malformed_numeric_policy_checkpoints_are_rejected(self):
        mutations = [
            lambda d: d.update(alpha=float("nan")),
            lambda d: d.update(alpha=0),
            lambda d: d.update(training_turns=True),
            lambda d: d.update(training_fingerprint="bad"),
            lambda d: d["action_counts"].__setitem__(0, float("inf")),
            lambda d: d["action_counts"].__setitem__(0, True),
            lambda d: d["tables"][0]["counts"][0].append(3),
            lambda d: d["tables"][0]["counts"][0].__setitem__(0, -1),
            lambda d: d["tables"][0].update(vocabulary=["x"] * 129),
            lambda d: d["tables"].append(deepcopy(d["tables"][0])),
        ]
        for mutate in mutations:
            damaged = deepcopy(self.policy.to_dict())
            mutate(damaged)
            with self.subTest(snapshot=damaged), self.assertRaises(ValueError):
                LearnedDialoguePolicy.from_dict(damaged)

    def test_feature_and_training_input_bounds(self):
        for features in (
            {"x": float("nan")},
            {"x": float("inf")},
            {"x": [1]},
            {"x": "\ud800"},
        ):
            with self.subTest(features=features), self.assertRaises(ValueError):
                self.policy.choose(features, ACTIONS)
        for value in ([], [[]], [training_dialogues()[0]] * 129):
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                dialogues_from_data(value)


class LearnedTokenGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = default_generator()

    def test_all_declared_training_targets_have_a_verified_decoded_response(self):
        for sequence in training_dialogues():
            for turn in sequence:
                with self.subTest(action=turn["action"], user=turn["user"]):
                    reply = self.generator.generate(
                        turn["action"], turn["slots"], turn["evidence"]
                    )
                    self.assertTrue(reply.grounded, reply.reason)
                    self.assertTrue(
                        self.generator.verify(
                            turn["action"], turn["slots"], turn["evidence"], reply
                        )
                    )
                    self.assertLessEqual(len(reply.tokens), 48)

    def test_default_copy_annotations_inflect_known_and_preserve_unseen_names(self):
        location = self.generator.generate(
            "answer", {"object": "ключ", "place": "ящик"}, [fact()]
        )
        self.assertEqual(location.text, "Ключ в ящике.")
        holder = self.generator.generate(
            "answer", {"holder": "маша"}, [fact(relation="holder", value="маша")]
        )
        self.assertEqual(holder.text, "Ключ у Маши.")
        novel = self.generator.generate(
            "answer", {}, [fact(subject="датчик-альфа", value="отсек-бета")]
        )
        self.assertEqual(novel.text, "Датчик-альфа в отсек-бета.")
        self.assertTrue(novel.grounded)

    def test_model_persists_token_counts_not_complete_responses(self):
        snapshot = self.generator.to_dict()
        self.assertNotIn("responses", snapshot)
        self.assertNotIn("examples", snapshot)
        self.assertNotIn("data", snapshot)
        self.assertTrue(snapshot["rows"])
        self.assertTrue(
            all(
                type(value) is int
                for row in snapshot["rows"]
                for value in row["counts"].values()
            )
        )
        self.assertIn("<object>", snapshot["vocabulary"])
        self.assertNotIn("ключ", snapshot["vocabulary"])

    def test_training_response_style_changes_actual_decoded_tokens(self):
        example = demonstration(
            "ack", "Учёл: <object> <prep> <place>.", evidence=[fact()]
        )
        model = LearnedTokenGenerator().fit([[example, deepcopy(example)]])
        before = model.generate("ack", {}, [fact()])
        self.assertTrue(before.text.startswith("Учёл:"))
        example["response"] = "Запомнил: <object> <prep> <place>."
        model.fit([[example, deepcopy(example)]])
        after = model.generate("ack", {}, [fact()])
        self.assertTrue(after.text.startswith("Запомнил:"))
        self.assertNotEqual(before.tokens, after.tokens)

    def test_negation_truth_location_and_holder_are_all_checked(self):
        for relation, value in (("location", "ящик"), ("holder", "маша")):
            for negated in (False, True):
                for truth in ("yes", "no"):
                    evidence = [fact(relation=relation, value=value, negated=negated)]
                    reply = self.generator.generate(
                        "answer", {"truth": truth}, evidence
                    )
                    with self.subTest(relation=relation, negated=negated, truth=truth):
                        self.assertTrue(reply.grounded, reply.reason)
                        self.assertTrue(
                            reply.text.startswith("Да." if truth == "yes" else "Нет.")
                        )
                        self.assertEqual(" не " in reply.text, negated)
                        self.assertTrue(
                            self.generator.verify(
                                "answer", {"truth": truth}, evidence, reply
                            )
                        )

    def test_verifier_rejects_wrong_names_relations_extra_claims_and_negation_flips(
        self,
    ):
        invalid = (
            "Паспорт в ящике.",
            "Ключ в сумке.",
            "Ключ не в ящике.",
            "Ключ на ящике.",
            "Ключ у Маши.",
            "Ключ в ящике. Паспорт в сумке.",
            "Да. Ключ в ящике.",
            "Ключ в ящике, и Миша в комнате.",
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertFalse(self.generator.verify("answer", {}, [fact()], text))
        self.assertFalse(
            self.generator.verify("answer", {}, [fact(negated=True)], "Ключ в ящике.")
        )

    def test_verifier_rejects_wrong_copy_payload_before_decoding(self):
        for slots in (
            {"object": "паспорт"},
            {"place": "сумка"},
            {"holder": "маша"},
            {"prep": "на"},
            {"source": "сообщение 999"},
        ):
            with self.subTest(slots=slots):
                reply = self.generator.generate("answer", slots, [fact()])
                self.assertFalse(reply.grounded)
                self.assertEqual(reply.text, "")

    def test_source_is_real_turn_provenance_not_event_id_guessed_as_turn(self):
        evidence = [fact(source="сообщение 7", event_id=2)]
        reply = self.generator.generate("explain", {}, evidence)
        self.assertTrue(reply.grounded)
        self.assertIn("Источник: сообщение 7.", reply.text)
        self.assertFalse(
            self.generator.verify(
                "explain",
                {},
                evidence,
                reply.text.replace("сообщение 7", "сообщение 2"),
            )
        )
        self.assertFalse(
            self.generator.generate(
                "explain", {"source": "сообщение 2"}, evidence
            ).grounded
        )

    def test_nonactual_acknowledgement_never_claims_execution(self):
        reply = self.generator.generate("nonactual", {}, [])
        self.assertTrue(reply.grounded)
        self.assertIn("Не отмечаю", reply.text)
        self.assertIn("выполненное", reply.text)
        self.assertFalse(
            self.generator.verify("nonactual", {}, [], reply.text.replace("Не ", ""))
        )
        self.assertFalse(
            self.generator.verify("nonactual", {}, [], "Действие выполнено.")
        )

    def test_social_and_unknown_output_never_sneaks_in_facts(self):
        for action in ("greet", "thanks", "help", "clarify", "unknown", "retracted"):
            with self.subTest(action=action):
                reply = self.generator.generate(action, {}, [])
                self.assertTrue(reply.grounded, reply.reason)
                self.assertFalse(
                    self.generator.verify(action, {}, [], reply.text + " Ключ в ящике.")
                )
        self.assertFalse(self.generator.verify("unknown", {}, [], "Я знаю."))

    def test_no_evidence_no_factual_claim_and_no_multi_fact_partial_answer(self):
        for action in ("answer", "explain"):
            self.assertFalse(self.generator.generate(action, {}, []).grounded)
        self.assertFalse(
            self.generator.generate(
                "answer", {}, [fact(), fact(subject="паспорт")]
            ).grounded
        )

    def test_query_topic_without_evidence_can_receive_an_unknown_response(self):
        reply = self.generator.generate("unknown", {"object": "ключ"}, [])
        self.assertTrue(reply.grounded, reply.reason)
        self.assertTrue(self.generator.verify("unknown", {"object": "ключ"}, [], reply))
        self.assertFalse(
            self.generator.generate("answer", {"object": "ключ"}, []).grounded
        )
        self.assertFalse(
            self.generator.generate(
                "unknown", {"object": "ключ", "holder": "маша"}, []
            ).grounded
        )

    def test_actual_tokens_and_rendered_text_must_agree_on_cached_reply(self):
        reply = self.generator.generate("answer", {}, [fact()])
        self.assertTrue(self.generator.verify("answer", {}, [fact()], reply))
        self.assertFalse(
            self.generator.verify(
                "answer", {}, [fact()], replace(reply, text="Паспорт в ящике.")
            )
        )
        self.assertFalse(
            self.generator.verify(
                "answer", {}, [fact()], replace(reply, tokens=("<holder>", "."))
            )
        )
        restored = GeneratedReply.from_dict(json.loads(json.dumps(reply.to_dict())))
        self.assertEqual(restored, reply)

    def test_untrained_and_shuffled_generator_controls_are_real(self):
        self.assertFalse(
            LearnedTokenGenerator().generate("answer", {}, [fact()]).grounded
        )
        shuffled = LearnedTokenGenerator().fit(
            training_dialogues(), shuffle_targets=True
        )
        self.assertNotEqual(
            shuffled.training_fingerprint, self.generator.training_fingerprint
        )
        self.assertNotEqual(
            shuffled.to_dict()["rows"], self.generator.to_dict()["rows"]
        )

    def test_bounded_decode_and_timeout_return_explicit_failures(self):
        short = self.generator.generate("answer", {}, [fact()], max_tokens=1)
        self.assertFalse(short.grounded)
        self.assertLessEqual(len(short.tokens), 1)
        with mock.patch(
            "text_factors.learning.dialogue_learning._Deadline.check",
            side_effect=LearningTimeout("test"),
        ):
            expired = self.generator.generate("answer", {}, [fact()])
        self.assertEqual(expired.reason, "generation_deadline")
        invalid_options: tuple[dict[str, Any], ...] = (
            {"max_tokens": 65},
            {"max_tokens": True},
            {"beam_width": 0},
            {"beam_width": 9},
            {"seconds": float("nan")},
        )
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generator.generate("answer", {}, [fact()], **kwargs)

    def test_failed_fit_does_not_partially_change_weights_or_copy_annotations(self):
        model = LearnedTokenGenerator.from_dict(self.generator.to_dict())
        before = model.to_dict()
        with (
            mock.patch(
                "text_factors.learning.dialogue_learning._Deadline.check",
                side_effect=LearningTimeout("test"),
            ),
            self.assertRaises(LearningTimeout),
        ):
            model.fit(training_dialogues())
        self.assertEqual(model.to_dict(), before)
        damaged = training_dialogues()
        damaged[0][1]["response"] = "Паспорт не в другом месте."
        with self.assertRaises(ValueError):
            model.fit(damaged)
        self.assertEqual(model.to_dict(), before)

    def test_generator_checkpoint_restores_identical_numeric_predictions(self):
        original = self.generator.to_dict()
        copied = json.loads(json.dumps(original))
        restored = LearnedTokenGenerator.from_dict(copied)
        self.assertEqual(restored.to_dict(), original)
        self.assertEqual(
            restored.generate("answer", {}, [fact()]),
            self.generator.generate("answer", {}, [fact()]),
        )
        first = next(iter(copied["rows"][0]["counts"]))
        copied["rows"][0]["counts"][first] = 999
        self.assertEqual(restored.to_dict(), original)
        self.assertEqual(
            LearnedTokenGenerator.from_dict(
                LearnedTokenGenerator().to_dict()
            ).training_turns,
            0,
        )

    def test_malformed_generator_checkpoints_rejected_before_allocation(self):
        original = self.generator.to_dict()
        token = next(iter(original["rows"][0]["counts"]))
        mutations = (
            lambda d: d.update(training_turns=True),
            lambda d: d.update(training_fingerprint="x"),
            lambda d: d.update(vocabulary=["x"] * 257),
            lambda d: d["rows"][0].update(context=["evil", "", ""]),
            lambda d: d["rows"][0]["counts"].update({token: float("inf")}),
            lambda d: d["rows"][0]["counts"].update({token: -1}),
            lambda d: d["rows"][0]["counts"].update({token: True}),
            lambda d: d["rows"].append(deepcopy(d["rows"][0])),
            lambda d: d["surface_forms"][0].update(role="invented"),
            lambda d: d["surface_forms"][0].update(forms=["\ud800"]),
        )
        for mutation in mutations:
            damaged = deepcopy(original)
            mutation(damaged)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                LearnedTokenGenerator.from_dict(damaged)

    def test_malformed_evidence_and_control_characters_never_produce_text(self):
        for evidence in (
            [fact(negated=1)],
            [fact(event_id=True)],
            [fact(value="ящик. Паспорт в сумке")],
            [fact(source="источник\nподмена")],
        ):
            with self.subTest(evidence=evidence):
                reply = self.generator.generate("answer", {}, evidence)
                self.assertFalse(reply.grounded)
                self.assertEqual(reply.text, "")


if __name__ == "__main__":
    unittest.main()

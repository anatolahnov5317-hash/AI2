"""Real fitted components on public examples; no sealed evaluation imports.

The one bounded fit in setUpClass is shared by all checks. These are development
regressions, not an estimate of performance on independently held-out language.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from text_factors.learning import runtime as runtime_module
from text_factors.learning.commands import _DEMO
from text_factors.learning.model import ModelBundle
from text_factors.learning.persistence import read_artifact, save_artifact
from text_factors.learning.runtime import SupervisedLearnedConversation
from text_factors.learning.schema import Event
from text_factors.learning.session import LearnedSession


def semantic_facts(rows: Any) -> list[tuple[str, str, str, bool, str]]:
    return sorted(
        (
            row["subject"],
            row["relation"],
            row["value"],
            row["negated"],
            row["spatial"],
        )
        for row in rows
    )


def without_elapsed(response: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in response.items() if key != "elapsed_seconds"}


class LearnedIntegrationTests(unittest.TestCase):
    bundle: ModelBundle
    checkpoint: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = ModelBundle.fit(seed=42, seconds=120)
        cls.checkpoint = cls.bundle.to_dict()

    def tearDown(self) -> None:
        # Every exercised inference path must leave all numeric parameters intact.
        self.assertEqual(self.bundle.to_dict(), self.checkpoint)

    def test_explicit_explanation_subject_cannot_reuse_another_objects_answer(self):
        session = LearnedSession(self.bundle)
        self.assert_response(session.respond("Ключ в ящике."), "ack")
        self.assert_response(session.respond("Где ключ?"), "answer")
        response = session.respond("Почему книга?")
        self.assert_response(response, ("unknown", "clarify"))
        self.assertEqual(response["assertions"], [])
        self.assertEqual(response["evidence"], [])
        self.assertEqual(
            LearnedSession.from_dict(session.to_dict(), self.bundle).to_dict(),
            session.to_dict(),
        )

    def assert_response(
        self, response: dict[str, Any], action: str | tuple[str, ...]
    ) -> dict[str, Any]:
        expected = (action,) if isinstance(action, str) else action
        self.assertIn(response["action"], expected, response)
        self.assertEqual(
            response["complete"], response["action"] not in {"unknown", "clarify"}
        )
        self.assertTrue(response["text"].strip())
        self.assertEqual(
            response["diagnostics"]["policy"]["action"], response["action"]
        )
        self.assertTrue(response["diagnostics"]["generation"]["grounded"])
        evidence = {row["id"]: row for row in response["evidence"]}
        for assertion in response["assertions"]:
            record = evidence[assertion["event_id"]]
            self.assertEqual(assertion["source"], record["source"])
            self.assertIn(
                {
                    "op": "exclude" if assertion["negated"] else "set",
                    "subject": assertion["subject"],
                    "relation": assertion["relation"],
                    "value": assertion["value"],
                    "spatial": assertion["spatial"],
                },
                record["effects"],
            )
        return response

    def test_received_and_given_have_identical_roles_and_world_effects(self) -> None:
        expected = Event("give", actor="петя", object="книга", recipient="маша")
        sessions = [LearnedSession(self.bundle), LearnedSession(self.bundle)]
        statements = ("Маша получила книгу от Пети", "Петя передал книгу Маше")
        meanings = []
        for session, text in zip(sessions, statements, strict=True):
            with self.subTest(text=text):
                response = self.assert_response(session.respond(text), "ack")
                self.assertEqual(response["meaning"]["event"], expected.to_dict())
                self.assertTrue(response["diagnostics"]["dynamics"]["supported"])
                meanings.append(response["meaning"])
                answer = self.assert_response(
                    session.respond("У кого книга?"), "answer"
                )
                self.assertEqual(
                    semantic_facts(answer["assertions"]),
                    [("книга", "holder", "маша", False, "in")],
                )
        self.assertEqual(meanings[0], meanings[1])
        self.assertEqual(sessions[0].world.facts(), sessions[1].world.facts())

    def test_entire_public_demo_preserves_scope_evidence_and_retraction(self) -> None:
        actions = (
            "greet",
            "ack",
            "answer",
            "ack",
            "answer",
            "explain",
            "nonactual",
            "answer",
            "corrected",
            "answer",
            "retracted",
            "answer",
        )
        expected_facts = {
            3: ("location", "ящик", "in", 2),
            5: ("holder", "петя", "in", 4),
            6: ("holder", "петя", "in", 4),
            8: ("holder", "петя", "in", 4),
            10: ("location", "стол", "on", 9),
            12: ("holder", "петя", "in", 4),
        }
        self.assertEqual(len(_DEMO), len(actions))
        session = LearnedSession(self.bundle)
        responses = []
        for turn_id, (text, action) in enumerate(zip(_DEMO, actions, strict=True), 1):
            with self.subTest(turn=turn_id, text=text):
                response = self.assert_response(session.respond(text), action)
                self.assertEqual(response["turn_id"], turn_id)
                responses.append(response)
                if turn_id in expected_facts:
                    relation, value, spatial, source_turn = expected_facts[turn_id]
                    self.assertEqual(
                        semantic_facts(response["assertions"]),
                        [("книга", relation, value, False, spatial)],
                    )
                    self.assertEqual(
                        response["assertions"][0]["source"],
                        f"сообщение {source_turn}",
                    )
                # Supervised chat reloads a saved receipt window before each turn.
                snapshot = json.loads(json.dumps(session.to_dict(), allow_nan=False))
                session = LearnedSession.from_dict(snapshot, self.bundle)
                self.assertEqual(session.to_dict(), snapshot)
        promise = responses[6]
        self.assertEqual(promise["meaning"]["event"]["predicate"], "promise")
        self.assertEqual(promise["meaning"]["event"]["content"]["recipient"], "маша")
        self.assertEqual(promise["assertions"], [])
        self.assertEqual(promise["diagnostics"]["dynamics"]["effects"], [])
        self.assertEqual(len(session.world.events), 5)
        self.assertEqual(session.world.events[-1]["target"], 4)
        self.assertEqual(
            semantic_facts(session.world.facts()),
            [("книга", "holder", "петя", False, "in")],
        )

    def test_promise_and_negated_transfer_do_not_move_the_object(self) -> None:
        session = LearnedSession(self.bundle)
        self.assert_response(session.respond("У Пети есть книга"), "ack")
        original = session.world.facts()
        texts = (
            "Петя обещал передать книгу Маше",
            "Петя не передал книгу Маше",
        )
        for text in texts:
            with self.subTest(text=text):
                response = self.assert_response(session.respond(text), "nonactual")
                self.assertEqual(response["assertions"], [])
                self.assertEqual(response["diagnostics"]["dynamics"]["effects"], [])
                self.assertEqual(session.world.facts(), original)
        negative = session.world.events[-1]["meaning"]["event"]
        self.assertEqual(negative["predicate"], "give")
        self.assertIs(negative["negated"], True)
        answer = self.assert_response(session.respond("У кого книга?"), "answer")
        self.assertEqual(answer["assertions"], list(original))

    def test_negative_location_answers_polarity_without_inventing_a_location(
        self,
    ) -> None:
        session = LearnedSession(self.bundle)
        statement = self.assert_response(session.respond("Книга не на столе"), "ack")
        self.assertEqual(
            semantic_facts(statement["assertions"]),
            [("книга", "location", "стол", True, "on")],
        )
        for text, truth in (("Книга на столе?", "no"), ("Книга не на столе?", "yes")):
            with self.subTest(text=text):
                answer = self.assert_response(session.respond(text), "answer")
                self.assertEqual(answer["assertions"], statement["assertions"])
                self.assertEqual(
                    answer["diagnostics"]["generation"]["segments"][0]["slots"][
                        "truth"
                    ],
                    truth,
                )
        missing = self.assert_response(session.respond("Где книга?"), "unknown")
        self.assertEqual(missing["reason"], "missing_evidence")
        self.assertEqual(missing["assertions"], [])

    def test_reference_resolution_tracks_object_and_person_roles(self) -> None:
        session = LearnedSession(self.bundle)
        self.assert_response(session.respond("Маша положила книгу в ящик"), "ack")
        answer = self.assert_response(session.respond("Где она?"), "answer")
        self.assertEqual(answer["meaning"]["query"]["subject"], "книга")
        moved = self.assert_response(
            session.respond("Она положила книгу на стол"), "ack"
        )
        event = moved["meaning"]["event"]
        self.assertEqual(event["actor"], "маша")
        self.assertEqual(event["object"], "книга")
        self.assertEqual(event["place"], "стол")
        self.assertEqual(event["spatial"], "on")
        answer = self.assert_response(session.respond("Где она?"), "answer")
        self.assertEqual(
            semantic_facts(answer["assertions"]),
            [("книга", "location", "стол", False, "on")],
        )

    def test_missing_reference_requests_clarification_without_world_write(self) -> None:
        session = LearnedSession(self.bundle)
        before = session.world.to_dict()
        answer = self.assert_response(session.respond("Где она?"), "clarify")
        self.assertIsNone(answer["meaning"])
        self.assertIn("reference", answer["reason"])
        self.assertEqual(answer["assertions"], [])
        self.assertEqual(session.world.to_dict(), before)

    def test_missing_fact_and_unsupported_language_abstain_without_fabrication(
        self,
    ) -> None:
        session = LearnedSession(self.bundle)
        self.assert_response(session.respond("Маша положила книгу в ящик"), "ack")
        before = session.world.to_dict()
        missing = self.assert_response(session.respond("Где телефон?"), "unknown")
        self.assertEqual(missing["meaning"]["query"]["subject"], "телефон")
        self.assertEqual(missing["reason"], "missing_evidence")
        unsupported = self.assert_response(
            session.respond("Объясни квантовую гравитацию"), ("unknown", "clarify")
        )
        self.assertIsNone(unsupported["meaning"])
        self.assertEqual(
            unsupported["diagnostics"]["understanding"]["reason"], "unknown_lexeme"
        )
        for response in (missing, unsupported):
            self.assertEqual(response["assertions"], [])
            self.assertEqual(response["evidence"], [])
        self.assertEqual(session.world.to_dict(), before)

    def test_model_contains_numeric_learned_parameters_and_data_fingerprints(
        self,
    ) -> None:
        saved = self.checkpoint
        understanding = saved["understanding"]
        self.assertTrue(understanding["trained"])
        for name in ("role_weights", "reference_weights"):
            weights = np.asarray(understanding[name])
            self.assertTrue(np.isfinite(weights).all())
            self.assertGreater(np.count_nonzero(weights), 0)
        weights = np.asarray(understanding["heads"]["act"]["weights"])
        self.assertGreater(np.count_nonzero(weights), 0)
        for bank in (
            saved["dynamics"]["transform"],
            saved["dynamics"]["experience"]["memory"],
        ):
            self.assertTrue(bank["clusters"])
            self.assertGreater(sum(bank["clusters"][0]["bit_hits"]), 0)
        self.assertGreater(saved["policy"]["training_turns"], 0)
        self.assertEqual(
            sum(saved["policy"]["action_counts"]), saved["policy"]["training_turns"]
        )
        self.assertGreater(
            sum(sum(row["counts"].values()) for row in saved["generator"]["rows"]),
            0,
        )
        for fingerprint in (
            understanding["fingerprint"],
            saved["dynamics"]["training"]["fingerprint"],
            saved["policy"]["training_fingerprint"],
            saved["generator"]["training_fingerprint"],
        ):
            self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertIs(saved["metadata"]["pretrained_model"], False)
        self.assertIs(saved["metadata"]["implicit_online_learning"], False)

    def test_numeric_checkpoint_and_session_roundtrip_without_training_replay(
        self,
    ) -> None:
        session = LearnedSession(self.bundle)
        self.assert_response(session.respond("Маша положила книгу в ящик"), "ack")
        self.assert_response(session.respond("Где книга?"), "answer")
        state = session.to_dict()
        with tempfile.TemporaryDirectory(prefix="ai2-v05-integration-") as directory:
            model_path, state_path = (
                Path(directory) / name for name in ("model.json", "state.json")
            )
            save_artifact(self.checkpoint, model_path, kind="model")
            save_artifact(state, state_path, kind="session")
            with ExitStack() as guards:
                for component in (
                    ModelBundle,
                    type(self.bundle.understanding),
                    type(self.bundle.dynamics),
                    type(self.bundle.policy),
                    type(self.bundle.generator),
                ):
                    guards.enter_context(
                        patch.object(
                            component,
                            "fit",
                            side_effect=AssertionError("unexpected fit"),
                        )
                    )
                for module, function in (
                    ("language_data", "training_examples"),
                    ("transition_data", "training_episodes"),
                    ("dialogue_data", "training_dialogues"),
                ):
                    guards.enter_context(
                        patch(
                            f"text_factors.learning.{module}.{function}",
                            side_effect=AssertionError(
                                "unexpected training data replay"
                            ),
                        )
                    )
                restored_bundle = ModelBundle.from_dict(
                    read_artifact(model_path, kind="model")
                )
                self.assertEqual(restored_bundle.to_dict(), self.checkpoint)
                self.assertEqual(restored_bundle.fingerprint, self.bundle.fingerprint)
                restored = LearnedSession.from_dict(
                    read_artifact(state_path, kind="session"), restored_bundle
                )
                self.assertEqual(restored.to_dict(), state)
                for text, action in (
                    ("Почему?", "explain"),
                    ("Нет, книга на столе", "corrected"),
                    ("Где она?", "answer"),
                ):
                    original_reply = self.assert_response(session.respond(text), action)
                    restored_reply = self.assert_response(
                        restored.respond(text), action
                    )
                    self.assertEqual(
                        without_elapsed(original_reply), without_elapsed(restored_reply)
                    )
                self.assertEqual(restored.world.to_dict(), session.world.to_dict())
                self.assertEqual(restored.context, session.context)
                self.assertEqual(restored_bundle.to_dict(), self.checkpoint)

    def test_retried_request_is_idempotent_and_conflicting_reuse_is_atomic(
        self,
    ) -> None:
        session = LearnedSession(self.bundle)
        response = self.assert_response(
            session.respond("Маша положила книгу в ящик", request_id="turn-a"), "ack"
        )
        before = session.to_dict()
        self.assertEqual(
            session.respond("Маша положила книгу в ящик", request_id="turn-a"),
            response,
        )
        with self.assertRaises(ValueError):
            session.respond("Нет, книга на столе", request_id="turn-a")
        self.assertEqual(session.to_dict(), before)
        restored = LearnedSession.from_dict(before, self.bundle)
        self.assertEqual(
            restored.respond("Маша положила книгу в ящик", request_id="turn-a"),
            response,
        )
        self.assertEqual(restored.to_dict(), before)

    def test_real_worker_commits_success_and_discards_timed_out_child_mutation(
        self,
    ) -> None:
        runtime = SupervisedLearnedConversation(self.checkpoint, seconds=10)
        self.assert_response(runtime.respond("Маша положила книгу в ящик"), "ack")
        self.assertEqual(runtime.last_status, "completed")
        answer = self.assert_response(runtime.respond("Где книга?"), "answer")
        self.assertEqual(answer["turn_id"], 2)
        self.assertIsNotNone(runtime.state)
        assert runtime.state is not None
        before = deepcopy(runtime.state)
        with tempfile.TemporaryDirectory(prefix="ai2-v05-slow-worker-") as directory:
            marker = Path(directory) / "child-state.json"
            module = Path(directory) / "integration_slow_worker.py"
            # Run the production worker and real mutation, then stall before its
            # result reaches the parent. Only this injected delay is synthetic.
            module.write_text(
                "import json\n"
                "import time\n"
                "from pathlib import Path\n"
                "from text_factors.learning.session import LearnedSession\n"
                "from text_factors.learning.worker import main\n"
                f"marker = Path({str(marker)!r})\n"
                "original = LearnedSession.respond\n"
                "def delayed(self, *args, **kwargs):\n"
                "    response = original(self, *args, **kwargs)\n"
                "    result = {'response': response, 'state': self.to_dict()}\n"
                "    marker.write_text(json.dumps(result), encoding='utf-8')\n"
                "    time.sleep(30)\n"
                "    return response\n"
                "LearnedSession.respond = delayed\n"
                "raise SystemExit(main())\n",
                encoding="utf-8",
            )
            python_path = os.pathsep.join(
                filter(None, (directory, os.environ.get("PYTHONPATH", "")))
            )
            # The hard deadline includes importing Python/NumPy and restoring
            # the actual numeric model. Use the same startup allowance as the
            # successful turns above, so slower CI reaches the injected stall.
            # The 30-second child delay must still be killed before it returns.
            runtime.seconds = 10.0
            with (
                patch.object(runtime_module, "WORKER", module.stem),
                patch.dict(os.environ, {"PYTHONPATH": python_path}),
            ):
                failed = runtime.respond("Нет, книга на столе")
            self.assertEqual(runtime.last_status, "timeout", failed)
            self.assertEqual(failed["action"], "limit")
            self.assertEqual(failed["reason"], "worker_timeout")
            self.assertFalse(failed["complete"])
            self.assertEqual(failed["assertions"], [])
            self.assertEqual(runtime.state, before)
            child = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(child["response"]["action"], "corrected")
            self.assertEqual(child["state"]["turn_count"], 3)
            self.assertNotEqual(child["state"]["world"], before["world"])
        runtime.seconds = 10
        recovered = self.assert_response(runtime.respond("Где книга?"), "answer")
        self.assertEqual(recovered["turn_id"], 3)
        self.assertEqual(
            semantic_facts(recovered["assertions"]),
            [("книга", "location", "ящик", False, "in")],
        )
        self.assertEqual(runtime.model, self.checkpoint)
        self.assertEqual(
            LearnedSession.from_dict(runtime.state, self.bundle).to_dict(),
            runtime.state,
        )


if __name__ == "__main__":
    unittest.main()

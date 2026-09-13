import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text_factors.conversation.engine import ConversationSession
from text_factors.conversation.runtime import SessionFileLock, SupervisedConversation


class SupervisedRuntimeTests(unittest.TestCase):
    def test_real_worker_create_then_restore_and_answer(self) -> None:
        runtime = SupervisedConversation(seconds=15)
        response = runtime.respond("Ключ в ящике", request_id="one")
        self.assertTrue(response.complete, response.to_dict())
        self.assertEqual(runtime.last_status, "completed")
        self.assertIsNotNone(runtime.state)
        restored = SupervisedConversation(seconds=15, state=runtime.state)
        answer = restored.respond("Где ключ?", request_id="two")
        self.assertEqual(answer.text, "Ключ в ящике.")
        before = copy.deepcopy(restored.state)
        repeated = restored.respond("Где ключ?", request_id="two")
        self.assertEqual(repeated, answer)
        self.assertEqual(restored.state, before)

    def test_hard_timeout_does_not_replace_previous_state(self) -> None:
        state = ConversationSession(train_defaults=False).to_dict()
        runtime = SupervisedConversation(seconds=0.000001, state=state)
        before = copy.deepcopy(runtime.state)
        response = runtime.respond("Ключ в ящике")
        self.assertEqual(runtime.last_status, "timeout")
        self.assertFalse(response.complete)
        self.assertEqual(response.reason, "turn_time_budget")
        self.assertEqual(runtime.state, before)

    def test_invalid_worker_result_does_not_replace_state(self) -> None:
        runtime = SupervisedConversation()
        result = {
            "status": "completed",
            "result": {"bad": True},
            "error": None,
            "elapsed_seconds": 0.01,
        }
        with patch(
            "text_factors.conversation.runtime.run_json_worker", return_value=result
        ):
            response = runtime.respond("Привет")
        self.assertFalse(response.complete)
        self.assertIsNone(runtime.state)

    def test_failed_worker_does_not_replace_state(self) -> None:
        runtime = SupervisedConversation()
        for status in ("error", "output_limit"):
            result = {
                "status": status,
                "result": None,
                "error": "bounded",
                "elapsed_seconds": 0.01,
            }
            with patch(
                "text_factors.conversation.runtime.run_json_worker", return_value=result
            ):
                response = runtime.respond("Привет")
            self.assertFalse(response.complete)
            self.assertIsNone(runtime.state)

    def test_timeout_configuration_is_bounded(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), True, 301):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SupervisedConversation(seconds=value)

    def test_session_lock_is_nonblocking_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            with (
                SessionFileLock(path),
                self.assertRaises(ValueError),
                SessionFileLock(path),
            ):
                self.fail("second writer acquired the lock")
            with SessionFileLock(path):
                pass
            self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink check")
    def test_lock_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            lock = path.with_name(path.name + ".lock")
            lock.symlink_to(Path(directory) / "unrelated")
            with self.assertRaises(ValueError), SessionFileLock(path):
                self.fail("symlink accepted")


if __name__ == "__main__":
    unittest.main()

"""CLI and parent-process integrity tests; mocked workers are not quality tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from text_factors.cli import build_parser
from text_factors.conversation.schema import ConversationLimits
from text_factors.learning import commands
from text_factors.learning.model import SCHEMA
from text_factors.learning.persistence import (
    envelope,
    read_artifact,
    save_artifact,
    unpack,
)
from text_factors.learning.runtime import SupervisedLearnedConversation


class LearnedArtifactTests(unittest.TestCase):
    def test_envelope_checksum_kind_version_and_input_copy(self):
        model = {"schema": SCHEMA, "fixture": "not a trained model"}
        wrapped = envelope(model, kind="model")
        self.assertEqual(unpack(wrapped, kind="model"), model)
        for key, replacement in (
            ("version", True),
            ("sha256", "0" * 64),
            ("kind", "ai2.learned.session"),
        ):
            bad = deepcopy(wrapped)
            bad[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                unpack(bad, kind="model")
        with self.assertRaises(ValueError):
            unpack(wrapped, kind="session")

    def test_atomic_save_rejects_unrelated_file_and_symlink(self):
        model = {"schema": SCHEMA, "fixture": True}
        with tempfile.TemporaryDirectory(prefix="ai2-v05-test-") as directory:
            path = Path(directory) / "model.json"
            save_artifact(model, path, kind="model")
            self.assertEqual(read_artifact(path, kind="model"), model)
            with self.assertRaises(FileExistsError):
                save_artifact(model, path, kind="model")
            save_artifact(model, path, kind="model", overwrite=True)
            unrelated = Path(directory) / "unrelated.json"
            unrelated.write_text('{"user":"data"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                save_artifact(model, unrelated, kind="model", overwrite=True)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), '{"user":"data"}')
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                read_artifact(link, kind="model")

    def test_model_envelope_bounded_finite_header(self):
        for value in ({"schema": SCHEMA, "number": float("nan")}, {"schema": "other"}):
            with self.assertRaises(ValueError):
                envelope(value, kind="model")
        with self.assertRaises(ValueError):
            invalid_kind: Any = []
            envelope({"schema": SCHEMA}, kind=invalid_kind)


class LearnedCommandTests(unittest.TestCase):
    def test_new_commands_do_not_replace_old_commands(self):
        parser = build_parser()
        for command in ("converse", "learned-chat"):
            argv = [command, "--demo"]
            if command == "learned-chat":
                argv += ["--model", "model.json"]
            self.assertTrue(callable(parser.parse_args(argv).handler))
        train = parser.parse_args(["learned-train", "--output", "model.json"])
        self.assertIs(train.handler, commands.learned_train)

    def test_training_timeout_never_creates_or_overwrites_model(self):
        with tempfile.TemporaryDirectory(prefix="ai2-v05-test-") as directory:
            path = Path(directory) / "model.json"
            args = build_parser().parse_args(
                ["learned-train", "--output", str(path), "--seconds", "0.01"]
            )
            result = {
                "status": "timeout",
                "result": None,
                "error": "deadline",
                "elapsed_seconds": 0.01,
            }
            with (
                patch.object(commands, "run_json_worker", return_value=result),
                patch("builtins.print"),
            ):
                self.assertEqual(commands.learned_train(args), 1)
                self.assertFalse(path.exists())
                save_artifact({"schema": SCHEMA, "fixture": True}, path, kind="model")
                before = path.read_bytes()
                args.overwrite = True
                self.assertEqual(commands.learned_train(args), 1)
                self.assertEqual(path.read_bytes(), before)

    def test_heldout_requires_explicit_freeze_before_worker(self):
        with tempfile.TemporaryDirectory(prefix="ai2-v05-test-") as directory:
            path = Path(directory) / "model.json"
            save_artifact({"schema": SCHEMA, "fixture": True}, path, kind="model")
            args = build_parser().parse_args(
                ["learned-evaluate", "--model", str(path), "--split", "held_out"]
            )
            with patch.object(commands, "run_json_worker") as worker:
                with self.assertRaises(ValueError):
                    commands.learned_evaluate(args)
                worker.assert_not_called()

    def test_export_contains_training_only_and_typed_annotations(self):
        with tempfile.TemporaryDirectory(prefix="ai2-v05-test-") as directory:
            path = Path(directory) / "training.json"
            args = build_parser().parse_args(
                ["learned-export-data", "--output", str(path)]
            )
            with patch("builtins.print"):
                self.assertEqual(commands.learned_export_data(args), 0)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(value), {"schema", "understanding", "transitions", "dialogues"}
            )
            self.assertGreater(len(value["understanding"]), 0)
            self.assertIn("links", value["understanding"][0])
            self.assertEqual(set(value["transitions"][0]), {"before", "event", "after"})

    def test_hard_evaluation_timeout_preserves_completed_prefix(self):
        with tempfile.TemporaryDirectory(prefix="ai2-v05-test-") as directory:
            model_path = Path(directory) / "model.json"
            report_path = Path(directory) / "report.json"
            save_artifact({"schema": SCHEMA, "fixture": True}, model_path, kind="model")
            args = build_parser().parse_args(
                [
                    "learned-evaluate",
                    "--model",
                    str(model_path),
                    "--split",
                    "development",
                    "--output",
                    str(report_path),
                ]
            )

            def killed_worker(command, payload, **kwargs):
                prefix = {
                    "status": "in_progress",
                    "complete": False,
                    "completed_cases": 2,
                    "requested_cases": 10,
                    "runs": [{"case_id": "fixture-1", "correct": True}],
                }
                commands.atomic_write_json(Path(payload["checkpoint"]), prefix)
                return {
                    "status": "timeout",
                    "result": None,
                    "elapsed_seconds": 0.01,
                    "error": "deadline",
                }

            with (
                patch.object(commands, "run_json_worker", side_effect=killed_worker),
                patch("builtins.print"),
            ):
                self.assertEqual(commands.learned_evaluate(args), 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["completed_cases"], 2)
            self.assertEqual(report["requested_cases"], 10)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["termination_reason"], "hard_worker_timeout")
            self.assertFalse(report["complete"])

    def test_invalid_training_budget_rejected_before_worker(self):
        for seconds in (float("nan"), float("inf"), -1.0, 0.0, 301.0):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                commands._seconds(seconds)


class LearnedRuntimeTests(unittest.TestCase):
    def _runtime(self, *, seconds: float = 1.0):
        # A fixture header exercises only parent-process behavior with a mocked
        # worker; real numeric model inference is tested in integration tests.
        model = {"schema": SCHEMA, "fixture": "parent-only"}
        return SupervisedLearnedConversation(model, seconds=seconds)

    def test_worker_timeout_and_error_preserve_parent_state(self):
        runtime = self._runtime()
        for status in ("timeout", "error", "output_limit"):
            result = {
                "status": status,
                "result": None,
                "elapsed_seconds": 0.1,
                "error": "fixture failure",
            }
            with patch(
                "text_factors.learning.runtime.run_json_worker", return_value=result
            ):
                response = runtime.respond("Где книга?")
            self.assertIsNone(runtime.state)
            self.assertFalse(response["complete"])
            self.assertEqual(response["assertions"], [])

    def test_nonblocking_concurrent_writer(self):
        runtime = self._runtime()
        runtime._lock.acquire()
        try:
            response = runtime.respond("Привет")
        finally:
            runtime._lock.release()
        self.assertEqual(response["reason"], "worker_busy")
        self.assertIsNone(runtime.state)

    def test_malformed_completed_worker_results_preserve_parent_state(self):
        runtime = self._runtime()
        malformed = (
            None,
            {"state": {}, "response": {}},
            {
                "state": {
                    "schema": "ai2-learned-dialogue-session-v1",
                    "model_fingerprint": runtime.model_fingerprint,
                    "limits": runtime.limits.to_dict(),
                    "turn_count": 1,
                },
                "response": {"action": ["answer"]},
            },
        )
        for output in malformed:
            result = {
                "status": "completed",
                "result": output,
                "elapsed_seconds": 0.01,
                "error": None,
            }
            with patch(
                "text_factors.learning.runtime.run_json_worker", return_value=result
            ):
                response = runtime.respond("Где книга?")
            self.assertEqual(response["action"], "error")
            self.assertIsNone(runtime.state)

    def test_supervised_combined_state_headroom(self):
        with self.assertRaises(ValueError):
            SupervisedLearnedConversation(
                {"schema": SCHEMA}, limits=ConversationLimits(max_state_bytes=4_000_000)
            )
        with self.assertRaises(ValueError):
            self._runtime(seconds=float("nan"))


if __name__ == "__main__":
    unittest.main()

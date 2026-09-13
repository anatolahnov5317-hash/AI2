import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text_factors.cli import main
from text_factors.conversation.persistence import atomic_write_json, read_json


class ConversationCommandTests(unittest.TestCase):
    def test_state_survives_separate_cli_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "converse",
                            "--state",
                            str(path),
                            "--message",
                            "Ключ на столе",
                            "--json",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "converse",
                            "--state",
                            str(path),
                            "--message",
                            "Где ключ?",
                            "--json",
                        ]
                    ),
                    0,
                )
            responses = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(responses[-1]["response"]["text"], "Ключ на столе.")
            self.assertEqual(read_json(path)["kind"], "ai2.conversation")

    def test_does_not_overwrite_an_unrelated_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            atomic_write_json(path, {"important": 123})
            original = path.read_bytes()
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["converse", "--state", str(path), "--message", "Привет"])
            self.assertEqual(path.read_bytes(), original)

    def test_hard_timeout_does_not_create_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "converse",
                        "--state",
                        str(path),
                        "--message",
                        "Привет",
                        "--timeout",
                        "0.000001",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertFalse(path.exists())

    def test_real_development_evaluation_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "dialogue-evaluate",
                        "--split",
                        "development",
                        "--seeds",
                        "42",
                        "--modes",
                        "oracle",
                        "--seconds",
                        "10",
                        "--output",
                        str(path),
                    ]
                )
            self.assertEqual(code, 0)
            report = read_json(path)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["by_mode"]["oracle"]["state_correct"], 10)
            self.assertEqual(report["supervisor"]["status"], "completed")

    def test_hard_kill_retains_last_completed_evaluation_prefix(self) -> None:
        def fake_worker(command, payload, **kwargs):
            atomic_write_json(
                payload["checkpoint"],
                {
                    "status": "running",
                    "completed_turns": 3,
                    "runs": [{"complete": False}],
                },
            )
            return {
                "status": "timeout",
                "result": None,
                "elapsed_seconds": 5.0,
                "error": "deadline",
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch(
                    "text_factors.conversation.commands.run_json_worker",
                    side_effect=fake_worker,
                ),
            ):
                code = main(["dialogue-evaluate", "--output", str(path)])
            self.assertEqual(code, 1)
            report = read_json(path)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["completed_turns"], 3)
            self.assertFalse(report["complete"])

    def test_existing_report_is_protected_before_work_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            atomic_write_json(path, {"old": True})
            with (
                contextlib.redirect_stderr(io.StringIO()),
                patch(
                    "text_factors.conversation.commands.run_json_worker",
                ) as worker,
                self.assertRaises(SystemExit),
            ):
                main(["dialogue-evaluate", "--output", str(path)])
            worker.assert_not_called()
            self.assertEqual(read_json(path), {"old": True})

    def test_invalid_evaluation_budget_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["dialogue-evaluate", "--seconds", "nan"])


if __name__ == "__main__":
    unittest.main()

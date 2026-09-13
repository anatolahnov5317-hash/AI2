import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from text_factors.cli import main
from text_factors.dialogue import GroundedDialogue


class IntegrationCliTests(unittest.TestCase):
    def test_timeout_reports_remain_incomplete_and_fail_exit_code(self) -> None:
        for command in (
            "context-integration",
            "coactivation-structure",
            "scene-integration",
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "partial.json"
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = main(
                        [command, "--seconds", "1e-9", "--output", str(destination)]
                    )
                self.assertEqual(code, 1)
                report = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "incomplete")
                self.assertIsNone(report["aggregate"])

    def test_factor_word_matching_policy_persists_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "dialogue.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "--demo",
                        "--grounding",
                        "factor",
                        "--state",
                        str(state),
                        "--message",
                        "покажи ab",
                        "--message",
                        "назови 1 метка",
                    ]
                )
            self.assertEqual(code, 0)
            first = GroundedDialogue.load(state)
            self.assertEqual(first.grounding_policy.mode, "factor")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "--demo",
                        "--state",
                        str(state),
                        "--message",
                        "что видишь",
                    ]
                )
            self.assertEqual(code, 0)
            resumed = GroundedDialogue.load(state)
            self.assertEqual(resumed.grounding_policy, first.grounding_policy)
            self.assertEqual(resumed.stats()["events"], first.stats()["events"])
            self.assertEqual(resumed.describe().text, first.describe().text)


if __name__ == "__main__":
    unittest.main()

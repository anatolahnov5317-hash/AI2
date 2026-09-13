import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from text_factors.cli import main


class ResearchCliTests(unittest.TestCase):
    def test_timeouts_preserve_a_partial_report_and_fail_the_process(self) -> None:
        for command in ("factor-recovery", "context-transfer"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp) / "partial.json"
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = main(
                        [
                            command,
                            "--seeds",
                            "7",
                            "--points",
                            "4",
                            "--seconds-per-seed",
                            "0.000000001",
                            "--output",
                            str(destination),
                        ]
                    )
                self.assertEqual(result, 1)
                report = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "incomplete")
                self.assertFalse(
                    any(
                        row["all_seeds_complete"]
                        for row in (
                            report["aggregate"].values()
                            if isinstance(report["aggregate"], dict)
                            else report["aggregate"]
                        )
                    )
                )


if __name__ == "__main__":
    unittest.main()

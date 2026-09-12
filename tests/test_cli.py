import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from text_factors.cli import main


class CommandLineTests(unittest.TestCase):
    def test_train_summary_and_analyze_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "train",
                        "--text",
                        "abcabcabcabc",
                        "--model",
                        str(model_path),
                        "--epochs",
                        "3",
                        "--points",
                        "128",
                        "--probation-after",
                        "2",
                        "--stable-after",
                        "3",
                        "--alphabet",
                        "abc",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(model_path.exists())
            self.assertEqual(json.loads(output.getvalue())["saved_to"], str(model_path))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "analyze",
                            "--model",
                            str(model_path),
                            "--text",
                            "abcabc",
                        ]
                    ),
                    0,
                )
            report = json.loads(output.getvalue())
            self.assertGreater(len(report["results"]), 0)
            self.assertIn("memory", report)


if __name__ == "__main__":
    unittest.main()

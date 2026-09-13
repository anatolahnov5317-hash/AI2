import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from text_factors.cli import main
from text_factors.evaluation.runner import EvaluationConfig, run_evaluation
from text_factors.evaluation.statistics import NoveltyGate, classification_metrics


class EvaluationRunnerTests(unittest.TestCase):
    def small_config(self):
        return EvaluationConfig(
            seeds=(7,),
            points=8,
            epochs=3,
            train_size=8,
            dev_size=16,
            test_size=16,
            noise_size=16,
            bootstrap_resamples=50,
        )

    def test_raw_traces_recompute_all_metrics(self):
        report = run_evaluation(self.small_config())
        json.dumps(report, allow_nan=False)
        self.assertEqual(
            report["manifest"]["scope"],
            "within-family combination transfer; NOT unseen-rule learning or AGI",
        )
        run = report["runs"][0]
        self.assertIn("pair_association", run["methods"])
        self.assertIn("pair_association", report["aggregate"])
        for result in run["methods"].values():
            gate = NoveltyGate.fit(
                [row["score"] for row in result["dev"] if not row["label"]],
                target_fpr=0.05,
            )
            self.assertEqual(gate.threshold, result["threshold"]["threshold"])
            metrics = classification_metrics(
                [row["label"] for row in result["test"]],
                [row["score"] for row in result["test"]],
                gate,
            )
            self.assertEqual(metrics, result["metrics"])

    def test_metrics_reproducible_but_timings_not_compared(self):
        first = run_evaluation(self.small_config())
        second = run_evaluation(self.small_config())
        self.assertEqual(first["aggregate"], second["aggregate"])
        self.assertEqual(
            first["runs"][0]["data_sha256"], second["runs"][0]["data_sha256"]
        )
        for name, result in first["runs"][0]["methods"].items():
            self.assertEqual(result["test"], second["runs"][0]["methods"][name]["test"])

    def test_cli_writes_report_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            args = [
                "evaluate",
                "--seeds",
                "7",
                "--points",
                "8",
                "--epochs",
                "3",
                "--train-size",
                "8",
                "--dev-size",
                "16",
                "--test-size",
                "16",
                "--noise-size",
                "16",
                "--bootstrap-resamples",
                "20",
                "--output",
                str(path),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            original = path.read_bytes()
            self.assertIn("aggregate", json.loads(original))
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(args)
            self.assertEqual(original, path.read_bytes())

    def test_microworld_cli(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(main(["microworld", "--seed", "7", "--episodes", "2"]), 0)
        self.assertIn("metrics", json.loads(stream.getvalue()))

    def test_config_validation(self):
        for kwargs in (
            {"seeds": ()},
            {"seeds": (1, 1)},
            {"points": 0},
            {"epochs": False},
            {"target_fpr": 1.0},
            {"noise_size": -1},
            {"noise_size": False},
        ):
            with self.assertRaises(ValueError):
                EvaluationConfig(**cast(dict[str, Any], kwargs))

    def test_noise_control_can_be_explicitly_disabled(self):
        report = run_evaluation(replace(self.small_config(), noise_size=0))
        self.assertEqual(len(report["runs"][0]["data"]["noise"]), 0)


if __name__ == "__main__":
    unittest.main()

"""Budget and persistence guardrails for the experimental real-data training path."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from text_factors import ModelConfig, TextFactorModel
from text_factors.cli import main
from text_factors.real_data import BudgetExceeded, ResourceBudget


def small_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 64,
        "active_bits_per_symbol": 4,
        "positions": 5,
        "frame_size": 3,
        "context_count": 5,
        "receptive_bits": 16,
        "point_count": 64,
        "output_bits": 16,
        "create_threshold": 3,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
        "max_clusters_per_point": 8,
        "seed": 7,
    }
    values.update(overrides)
    return ModelConfig(**values)


class RealDataTrainingGuardrailTests(unittest.TestCase):
    def test_fit_text_stops_before_window_beyond_step_budget(self):
        model = TextFactorModel(small_config(), alphabet="abc")
        with self.assertRaises(BudgetExceeded) as raised:
            model.fit_text(
                "abcabc",
                budget=ResourceBudget(
                    max_steps=1,
                    max_items=10,
                    max_bytes=10_000,
                    max_wall_seconds=30.0,
                    max_clusters_total=10_000,
                    checkpoint_every_steps=1,
                ),
            )
        self.assertEqual(raised.exception.reason, "max_steps")
        self.assertEqual(raised.exception.snapshot.steps, 1)
        self.assertEqual(model.memory.step, 1)
        self.assertEqual(model.memory.cluster_count, model.memory.stats()["clusters"])

    def test_fit_text_emits_completed_progress_snapshot(self):
        model = TextFactorModel(small_config(), alphabet="abc")
        progress: list[dict] = []
        model.fit_text(
            "abc",
            budget=ResourceBudget(
                max_steps=10,
                checkpoint_every_steps=1,
                max_clusters_total=10_000,
            ),
            progress=progress.append,
        )
        self.assertEqual(progress[-1]["phase"], "completed")
        self.assertTrue(progress[-1]["budget"]["complete"])
        self.assertEqual(progress[-1]["clusters"], model.memory.cluster_count)

    def test_model_save_rejects_artifact_above_explicit_limit(self):
        model = TextFactorModel(small_config(), alphabet="abc")
        model.fit_text("abcabcabc", epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too-small.npz"
            with self.assertRaisesRegex(ValueError, "max_file_bytes"):
                model.save(path, max_file_bytes=1)
            self.assertFalse(path.exists())

    def test_cli_budget_stop_does_not_publish_partial_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "train",
                        "--text",
                        "abcabc",
                        "--model",
                        str(path),
                        "--points",
                        "64",
                        "--alphabet",
                        "abc",
                        "--max-windows",
                        "1",
                        "--progress-every",
                        "1",
                    ]
                )
            self.assertEqual(code, 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["stop_reason"], "max_steps")
            self.assertIsNone(report["saved_to"])
            self.assertFalse(path.exists())

    def test_cli_success_records_explicit_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "train",
                        "--text",
                        "abc",
                        "--model",
                        str(path),
                        "--points",
                        "64",
                        "--alphabet",
                        "abc",
                        "--max-windows",
                        "10",
                        "--max-wall-seconds",
                        "30",
                        "--max-clusters-total",
                        "10000",
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["training_budget"]["max_windows"], 10)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

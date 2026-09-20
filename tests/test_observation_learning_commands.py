"""Frozen experiment boundaries, stage recovery, and language-gated proposals."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from test_observation_learning_data import corpus_fixture

from text_factors.cli import build_parser
from text_factors.observations.learning import LearningConfig
from text_factors.observations.learning_commands import (
    evaluate_bundle,
    load_bundle,
    propose_bundle,
    train_stages,
    write_artifact,
)
from text_factors.observations.learning_data import fingerprint


class LearningCommandsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.corpus = corpus_fixture()
        self.config = LearningConfig(epochs=1, feature_dim=128)

    def tearDown(self):
        self.directory.cleanup()

    def _train(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_stages(self.corpus, self.root, self.config, **kwargs)

    def test_calibration_failure_keeps_trained_stage_and_resume_does_not_refit(self):
        with (
            patch(
                "text_factors.observations.learning_commands.calibrate_model",
                side_effect=ValueError("interrupted"),
            ),
            self.assertRaisesRegex(ValueError, "interrupted"),
        ):
            self._train()
        self.assertTrue((self.root / "trained.json").exists())
        self.assertFalse((self.root / "model.json").exists())
        with patch("text_factors.observations.learning_commands.train_model") as fit:
            self._train(resume=True)
            fit.assert_not_called()
        original = (self.root / "model.json").read_bytes()
        with patch("text_factors.observations.learning_commands.train_model") as fit:
            self._train(resume=True)
            fit.assert_not_called()
        self.assertEqual((self.root / "model.json").read_bytes(), original)

    def test_resume_refuses_different_labels_or_implementation(self):
        self._train()
        old = (self.root / "model.json").read_bytes()
        with (
            patch(
                "text_factors.observations.learning_commands.implementation_fingerprint",
                return_value="changed",
            ),
            self.assertRaisesRegex(ValueError, "specification"),
        ):
            self._train(resume=True)
        self.corpus["documents"][0]["mentions"][0]["entity_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "specification"):
            self._train(resume=True)
        self.assertEqual((self.root / "model.json").read_bytes(), old)

    def test_test_labels_cannot_change_training_weights_and_evaluation_is_frozen(self):
        first = self._train()
        changed = deepcopy(self.corpus)
        changed["documents"][2]["mentions"][1]["entity_id"] = "other"
        with contextlib.redirect_stdout(io.StringIO()):
            second = train_stages(changed, self.root / "another", self.config)
        self.assertEqual(first["model"], second["model"])
        self.assertEqual(first["policy"], second["policy"])
        with self.assertRaisesRegex(ValueError, "frozen experiment"):
            evaluate_bundle(self.root / "model.json", changed)
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_bundle(self.root / "model.json", self.corpus)
        self.assertEqual(result["corpus"]["splits"]["test"]["mentions"], 2)
        self.assertFalse(result["production_ready"])

    def test_checkpoint_policy_is_validated_even_on_completed_resume(self):
        self._train()
        path = self.root / "model.json"
        damaged = json.loads(path.read_text())
        damaged["payload"]["policy"]["model_fingerprint"] = "wrong"
        damaged["fingerprint"] = fingerprint(damaged["payload"])
        path.write_text(json.dumps(damaged))
        with self.assertRaises(ValueError):
            load_bundle(path)
        with self.assertRaises(ValueError):
            self._train(resume=True)

    def test_language_selection_requires_training_and_calibration_coverage(self):
        self.corpus["documents"][0]["language"] = "en"
        self.corpus["documents"][1]["language"] = "de"
        self._train()
        for language in ("en", "de", "ru"):
            with patch(
                "text_factors.observations.learning_commands.propose",
                return_value={"mentions": []},
            ) as proposal:
                result = propose_bundle(
                    self.root / "model.json", "Новые слова", language=language
                )
            self.assertFalse(proposal.call_args.kwargs["allow_selection"])
            self.assertFalse(result["archive_mutated"])
        self.assertFalse(result["language_in_calibration"])

    def test_cli_proposals_preserve_bytes_and_do_not_write_into_archive(self):
        self._train()
        original = "\ufeffНовый №42\r\n你好 😀"
        input_path = self.root / "input.txt"
        input_path.write_bytes(original.encode("utf-8"))
        output_path = self.root / "proposals.json"
        args = build_parser().parse_args(
            [
                "mention-learning",
                "propose",
                "--model",
                str(self.root / "model.json"),
                "--input",
                str(input_path),
                "--language",
                "en",
                "--output",
                str(output_path),
            ]
        )
        self.assertEqual(args.handler(args), 0)
        result = json.loads(output_path.read_text())
        self.assertEqual(result["text_fingerprint"], fingerprint(original))
        self.assertTrue(
            all(mention["selected"] is None for mention in result["mentions"])
        )
        self.assertFalse(result["archive_mutated"])
        with self.assertRaises(ValueError):
            write_artifact(output_path, {"schema": "unrelated"}, overwrite=True)


if __name__ == "__main__":
    unittest.main()

"""P03 candidate ceiling, nesting, public-only comparison and time bounds."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text_factors.observations.diagnostics import (
    NGramSpanControl,
    calibrate_ngram_control,
    diagnose_mentions,
    score_ngram_control,
)
from text_factors.observations.learning import LearningConfig, _span_indices, _tokens
from text_factors.observations.learning_data import fingerprint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_open_mentions.py"
SPEC = importlib.util.spec_from_file_location("diagnose_open_mentions", SCRIPT)
assert SPEC and SPEC.loader
diagnostic_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic_script)


def _doc(identifier: str, split: str, text: str, spans: list[tuple[int, int]]) -> dict:
    return {
        "document_id": identifier,
        "group_id": identifier,
        "split": split,
        "coverage": "complete",
        "text": text,
        "mentions": [
            {"start": start, "end": end, "entity_id": f"{identifier}:{index}"}
            for index, (start, end) in enumerate(spans)
        ],
    }


class _Model:
    config = LearningConfig(max_span_tokens=2)

    def span_scores(self, text: str) -> list[dict]:
        tokens = _tokens(text, self.config)
        accepted = {(0, 5), (0, 10), (6, 10), (11, 16)}
        return [
            {
                "start": tokens[left].start,
                "end": tokens[right - 1].end,
                "score": (
                    0.9
                    if (tokens[left].start, tokens[right - 1].end) in accepted
                    else 0.1
                ),
            }
            for left, right in _span_indices(tokens, self.config)
        ]


class P03DiagnosticsTests(unittest.TestCase):
    def test_candidate_ceiling_keeps_long_unaligned_and_nested_gold(self):
        text = "Alpha Beta Gamma Delta"
        gold = [(0, 5), (0, 10), (6, 10), (0, 16), (1, 5), (17, 22)]
        report = diagnose_mentions(
            _Model(), [_doc("validation-a", "validation", text, gold)], threshold=0.5
        )
        self.assertEqual(report["candidate_coverage"]["represented_count"], 4)
        self.assertEqual(report["candidate_coverage"]["gold_count"], 6)
        self.assertAlmostEqual(report["candidate_coverage"]["recall"], 4 / 6)
        self.assertEqual(report["selected_spans"]["true_positive"], 3)
        self.assertEqual(report["selected_spans"]["false_positive"], 1)
        self.assertEqual(report["selected_spans"]["false_negative"], 3)
        self.assertEqual(
            report["span_length_in_unicode_tokens"]["2-4"]["candidate_coverage"][
                "unsupported_count"
            ],
            1,
        )
        self.assertEqual(
            report["span_length_in_unicode_tokens"]["unaligned"]["candidate_coverage"][
                "unsupported_count"
            ],
            1,
        )
        self.assertEqual(report["gold_nesting"]["nested"]["gold_count"], 5)
        self.assertEqual(report["gold_nesting"]["flat"]["gold_count"], 1)
        self.assertEqual(report["per_document"][0]["characters"], len(text))

    def test_only_public_splits_and_finite_threshold_are_accepted(self):
        text = "Alpha Beta"
        for split in ("train", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                diagnose_mentions(
                    _Model(), [_doc("sealed", split, text, [(0, 5)])], threshold=0.5
                )
        with self.assertRaises(ValueError):
            diagnose_mentions(
                _Model(),
                [_doc("v", "validation", text, [(0, 5)])],
                threshold=float("nan"),
            )

    def test_long_public_document_keeps_overwide_gold_in_denominator(self):
        text = " ".join(f"token{i}" for i in range(160))
        report = diagnose_mentions(
            _Model(),
            [_doc("validation-long", "validation", text, [(0, len(text))])],
            threshold=0.5,
        )
        self.assertEqual(report["per_document"][0]["tokens"], 160)
        self.assertEqual(report["candidate_coverage"]["unsupported_count"], 1)
        self.assertEqual(report["candidate_coverage"]["recall"], 0.0)
        self.assertEqual(
            report["span_length_in_unicode_tokens"]["17+"]["candidate_coverage"][
                "unsupported_count"
            ],
            1,
        )

    def test_ngram_adapter_fits_train_only_and_calibrates_validation_only(self):
        config = LearningConfig(max_span_tokens=2)
        train = [_doc("train-a", "train", "Ann saw Ann", [(0, 3), (8, 11)])]
        validation = [_doc("validation-b", "validation", "Ann met Bob", [(0, 3)])]
        control = NGramSpanControl(config)
        with self.assertRaises(ValueError):
            control.fit(validation)
        control.fit(train)
        self.assertEqual(
            control.span_scores("Ann")[:1][0]["score"], control.scorer.score("Ann")
        )
        with self.assertRaises(ValueError):
            calibrate_ngram_control(control, [_doc("test", "test", "Ann", [(0, 3)])])
        calibrated = calibrate_ngram_control(control, validation)
        self.assertEqual(
            calibrated["control_kind"],
            "new_span_adapter_to_existing_ngram_string_memory",
        )
        self.assertEqual(calibrated["calibration_split"], "validation")
        self.assertEqual(calibrated["validation_spans"]["true_positive"], 1)
        self.assertEqual(
            score_ngram_control(control, validation, calibrated)["selected_spans"],
            calibrated["validation_spans"],
        )
        with self.assertRaises(ValueError):
            score_ngram_control(
                control, [_doc("test", "test", "Ann", [(0, 3)])], calibrated
            )

    def test_long_document_benchmark_has_independent_worker_timeout_and_no_gold(self):
        docs = [
            _doc("short", "validation", "A B", [(0, 1)]),
            _doc("long", "validation", "A B C D", [(0, 1)]),
        ]
        with patch.object(
            diagnostic_script.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["python"], 1),
        ) as call:
            report = diagnostic_script._measure_long_documents(
                Path("model.json"), docs, maximum=1, timeout=1
            )
        self.assertEqual(report["requested_documents"], 1)
        self.assertEqual(report["completed_documents"], 0)
        self.assertEqual(report["measurements"][0]["document_id"], "long")
        self.assertEqual(report["measurements"][0]["status"], "timed_out")
        self.assertEqual(
            json.loads(call.call_args.kwargs["input"]), {"text": "A B C D"}
        )

    def test_public_input_rejects_closed_key_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public.json"
            public.write_text(
                json.dumps(
                    {
                        "schema": "ai2-p03-public-splits-v1",
                        "train": [],
                        "validation": [],
                        "test": [],
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(
                    diagnostic_script,
                    "load_bundle",
                    side_effect=AssertionError("model opened"),
                ),
                self.assertRaisesRegex(ValueError, "only train and validation"),
            ):
                diagnostic_script.diagnose(public, Path("model.json"))

    def test_public_report_uses_frozen_split_fingerprints(self):
        train = [_doc("train-a", "train", "Ann met Bob", [(0, 3), (8, 11)])]
        validation = [_doc("validation-a", "validation", "Ann saw Bob", [(0, 3)])]
        payload = {
            "provenance": {
                "spec": {
                    "train_fingerprint": fingerprint(train),
                    "validation_fingerprint": fingerprint(validation),
                }
            },
            "policy": {"mention_threshold": 0.5},
            "model": {"source": "synthetic-fixture"},
        }
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "public.json"

            def write(documents: list[dict]) -> None:
                public.write_text(
                    json.dumps(
                        {
                            "schema": "ai2-p03-public-splits-v1",
                            "train": train,
                            "validation": documents,
                        }
                    ),
                    encoding="utf-8",
                )

            write(validation)
            with (
                patch.object(
                    diagnostic_script, "load_bundle", return_value=(_Model(), payload)
                ),
                patch.object(
                    diagnostic_script, "_measure_long_documents", return_value={}
                ),
            ):
                report = diagnostic_script.diagnose(public, Path("fixture-model.json"))
                self.assertEqual(report["splits_read"], ["train", "validation"])
                self.assertFalse(report["test_split_accessed"])
                self.assertEqual(report["validation_sha256"], fingerprint(validation))
                self.assertEqual(
                    report["ngram_span_adapter"]["calibration_split"], "validation"
                )
                corrupted = json.loads(json.dumps(validation))
                corrupted[0]["mentions"][0]["end"] = 2
                write(corrupted)
                with self.assertRaisesRegex(ValueError, "frozen model"):
                    diagnostic_script.diagnose(public, Path("fixture-model.json"))


if __name__ == "__main__":
    unittest.main()

"""Tests for the frozen Block-1 mention/coreference quality gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from text_factors.observations.learning import LearningConfig
from text_factors.observations.learning_commands import train_stages
from text_factors.observations.quality_gate import (
    evaluate_quality_gate,
    freeze_quality_gate,
    validate_quality_gate,
)


def _document(
    document_id: str,
    split: str,
    text: str,
    mentions: list[tuple[int, int, str | None]],
) -> dict:
    return {
        "document_id": document_id,
        "group_id": f"group-{document_id}",
        "split": split,
        "language": "en",
        "coverage": "complete",
        "text": text,
        "provenance": {
            "source_url": f"https://example.invalid/{document_id}",
            "license": "test-only",
            "annotation_origin": "unit-test",
        },
        "mentions": [
            {"start": start, "end": end, "entity_id": entity}
            for start, end, entity in mentions
        ],
    }


def _corpus() -> dict:
    train_text = "Alice saw Bob. Alice waved."
    validation_text = "Carol met Dan. Carol smiled."
    test_text = "X17 met X18. X17 moved."
    return {
        "schema": "ai2-mention-learning-corpus-v1",
        "provenance": {"annotation_policy": "complete mentions for unit tests"},
        "documents": [
            _document(
                "train-1",
                "train",
                train_text,
                [
                    (0, 5, "alice"),
                    (10, 13, "bob"),
                    (15, 20, "alice"),
                ],
            ),
            _document(
                "validation-1",
                "validation",
                validation_text,
                [
                    (0, 5, "carol"),
                    (10, 13, "dan"),
                    (15, 20, "carol"),
                ],
            ),
            _document(
                "test-1",
                "test",
                test_text,
                [
                    (0, 3, "x17-a"),
                    (8, 11, "x18"),
                    (13, 16, "x17-a"),
                ],
            ),
        ],
    }


class Block1QualityGateTests(unittest.TestCase):
    def test_gate_freeze_is_reproducible_and_tamper_evident(self):
        corpus = _corpus()
        gate = freeze_quality_gate(
            corpus,
            requirements={
                "max_unsupported_gold_rate": 1.0,
                "min_unseen_surface_recall": 0.0,
                "max_same_surface_false_merges": 100,
                "min_accepted_link_precision": 0.0,
                "min_accepted_link_decisions": 0,
                "min_mention_f1_delta_from_baseline": -1.0,
                "min_coreference_f1_delta_from_baseline": -1.0,
            },
        )
        validate_quality_gate(gate, corpus)
        same = freeze_quality_gate(
            corpus,
            requirements=gate["requirements"],
        )
        self.assertEqual(gate, same)
        changed = {**gate, "test_predictions_seen_before_freeze": True}
        with self.assertRaisesRegex(ValueError, "changed after freeze"):
            validate_quality_gate(changed, corpus)

    def test_gate_rejects_corpus_change_after_freeze(self):
        corpus = _corpus()
        gate = freeze_quality_gate(corpus)
        changed = _corpus()
        changed["documents"][-1]["text"] = "X17 met X18. X18 moved."
        with self.assertRaisesRegex(ValueError, "frozen corpus"):
            validate_quality_gate(gate, changed)

    def test_quality_gate_evaluates_frozen_bundle_without_test_tuning(self):
        corpus = _corpus()
        requirements = {
            "max_unsupported_gold_rate": 1.0,
            "min_unseen_surface_recall": 0.0,
            "max_same_surface_false_merges": 100,
            "min_accepted_link_precision": 0.0,
            "min_accepted_link_decisions": 0,
            "min_mention_f1_delta_from_baseline": -1.0,
            "min_coreference_f1_delta_from_baseline": -1.0,
        }
        gate = freeze_quality_gate(corpus, requirements=requirements)
        config = LearningConfig(
            seed=17,
            epochs=2,
            feature_dim=256,
            max_char_ngrams=16,
            max_span_tokens=4,
            max_antecedents=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_stages(corpus, output, config)
            report = evaluate_quality_gate(output / "model.json", corpus, gate)
        self.assertEqual(report["schema"], "ai2-block1-quality-gate-report-v1")
        self.assertTrue(report["passed"])
        self.assertFalse(report["thresholds_changed_on_test"])
        self.assertEqual(report["test_fingerprint"], gate["test_fingerprint"])
        self.assertIn("unseen_surfaces", report["slices"])
        self.assertIn("same_surface_multiple_entities", report["slices"])

    def test_default_gate_can_fail_honestly_without_mutating_requirements(self):
        corpus = _corpus()
        gate = freeze_quality_gate(corpus)
        config = LearningConfig(
            seed=17,
            epochs=2,
            feature_dim=256,
            max_char_ngrams=16,
            max_span_tokens=4,
            max_antecedents=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_stages(corpus, output, config)
            report = evaluate_quality_gate(output / "model.json", corpus, gate)
        self.assertFalse(report["passed"])
        self.assertTrue(report["failure_reasons"])
        self.assertEqual(report["requirements"], gate["requirements"])


if __name__ == "__main__":
    unittest.main()

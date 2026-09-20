"""Tests for the frozen Block 1 mention/coreference quality gate."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from text_factors.observations.block1_gate import (
    Block1GateConfig,
    evaluate_block1_gate,
)


class Block1QualityGateTests(unittest.TestCase):
    def test_selection_is_fixed_disjoint_from_original_pilot(self):
        root = Path(__file__).resolve().parents[1]
        selection = json.loads(
            (root / "docs" / "GUM_BLOCK1_GATE_SELECTION.json").read_text(
                encoding="utf-8"
            )
        )
        pilot = json.loads(
            (root / "docs" / "GUM_PILOT_MANIFEST.json").read_text(encoding="utf-8")
        )
        split_ids = selection["splits"]
        self.assertEqual(
            {split: len(values) for split, values in split_ids.items()},
            {"train": 12, "validation": 4, "test": 4},
        )
        selected = {
            document_id for values in split_ids.values() for document_id in values
        }
        self.assertEqual(len(selected), 20)
        self.assertFalse(
            selected & {document["document_id"] for document in pilot["documents"]}
        )
        self.assertTrue(
            all(value.startswith(("GUM_academic_", "GUM_bio_")) for value in selected)
        )

    def test_gate_passes_only_when_all_predeclared_criteria_pass(self):
        policy = {
            "link_gate": {
                "enabled": True,
                "validation_empirical_precision": 0.95,
                "validation_evaluable_decisions": 20,
            }
        }
        metrics = {
            "mentions": {
                "gold_count": 100,
                "unsupported_gold_spans": 2,
            },
            "accepted_links": {
                "precision": 0.9,
                "evaluable_accepted_count": 20,
            },
            "coreference_oracle_mentions": {"f1": 0.6},
            "baseline": {
                "coreference_oracle_mentions": {"f1": 0.5},
            },
            "comparison": {
                "mention_f1_delta_from_baseline": 0.05,
            },
        }
        report = evaluate_block1_gate(policy, metrics)
        self.assertTrue(report["passed"])
        self.assertTrue(all(item["passed"] for item in report["criteria"].values()))

        failed = json.loads(json.dumps(metrics))
        failed["accepted_links"]["precision"] = 0.5
        report = evaluate_block1_gate(policy, failed)
        self.assertFalse(report["passed"])
        self.assertFalse(report["criteria"]["test_accepted_links"]["passed"])

    def test_gate_config_is_strictly_validated(self):
        with self.assertRaises(ValueError):
            Block1GateConfig(max_unsupported_gold_rate=1.5)
        with self.assertRaises(ValueError):
            Block1GateConfig(min_test_evaluable_links=-1)


if __name__ == "__main__":
    unittest.main()

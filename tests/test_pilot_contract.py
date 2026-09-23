"""P01 gate tests: an honest draft passes, false readiness and leakage fail."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_pilot_contract import REQUIRED_DECISIONS, validate_contract

CONTRACT = Path(__file__).resolve().parents[1] / "docs/real_data/pilot_contract.yaml"


class PilotContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_real_draft_is_valid_but_cannot_claim_finalization(self):
        errors, blockers = validate_contract(self.contract)
        self.assertEqual(errors, [])
        self.assertIn("independence.sealed_test_source_reference", blockers)
        self.assertIn("scope.data_controller", blockers)
        self.assertIn("evaluation.block1_quality_gates_reference", blockers)
        self.contract["status"] = "finalized"
        errors, _ = validate_contract(self.contract)
        self.assertIn("finalized contract still has unresolved decisions", errors)

    def test_sealed_evaluation_cannot_reuse_disclosed_corpus(self):
        self.contract["independence"]["sealed_test_source_reference"] = (
            "docs/GUM_PILOT_MANIFEST.json"
        )
        errors, _ = validate_contract(self.contract)
        self.assertIn("already disclosed GUM cannot be a new sealed_test", errors)

    def test_metrics_fail_if_abstentions_or_denominators_are_hidden(self):
        contract = copy.deepcopy(self.contract)
        contract["evaluation"]["useful_coverage"]["denominator"] = (
            "only_answers_the_model_chose_to_accept"
        )
        contract["evaluation"]["unsupported_answer_risk"]["zero_denominator"] = (
            "zero_risk"
        )
        errors, _ = validate_contract(contract)
        self.assertIn("useful_coverage: unexpected denominator", errors)
        self.assertIn("zero accepted answers cannot imply zero risk", errors)

    def test_independence_requires_all_groups_and_separate_calibration(self):
        self.contract["independence"]["splits"] = ["train", "test"]
        self.contract["evaluation"]["unsupported_answer_risk"]["trial_unit"] = (
            "all_questions_from_same_task"
        )
        errors, _ = validate_contract(self.contract)
        self.assertIn("all five ordered evaluation splits are required", errors)
        self.assertIn("risk: correlated queries cannot be binomial trials", errors)

    def test_a_synthetic_complete_contract_can_finalize_without_model_training(self):
        contract = copy.deepcopy(self.contract)
        contract["status"] = "finalized"
        for path in REQUIRED_DECISIONS:
            target = contract
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = 1 if path[-1].endswith(("gib", "seconds")) else (
                "synthetic-example-reference"
            )
        for source in contract["data"]["sources"]:
            source["status"] = "available"
            source["location_reference"] = "synthetic-example-reference"
            source["rights_evidence_reference"] = "synthetic-example-reference"
            source["collection_owner"] = "synthetic-example-reference"
        errors, blockers = validate_contract(contract)
        self.assertEqual((errors, blockers), ([], []))


if __name__ == "__main__":
    unittest.main()

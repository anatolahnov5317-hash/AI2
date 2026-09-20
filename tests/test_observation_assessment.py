"""Split discipline, conservative linking, and honest end-to-end denominators."""

from __future__ import annotations

import re
import unittest
from copy import deepcopy
from dataclasses import asdict, dataclass

from text_factors.observations.assessment import (
    _apply_gate,
    calibrate_model,
    evaluate_model,
    propose,
    validate_policy,
)


@dataclass
class StubConfig:
    max_antecedents: int = 3


class StubModel:
    def __init__(self, *, pair_enabled=True, spans=None, tied=False):
        self.config = StubConfig()
        self.training_summary = {
            "pair_training_enabled": pair_enabled,
            "train_document_ids": ["train-1"],
            "train_group_ids": ["train-group"],
        }
        self.overrides = spans or {}
        self.tied = tied
        self.span_calls = 0
        self.link_calls = 0

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "training_summary": self.training_summary,
            "overrides": self.overrides,
            "tied": self.tied,
        }

    def span_scores(self, text):
        self.span_calls += 1
        return deepcopy(
            self.overrides.get(
                text,
                [
                    {"start": match.start(), "end": match.end(), "score": 0.99}
                    for match in re.finditer(r"\S+", text)
                ],
            )
        )

    def link_score(self, text, left, right):
        self.link_calls += 1
        # Gold identities must not reach the inference interface.
        assert set(left) <= {"start", "end", "score"}
        assert set(right) <= {"start", "end", "score"}
        if not self.training_summary["pair_training_enabled"]:
            return 0.5
        if self.tied:
            return 0.98
        return 0.98 if left["end"] + 1 == right["start"] else 0.1


def document(text, split, entities=None, mentions=None):
    spans = list(re.finditer(r"\S+", text))
    if mentions is None:
        mentions = [
            {
                "start": match.start(),
                "end": match.end(),
                "entity_id": entities[index] if entities is not None else "entity",
            }
            for index, match in enumerate(spans)
        ]
    return {
        "document_id": f"{split}-1",
        "group_id": f"{split}-group",
        "split": split,
        "text": text,
        "mentions": mentions,
        "language": "en",
    }


def validation(count=12):
    return document(" ".join(["a"] * count), "validation")


def training():
    return [document("a b c", "train", ["a", "b", "c"])]


class ObservationAssessmentTests(unittest.TestCase):
    def test_calibration_requires_validation_and_evaluation_requires_test(self):
        model = StubModel()
        for split in ("train", "test"):
            with self.assertRaisesRegex(ValueError, "only validation"):
                calibrate_model(model, [document("a", split)])
        policy = calibrate_model(model, [validation()])
        with self.assertRaisesRegex(ValueError, "only test"):
            evaluate_model(model, [validation()], policy, training())
        with self.assertRaisesRegex(ValueError, "only train"):
            evaluate_model(model, [document("a", "test")], policy, [validation()])

    def test_gate_needs_ten_evaluable_decisions_and_empirical_precision(self):
        model = StubModel()
        insufficient = calibrate_model(model, [validation(10)])
        self.assertFalse(insufficient["link_gate"]["enabled"])
        enough = calibrate_model(model, [validation(11)])
        self.assertTrue(enough["link_gate"]["enabled"])
        self.assertEqual(enough["link_gate"]["validation_evaluable_decisions"], 10)
        self.assertEqual(enough["link_gate"]["validation_empirical_precision"], 1.0)
        self.assertFalse(enough["link_gate"]["statistical_guarantee"])
        bad_labels = validation(11)
        for index, mention in enumerate(bad_labels["mentions"]):
            mention["entity_id"] = str(index)
        self.assertFalse(calibrate_model(model, [bad_labels])["link_gate"]["enabled"])

    def test_disabled_pair_training_never_auto_links_and_keeps_alternatives(self):
        model = StubModel(pair_enabled=False)
        policy = calibrate_model(model, [validation(20)])
        self.assertEqual(policy["link_gate"]["reason"], "pair_training_disabled")
        result = propose(model, "a b c d", policy)
        self.assertTrue(all(row["selected"] is None for row in result["mentions"]))
        self.assertEqual(len(result["mentions"][-1]["candidates"]), 3)
        self.assertEqual(result["mentions"][0]["candidates"], [])
        self.assertFalse(result["archive_mutation"])
        self.assertFalse(result["entity_creation"])

    def test_equal_scores_preserve_ambiguity_and_stable_recency_tie_break(self):
        model = StubModel(tied=True)
        # Only the first candidate-bearing row has a positive margin.
        policy = calibrate_model(model, [validation(30)])
        self.assertFalse(policy["link_gate"]["enabled"])
        result = propose(model, "a b c d", policy)
        last = result["mentions"][-1]
        self.assertEqual(last["margin"], 0.0)
        self.assertEqual(
            [candidate["mention_id"] for candidate in last["candidates"]],
            ["m000002", "m000001", "m000000"],
        )
        self.assertIsNone(last["selected"])

    def test_nested_spans_are_retained(self):
        text = "a b"
        model = StubModel(
            pair_enabled=False,
            spans={
                text: [
                    {"start": 0, "end": 1, "score": 0.99},
                    {"start": 0, "end": 3, "score": 0.99},
                    {"start": 2, "end": 3, "score": 0.99},
                ]
            },
        )
        policy = calibrate_model(model, [validation()])
        result = propose(model, text, policy)
        self.assertEqual(
            [(row["start"], row["end"]) for row in result["mentions"]],
            [(0, 1), (0, 3), (2, 3)],
        )
        self.assertNotIn("entity_id", str(result))

    def test_link_gate_can_require_high_confidence_mention_endpoints(self):
        rows = [
            {
                "mention_id": "m000000",
                "start": 0,
                "end": 1,
                "score": 0.99,
                "surface": "a",
                "candidates": [],
                "considered_antecedents": 0,
                "margin": None,
            },
            {
                "mention_id": "m000001",
                "start": 2,
                "end": 3,
                "score": 0.70,
                "surface": "b",
                "candidates": [
                    {
                        "mention_id": "m000000",
                        "start": 0,
                        "end": 1,
                        "score": 0.99,
                    }
                ],
                "considered_antecedents": 1,
                "margin": 0.99,
            },
            {
                "mention_id": "m000002",
                "start": 4,
                "end": 5,
                "score": 0.99,
                "surface": "c",
                "candidates": [
                    {
                        "mention_id": "m000000",
                        "start": 0,
                        "end": 1,
                        "score": 0.99,
                    }
                ],
                "considered_antecedents": 2,
                "margin": 0.99,
            },
        ]
        gated = _apply_gate(
            rows,
            {
                "enabled": True,
                "score_threshold": 0.9,
                "margin_threshold": 0.1,
                "mention_score_threshold": 0.9,
            },
        )
        self.assertIsNone(gated[1]["selected"])
        self.assertEqual(gated[2]["selected"], "m000000")

    def test_calibration_records_oracle_link_diagnostics(self):
        model = StubModel()
        policy = calibrate_model(model, [validation(11)])
        self.assertIn("oracle_link_grid", policy)
        self.assertTrue(policy["oracle_link_grid"])
        if policy["link_gate"]["enabled"]:
            self.assertIn("mention_score_threshold", policy["link_gate"])

    def test_fixed_grid_caches_scores_and_respects_antecedent_budget(self):
        model = StubModel()
        model.config.max_antecedents = 2
        calibrate_model(model, [validation(11)])
        self.assertEqual(model.span_calls, 1)
        self.assertEqual(model.link_calls, 19)

    def test_test_labels_never_change_policy_or_text_only_proposals(self):
        model = StubModel()
        policy = calibrate_model(model, [validation()])
        original_policy = deepcopy(policy)
        before = propose(model, "a b c", policy)
        first = evaluate_model(
            model, [document("a b c", "test", ["x", "y", "x"])], policy, training()
        )
        second = evaluate_model(
            model, [document("a b c", "test", ["x", "x", "x"])], policy, training()
        )
        self.assertEqual(policy, original_policy)
        self.assertEqual(propose(model, "a b c", policy), before)
        self.assertEqual(first["coreference_end_to_end"]["false_merge_pairs"], 2)
        self.assertEqual(first["coreference_end_to_end"]["true_positive"], 1)
        self.assertEqual(first["accepted_links"]["precision"], 0.0)
        self.assertEqual(second["coreference_end_to_end"]["false_merge_pairs"], 0)

    def test_missing_mentions_count_as_false_negatives_and_false_splits(self):
        model = StubModel(spans={"a b c": [{"start": 0, "end": 1, "score": 0.99}]})
        policy = calibrate_model(model, [validation()])
        metrics = evaluate_model(
            model, [document("a b c", "test", ["x", "x", "x"])], policy, training()
        )
        self.assertEqual(metrics["mentions"]["false_negative"], 2)
        self.assertEqual(metrics["mentions"]["unsupported_gold_spans"], 2)
        self.assertEqual(metrics["coreference_end_to_end"]["false_split_pairs"], 3)
        self.assertEqual(metrics["coreference_oracle_mentions"]["false_split_pairs"], 0)

    def test_unknown_gold_is_excluded_only_from_identity_denominators(self):
        model = StubModel()
        policy = calibrate_model(model, [validation()])
        metrics = evaluate_model(
            model, [document("a b c", "test", ["x", None, "x"])], policy, training()
        )
        self.assertEqual(metrics["mentions"]["gold_count"], 3)
        self.assertEqual(metrics["coreference_end_to_end"]["pair_count"], 1)
        self.assertEqual(
            metrics["coreference_end_to_end"]["unknown_gold_pairs_excluded"], 2
        )
        self.assertEqual(metrics["accepted_links"]["unknown_gold_excluded_count"], 2)
        self.assertIsNone(metrics["accepted_links"]["precision"])

    def test_baseline_learns_only_train_surfaces_and_reports_model_underperformance(
        self,
    ):
        model = StubModel(spans={"a a unseen": []})
        policy = calibrate_model(model, [validation()])
        metrics = evaluate_model(
            model,
            [document("a a unseen", "test", ["x", "x", "z"])],
            policy,
            training(),
        )
        self.assertEqual(metrics["baseline"]["learned_surface_count"], 3)
        self.assertEqual(metrics["baseline"]["mentions"]["true_positive"], 2)
        self.assertEqual(metrics["baseline"]["mentions"]["false_negative"], 1)
        self.assertTrue(metrics["comparison"]["mention_below_baseline"])
        self.assertTrue(metrics["comparison"]["coreference_below_baseline"])

    def test_frozen_model_and_policy_tampering_is_rejected(self):
        model = StubModel()
        policy = calibrate_model(model, [validation()])
        changed = deepcopy(policy)
        changed["mention_threshold"] = 0.0
        with self.assertRaisesRegex(ValueError, "policy changed"):
            validate_policy(model, changed)
        with self.assertRaisesRegex(ValueError, "policy changed"):
            propose(model, "a", changed)
        model.tied = True
        with self.assertRaisesRegex(ValueError, "model changed"):
            evaluate_model(model, [document("a", "test")], policy, training())

    def test_caller_can_disable_selection_without_discarding_candidates(self):
        model = StubModel()
        events = []
        policy = calibrate_model(model, [validation()], progress=events.append)
        self.assertTrue(policy["link_gate"]["enabled"])
        result = propose(model, "a b c", policy, allow_selection=False)
        self.assertFalse(result["selection_allowed"])
        self.assertTrue(all(row["selected"] is None for row in result["mentions"]))
        self.assertEqual(len(result["mentions"][-1]["candidates"]), 2)
        self.assertTrue(policy["link_gate"]["enabled"])
        evaluate_model(
            model,
            [document("a b", "test")],
            policy,
            training(),
            progress=events.append,
        )
        self.assertEqual(
            [event["phase"] for event in events],
            ["validation_spans", "validation_links", "test_evaluation"],
        )
        self.assertTrue(
            all(event["completed"] == event["total"] == 1 for event in events)
        )

    def test_split_group_overlap_is_rejected(self):
        model = StubModel()
        overlapping = validation()
        overlapping["group_id"] = "train-group"
        with self.assertRaisesRegex(ValueError, "training overlap"):
            calibrate_model(model, [overlapping])
        policy = calibrate_model(model, [validation()])
        test = document("a", "test")
        test["group_id"] = "validation-group"
        with self.assertRaisesRegex(ValueError, "overlap across splits"):
            evaluate_model(model, [test], policy, training())


if __name__ == "__main__":
    unittest.main()

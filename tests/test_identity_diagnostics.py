"""Diagnose homonyms, Russian inflection, long links and sealed split isolation."""

from __future__ import annotations

import re
import unittest
from copy import deepcopy
from dataclasses import asdict, dataclass

from text_factors.observations.assessment import calibrate_model, propose
from text_factors.observations.identity_diagnostics import (
    _training_digest,
    diagnose_identity,
)
from text_factors.observations.learning import train_model


@dataclass
class Config:
    max_antecedents: int = 2


class LocalModel:
    def __init__(self, *, window: int = 2, pair_enabled: bool = False):
        self.config = Config(window)
        self.training_summary = {
            "pair_training_enabled": pair_enabled,
            "train_document_ids": ["train-1"],
            "train_group_ids": ["train-family"],
        }
        self.score_calls = 0

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "training_summary": self.training_summary,
        }

    def span_scores(self, text):
        return [
            {"start": match.start(), "end": match.end(), "score": 0.99}
            for match in re.finditer(r"\S+", text)
        ]

    def link_score(self, text, left, right):
        self.score_calls += 1
        # Gold identity and annotation provenance must stay outside inference.
        assert set(left) == set(right) == {"start", "end", "score"}
        spans = list(re.finditer(r"\S+", text))
        right_index = next(
            i for i, span in enumerate(spans) if span.start() == right["start"]
        )
        left_index = next(
            i for i, span in enumerate(spans) if span.start() == left["start"]
        )
        return 0.99 if right_index - left_index == 1 else 0.1


def document(text, entities, split):
    tokens = list(re.finditer(r"\S+", text))
    assert len(tokens) == len(entities)
    return {
        "document_id": f"{split}-1",
        "group_id": f"{split}-family",
        "split": split,
        "coverage": "complete",
        "text": text,
        "mentions": [
            {"start": token.start(), "end": token.end(), "entity_id": entity}
            for token, entity in zip(tokens, entities, strict=True)
        ],
    }


def bind_training(model, train):
    model.training_summary["train_data_sha256"] = _training_digest([train])


class IdentityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.validation = document("а " * 12, ["same"] * 12, "validation")

    def test_russian_forms_homonyms_unknown_and_distant_mentions(self):
        model = LocalModel()
        train = document(
            "Иван Ивана Иван кто Машу Ивану",
            ["person-a", "person-a", "person-b", None, "person-c", "person-a"],
            "train",
        )
        bind_training(model, train)
        policy = calibrate_model(model, [self.validation])
        original = deepcopy(policy)
        diagnostic = diagnose_identity(model, [train], policy, split="train")
        counts = diagnostic["counts"]
        self.assertEqual(counts["gold_mentions"], 6)
        self.assertEqual(counts["unknown_identity_mentions"], 1)
        self.assertEqual(counts["same_surface_different_entity_pairs"], 1)
        self.assertEqual(counts["different_surface_same_entity_pairs"], 3)
        self.assertEqual(counts["different_surface_false_split_pairs"], 3)
        self.assertEqual(counts["false_merge_pairs"], 0)
        self.assertEqual(counts["anaphoric_known_mentions"], 2)
        self.assertEqual(counts["distant_only_mentions"], 1)
        self.assertEqual(counts["gold_antecedent_inside_window"], 1)
        self.assertEqual(counts["oracle_top3_retrieval_hits"], 1)
        self.assertEqual(diagnostic["oracle_top3_recall"], 0.5)
        self.assertFalse(diagnostic["link_gate_enabled"])
        self.assertEqual(policy, original)

    def test_supported_gate_reports_homonym_merges_without_retuning(self):
        model = LocalModel(pair_enabled=True)
        text = "Иван Иван Иван"
        train = document(text, ["first", "second", "second"], "train")
        bind_training(model, train)
        policy = calibrate_model(model, [self.validation])
        self.assertTrue(policy["link_gate"]["enabled"])
        before = propose(model, text, policy)
        report = diagnose_identity(model, [train], policy, split="train")
        counts = report["counts"]
        self.assertEqual(counts["same_surface_different_entity_pairs"], 2)
        self.assertEqual(counts["same_surface_false_merge_pairs"], 2)
        self.assertEqual(counts["false_merge_pairs"], 2)
        self.assertEqual(counts["false_split_pairs"], 0)
        self.assertEqual(before, propose(model, text, policy))

    def test_true_antecedent_inside_window_can_be_lost_from_top_three(self):
        model = LocalModel(window=4)
        train = document("А Б В Г А", ["a", "b", "c", "d", "a"], "train")
        bind_training(model, train)
        policy = calibrate_model(model, [self.validation])
        report = diagnose_identity(model, [train], policy, split="train")
        counts = report["counts"]
        self.assertEqual(counts["distant_only_mentions"], 0)
        self.assertEqual(counts["gold_antecedent_inside_window"], 1)
        self.assertEqual(counts["oracle_top3_ranking_misses"], 1)
        self.assertEqual(counts["oracle_top3_retrieval_hits"], 0)

    def test_only_registered_train_and_frozen_validation_can_be_diagnosed(self):
        model = LocalModel()
        train = document("Иван Ивана", ["a", "a"], "train")
        bind_training(model, train)
        policy = calibrate_model(model, [self.validation])
        diagnostic = diagnose_identity(
            model, [self.validation], policy, split="validation"
        )
        self.assertEqual(diagnostic["split"], "validation")
        changed = deepcopy(self.validation)
        changed["mentions"][0]["entity_id"] = "other"
        with self.assertRaisesRegex(ValueError, "differ from frozen calibration"):
            diagnose_identity(model, [changed], policy, split="validation")
        with self.assertRaisesRegex(ValueError, "only train or validation"):
            diagnose_identity(
                model, [document("x", ["x"], "test")], policy, split="test"
            )
        altered_gold = deepcopy(train)
        altered_gold["mentions"][0]["entity_id"] = "b"
        with self.assertRaisesRegex(ValueError, "model training fingerprint"):
            diagnose_identity(model, [altered_gold], policy, split="train")
        with self.assertRaisesRegex(ValueError, "full corpus"):
            diagnose_identity(
                model,
                [document("x", ["x"], "train") | {"document_id": "alien"}],
                policy,
                split="train",
            )

    def test_training_fingerprint_matches_actual_learner(self):
        train = document("Иван Ивана Иван", ["a", "a", "b"], "train")
        model = train_model([train])
        self.assertEqual(
            _training_digest([train]), model.training_summary["train_data_sha256"]
        )


if __name__ == "__main__":
    unittest.main()

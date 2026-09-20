"""Candidate learning: supervision boundaries, open input, offsets and budgets."""

from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace
from typing import Any

from text_factors.observations.learning import (
    CandidateModel,
    LearningConfig,
    train_model,
)


def _document(
    document_id: str, text: str, mentions: list[tuple[int, int, str | None]]
) -> dict:
    return {
        "document_id": document_id,
        "group_id": f"group-{document_id}",
        "split": "train",
        "coverage": "complete",
        "text": text,
        "mentions": [
            {"start": start, "end": end, "entity_id": entity}
            for start, end, entity in mentions
        ],
    }


def _training() -> list[dict]:
    # Both equal and unequal surfaces can denote the same or different instance.
    return [
        _document(
            "one",
            "Мира видит Мира рядом с ней.",
            [(0, 4, "a"), (11, 15, "b"), (24, 27, "a")],
        ),
        _document(
            "two",
            "Лена видит Лену рядом с ним.",
            [(0, 4, "a"), (11, 15, "a"), (24, 27, "b")],
        ),
        _document(
            "three",
            "Мира видит Мира рядом с ним.",
            [(0, 4, "c"), (11, 15, "c"), (24, 27, "d")],
        ),
    ]


class ObservationLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LearningConfig(feature_dim=256, epochs=8, max_char_ngrams=24)
        cls.model = train_model(_training(), cls.config)

    def test_unseen_unicode_names_and_nested_spans_keep_original_offsets(self):
        text = "\ufeffÆlvíra e\u0301\r\n王明 😀"
        scores = self.model.span_scores(text)
        spans = {(item["start"], item["end"]): item["score"] for item in scores}
        for surface in ("Ælvíra", "e\u0301", "王明", "😀", "Ælvíra e\u0301"):
            start = text.index(surface)
            self.assertIn((start, start + len(surface)), spans)
        self.assertTrue(
            all(math.isfinite(score) and 0 <= score <= 1 for score in spans.values())
        )
        self.assertEqual(len(spans), len(scores))
        # The unmodified source coordinate still refers to decomposed Unicode.
        start = text.index("e")
        self.assertEqual(text[start : start + 2], "e\u0301")

    def test_supervision_changes_scores_and_json_roundtrip_is_exact(self):
        scores = self.model.span_scores("Мира видит Мира рядом с ней.")
        by_span = {(item["start"], item["end"]): item["score"] for item in scores}
        self.assertGreater(by_span[(0, 4)], by_span[(5, 10)])
        self.assertNotEqual(by_span[(0, 4)], 0.5)
        payload = json.loads(json.dumps(self.model.to_dict(), allow_nan=False))
        restored = CandidateModel.from_dict(payload)
        self.assertEqual(restored.to_dict(), self.model.to_dict())
        self.assertEqual(
            restored.span_scores("Новая Ωmega 王明"),
            self.model.span_scores("Новая Ωmega 王明"),
        )
        text = "Мира видит Мира рядом с ней."
        self.assertEqual(
            restored.link_score(text, {"start": 0, "end": 4}, {"start": 11, "end": 15}),
            self.model.link_score(
                text, {"start": 0, "end": 4}, {"start": 11, "end": 15}
            ),
        )

    def test_reproducible_sampling_and_training_ignore_document_order(self):
        again = train_model(list(reversed(_training())), self.config)
        self.assertEqual(again.to_dict(), self.model.to_dict())
        summary = self.model.training_summary
        self.assertEqual(summary["train_document_ids"], ["one", "three", "two"])
        self.assertGreater(summary["span_negative_examples"], 0)
        self.assertGreater(summary["pair_positive_examples"], 0)
        self.assertGreater(summary["pair_negative_examples"], 0)

    def test_validation_and_test_documents_are_never_training_inputs(self):
        for split in ("validation", "test", "TRAIN", None):
            docs = _training()
            docs[-1]["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "train"):
                train_model(docs, self.config)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            train_model([_training()[0], _training()[0]], self.config)

    def test_training_requires_explicit_complete_annotation_coverage(self):
        for coverage in ("partial", "unknown", None, True):
            docs = _training()
            docs[-1]["coverage"] = coverage
            with (
                self.subTest(coverage=coverage),
                self.assertRaisesRegex(ValueError, "coverage='complete'"),
            ):
                train_model(docs, self.config)
        docs = _training()
        del docs[-1]["coverage"]
        with self.assertRaisesRegex(ValueError, "coverage='complete'"):
            train_model(docs, self.config)

    def test_inference_ignores_gold_entity_ids_and_equal_surface_is_not_union(self):
        text = "Мира видит Мира рядом с ней."
        left = {"start": 0, "end": 4}
        right = {"start": 11, "end": 15}
        score = self.model.link_score(text, left, right)
        self.assertTrue(0 < score < 1)
        self.assertEqual(
            score,
            self.model.link_score(
                text, {**left, "entity_id": "same"}, {**right, "entity_id": "same"}
            ),
        )
        self.assertEqual(
            score,
            self.model.link_score(
                text,
                {**left, "entity_id": "different"},
                {**right, "entity_id": "other"},
            ),
        )
        # Changing only the arbitrary names of gold IDs does not change weights.
        renamed = _training()
        for document in renamed:
            for mention in document["mentions"]:
                mention["entity_id"] = "renamed-" + mention["entity_id"]
        model = train_model(renamed, self.config)
        self.assertEqual(
            model.to_dict()["link_weights"], self.model.to_dict()["link_weights"]
        )

    def test_unknown_identity_is_not_a_negative_pair_and_one_class_is_disabled(self):
        text = "Мира видит Лена рядом"
        document = _document(
            "unknown", text, [(0, 4, "a"), (11, 15, None), (16, 21, "a")]
        )
        # Fix the last span to the exact remaining source length.
        document["mentions"][-1]["end"] = len(text)
        model = train_model([document], self.config)
        summary = model.training_summary
        self.assertEqual(summary["pair_positive_examples"], 1)
        self.assertEqual(summary["pair_negative_examples"], 0)
        self.assertEqual(summary["unknown_identity_pairs_ignored"], 2)
        self.assertFalse(summary["pair_training_enabled"])
        self.assertEqual(
            model.link_score(text, {"start": 0, "end": 4}, {"start": 11, "end": 15}),
            0.5,
        )

    def test_gold_candidate_representability_and_antecedent_window_are_reported(self):
        text = "One two three"
        doc = _document(
            "ceiling",
            text,
            [(0, 3, "a"), (1, 3, "b"), (0, 13, "a"), (4, 7, "b"), (8, 13, "a")],
        )
        model = train_model(
            [doc], replace(self.config, max_span_tokens=1, max_antecedents=1)
        )
        self.assertEqual(model.training_summary["unsupported_gold_spans"], 2)
        self.assertEqual(model.training_summary["span_positive_examples"], 3)
        self.assertEqual(
            model.training_summary["pair_positive_examples"]
            + model.training_summary["pair_negative_examples"],
            4,
        )

    def test_pair_negative_sampling_retains_eligible_counts_and_all_positives(self):
        text = "A B C D E F"
        mentions: list[tuple[int, int, str | None]] = [
            (position, position + 1, "shared" if position < 3 else str(position))
            for position in range(0, len(text), 2)
        ]
        model = train_model([_document("sample", text, mentions)], self.config)
        summary = model.training_summary
        self.assertEqual(summary["pair_positive_examples"], 1)
        self.assertEqual(summary["pair_eligible_positive_examples"], 1)
        self.assertEqual(summary["pair_eligible_negative_examples"], 14)
        self.assertEqual(summary["pair_negative_examples"], 3)

    def test_identity_conflict_negatives_are_never_sampled_away(self):
        text = "Alex met Alex. Alex saw Alex."
        starts = []
        offset = 0
        while True:
            offset = text.find("Alex", offset)
            if offset < 0:
                break
            starts.append(offset)
            offset += 4
        mentions: list[tuple[int, int, str | None]] = [
            (start, start + 4, entity)
            for start, entity in zip(starts, ("a", "b", "a", "b"), strict=True)
        ]
        model = train_model(
            [_document("same-name", text, mentions)],
            replace(self.config, negative_ratio=1),
        )
        summary = model.training_summary
        self.assertGreater(summary["pair_eligible_critical_negative_examples"], 0)
        self.assertEqual(
            summary["pair_critical_negative_examples"],
            summary["pair_eligible_critical_negative_examples"],
        )
        self.assertGreaterEqual(
            summary["pair_negative_examples"],
            summary["pair_critical_negative_examples"],
        )
        self.assertEqual(
            summary["pair_negative_sampling"],
            "retain_identity_conflicts_then_hard_similarity_distance_v2",
        )

    def test_resource_limits_reject_oversized_inputs_without_silent_truncation(self):
        doc = _document("small", "A B", [(0, 1, "a")])
        model = train_model(
            [doc],
            replace(self.config, max_tokens=2, max_candidates=3, max_document_chars=20),
        )
        self.assertEqual(len(model.span_scores("A B")), 3)
        for text in ("A B C", "a" * 21, "\ud800"):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                model.span_scores(text)
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            train_model([doc], replace(self.config, max_candidates=2))
        with self.assertRaisesRegex(ValueError, "max_training_examples"):
            train_model(_training(), replace(self.config, max_training_examples=2))
        with self.assertRaisesRegex(ValueError, "max_pair_examples"):
            train_model(_training(), replace(self.config, max_pair_examples=1))
        with self.assertRaisesRegex(ValueError, "max_training_tokens"):
            train_model(_training(), replace(self.config, max_training_tokens=2))
        with self.assertRaises(ValueError):
            train_model(_training(), replace(self.config, max_documents=2))

    def test_model_loading_rejects_nonfinite_weights_dimensions_and_tampering(self):
        for invalid in (float("nan"), float("inf"), True, 10**1000):
            payload = self.model.to_dict()
            payload["span_weights"][0] = invalid
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                CandidateModel.from_dict(payload)
        payload = self.model.to_dict()
        payload["link_weights"].pop()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CandidateModel.from_dict(payload)
        payload = self.model.to_dict()
        payload["config"]["max_tokens"] = 10**20
        with self.assertRaises(ValueError):
            CandidateModel.from_dict(payload)
        payload = self.model.to_dict()
        payload["span_weights"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            CandidateModel.from_dict(payload)
        payload = self.model.to_dict()
        payload["gold_labels"] = []
        with self.assertRaises(ValueError):
            CandidateModel.from_dict(payload)

    def test_invalid_offsets_and_boolean_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            invalid: Any = True
            LearningConfig(epochs=invalid)
        for start, end in ((True, 4), (0, 0), (-1, 4), (0, 999)):
            doc = _training()[0]
            doc["mentions"][0].update(start=start, end=end)
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                train_model([doc], self.config)
        with self.assertRaisesRegex(ValueError, "itself"):
            self.model.link_score("A", {"start": 0, "end": 1}, {"start": 0, "end": 1})

    def test_progress_reports_real_training_updates_and_summary_is_a_copy(self):
        events = []
        model = train_model(
            _training(), replace(self.config, epochs=2), progress=events.append
        )
        for phase in ("prepare_examples", "span_training", "pair_training"):
            self.assertTrue(any(event["phase"] == phase for event in events))
        self.assertEqual(events[-1]["completed_updates"], events[-1]["total_updates"])
        summary = model.training_summary
        original = copy.deepcopy(summary)
        summary["train_document_ids"].append("test-leak")
        self.assertEqual(model.training_summary, original)


if __name__ == "__main__":
    unittest.main()

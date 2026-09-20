"""Training coverage, split integrity, and external annotation archive bridge."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from text_factors.observations import ObservationArchive
from text_factors.observations.corpus import validate_corpus
from text_factors.observations.learning_data import (
    LEARNING_CORPUS_SCHEMA,
    archive_learning_corpus,
    corpus_summary,
    fingerprint,
    split_documents,
    validate_learning_corpus,
)


def corpus_fixture():
    texts = ["😀 Арда видит Арду.", "Бора видит Бору.", "Кира видит Киру."]
    documents = []
    for index, (split, text) in enumerate(
        zip(("train", "validation", "test"), texts, strict=True)
    ):
        first = text.index(("Арда", "Бора", "Кира")[index])
        last = text.index(("Арду", "Бору", "Киру")[index])
        documents.append(
            {
                "document_id": f"d{index}",
                "group_id": f"g{index}",
                "split": split,
                "language": "ru",
                "text": text,
                "coverage": "complete",
                "provenance": {
                    "source_url": "synthetic:test",
                    "license": "project test fixture",
                    "annotation_origin": "explicit synthetic fixture",
                },
                "mentions": [
                    {"start": start, "end": start + 4, "entity_id": "one"}
                    for start in (first, last)
                ],
            }
        )
    return {
        "schema": LEARNING_CORPUS_SCHEMA,
        "provenance": {"annotation_policy": "two explicit person mentions per fixture"},
        "documents": documents,
    }


class LearningDataTests(unittest.TestCase):
    def test_exact_content_and_source_group_leakage_are_rejected(self):
        original = corpus_fixture()
        for same_content in (False, True):
            value = deepcopy(original)
            if same_content:
                value["documents"][1]["text"] = value["documents"][0]["text"]
            else:
                value["documents"][1]["group_id"] = value["documents"][0]["group_id"]
            with (
                self.subTest(same_content=same_content),
                self.assertRaisesRegex(ValueError, "leaks"),
            ):
                validate_learning_corpus(value)

    def test_incomplete_coverage_and_bad_coordinates_are_not_training_negatives(self):
        for kind in (
            "coverage",
            "range",
            "bool",
            "surface",
            "duplicate",
            "missing_label",
        ):
            value = corpus_fixture()
            document = value["documents"][0]
            mention = document["mentions"][0]
            if kind == "coverage":
                document["coverage"] = "partial"
            elif kind == "range":
                mention["end"] = len(document["text"]) + 1
            elif kind == "bool":
                mention["start"] = True
            elif kind == "surface":
                mention["surface"] = "wrong"
            elif kind == "duplicate":
                document["mentions"].append(dict(mention))
            else:
                del mention["entity_id"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_learning_corpus(value)

    def test_split_selection_preserves_labels_and_unseen_counts(self):
        corpus = validate_learning_corpus(corpus_fixture())
        before = fingerprint(corpus)
        train = split_documents(corpus, "train")
        summary = corpus_summary(corpus)
        self.assertEqual([doc["document_id"] for doc in train], ["d0"])
        self.assertEqual(summary["splits"]["test"]["unseen_surface_mentions"], 2)
        self.assertFalse(summary["near_duplicate_independence_verified"])
        self.assertEqual(fingerprint(corpus), before)

    def test_archive_bridge_pins_original_coordinates_and_is_idempotent(self):
        corpus = corpus_fixture()
        with (
            TemporaryDirectory() as temporary,
            ObservationArchive(Path(temporary) / "a.sqlite", create=True) as archive,
        ):
            frozen = archive_learning_corpus(archive, corpus, namespace="test")
            self.assertEqual(
                archive_learning_corpus(archive, corpus, namespace="test"), frozen
            )
            self.assertEqual(validate_corpus(archive, frozen)["pinned_annotations"], 6)
            sources = archive.sources("test")
            self.assertEqual(len(sources), 3)
            instances = set()
            for source in sources:
                annotations = archive.annotations(source.source_id, source.version)
                self.assertEqual(
                    annotations[0]["binding"]["selected_id"],
                    annotations[1]["binding"]["selected_id"],
                )
                instances.add(annotations[0]["binding"]["selected_id"])
                self.assertEqual(
                    annotations[0]["binding"]["origin"], "external_annotation"
                )
            self.assertEqual(len(instances), 3)


if __name__ == "__main__":
    unittest.main()

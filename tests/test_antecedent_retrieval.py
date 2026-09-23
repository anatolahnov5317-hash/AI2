"""Bounded lexical retrieval proposes alternatives without merging identities."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from text_factors.observations.antecedent_retrieval import BoundedSurfaceRetrieval
from text_factors.observations.assessment import _ranked

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_identity_retrieval_dev.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_identity_retrieval_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic_script)
_public_documents = diagnostic_script._public_documents


def spans(text: str, words: list[str]) -> list[dict]:
    cursor = 0
    result = []
    for word in words:
        begin = text.index(word, cursor)
        result.append({"start": begin, "end": begin + len(word), "score": 0.99})
        cursor = begin + len(word)
    return result


class FakeModel:
    def __init__(self, budget: int = 4):
        self.config = SimpleNamespace(max_antecedents=budget)

    def link_score(self, text, left, right):
        return (
            0.9
            if text[left["start"] : left["end"]] == text[right["start"] : right["end"]]
            else 0.1
        )


class RetrievalTests(unittest.TestCase):
    def test_distant_exact_surface_retrieved_within_original_budget(self):
        words = ["Книга", "первый", "второй", "третий", "четвёртый", "Книга"]
        text = " ".join(words)
        candidates = spans(text, words)
        index = BoundedSurfaceRetrieval(text, candidates)
        self.assertEqual(index(5, 4), (0, 2, 3, 4))
        self.assertEqual(len(index(5, 4)), 4)
        rows = _ranked(FakeModel(), text, candidates, antecedent_selector=index)
        default = _ranked(FakeModel(), text, candidates)
        self.assertEqual(rows[-1]["candidates"][0]["mention_id"], "m000000")
        self.assertNotIn(
            "m000000", {c["mention_id"] for c in default[-1]["candidates"]}
        )
        self.assertNotIn("selected", rows[-1])

    def test_russian_inflected_forms_are_candidates_only(self):
        words = ["Иван", "промежуток", "ещё", "текст", "Ивана"]
        text = " ".join(words)
        candidates = spans(text, words)
        self.assertEqual(BoundedSurfaceRetrieval(text, candidates)(4, 3), (1, 2, 3))
        self.assertEqual(
            BoundedSurfaceRetrieval(text, candidates, prefix_forms=True)(4, 3),
            (0, 2, 3),
        )

    def test_same_name_multiple_entities_remain_distinct_candidates(self):
        words = ["Иван", "Иван"] + [f"текст{n}" for n in range(9)] + ["Иван"]
        text = " ".join(words)
        candidates = spans(text, words)
        selector = BoundedSurfaceRetrieval(text, candidates)
        self.assertEqual(selector(11, 8), (0, 1, 5, 6, 7, 8, 9, 10))
        # The scorer receives no gold IDs; both homonyms remain separate options.
        ranked = _ranked(FakeModel(8), text, candidates, antecedent_selector=selector)
        previous = {item["mention_id"] for item in ranked[-1]["candidates"]}
        self.assertTrue({"m000000", "m000001"} <= previous)
        self.assertNotIn("selected", ranked[-1])

    def test_prefix_is_bounded_and_unknown_short_names_keep_recency(self):
        words = ["Иван"] * 10 + ["он", "она", "он", "Ивану"]
        text = " ".join(words)
        candidates = spans(text, words)
        selector = BoundedSurfaceRetrieval(text, candidates, prefix_forms=True)
        matched = selector(13, 8)
        self.assertEqual(len(matched), 8)
        self.assertEqual(matched, tuple(sorted(set(matched))))
        self.assertTrue(any(index < 5 for index in matched))
        self.assertEqual(selector(12, 8), tuple(range(4, 12)))
        with self.assertRaises(ValueError):
            selector(len(candidates), 8)
        with self.assertRaises(ValueError):
            selector(12, 0)

    def test_ablation_rejects_sealed_and_overlapping_documents(self):
        public = {
            "schema": "ai2-p03-public-splits-v1",
            "train": [{"split": "train", "document_id": "t", "group_id": "a"}],
            "validation": [
                {"split": "validation", "document_id": "v", "group_id": "b"}
            ],
        }
        self.assertEqual(len(_public_documents(public)[0]), 1)
        with self.assertRaisesRegex(ValueError, "only train and validation"):
            _public_documents({**public, "test": []})
        with self.assertRaisesRegex(ValueError, "forbidden"):
            _public_documents({**public, "validation": [{"split": "test"}]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            _public_documents(
                {
                    **public,
                    "validation": [
                        {"split": "validation", "document_id": "t", "group_id": "b"}
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()

"""Explicit identity is independent of names and never silently inferred."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from text_factors.observations import ArchiveLimits, ObservationArchive
from text_factors.observations.archive import ANNOTATION_SCHEMA


def batch_for(source, text):
    first = text.index("ключ")
    second = text.index("ключ", first + 1)
    return {
        "schema": ANNOTATION_SCHEMA,
        "source_id": source.source_id,
        "source_version": source.version,
        "annotator": "test-human",
        "evidence": "explicit ordinal annotation",
        "instances": [
            {"ref": "first", "external_key": "item-1", "label": "ключ"},
            {"ref": "second", "external_key": "item-2", "label": "ключ"},
        ],
        "mentions": [
            {
                "start": start,
                "end": start + 4,
                "surface": "ключ",
                "candidates": [ref],
                "selected": ref,
                "expected_version": 0,
            }
            for start, ref in ((first, "first"), (second, "second"))
        ],
    }


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-open-identity-")
        self.path = Path(self.directory.name) / "archive.sqlite"
        self.archive = ObservationArchive(
            self.path, create=True, limits=ArchiveLimits(chunk_chars=2)
        )
        self.text = "Первый ключ здесь. Второй ключ там. Его перенесли."
        self.source = self.archive.import_text(
            self.text,
            namespace="user-a",
            external_key="story",
            group_id="story-family",
        )

    def tearDown(self):
        self.archive.close()
        self.directory.cleanup()

    def test_same_name_has_distinct_persistent_instances_and_repeat_is_idempotent(self):
        batch = batch_for(self.source, self.text)
        result = self.archive.annotate(batch)
        first, second = result["instances"]["first"], result["instances"]["second"]
        self.assertEqual(first["label"], second["label"])
        self.assertNotEqual(first["instance_id"], second["instance_id"])
        self.assertEqual(self.archive.annotate(batch), result)
        self.assertEqual(self.archive.verify()["binding_versions"], 2)
        with ObservationArchive(self.path) as restarted:
            self.assertEqual(
                restarted.get_instance(first["instance_id"]).to_dict(), first
            )
            self.assertEqual(
                restarted.annotations(self.source.source_id, 1),
                tuple(result["annotations"]),
            )

    def test_ambiguous_reference_retains_alternatives_and_correction_history(self):
        batch = batch_for(self.source, self.text)
        initial = self.archive.annotate(batch)
        start = self.text.index("Его")
        batch["mentions"] = [
            {
                "start": start,
                "end": start + 3,
                "surface": "Его",
                "candidates": ["first", "second"],
                "selected": None,
                "expected_version": 0,
            }
        ]
        ambiguous = self.archive.annotate(batch)["annotations"][0]
        self.assertEqual(ambiguous["binding"]["status"], "ambiguous")
        self.assertIsNone(ambiguous["binding"]["selected_id"])
        batch["mentions"][0]["selected"] = "second"
        batch["mentions"][0]["expected_version"] = 1
        batch["evidence"] = "speaker explicitly clarified the second instance"
        revised = self.archive.annotate(batch)["annotations"][0]
        self.assertEqual(
            revised["binding"]["selected_id"],
            initial["instances"]["second"]["instance_id"],
        )
        mention_id = revised["mention"]["mention_id"]
        old = self.archive.get_binding(mention_id, 1)
        self.assertIsNotNone(old)
        assert old is not None
        self.assertEqual(old.status, "ambiguous")
        current = self.archive.get_binding(mention_id)
        assert current is not None
        self.assertEqual(current.version, 2)
        self.assertEqual(
            self.archive.read_span(self.source.source_id, 1, start, start + 3), "Его"
        )

    def test_unresolved_mention_does_not_create_an_instance_or_world_fact(self):
        batch = batch_for(self.source, self.text)
        batch["instances"] = []
        batch["mentions"] = [dict(batch["mentions"][0], candidates=[], selected=None)]
        result = self.archive.annotate(batch)
        self.assertEqual(result["instances"], {})
        self.assertEqual(result["annotations"][0]["binding"]["status"], "unresolved")
        self.assertEqual(result["semantic_status"], "externally_annotated")

    def test_invalid_later_annotation_rolls_back_instances_mentions_and_bindings(self):
        batch = batch_for(self.source, self.text)
        batch["mentions"][1]["surface"] = "wrong"
        with self.assertRaisesRegex(ValueError, "surface"):
            self.archive.annotate(batch)
        self.assertEqual(self.archive.annotations(self.source.source_id, 1), ())
        self.assertEqual(self.archive.verify()["binding_versions"], 0)
        # Reusing the proposed keys with other labels would fail if a partial
        # transaction had leaked an instance.
        fixed = batch_for(self.source, self.text)
        fixed["instances"][0]["label"] = "new label"
        self.archive.annotate(fixed)

    def test_stale_writer_cannot_replace_binding_and_batch_is_atomic(self):
        batch = batch_for(self.source, self.text)
        first = self.archive.annotate(batch)
        stale = deepcopy(batch)
        stale["evidence"] = "another writer based on version zero"
        stale["mentions"][0]["candidates"] = ["second"]
        stale["mentions"][0]["selected"] = "second"
        with (
            ObservationArchive(self.path) as other,
            self.assertRaisesRegex(ValueError, "stale"),
        ):
            other.annotate(stale)
        self.assertEqual(
            self.archive.annotations(self.source.source_id, 1),
            tuple(first["annotations"]),
        )

    def test_cross_namespace_reference_is_rejected(self):
        result = self.archive.annotate(batch_for(self.source, self.text))
        other = self.archive.import_text(
            self.text,
            namespace="user-b",
            external_key="story",
            group_id="story-family",
        )
        batch = batch_for(other, self.text)
        batch["instances"][0] = {
            "ref": "first",
            "instance_id": result["instances"]["first"]["instance_id"],
        }
        with self.assertRaisesRegex(ValueError, "namespace"):
            self.archive.annotate(batch)
        self.assertEqual(self.archive.annotations(other.source_id, 1), ())

    def test_revised_source_never_retargets_old_mention_offsets(self):
        result = self.archive.annotate(batch_for(self.source, self.text))
        updated = self.archive.import_text(
            "Добавление. " + self.text,
            namespace="user-a",
            external_key="story",
            group_id="story-family",
        )
        self.assertEqual(updated.version, 2)
        mention = self.archive.get_mention(
            result["annotations"][0]["mention"]["mention_id"]
        )
        self.assertEqual(mention.source_version, 1)
        self.assertEqual(mention.surface, "ключ")
        self.assertEqual(self.archive.annotations(updated.source_id, 2), ())

    def test_annotation_rejects_unpinned_version_unknown_fields_and_duplicate_refs(
        self,
    ):
        good = batch_for(self.source, self.text)
        mutations = []
        for version in (None, True, 0):
            mutations.append({**good, "source_version": version})
        mutations.append({**good, "hidden_teacher": True})
        mutations.append({**good, "instances": [good["instances"][0]] * 2})
        bad = deepcopy(good)
        bad["mentions"][0]["candidates"] = ["first", "first"]
        mutations.append(bad)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.archive.annotate(value)
        self.assertEqual(self.archive.verify()["mentions"], 0)


if __name__ == "__main__":
    unittest.main()

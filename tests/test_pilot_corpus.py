"""Independent P02 corpus gates; all records are local synthetic fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from text_factors.observations import ArchiveLimits, ObservationArchive
from text_factors.observations.corpus import freeze_corpus
from text_factors.observations.pilot_corpus import (
    PILOT_ASSIGNMENTS_SCHEMA,
    freeze_pilot_corpus,
    group_commitment,
    normalized_sha256,
    pilot_training_view,
    screening_sha256,
    validate_pilot_corpus,
)
from text_factors.observations.schema import canonical_json

GROUPS = [f"00000000-0000-4000-8000-{index:012x}" for index in range(1, 12)]


def metadata(*, coverage: str = "complete") -> dict:
    return {
        "origin": "explicit synthetic test",
        "language": "ru",
        "rights_evidence_reference": "test:rights",
        "access_policy_reference": "test:policy",
        "annotation_policy_reference": "test:annotations",
        "grouping_evidence_reference": "test:families",
        "annotation_coverage": {"mentions": coverage, "identity": "partial"},
    }


class PilotCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory(prefix="ai2-p02-")
        self.root = Path(self.directory.name)
        self.archive_path = self.root / "open.sqlite"
        self.archive = ObservationArchive(
            self.archive_path, create=True, limits=ArchiveLimits(chunk_chars=3)
        )
        self.records = [
            self.archive.import_text(
                "Ёж 😀 и ёж.\r\n",
                namespace="pilot",
                external_key="task-1-v1",
                group_id=GROUPS[0],
                metadata=metadata(),
            ),
            self.archive.import_text(
                "Документ о втором деле.",
                namespace="pilot",
                external_key="task-2-v1",
                group_id=GROUPS[1],
                metadata=metadata(),
            ),
            self.archive.import_text(
                "Акт принят по третьему делу.",
                namespace="pilot",
                external_key="task-3-v1",
                group_id=GROUPS[2],
                metadata=metadata(),
            ),
        ]
        self.assignments = [
            {"source_id": record.source_id, "source_version": 1, "split": split}
            for record, split in zip(
                self.records, ("train", "development", "calibration"), strict=True
            )
        ]
        self.restricted = {
            "sealed_test": {
                "members": [
                    {
                        "source_sha256": "a" * 64,
                        "normalized_sha256": "b" * 64,
                        "group_commitment": group_commitment(GROUPS[3]),
                    }
                ]
            },
            "future_stream": {"members": []},
        }
        screened_groups = sorted(
            [group_commitment(record.group_id) for record in self.records]
            + [group_commitment(GROUPS[3])]
        )
        self.review = {
            "status": "completed",
            "reviewer": "independent-synthetic-reviewer",
            "method": "manual origin and near-duplicate review",
            "evidence_reference": "test:review-record",
            "reviewed_group_commitments": screened_groups,
            "screening_reference": "test:screening-log",
            "screening_sha256": screening_sha256(screened_groups, []),
            "candidate_pairs": [],
            "reviewed_pairs": [],
        }

    def tearDown(self) -> None:
        self.archive.close()
        self.directory.cleanup()

    def freeze(self) -> dict:
        return freeze_pilot_corpus(
            self.archive,
            self.assignments,
            restricted_receipts=self.restricted,
            near_duplicate_review=self.review,
        )

    def _annotate(self) -> None:
        record = self.records[0]
        self.archive.annotate(
            {
                "schema": "ai2-open-annotations-v1",
                "source_id": record.source_id,
                "source_version": record.version,
                "annotator": "test-person",
                "evidence": "test:annotation-proof",
                "instances": [
                    {"ref": "a", "external_key": "hedgehog-1", "label": "ёж"}
                ],
                "mentions": [
                    {
                        "start": 0,
                        "end": 2,
                        "surface": "Ёж",
                        "candidates": ["a"],
                        "selected": "a",
                        "expected_version": 0,
                    },
                    {
                        "start": 7,
                        "end": 9,
                        "surface": "ёж",
                        "candidates": [],
                        "selected": None,
                        "expected_version": 0,
                    },
                ],
            }
        )

    def test_round_trip_versions_provenance_and_unknown_identity(self) -> None:
        self._annotate()
        manifest = self.freeze()
        self.assertEqual(
            manifest["payload"]["split_counts"],
            {
                "train": 1,
                "development": 1,
                "calibration": 1,
                "sealed_test": 1,
                "future_stream": 0,
            },
        )
        self.assertEqual(
            validate_pilot_corpus(self.archive, manifest)["status"], "validated"
        )
        source = self.records[0]
        duplicate = self.archive.import_text(
            "Ёж 😀 и ёж.\r\n",
            namespace="pilot",
            external_key="task-1-v1",
            group_id=GROUPS[0],
            metadata=metadata(),
        )
        self.assertEqual(
            (source.source_id, source.version), (duplicate.source_id, duplicate.version)
        )
        original_chunks = list(self.archive.iter_observations(source.source_id, 1))
        self.assertEqual(
            "".join(chunk.text for chunk in original_chunks), "Ёж 😀 и ёж.\r\n"
        )
        self.assertEqual(original_chunks[-1].byte_end, len("Ёж 😀 и ёж.\r\n".encode()))
        self.assertEqual(self.freeze(), manifest)
        view = pilot_training_view(self.archive, manifest)
        self.assertEqual(len(view["documents"]), 1)
        annotations = view["documents"][0]["annotations"]
        self.assertEqual(
            [item["status"] for item in annotations], ["annotated", "unresolved"]
        )
        self.assertIsNone(annotations[1]["selected_id"])
        # An unknown identity never becomes a negative pair by omission.
        self.assertNotIn("pair_labels", view)
        self.assertNotIn("negative_pairs", view)
        self.assertEqual(annotations[0]["annotator"], "test-person")
        self.assertEqual(annotations[0]["evidence"], "test:annotation-proof")
        self.archive.import_text(
            "Изменённая редакция",
            namespace="pilot",
            external_key="task-1-v1",
            group_id=GROUPS[0],
            metadata=metadata(),
        )
        self.assertEqual(
            validate_pilot_corpus(self.archive, manifest)["status"], "validated"
        )
        self.assertEqual(pilot_training_view(self.archive, manifest), view)

    def test_normalized_duplicates_in_open_and_sealed_splits_are_rejected(self) -> None:
        duplicate = self.archive.import_text(
            "  ёЖ  😀  И  ёж.\n",
            namespace="pilot",
            external_key="copy",
            group_id=GROUPS[4],
            metadata=metadata(),
        )
        self.assertEqual(
            normalized_sha256("Ёж 😀 и ёж.\r\n"),
            normalized_sha256("  ёЖ  😀  И  ёж.\n"),
        )
        self.assignments.append(
            {
                "source_id": duplicate.source_id,
                "source_version": 1,
                "split": "development",
            }
        )
        with self.assertRaisesRegex(ValueError, "normalized duplicate leaks"):
            self.freeze()
        self.assignments.pop()
        self.restricted["sealed_test"]["members"][0]["normalized_sha256"] = (
            normalized_sha256("Ёж 😀 и ёж.\r\n")
        )
        with self.assertRaisesRegex(ValueError, "normalized duplicate leaks"):
            self.freeze()

    def test_family_overlap_and_unfinished_review_block_freeze(self) -> None:
        self.restricted["sealed_test"]["members"][0]["group_commitment"] = (
            group_commitment(self.records[0].group_id)
        )
        with self.assertRaisesRegex(ValueError, "related source family leaks"):
            self.freeze()
        self.restricted["sealed_test"]["members"][0]["group_commitment"] = (
            group_commitment(GROUPS[3])
        )
        flagged = deepcopy(self.review)
        left, right = sorted(
            [group_commitment(self.records[0].group_id), group_commitment(GROUPS[3])]
        )
        flagged["candidate_pairs"] = [{"left": left, "right": right}]
        flagged["screening_sha256"] = screening_sha256(
            flagged["reviewed_group_commitments"], flagged["candidate_pairs"]
        )
        for broken in (
            None,
            {**self.review, "status": "pending"},
            flagged,
            {**self.review, "reviewed_group_commitments": []},
        ):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                freeze_pilot_corpus(
                    self.archive,
                    self.assignments,
                    restricted_receipts=self.restricted,
                    near_duplicate_review=broken,
                )

    def test_flagged_reprint_needs_recorded_independent_decision(self) -> None:
        left, right = sorted(
            [group_commitment(self.records[0].group_id), group_commitment(GROUPS[3])]
        )
        self.review["candidate_pairs"] = [{"left": left, "right": right}]
        self.review["screening_sha256"] = screening_sha256(
            self.review["reviewed_group_commitments"], self.review["candidate_pairs"]
        )
        with self.assertRaisesRegex(ValueError, "matching review"):
            self.freeze()
        self.review["reviewed_pairs"] = [
            {
                "left": left,
                "right": right,
                "decision": "related",
                "evidence_reference": "test:cannot-split",
            }
        ]
        with self.assertRaisesRegex(ValueError, "related or unresolved"):
            self.freeze()
        self.review["reviewed_pairs"][0]["decision"] = "independent"
        self.assertEqual(
            validate_pilot_corpus(self.archive, self.freeze())["status"], "validated"
        )
        self.review["screening_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "screening inventory changed"):
            self.freeze()

    def test_closed_receipt_repeats_and_predictable_group_names_rejected(self) -> None:
        original = dict(self.restricted["sealed_test"]["members"][0])
        for key in ("source_sha256", "normalized_sha256", "group_commitment"):
            second = {
                "source_sha256": "c" * 64,
                "normalized_sha256": "d" * 64,
                "group_commitment": group_commitment(GROUPS[7]),
            }
            second[key] = original[key]
            self.restricted["sealed_test"]["members"].append(second)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "duplicate"):
                self.freeze()
            self.restricted["sealed_test"]["members"].pop()
        with self.assertRaisesRegex(ValueError, "opaque UUIDv4"):
            group_commitment("project-main-docs")

    def test_train_requires_provenance_complete_coverage_and_no_closed_assignment(
        self,
    ) -> None:
        incomplete = self.archive.import_text(
            "новый документ",
            namespace="pilot",
            external_key="incomplete",
            group_id=GROUPS[5],
            metadata=metadata(coverage="partial"),
        )
        self.assignments.append(
            {"source_id": incomplete.source_id, "source_version": 1, "split": "train"}
        )
        with self.assertRaisesRegex(ValueError, "complete mention"):
            self.freeze()
        self.assignments.pop()
        self.assignments[0]["split"] = "sealed_test"
        with self.assertRaisesRegex(ValueError, "detached receipts"):
            self.freeze()
        self.assignments[0]["split"] = "train"
        without_rights = self.archive.import_text(
            "новый документ",
            namespace="pilot",
            external_key="without-rights",
            group_id=GROUPS[6],
            metadata={"origin": "test", "language": "ru"},
        )
        self.assignments.append(
            {
                "source_id": without_rights.source_id,
                "source_version": 1,
                "split": "development",
            }
        )
        with self.assertRaisesRegex(ValueError, "rights_evidence_reference"):
            self.freeze()

    def test_training_view_has_no_sealed_labels_or_closed_source_text(self) -> None:
        sealed_text = "PRIVATE SYNTHETIC GOLD: никому не показывать"
        with ObservationArchive(self.root / "closed.sqlite", create=True) as custodian:
            hidden = custodian.import_text(
                sealed_text,
                namespace="sealed",
                external_key="hidden-source",
                group_id=GROUPS[3],
            )
            self.restricted["sealed_test"]["members"][0] = {
                "source_sha256": hidden.sha256,
                "normalized_sha256": normalized_sha256(sealed_text),
                "group_commitment": group_commitment(hidden.group_id),
            }
        manifest = self.freeze()
        view = pilot_training_view(self.archive, manifest)
        serialized = json.dumps([manifest, view], ensure_ascii=False)
        self.assertNotIn("PRIVATE SYNTHETIC GOLD", serialized)
        self.assertNotIn("sealed_test", json.dumps(view, ensure_ascii=False))
        self.assertEqual(len(view["documents"]), 1)
        self.assertNotIn(
            "text",
            manifest["payload"]["restricted_receipts"]["sealed_test"]["members"][0],
        )

    def test_cli_round_trip_and_legacy_split_compatibility(self) -> None:
        legacy = freeze_corpus(
            self.archive,
            [
                {
                    "source_id": self.records[1].source_id,
                    "source_version": 1,
                    "split": "validation",
                },
                {
                    "source_id": self.records[2].source_id,
                    "source_version": 1,
                    "split": "test",
                },
            ],
        )
        self.assertEqual(
            {item["split"] for item in legacy["payload"]["members"]},
            {"validation", "test"},
        )
        spec = self.root / "pilot-input.json"
        result = self.root / "pilot.json"
        spec.write_text(
            json.dumps(
                {
                    "schema": PILOT_ASSIGNMENTS_SCHEMA,
                    "assignments": self.assignments,
                    "restricted_receipts": self.restricted,
                    "near_duplicate_review": self.review,
                }
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        for operation, more in (
            ("freeze-pilot", ["--output", str(result)]),
            ("validate-pilot", []),
        ):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "text_factors",
                    "observations",
                    operation,
                    "--archive",
                    str(self.archive_path),
                    "--data",
                    str(spec if operation == "freeze-pilot" else result),
                    *more,
                ],
                capture_output=True,
                text=True,
                timeout=20,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            json.loads(result.read_text(encoding="utf-8"))["schema"],
            "ai2-pilot-corpus-v1",
        )

    def test_pinned_annotation_version_survives_later_correction(self) -> None:
        self._annotate()
        frozen = self.freeze()
        old = pilot_training_view(self.archive, frozen)
        mention = self.archive.annotations(self.records[0].source_id, 1)[1]["mention"]
        self.archive.annotate(
            {
                "schema": "ai2-open-annotations-v1",
                "source_id": self.records[0].source_id,
                "source_version": 1,
                "annotator": "second-person",
                "evidence": "test:correction",
                "instances": [
                    {"ref": "a", "external_key": "hedgehog-1", "label": "ёж"}
                ],
                "mentions": [
                    {
                        "start": mention["char_start"],
                        "end": mention["char_end"],
                        "surface": mention["surface"],
                        "candidates": ["a"],
                        "selected": "a",
                        "expected_version": 1,
                    }
                ],
            }
        )
        self.assertEqual(
            validate_pilot_corpus(self.archive, frozen)["status"], "validated"
        )
        self.assertEqual(pilot_training_view(self.archive, frozen), old)
        self.assertNotEqual(self.freeze(), frozen)

    def test_explicit_identity_pairs_are_versioned_and_unknown_is_not_negative(
        self,
    ) -> None:
        self._annotate()
        source = self.records[0]
        annotated = self.archive.annotations(source.source_id, source.version)
        first, second = [record["mention"]["mention_id"] for record in annotated]
        batch = {
            "schema": "ai2-open-identity-pairs-v1",
            "source_id": source.source_id,
            "source_version": source.version,
            "start": 0,
            "end": source.char_count,
            "coverage": "complete",
            "policy_reference": "test:identity-policy",
            "annotator": "independent-pair-reviewer",
            "evidence": "test:pair-review-record",
            "mention_ids": [first, second],
            "pairs": [
                {
                    "left_mention_id": second,
                    "right_mention_id": first,
                    "label": "unknown",
                }
            ],
            "expected_version": 0,
        }
        initial = self.archive.annotate_identity_pairs(batch)
        self.assertEqual(initial["version"], 1)
        self.assertEqual(self.archive.annotate_identity_pairs(batch), initial)
        pinned = self.freeze()
        view = pilot_training_view(self.archive, pinned)
        document = view["documents"][0]
        self.assertEqual(document["identity_scopes"][0]["coverage"], "complete")
        self.assertEqual(document["identity_pairs"][0]["label"], "unknown")
        self.assertNotIn("negative_pairs", document)
        self.assertNotIn("sealed_test", json.dumps(view, ensure_ascii=False))
        self.assertEqual(
            document["identity_pairs"][0]["annotator"], "independent-pair-reviewer"
        )
        revised = deepcopy(batch)
        revised["pairs"][0]["label"] = "different"
        revised["expected_version"] = 1
        revised["evidence"] = "test:re-reviewed-difference"
        correction = self.archive.annotate_identity_pairs(revised)
        self.assertEqual(correction["version"], 2)
        self.assertEqual(
            self.archive.get_identity_scope(initial["scope_id"], 1), initial
        )
        self.assertEqual(pilot_training_view(self.archive, pinned), view)
        updated_train = next(
            item
            for item in self.freeze()["payload"]["open_members"]
            if item["split"] == "train"
        )
        self.assertEqual(updated_train["identity_scopes"][0]["version"], 2)
        with ObservationArchive(self.archive_path) as reopened:
            self.assertEqual(
                reopened.get_identity_scope(initial["scope_id"], 1), initial
            )
            self.assertEqual(reopened.verify()["identity_scope_versions"], 2)
        data_path = self.root / "pair-review.json"
        data_path.write_text(json.dumps(revised), encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        for operation, arguments in (
            ("annotate-pairs", ["--data", str(data_path)]),
            (
                "identity-scopes",
                ["--source", source.source_id, "--version", str(source.version)],
            ),
        ):
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "text_factors",
                    "observations",
                    operation,
                    "--archive",
                    str(self.archive_path),
                    *arguments,
                ],
                capture_output=True,
                text=True,
                timeout=20,
                env=env,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(initial["scope_id"], process.stdout)

    def test_partial_scope_omissions_are_not_negative_and_complete_requires_all_pairs(
        self,
    ) -> None:
        self._annotate()
        source = self.records[0]
        self.archive.annotate(
            {
                "schema": "ai2-open-annotations-v1",
                "source_id": source.source_id,
                "source_version": source.version,
                "annotator": "test-person",
                "evidence": "test:third-mention",
                "instances": [],
                "mentions": [
                    {
                        "start": 3,
                        "end": 4,
                        "surface": "😀",
                        "candidates": [],
                        "selected": None,
                        "expected_version": 0,
                    }
                ],
            }
        )
        ids = [
            item["mention"]["mention_id"]
            for item in self.archive.annotations(source.source_id, source.version)
        ]
        batch = {
            "schema": "ai2-open-identity-pairs-v1",
            "source_id": source.source_id,
            "source_version": source.version,
            "start": 0,
            "end": source.char_count,
            "coverage": "complete",
            "policy_reference": "test:policy",
            "annotator": "test-reviewer",
            "evidence": "test:evidence",
            "mention_ids": ids,
            "pairs": [
                {"left_mention_id": ids[0], "right_mention_id": ids[1], "label": "same"}
            ],
            "expected_version": 0,
        }
        with self.assertRaisesRegex(ValueError, "explicit label for every pair"):
            self.archive.annotate_identity_pairs(batch)
        batch["pairs"] = [
            {"left_mention_id": ids[0], "right_mention_id": ids[1], "label": "same"},
            {"left_mention_id": ids[1], "right_mention_id": ids[2], "label": "same"},
            {
                "left_mention_id": ids[0],
                "right_mention_id": ids[2],
                "label": "different",
            },
        ]
        with self.assertRaisesRegex(ValueError, "contradicts reviewed same"):
            self.archive.annotate_identity_pairs(batch)
        batch["coverage"] = "partial"
        batch["pairs"] = batch["pairs"][:1]
        scope = self.archive.annotate_identity_pairs(batch)
        self.assertEqual(len(scope["pairs"]), 1)
        view = pilot_training_view(self.archive, self.freeze())
        self.assertEqual(view["documents"][0]["identity_pairs"][0]["label"], "same")
        self.assertEqual(len(view["documents"][0]["identity_pairs"]), 1)
        batch["coverage"] = "unknown"
        batch["expected_version"] = 1
        with self.assertRaisesRegex(ValueError, "invalid explicit identity pair label"):
            self.archive.annotate_identity_pairs(batch)
        batch["coverage"] = "complete"
        batch["mention_ids"] = ids[:2]
        batch["pairs"] = [
            {"left_mention_id": ids[0], "right_mention_id": ids[1], "label": "same"}
        ]
        with self.assertRaisesRegex(ValueError, "all current mentions"):
            self.archive.annotate_identity_pairs(batch)

    def test_identity_scope_rejects_cross_source_and_forged_manifest(self) -> None:
        self._annotate()
        source = self.records[0]
        ids = [
            item["mention"]["mention_id"]
            for item in self.archive.annotations(source.source_id, source.version)
        ]
        batch = {
            "schema": "ai2-open-identity-pairs-v1",
            "source_id": source.source_id,
            "source_version": source.version,
            "start": 0,
            "end": source.char_count,
            "coverage": "partial",
            "policy_reference": "test:policy",
            "annotator": "test-reviewer",
            "evidence": "test:evidence",
            "mention_ids": ids,
            "pairs": [
                {
                    "left_mention_id": ids[0],
                    "right_mention_id": ids[1],
                    "label": "different",
                }
            ],
            "expected_version": 0,
        }
        scope = self.archive.annotate_identity_pairs(batch)
        batch["source_id"] = self.records[1].source_id
        batch["end"] = self.records[1].char_count
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            self.archive.annotate_identity_pairs(batch)
        manifest = self.freeze()
        tampered = deepcopy(manifest)
        second = next(
            item
            for item in tampered["payload"]["open_members"]
            if item["split"] == "development"
        )
        second["identity_scopes"] = [{"scope_id": scope["scope_id"], "version": 1}]
        tampered["fingerprint"] = hashlib.sha256(
            canonical_json(tampered["payload"]).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "cross-source pinned identity scope"):
            validate_pilot_corpus(self.archive, tampered)

    def test_legacy_complete_binding_claim_is_not_complete_pair_supervision(
        self,
    ) -> None:
        old = self.freeze()
        source = self.records[0]
        changed = metadata()
        changed["annotation_coverage"]["identity"] = "complete"
        second_revision = self.archive.import_text(
            "Ёж 😀 и ёж.\r\n",
            namespace="pilot",
            external_key=source.external_key,
            group_id=source.group_id,
            metadata=changed,
        )
        self.assertEqual(second_revision.version, 2)
        legacy = deepcopy(old)
        train = next(
            item
            for item in legacy["payload"]["open_members"]
            if item["split"] == "train"
        )
        train["source_version"] = 2
        train["annotations"] = []
        train["metadata_sha256"] = hashlib.sha256(
            canonical_json(changed).encode("utf-8")
        ).hexdigest()
        train.pop("identity_scopes")
        legacy["fingerprint"] = hashlib.sha256(
            canonical_json(legacy["payload"]).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            validate_pilot_corpus(self.archive, legacy)["status"], "validated"
        )
        document = pilot_training_view(self.archive, legacy)["documents"][0]
        self.assertEqual(document["annotation_coverage"]["identity"], "unknown")
        self.assertEqual(document["pair_supervision"], "explicit_pairs_only")
        self.assertEqual(document["identity_pairs"], [])
        self.assignments[0]["source_version"] = 2
        with self.assertRaisesRegex(ValueError, "full-source pair scope"):
            self.freeze()

    def test_explicit_pairs_must_agree_with_pinned_selected_instance_versions(
        self,
    ) -> None:
        self._annotate()
        source = self.records[0]
        records = self.archive.annotations(source.source_id, source.version)
        first, second = [item["mention"]["mention_id"] for item in records]

        def correct_second(selected: str, expected_version: int) -> None:
            self.archive.annotate(
                {
                    "schema": "ai2-open-annotations-v1",
                    "source_id": source.source_id,
                    "source_version": source.version,
                    "annotator": "test-human",
                    "evidence": f"test:second-binding-{expected_version + 1}",
                    "instances": [
                        {"ref": "a", "external_key": "hedgehog-1", "label": "ёж"},
                        {"ref": "b", "external_key": "hedgehog-2", "label": "ёж"},
                    ],
                    "mentions": [
                        {
                            "start": 7,
                            "end": 9,
                            "surface": "ёж",
                            "candidates": ["a", "b"],
                            "selected": selected,
                            "expected_version": expected_version,
                        }
                    ],
                }
            )

        correct_second("b", 1)
        batch = {
            "schema": "ai2-open-identity-pairs-v1",
            "source_id": source.source_id,
            "source_version": source.version,
            "start": 0,
            "end": source.char_count,
            "coverage": "complete",
            "policy_reference": "test:identity-policy",
            "annotator": "test-reviewer",
            "evidence": "test:first-pair-review",
            "mention_ids": [first, second],
            "pairs": [
                {"left_mention_id": first, "right_mention_id": second, "label": "same"}
            ],
            "expected_version": 0,
        }
        self.archive.annotate_identity_pairs(batch)
        with self.assertRaisesRegex(ValueError, "contradicts pinned selected"):
            self.freeze()
        correct_second("a", 2)
        pinned_same = self.freeze()
        previous_view = pilot_training_view(self.archive, pinned_same)
        self.assertEqual(
            previous_view["documents"][0]["identity_pairs"][0]["label"], "same"
        )
        correction = deepcopy(batch)
        correction["pairs"][0]["label"] = "different"
        correction["expected_version"] = 1
        correction["evidence"] = "test:second-pair-review"
        self.archive.annotate_identity_pairs(correction)
        with self.assertRaisesRegex(ValueError, "contradicts pinned selected"):
            self.freeze()
        correct_second("b", 3)
        pinned_different = self.freeze()
        self.assertEqual(pilot_training_view(self.archive, pinned_same), previous_view)
        self.assertEqual(
            pilot_training_view(self.archive, pinned_different)["documents"][0][
                "identity_pairs"
            ][0]["label"],
            "different",
        )
        for manifest, conflicting_binding_version in (
            (pinned_same, 4),
            (pinned_different, 3),
        ):
            forged = deepcopy(manifest)
            train = next(
                item
                for item in forged["payload"]["open_members"]
                if item["split"] == "train"
            )
            second_record = next(
                item for item in train["annotations"] if item["mention_id"] == second
            )
            second_record["binding_version"] = conflicting_binding_version
            forged["fingerprint"] = hashlib.sha256(
                canonical_json(forged["payload"]).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "contradicts pinned selected"):
                validate_pilot_corpus(self.archive, forged)

    def test_development_and_calibration_pair_labels_do_not_enter_training_view(
        self,
    ) -> None:
        for source in self.records[1:]:
            text = "".join(
                item.text
                for item in self.archive.iter_observations(
                    source.source_id, source.version
                )
            )
            first = text.split()[0]
            self.archive.annotate(
                {
                    "schema": "ai2-open-annotations-v1",
                    "source_id": source.source_id,
                    "source_version": source.version,
                    "annotator": "test-human",
                    "evidence": "test:non-train-mention",
                    "instances": [],
                    "mentions": [
                        {
                            "start": 0,
                            "end": len(first),
                            "surface": first,
                            "candidates": [],
                            "selected": None,
                            "expected_version": 0,
                        }
                    ],
                }
            )
            mention_id = self.archive.annotations(source.source_id, 1)[0]
            mention_id = mention_id["mention"]["mention_id"]
            self.archive.annotate_identity_pairs(
                {
                    "schema": "ai2-open-identity-pairs-v1",
                    "source_id": source.source_id,
                    "source_version": source.version,
                    "start": 0,
                    "end": source.char_count,
                    "coverage": "partial",
                    "policy_reference": "test:non-train-policy",
                    "annotator": "test-reviewer",
                    "evidence": "test:non-train-pair-review",
                    "mention_ids": [mention_id],
                    "pairs": [],
                    "expected_version": 0,
                }
            )
        view = pilot_training_view(self.archive, self.freeze())
        self.assertEqual(len(view["documents"]), 1)
        self.assertEqual(view["documents"][0]["source_id"], self.records[0].source_id)
        self.assertNotIn("test:non-train-pair-review", json.dumps(view))
        for source in self.records[1:]:
            self.assertNotIn(source.source_id, json.dumps(view))


if __name__ == "__main__":
    unittest.main()

"""Real CLI workflows and corpus independence/version contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from text_factors.cli import build_parser
from text_factors.observations import ObservationArchive
from text_factors.observations.corpus import freeze_corpus, validate_corpus
from text_factors.observations.schema import canonical_json


class ObservationCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-observation-cli-")
        self.root = Path(self.directory.name)
        self.path = self.root / "data.sqlite"

    def tearDown(self):
        self.directory.cleanup()

    def _run(self, operation, *args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "text_factors",
                "observations",
                operation,
                "--archive",
                str(self.path),
                *map(str, args),
            ],
            capture_output=True,
            timeout=20,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
        )

    def test_import_show_annotate_freeze_and_reload_via_real_cli(self):
        original = "Первый адаптер Ω здесь. Второй адаптер Ω там.\r\n"
        input_path = self.root / "new.txt"
        input_path.write_bytes(original.encode())
        imported = self._run(
            "import",
            "--input",
            input_path,
            "--namespace",
            "pilot",
            "--source-key",
            "record-1",
            "--group",
            "family-1",
            "--origin",
            "test supplied input",
            "--chunk-chars",
            "7",
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        source = json.loads(imported.stdout)["source"]
        shown = self._run("show", "--source", source["source_id"], "--limit", "100")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertEqual(
            "".join(o["text"] for o in json.loads(shown.stdout)["observations"]),
            original,
        )
        start = original.index("адаптер")
        data = {
            "schema": "ai2-open-annotations-v1",
            "source_id": source["source_id"],
            "source_version": 1,
            "annotator": "external-labeler",
            "evidence": "source annotation",
            "instances": [
                {"ref": "a", "external_key": "adapter-first", "label": "адаптер Ω"}
            ],
            "mentions": [
                {
                    "start": start,
                    "end": start + len("адаптер Ω"),
                    "surface": "адаптер Ω",
                    "candidates": ["a"],
                    "selected": "a",
                    "expected_version": 0,
                }
            ],
        }
        annotation_path = self.root / "annotation.json"
        annotation_path.write_text(json.dumps(data, ensure_ascii=False))
        annotated = self._run("annotate", "--data", annotation_path)
        self.assertEqual(annotated.returncode, 0, annotated.stderr)
        self.assertEqual(
            json.loads(annotated.stdout)["annotations"][0]["binding"]["status"],
            "annotated",
        )
        assignments = self.root / "assignments.json"
        assignments.write_text(
            json.dumps(
                {
                    "schema": "ai2-open-corpus-splits-v1",
                    "assignments": [
                        {
                            "source_id": source["source_id"],
                            "source_version": 1,
                            "split": "validation",
                        }
                    ],
                }
            )
        )
        manifest = self.root / "corpus.json"
        frozen = self._run("freeze", "--data", assignments, "--output", manifest)
        self.assertEqual(frozen.returncode, 0, frozen.stderr)
        validated = self._run("validate-corpus", "--data", manifest)
        self.assertEqual(validated.returncode, 0, validated.stderr)
        self.assertEqual(json.loads(validated.stdout)["pinned_annotations"], 1)
        verified = self._run("verify")
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(
            json.loads(verified.stdout)["bytes_verified"], len(original.encode())
        )

    def test_failed_cli_import_reports_failure_and_commits_no_source(self):
        path = self.root / "broken.txt"
        path.write_bytes(b"valid prefix\xff")
        result = self._run(
            "import",
            "--input",
            path,
            "--namespace",
            "n",
            "--source-key",
            "k",
            "--group",
            "g",
            "--origin",
            "fixture",
            "--chunk-chars",
            "2",
        )
        self.assertNotEqual(result.returncode, 0)
        with ObservationArchive(self.path) as archive:
            self.assertEqual(archive.sources("n"), ())

    def test_old_commands_are_still_registered(self):
        parser = build_parser()
        for arguments in (
            ["converse", "--demo"],
            ["learned-chat", "--model", "model.json", "--demo"],
            ["observations", "verify", "--archive", "observations.sqlite"],
        ):
            self.assertTrue(callable(parser.parse_args(arguments).handler))


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-open-corpus-")
        self.archive = ObservationArchive(
            Path(self.directory.name) / "corpus.sqlite", create=True
        )

    def tearDown(self):
        self.archive.close()
        self.directory.cleanup()

    def _source(self, text, key, group):
        return self.archive.import_text(
            text, namespace="pilot", external_key=key, group_id=group
        )

    @staticmethod
    def _assignment(source, split):
        return {
            "source_id": source.source_id,
            "source_version": source.version,
            "split": split,
        }

    def test_family_and_exact_duplicate_leakage_rejected(self):
        a = self._source("original", "a", "family")
        b = self._source("edited", "b", "family")
        c = self._source("original", "c", "different-family")
        for other in (b, c):
            with (
                self.subTest(other=other.external_key),
                self.assertRaisesRegex(ValueError, "leaks"),
            ):
                freeze_corpus(
                    self.archive,
                    [self._assignment(a, "train"), self._assignment(other, "test")],
                )

    def test_new_revisions_do_not_change_frozen_sources(self):
        a = self._source("old source", "a", "a")
        manifest = freeze_corpus(self.archive, [self._assignment(a, "validation")])
        self._source("new source", "a", "a")
        result = validate_corpus(self.archive, manifest)
        self.assertEqual(result["status"], "validated")
        self.assertFalse(result["near_duplicate_independence_verified"])
        self.assertEqual(manifest["payload"]["members"][0]["source_version"], 1)

    def test_fingerprint_tampering_and_duplicate_member_rejected(self):
        a = self._source("a", "a", "a")
        assignment = self._assignment(a, "test")
        manifest = freeze_corpus(self.archive, [assignment])
        bad = deepcopy(manifest)
        bad["payload"]["members"][0]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_corpus(self.archive, bad)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            freeze_corpus(self.archive, [assignment, assignment])

    def test_annotation_versions_remain_pinned_after_correction(self):
        source = self._source("имя", "a", "a")
        batch = {
            "schema": "ai2-open-annotations-v1",
            "source_id": source.source_id,
            "source_version": 1,
            "annotator": "human",
            "evidence": "initial",
            "instances": [{"ref": "x", "external_key": "x", "label": "имя"}],
            "mentions": [
                {
                    "start": 0,
                    "end": 3,
                    "surface": "имя",
                    "candidates": ["x"],
                    "selected": None,
                    "expected_version": 0,
                }
            ],
        }
        self.archive.annotate(batch)
        manifest = freeze_corpus(self.archive, [self._assignment(source, "train")])
        batch["mentions"][0]["selected"] = "x"
        batch["mentions"][0]["expected_version"] = 1
        batch["evidence"] = "explicit clarification"
        self.archive.annotate(batch)
        self.assertEqual(
            validate_corpus(self.archive, manifest)["pinned_annotations"], 1
        )
        self.assertEqual(
            manifest["payload"]["members"][0]["annotations"][0]["binding_version"], 1
        )
        bad = deepcopy(manifest)
        bad["payload"]["members"][0]["annotations"][0]["binding_version"] = None
        bad["fingerprint"] = hashlib.sha256(
            canonical_json(bad["payload"]).encode()
        ).hexdigest()
        with self.assertRaises(ValueError):
            validate_corpus(self.archive, bad)


if __name__ == "__main__":
    unittest.main()

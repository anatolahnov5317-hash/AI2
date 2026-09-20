"""Pinned corpus conversion contracts using original synthetic examples only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gum_mentions.py"
SPEC = importlib.util.spec_from_file_location("prepare_gum_mentions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gum)


class GumMentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-gum-conversion-")
        self.root = Path(self.directory.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.manifest_path = self.root / "manifest.json"

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def _conll(rows, name="Synthetic_A"):
        body = "\n".join(f"{i}\t{word}\t{ann}" for i, (word, ann) in enumerate(rows))
        return f"# begin document {name}\n{body}\n# end document\n"

    def _document(self, name="Synthetic_A", split="train", words=None):
        words = words or ["Æther", "glows", "."]
        conll = self._conll(
            [(word, "(opaque)" if i == 0 else "_") for i, word in enumerate(words)],
            name,
        )
        # This metadata resembles GUM's independent summaries; none may enter text.
        tsv = "#Summary1=(model) SECRET_SUMMARY_SENTINEL\n"
        offset = 0
        for i, word in enumerate(words):
            tsv += f"1-{i + 1}\t{offset}-{offset + len(word)}\t{word}\t_\n"
            offset += len(word) + 1
        official = "dev" if split == "validation" else split
        url = f"https://example.invalid/{name}"
        xml = (
            f'<text id="{name}" partition="{official}" sourceURL="{url}" '
            'author="Synthetic test author" summary1="SECRET_SUMMARY_SENTINEL">'
            "<p>XML_BODY_IS_NOT_MODEL_INPUT</p></text>"
        )
        files = {}
        for extension, contents in (("conll", conll), ("tsv", tsv), ("xml", xml)):
            relative = f"{name}.{extension}"
            data = contents.encode("utf-8")
            (self.input_dir / relative).write_bytes(data)
            files[extension] = {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "url": f"https://example.invalid/{relative}",
            }
        return {
            "document_id": name,
            "group_id": name,
            "split": split,
            "source_url": url,
            "license": "Synthetic test data",
            "files": files,
        }

    def _manifest(self, documents):
        value = {
            "schema": gum.MANIFEST_SCHEMA,
            "revision": "synthetic-test-revision",
            "repository": "https://example.invalid/synthetic",
            "selection_policy": "Predetermined test fixture, not model selection",
            "licenses": {"test": "Synthetic test data"},
            "documents": documents,
        }
        self.manifest_path.write_text(json.dumps(value), encoding="utf-8")
        return value

    def _cli(self, output, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--input-dir",
                str(self.input_dir),
                "--manifest",
                str(self.manifest_path),
                "--output",
                str(output),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_preserves_unicode_nested_crossing_and_repeated_entity_mentions(self):
        data = self._conll(
            [
                ("A", "(outer"),
                ("🦋", "(inner)"),
                ("Ω", "outer)(cross"),
                ("end", "cross)"),
                ("again", "(inner)"),
            ]
        )
        text, mentions, tokens = gum.parse_conll(data, "Synthetic_A")
        self.assertEqual(text, "A 🦋 Ω end again")
        self.assertEqual(tokens, ["A", "🦋", "Ω", "end", "again"])
        self.assertEqual(
            [(text[m["start"] : m["end"]], m["entity_id"]) for m in mentions],
            [
                ("A 🦋 Ω", "outer"),
                ("🦋", "inner"),
                ("Ω end", "cross"),
                ("again", "inner"),
            ],
        )
        self.assertEqual(mentions[1]["end"] - mentions[1]["start"], 1)

    def test_duplicate_exact_span_with_distinct_entities_is_not_silently_removed(self):
        data = self._conll([("word", "(first)(second)")])
        with self.assertRaisesRegex(ValueError, "duplicate exact mention span"):
            gum.parse_conll(data, "Synthetic_A")

    def test_rejects_malformed_boundaries_discontinuous_encoding_and_token_indices(
        self,
    ):
        invalid = [
            self._conll([("word", "unopened)")]),
            self._conll([("word", "(unclosed")]),
            self._conll([("word", "(span[1_2])")]),
            self._conll([("word", "(entity)|")]),
            self._conll([("word", "_")]).replace("0\t", "3\t"),
            self._conll([("word", "_")]).replace("# end document", ""),
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                gum.parse_conll(data, "Synthetic_A")

    def test_cli_preserves_splits_provenance_and_excludes_all_summaries(self):
        entries = [
            self._document(),
            self._document("Synthetic_B", "validation", ["Ω", "flows"]),
            self._document("Synthetic_C", "test", ["Future", "rain"]),
        ]
        self._manifest(entries)
        output = self.root / "corpus.json"
        result = self._cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        corpus = json.loads(output.read_text())
        self.assertEqual(corpus["schema"], gum.CORPUS_SCHEMA)
        self.assertEqual(
            [d["split"] for d in corpus["documents"]], ["train", "validation", "test"]
        )
        self.assertEqual(
            [d["group_id"] for d in corpus["documents"]],
            [d["document_id"] for d in entries],
        )
        self.assertNotIn("SECRET_SUMMARY_SENTINEL", output.read_text())
        self.assertNotIn("XML_BODY_IS_NOT_MODEL_INPUT", output.read_text())
        self.assertFalse(
            corpus["documents"][0]["provenance"]["original_website_bytes_preserved"]
        )
        self.assertIn("native GUM", corpus["provenance"]["annotation_policy"])
        self.assertEqual(json.loads(result.stdout)["total_documents"], 3)

    def test_source_tampering_fails_cli_and_retains_existing_output(self):
        entry = self._document()
        self._manifest([entry])
        output = self.root / "corpus.json"
        output.write_bytes(b"previous checkpoint\n")
        source = self.input_dir / entry["files"]["conll"]["path"]
        source.write_bytes(source.read_bytes() + b"unexpected extra data")
        result = self._cli(output, "--overwrite")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertEqual(output.read_bytes(), b"previous checkpoint\n")
        self.assertFalse(list(self.root.glob(".gum-*")))

    def test_changed_official_partition_and_fragment_group_are_rejected(self):
        entry = self._document()
        entry["split"] = "test"
        self._manifest([entry])
        with self.assertRaisesRegex(ValueError, "official partition"):
            gum.prepare_corpus(self.input_dir, self.manifest_path)
        entry["split"] = "train"
        entry["group_id"] = "fragment-2"
        self._manifest([entry])
        with self.assertRaisesRegex(ValueError, "complete document"):
            gum.prepare_corpus(self.input_dir, self.manifest_path)

    def test_identical_text_cannot_appear_in_distinct_document_splits(self):
        entries = [self._document(), self._document("Synthetic_B", "test")]
        self._manifest(entries)
        with self.assertRaisesRegex(ValueError, "duplicate document text"):
            gum.prepare_corpus(self.input_dir, self.manifest_path)

    def test_bounded_reads_token_budget_and_path_confinement(self):
        data = self.root / "large.txt"
        data.write_bytes(b"123456")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            gum.read_bounded(data, 5)
        entry = self._document()
        self._manifest([entry])
        with self.assertRaisesRegex(ValueError, "source tokens"):
            gum.prepare_corpus(self.input_dir, self.manifest_path, max_tokens=2)
        entry["files"]["conll"]["path"] = "../large.txt"
        self._manifest([entry])
        with self.assertRaisesRegex(ValueError, "within the input directory"):
            gum.prepare_corpus(self.input_dir, self.manifest_path)

    def test_tsv_offsets_are_verified_independently_of_conll(self):
        entry = self._document()
        path = self.input_dir / entry["files"]["tsv"]["path"]
        changed = path.read_text().replace("0-5", "0-6").encode()
        path.write_bytes(changed)
        entry["files"]["tsv"]["sha256"] = hashlib.sha256(changed).hexdigest()
        self._manifest([entry])
        with self.assertRaisesRegex(ValueError, "offset mismatch"):
            gum.prepare_corpus(self.input_dir, self.manifest_path)

    def test_atomic_creation_refuses_overwrite_without_explicit_flag(self):
        self._manifest([self._document()])
        output = self.root / "corpus.json"
        output.write_text("existing file")
        result = self._cli(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "existing file")
        result = self._cli(output, "--overwrite")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["schema"], gum.CORPUS_SCHEMA)
        self.assertFalse(list(self.root.glob(".gum-*")))


if __name__ == "__main__":
    unittest.main()

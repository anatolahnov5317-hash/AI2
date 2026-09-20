"""Losslessness, revision identity, corruption and actual process-death recovery."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from text_factors.observations import ArchiveLimits, ObservationArchive
from text_factors.observations.encoding import (
    casefold_view,
    decode_bytes,
    encode_bytes,
    utf8_chunks,
)


class OpenEncodingTests(unittest.TestCase):
    def test_arbitrary_unicode_is_lossless_across_every_byte_boundary(self):
        text = "\ufeffКошелёк №42\r\n你好 שלום 😀 e\u0301 Straße\x00\t"
        raw = text.encode("utf-8")
        chunks = tuple(
            utf8_chunks(
                (bytes([value]) for value in raw),
                chunk_chars=3,
                max_bytes=len(raw),
            )
        )
        self.assertEqual("".join(chunks), text)
        self.assertEqual(b"".join(chunk.encode("utf-8") for chunk in chunks), raw)
        self.assertEqual(decode_bytes(encode_bytes(text)), text)

    def test_invalid_bytes_and_tokens_are_not_replaced_or_ignored(self):
        for raw in (b"ok\xff", b"\xf0\x9f", b"\xed\xa0\x80"):
            with self.subTest(raw=raw), self.assertRaises(UnicodeError):
                tuple(utf8_chunks([raw], chunk_chars=1, max_bytes=100))
        for values in ([True], [-1], [256], [1.0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                invalid: Any = values
                decode_bytes(invalid)
        with self.assertRaises(UnicodeError):
            encode_bytes("\ud800")

    def test_casefold_expansion_maps_to_original_coordinates(self):
        original = "Ёж Straße İ"
        view = casefold_view(original)
        self.assertEqual(view.text, "ёж strasse i\u0307")
        start = view.text.index("ss")
        left, right = view.original_span(start, start + 2)
        self.assertEqual(original[left:right], "ß")
        self.assertEqual(original, decode_bytes(encode_bytes(original)))
        with self.assertRaises(ValueError):
            view.original_span(True, 2)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ai2-open-observation-")
        self.path = Path(self.directory.name) / "sources.sqlite"
        self.archive = ObservationArchive(
            self.path,
            create=True,
            limits=ArchiveLimits(chunk_chars=3),
        )

    def tearDown(self):
        self.archive.close()
        self.directory.cleanup()

    def _import(self, text, key="source", **kwargs):
        return self.archive.import_text(
            text,
            namespace="test",
            external_key=key,
            group_id="document-family",
            **kwargs,
        )

    def test_long_multisentence_unknown_text_and_raw_file_roundtrip(self):
        original = ("Кошелёк аккуратно лежит. 你好 😀\r\n" * 100).encode("utf-8")
        path = Path(self.directory.name) / "input.txt"
        path.write_bytes(original)
        source = self.archive.import_file(
            path,
            namespace="test",
            external_key="file",
            group_id="family",
            metadata={"author": None, "kind": "unclassified"},
        )
        restored = b"".join(
            o.raw for o in self.archive.iter_observations(source.source_id, 1)
        )
        self.assertEqual(restored, original)
        self.assertEqual(source.sha256, hashlib.sha256(original).hexdigest())
        self.assertGreater(source.char_count, 2048)
        self.assertIsNone(self.archive.verify()["semantic_accuracy"])

    def test_reimport_is_idempotent_and_changed_source_has_immutable_revision(self):
        original = self._import("Первый текст", metadata={"type": "statement"})
        repeat = self._import("Первый текст", metadata={"type": "statement"})
        changed = self._import("Новый текст", metadata={"type": "statement"})
        metadata_changed = self._import("Новый текст", metadata={"type": "quotation"})
        self.assertEqual(original, repeat)
        self.assertEqual(changed.source_id, original.source_id)
        self.assertEqual((changed.version, metadata_changed.version), (2, 3))
        self.assertEqual(
            self.archive.read_span(original.source_id, 1, 0, 12), "Первый текст"
        )
        self.assertEqual(self.archive.get_source(original.source_id).version, 3)
        self.assertEqual(self.archive.verify()["source_versions"], 3)

    def test_source_family_cannot_drift_across_revisions(self):
        source = self._import("a")
        with self.assertRaisesRegex(ValueError, "group"):
            self.archive.import_text(
                "b",
                namespace="test",
                external_key="source",
                group_id="test-set",
            )
        self.assertEqual(self.archive.get_source(source.source_id), source)

    def test_empty_source_is_preserved_and_verified(self):
        source = self._import("")
        self.assertEqual(source.byte_count, 0)
        self.assertEqual(source.observation_count, 0)
        self.assertEqual(tuple(self.archive.iter_observations(source.source_id, 1)), ())
        self.assertEqual(self.archive.verify()["source_versions"], 1)

    def test_utf8_error_rolls_back_all_chunks_and_preserves_prior_revision(self):
        source = self._import("before")
        with self.assertRaises(UnicodeError):
            self.archive.import_blocks(
                [b"many chunks inserted before the error", b"\xf0"],
                namespace="test",
                external_key="source",
                group_id="document-family",
            )
        self.assertEqual(self.archive.get_source(source.source_id), source)
        self.assertEqual(self.archive.verify()["source_versions"], 1)

    def test_byte_budget_failure_leaves_no_partial_source(self):
        with (
            ObservationArchive(
                self.path, limits=ArchiveLimits(chunk_chars=1, max_source_bytes=4)
            ) as small,
            self.assertRaisesRegex(ValueError, "budget"),
        ):
            small.import_blocks(
                [b"abcd", b"e"],
                namespace="test",
                external_key="bad",
                group_id="g",
            )
        self.assertEqual(self.archive.sources("test"), ())
        self.assertEqual(self.archive.verify()["observations"], 0)

    def test_page_and_span_coordinates_cross_multibyte_chunks(self):
        text = "😀Alpha\r\nאבгде"
        source = self._import(text)
        page = self.archive.observations(source.source_id, 1, offset=1, limit=2)
        self.assertEqual([o.ordinal for o in page], [1, 2])
        for observation in page:
            self.assertEqual(
                text[observation.char_start : observation.char_end], observation.text
            )
            self.assertEqual(
                text.encode()[observation.byte_start : observation.byte_end],
                observation.raw,
            )
        self.assertEqual(self.archive.read_span(source.source_id, 1, 1, 10), text[1:10])
        for start, end in ((-1, 3), (0, 99), (True, 4), (2, 2)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.archive.read_span(source.source_id, 1, start, end)

    def test_raw_records_cannot_be_updated_or_deleted(self):
        self._import("original")
        with sqlite3.connect(self.path) as connection:
            for sql in ("UPDATE observations SET raw=x'61'", "DELETE FROM revisions"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(sql)

    def test_corruption_is_detected_even_if_sqlite_file_is_structurally_valid(self):
        source = self._import("original")
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TRIGGER observations_update")
            connection.execute("UPDATE observations SET raw=x'616263' WHERE ordinal=0")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.archive.observations(source.source_id, 1)
        with self.assertRaises(ValueError):
            self.archive.verify()

    def test_unrelated_archive_and_symlink_are_not_overwritten(self):
        path = Path(self.directory.name) / "other.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE user_data (value TEXT)")
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            ObservationArchive(path, create=True)
        self.assertEqual(path.read_bytes(), before)
        link = Path(self.directory.name) / "link.sqlite"
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            ObservationArchive(link)

    def test_process_death_during_import_restores_last_committed_revision(self):
        source = self._import("committed original")
        code = """
import os, sys
from text_factors.observations import ArchiveLimits, ObservationArchive
def blocks():
    yield b"uncommitted chunks"
    os._exit(79)
with ObservationArchive(sys.argv[1], limits=ArchiveLimits(chunk_chars=2)) as archive:
    archive.import_blocks(
        blocks(), namespace="test", external_key="source", group_id="document-family"
    )
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.path)],
            timeout=15,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
        )
        self.assertEqual(result.returncode, 79, result.stderr.decode())
        with ObservationArchive(self.path) as recovered:
            self.assertEqual(recovered.get_source(source.source_id), source)
            self.assertEqual(recovered.verify()["source_versions"], 1)


if __name__ == "__main__":
    unittest.main()

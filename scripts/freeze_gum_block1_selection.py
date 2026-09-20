#!/usr/bin/env python3
"""Materialize a checksum-complete GUM manifest from a pre-frozen selection.

This script never sees model predictions. It validates the committed selection
against the pinned GUM splits, hashes the exact source files, and writes the
manifest consumed by prepare_gum_mentions.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SELECTION_SCHEMA = "ai2-block1-gum-selection-v1"
MANIFEST_SCHEMA = "ai2-gum-pilot-manifest-v1"
MAX_FILE_BYTES = 4 * 1024 * 1024
GENRE_PREFIXES = ("GUM_academic_", "GUM_conversation_")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"selection file exceeds {MAX_FILE_BYTES} bytes")
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if type(value) is not dict:
        raise ValueError("selection must be a JSON object")
    return value


def _read_bounded(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected regular source file: {path}")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"source file exceeds {MAX_FILE_BYTES} bytes: {path}")
    return data


def _parse_splits(path: Path) -> dict[str, list[str]]:
    text = _read_bounded(path).decode("utf-8", errors="strict")
    current: str | None = None
    result: dict[str, list[str]] = {"train": [], "dev": [], "test": [], "test2": []}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            heading = line[3:].strip()
            current = heading if heading in result else None
            continue
        if current is None:
            continue
        match = re.fullmatch(r"\*\s+([A-Za-z0-9_]+)", line)
        if match:
            result[current].append(match.group(1))
    if any(not result[key] for key in ("train", "dev", "test")):
        raise ValueError("could not parse required GUM splits")
    return result


def _expected_selection(splits: dict[str, list[str]]) -> list[tuple[str, str]]:
    expected: list[tuple[str, str]] = []
    for prefix in GENRE_PREFIXES:
        train = [item for item in splits["train"] if item.startswith(prefix)]
        dev = [item for item in splits["dev"] if item.startswith(prefix)]
        test = [item for item in splits["test"] if item.startswith(prefix)]
        if len(train) < 8 or len(dev) != 2 or len(test) != 2:
            raise ValueError(f"unexpected pinned split cardinality for {prefix}")
        expected.extend((item, "train") for item in train[:8])
    for prefix in GENRE_PREFIXES:
        dev = [item for item in splits["dev"] if item.startswith(prefix)]
        expected.extend((item, "validation") for item in dev)
    for prefix in GENRE_PREFIXES:
        test = [item for item in splits["test"] if item.startswith(prefix)]
        expected.extend((item, "test") for item in test)
    return expected


def _validate_selection(
    selection: dict[str, Any],
    gum_root: Path,
) -> list[dict[str, Any]]:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unsupported frozen selection schema")
    revision = selection.get("revision")
    if type(revision) is not str or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("selection requires a pinned 40-hex GUM revision")
    if selection.get("test_predictions_seen_before_freeze") is not False:
        raise ValueError("selection must be frozen before test predictions")
    documents = selection.get("documents")
    if type(documents) is not list or len(documents) != 24:
        raise ValueError("frozen selection must contain exactly 24 documents")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in documents:
        if type(item) is not dict:
            raise ValueError("selection document must be an object")
        if set(item) != {"document_id", "group_id", "split", "language"}:
            raise ValueError("unexpected frozen selection document fields")
        document_id = item["document_id"]
        if (
            type(document_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_]+", document_id)
            or document_id in seen
        ):
            raise ValueError("invalid or duplicate frozen document ID")
        seen.add(document_id)
        if item["group_id"] != document_id:
            raise ValueError("whole frozen documents must remain their own groups")
        if item["split"] not in {"train", "validation", "test"}:
            raise ValueError("invalid frozen split")
        if item["language"] != "en":
            raise ValueError("this frozen experiment is English-only")
        normalized.append(dict(item))

    splits = _parse_splits(gum_root / "splits.md")
    expected = _expected_selection(splits)
    actual = [(item["document_id"], item["split"]) for item in normalized]
    if actual != expected:
        raise ValueError(
            "committed selection does not match the predeclared pinned policy"
        )
    return normalized


def _file_record(path: Path, gum_root: Path) -> dict[str, Any]:
    data = _read_bounded(path)
    return {
        "path": str(path.relative_to(gum_root)).replace("\\", "/"),
        "sha256": _sha256(data),
        "bytes": len(data),
    }


def build_manifest(
    selection_path: Path,
    gum_root: Path,
) -> dict[str, Any]:
    selection = _read_json(selection_path)
    documents = _validate_selection(selection, gum_root)
    selection_sha = _sha256(_canonical_bytes(selection))
    manifest_documents: list[dict[str, Any]] = []
    for item in documents:
        document_id = item["document_id"]
        conll = gum_root / "coref" / "gum" / "conll" / f"{document_id}.conll"
        tsv = gum_root / "coref" / "gum" / "tsv" / f"{document_id}.tsv"
        xml = gum_root / "xml" / f"{document_id}.xml"
        xml_data = _read_bounded(xml)
        metadata = ET.fromstring(xml_data)
        if metadata.tag != "text" or metadata.get("id") != document_id:
            raise ValueError(f"{document_id}: invalid XML identity")
        expected_partition = "dev" if item["split"] == "validation" else item["split"]
        if metadata.get("partition") != expected_partition:
            raise ValueError(f"{document_id}: XML partition mismatch")
        source_url = metadata.get("sourceURL")
        if not source_url:
            raise ValueError(f"{document_id}: missing source URL")
        manifest_documents.append(
            {
                **item,
                "source_url": source_url,
                "license": "Texts: CC-BY-2.5; annotations: CC-BY-4.0",
                "files": {
                    "conll": _file_record(conll, gum_root),
                    "tsv": _file_record(tsv, gum_root),
                    "xml": _file_record(xml, gum_root),
                },
            }
        )

    return {
        "schema": MANIFEST_SCHEMA,
        "repository": selection["repository"],
        "revision": selection["revision"],
        "source_commit_url": (
            f"{selection['repository']}/commit/{selection['revision']}"
        ),
        "selection_spec_sha256": selection_sha,
        "selection_policy": selection["selection_policy"],
        "test_predictions_seen_before_freeze": False,
        "licenses": {
            "texts": {
                "id": "CC-BY-2.5",
                "url": "https://creativecommons.org/licenses/by/2.5/",
            },
            "annotations": {
                "id": "CC-BY-4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
        },
        "documents": manifest_documents,
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--gum-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.selection, args.gum_root)
    _write_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "frozen_manifest_materialized",
                "documents": len(manifest["documents"]),
                "selection_spec_sha256": manifest["selection_spec_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

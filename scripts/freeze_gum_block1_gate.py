"""Freeze the predeclared GUM Block 1 selection into a checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SELECTION_SCHEMA = "ai2-gum-block1-gate-selection-v1"
MANIFEST_SCHEMA = "ai2-gum-pilot-manifest-v1"
LICENSE = "Texts: CC-BY-2.5; annotations: CC-BY-4.0"


def _read(path: Path, limit: int = 8_000_000) -> bytes:
    data = path.read_bytes()
    if not data or len(data) > limit:
        raise ValueError(f"invalid or oversized source file: {path}")
    return data


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _record(root: Path, relative: str, revision: str) -> dict[str, Any]:
    data = _read(root / relative)
    return {
        "path": relative,
        "url": (
            "https://raw.githubusercontent.com/amir-zeldes/gum/"
            f"{revision}/{relative}"
        ),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def freeze(input_dir: Path, selection_path: Path) -> dict[str, Any]:
    selection = json.loads(_read(selection_path, 1_000_000))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unsupported Block 1 selection schema")
    revision = selection.get("revision")
    if type(revision) is not str or len(revision) != 40:
        raise ValueError("selection requires an exact 40-character revision")
    if _git_head(input_dir) != revision:
        raise ValueError("GUM checkout revision differs from frozen selection")

    split_map = selection.get("splits")
    if type(split_map) is not dict or set(split_map) != {"train", "validation", "test"}:
        raise ValueError("selection must contain train/validation/test")
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        ids = split_map[split]
        if type(ids) is not list or not ids:
            raise ValueError(f"{split} selection must be a nonempty list")
        for document_id in ids:
            if (
                type(document_id) is not str
                or not document_id.startswith(("GUM_academic_", "GUM_bio_"))
                or document_id in seen
            ):
                raise ValueError("invalid or duplicate selected document")
            seen.add(document_id)
            ordered.append((split, document_id))

    documents = []
    for split, document_id in ordered:
        files = {
            "conll": _record(
                input_dir,
                f"coref/gum/conll/{document_id}.conll",
                revision,
            ),
            "tsv": _record(
                input_dir,
                f"coref/gum/tsv/{document_id}.tsv",
                revision,
            ),
            "xml": _record(input_dir, f"xml/{document_id}.xml", revision),
        }
        xml_bytes = _read(input_dir / files["xml"]["path"])
        root = ET.fromstring(xml_bytes)
        if root.tag != "text" or root.get("id") != document_id:
            raise ValueError(f"{document_id}: invalid XML identity")
        expected_partition = "dev" if split == "validation" else split
        if root.get("partition") != expected_partition:
            raise ValueError(f"{document_id}: published partition mismatch")
        source_url = root.get("sourceURL")
        if not source_url:
            raise ValueError(f"{document_id}: missing sourceURL")
        documents.append(
            {
                "document_id": document_id,
                "group_id": document_id,
                "split": split,
                "language": "en",
                "source_url": source_url,
                "license": LICENSE,
                "files": files,
            }
        )

    splits = _record(input_dir, "splits.md", revision)
    annotation = _record(input_dir, "coref/gum/README.md", revision)
    return {
        "schema": MANIFEST_SCHEMA,
        "repository": selection["repository"],
        "revision": revision,
        "source_commit_url": f"https://github.com/amir-zeldes/gum/commit/{revision}",
        "license_url": f"https://github.com/amir-zeldes/gum/blob/{revision}/LICENSE.md",
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
        "selection_policy": selection["selection_policy"],
        "quality_gate": selection["quality_gate"],
        "official_splits": splits,
        "annotation_documentation": {
            "source": "https://gucorpling.org/gum/annotations.html#ref",
            "native_scheme": annotation,
        },
        "text_representation": (
            "Exact CoNLL tokens joined by one ASCII space, independently checked "
            "against WebAnno TSV token offsets; original website bytes/typography "
            "not reconstructed."
        ),
        "excluded": (
            "Summary fields, POS, lemmas, syntax, entity-type features, Wikification "
            "features, bridging and split-antecedent membership edges. Native opaque "
            "cluster labels are identifiers only."
        ),
        "documents": documents,
        "block1_selection": selection,
    }


def _write_atomic(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".block1-gate-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise ValueError("output exists; use --overwrite")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "GUM_BLOCK1_GATE_SELECTION.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = freeze(args.input_dir, args.selection)
        _write_atomic(args.output, manifest, args.overwrite)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ET.ParseError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Block 1 manifest freeze failed: {error}")
        return 2
    print(
        json.dumps(
            {
                "status": "frozen",
                "documents": len(manifest["documents"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

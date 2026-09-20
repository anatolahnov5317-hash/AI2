#!/usr/bin/env python3
"""Convert pinned native GUM CoNLL mentions to a local learning corpus.

This adapter uses only tokens and existing coreference annotations. The result
preserves the corpus token strings separated by one ASCII space, not the source
website's original typography. It does not perform or repair linguistic markup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "ai2-gum-pilot-manifest-v1"
CORPUS_SCHEMA = "ai2-mention-learning-corpus-v1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_DOCUMENTS = 1000
ANNOTATION_POLICY = (
    "Complete only within the native GUM coreference scheme: includes "
    "singletons, nested and overlapping contiguous mentions, non-named "
    "compounds, predication, indefinite and verbal mentions. Native CoNLL "
    "identity clusters are preserved as opaque document-local IDs. Bridging "
    "and split-antecedent membership edges are not represented or evaluated. "
    "Discontinuous mention encodings are unsupported and rejected. No "
    "summaries, entity-type features, Wikification features, or generated "
    "annotations are used by this conversion."
)
TEXT_REPRESENTATION = (
    "Lossless preservation of source CoNLL token strings joined by a single "
    "ASCII space; offsets are Unicode code points in that reconstructed "
    "corpus-tokenized text. Original website bytes, whitespace, and document "
    "formatting are not reconstructed."
)


def read_bounded(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    """Read at most limit + 1 bytes, with no unbounded allocation on bad input."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular, non-symlink file: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds {limit} bytes: {path}")
    return data


def parse_conll(
    data: str, document_id: str, max_tokens: int = 10000
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Read native three-column GUM format, preserving nested/crossing spans."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    open_mentions: dict[str, list[int]] = {}
    mentions: list[dict[str, Any]] = []
    occupied: dict[tuple[int, int], str] = {}
    begun = ended = False
    text_end = 0
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line:
            continue
        if line.startswith("# begin document "):
            if begun or ended or line != f"# begin document {document_id}":
                raise ValueError(f"{document_id}: invalid document header")
            begun = True
            continue
        if line == "# end document":
            if not begun or ended:
                raise ValueError(f"{document_id}: invalid document footer")
            ended = True
            continue
        if line.startswith("#"):
            # Corpus summaries/comments never become model input or gold.
            continue
        if not begun or ended:
            raise ValueError(f"{document_id}:{line_number}: token outside document")
        fields = line.split("\t")
        if len(fields) != 3 or fields[0] != str(len(tokens)):
            raise ValueError(f"{document_id}:{line_number}: invalid token row/index")
        token, annotations = fields[1:]
        if not token or any(char.isspace() for char in token):
            raise ValueError(f"{document_id}:{line_number}: invalid token string")
        if len(tokens) >= max_tokens:
            raise ValueError(f"{document_id}: exceeds {max_tokens} source tokens")
        start = text_end + bool(tokens)
        text_end = start + len(token)
        offsets.append((start, text_end))
        tokens.append(token)
        if annotations == "_":
            continue
        position = 0
        while position < len(annotations):
            opening = annotations[position] == "("
            if opening:
                position += 1
            match = re.match(r"[^()|\[\]\s]+", annotations[position:])
            if match is None:
                raise ValueError(
                    f"{document_id}:{line_number}: malformed/unsupported "
                    "annotation; discontinuous spans are not supported"
                )
            entity = match.group()
            position += len(entity)
            closing = position < len(annotations) and annotations[position] == ")"
            if closing:
                position += 1
            if not opening and not closing:
                raise ValueError(f"{document_id}:{line_number}: missing span boundary")
            if opening:
                open_mentions.setdefault(entity, []).append(len(tokens) - 1)
            if closing:
                if not open_mentions.get(entity):
                    raise ValueError(f"{document_id}:{line_number}: unmatched close")
                first = open_mentions[entity].pop()
                span = (offsets[first][0], text_end)
                if span in occupied:
                    raise ValueError(
                        f"{document_id}: duplicate exact mention span {span}; "
                        f"entities {occupied[span]!r} and {entity!r}"
                    )
                occupied[span] = entity
                mentions.append({"start": span[0], "end": span[1], "entity_id": entity})
            if position < len(annotations) and annotations[position] == "|":
                position += 1
                if position == len(annotations):
                    raise ValueError(f"{document_id}: trailing annotation separator")
    if not begun or not ended or not tokens:
        raise ValueError(f"{document_id}: incomplete/empty document")
    if any(open_mentions.values()):
        raise ValueError(f"{document_id}: unclosed mention")
    mentions.sort(key=lambda mention: (mention["start"], mention["end"]))
    return " ".join(tokens), mentions, tokens


def verify_tsv_tokens(data: str, tokens: list[str], document_id: str) -> None:
    """Independently check native WebAnno token strings and code-point offsets."""
    index = 0
    cursor = 0
    for line in data.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3 or index >= len(tokens):
            raise ValueError(f"{document_id}: incompatible TSV token count")
        end = cursor + len(tokens[index])
        if fields[1] != f"{cursor}-{end}" or fields[2] != tokens[index]:
            raise ValueError(f"{document_id}: TSV/CoNLL token or offset mismatch")
        index += 1
        cursor = end + 1
    if index != len(tokens):
        raise ValueError(f"{document_id}: incompatible TSV token count")


def _source_file(root: Path, record: dict[str, Any]) -> bytes:
    relative = record.get("path")
    checksum = record.get("sha256")
    blob_sha = record.get("git_blob_sha1")
    if not isinstance(relative, str):
        raise ValueError("source record requires path")
    if (checksum is None) == (blob_sha is None):
        raise ValueError(
            "source record requires exactly one of sha256 or git_blob_sha1"
        )
    if checksum is not None and not isinstance(checksum, str):
        raise ValueError("sha256 must be a string")
    if blob_sha is not None and not isinstance(blob_sha, str):
        raise ValueError("git_blob_sha1 must be a string")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("source path must stay within the input directory")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("source path escapes the input directory")
    data = read_bounded(path)
    if checksum is not None and hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError(f"source checksum mismatch: {relative}")
    if blob_sha is not None:
        material = b"blob " + str(len(data)).encode() + b"\0" + data
        if hashlib.sha1(material).hexdigest() != blob_sha:
            raise ValueError(f"source git blob mismatch: {relative}")
    return data


def prepare_corpus(
    input_dir: Path, manifest_path: Path, max_tokens: int = 10000
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a predetermined manifest; do not choose examples by model output."""
    manifest_bytes = read_bounded(manifest_path)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest must use schema {MANIFEST_SCHEMA}")
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_DOCUMENTS:
        raise ValueError(f"manifest needs 1..{MAX_DOCUMENTS} documents")
    for key in ("revision", "repository", "selection_policy", "licenses"):
        if not manifest.get(key):
            raise ValueError(f"manifest missing {key}")
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_text: dict[str, str] = {}
    counts: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("document entries must be objects")
        name = entry.get("document_id")
        split = entry.get("split")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("invalid document_id")
        if name in seen:
            raise ValueError(f"duplicate document: {name}")
        seen.add(name)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split for {name}")
        if entry.get("group_id") != name:
            raise ValueError(f"{name}: group_id must retain the complete document")
        sources = entry.get("files")
        if not isinstance(sources, dict) or set(sources) != {"conll", "tsv", "xml"}:
            raise ValueError(f"{name}: need conll, tsv and xml source files")
        raw = {key: _source_file(input_dir, record) for key, record in sources.items()}
        text, mentions, tokens = parse_conll(
            raw["conll"].decode("utf-8"), name, max_tokens
        )
        verify_tsv_tokens(raw["tsv"].decode("utf-8"), tokens, name)
        metadata = ET.fromstring(raw["xml"])
        if metadata.tag != "text" or metadata.get("id") != name:
            raise ValueError(f"{name}: incompatible XML document identity")
        official_split = "dev" if split == "validation" else split
        if metadata.get("partition") != official_split:
            raise ValueError(f"{name}: manifest contradicts the official partition")
        source_url = metadata.get("sourceURL")
        declared_source_url = entry.get("source_url")
        if not source_url:
            raise ValueError(f"{name}: XML source URL is missing")
        if declared_source_url is not None and source_url != declared_source_url:
            raise ValueError(f"{name}: inconsistent source URL")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_text:
            raise ValueError(
                f"{name}: duplicate document text with {seen_text[text_hash]}"
            )
        seen_text[text_hash] = name
        documents.append(
            {
                "document_id": name,
                "group_id": name,
                "split": split,
                "text": text,
                "mentions": mentions,
                "coverage": "complete",
                "language": "en",
                "provenance": {
                    "source_url": source_url,
                    "license": entry["license"],
                    "annotation_origin": "GUM published manually reviewed coreference",
                    "revision": manifest["revision"],
                    "annotation_scheme": "GUM native CoNLL identity clusters",
                    "text_representation": TEXT_REPRESENTATION,
                    "original_website_bytes_preserved": False,
                    "token_strings_preserved": True,
                    "source_files": sources,
                    "author": metadata.get("author", "GUM contributors"),
                    "summaries_excluded": True,
                },
            }
        )
        counts.append(
            {
                "document_id": name,
                "split": split,
                "tokens": len(tokens),
                "mentions": len(mentions),
                "entities": len({mention["entity_id"] for mention in mentions}),
            }
        )
    provenance = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "licenses": manifest["licenses"],
        "selection_policy": manifest["selection_policy"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "annotation_policy": ANNOTATION_POLICY,
        "text_representation": TEXT_REPRESENTATION,
        "annotation_modifications": "Deterministic format conversion; no relabeling",
        "language_scope": "English technical pilot; no claim of Russian accuracy",
        "discontinuous_spans_supported": False,
        "exact_duplicate_documents_checked": True,
        "semantic_independence_verified": False,
    }
    corpus = {"schema": CORPUS_SCHEMA, "documents": documents, "provenance": provenance}
    receipt = {"documents": counts, "total_documents": len(counts)}
    return corpus, receipt


def write_atomic(path: Path, value: dict[str, Any], overwrite: bool = False) -> None:
    """Create an atomic complete JSON file; failed conversion never changes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".gum-", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "GUM_PILOT_MANIFEST.json",
    )
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        corpus, receipt = prepare_corpus(args.input_dir, args.manifest, args.max_tokens)
        write_atomic(args.output, corpus, args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
        print(f"GUM conversion failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

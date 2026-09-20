"""Explicit supervised coverage and provenance for open mention learning.

An omitted mention is a negative example only within a declared complete
annotation policy. Corpus labels are external evidence, never model output.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from ..conversation.persistence import read_json
from .archive import ObservationArchive
from .corpus import freeze_corpus
from .schema import canonical_json, integer, text_field

LEARNING_CORPUS_SCHEMA = "ai2-mention-learning-corpus-v1"
MAX_CORPUS_BYTES = 8 * 1024 * 1024


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_learning_corpus(value: Any) -> dict[str, Any]:
    """Validate all splits without fitting a representation on their contents."""
    if type(value) is not dict or value.get("schema") != LEARNING_CORPUS_SCHEMA:
        raise ValueError("unsupported mention learning corpus schema")
    documents = value.get("documents")
    if type(documents) is not list or not 1 <= len(documents) <= 256:
        raise ValueError("corpus requires 1..256 documents")
    if type(value.get("provenance")) is not dict:
        raise ValueError("corpus requires explicit provenance and annotation policy")
    text_field(value["provenance"].get("annotation_policy"), "annotation policy")
    identities: set[str] = set()
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    total_chars = 0
    for document in documents:
        if type(document) is not dict:
            raise ValueError("document must be an object")
        for key in ("document_id", "group_id", "language"):
            text_field(document.get(key), key, cap=512)
        identity = document["document_id"]
        if identity in identities:
            raise ValueError("duplicate document identity")
        identities.add(identity)
        split = document.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("explicit train/validation/test split required")
        text = document.get("text")
        if type(text) is not str or not 1 <= len(text) <= 200_000:
            raise ValueError("document must contain 1..200000 codepoints")
        text.encode("utf-8", errors="strict")
        total_chars += len(text)
        if total_chars > 2_000_000:
            raise ValueError("corpus exceeds its character budget")
        if document.get("coverage") != "complete":
            raise ValueError("training negatives require declared complete coverage")
        provenance = document.get("provenance")
        if type(provenance) is not dict:
            raise ValueError("document requires annotation provenance")
        for key in ("source_url", "license", "annotation_origin"):
            text_field(provenance.get(key), key, cap=4096)
        for mapping, key, description in (
            (groups, document["group_id"], "source group"),
            (contents, hashlib.sha256(text.encode()).hexdigest(), "identical text"),
        ):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"{description} leaks between splits")
            mapping[key] = split
        mentions = document.get("mentions")
        if type(mentions) is not list or len(mentions) > 5000:
            raise ValueError("invalid mention count")
        spans = set()
        for mention in mentions:
            if type(mention) is not dict:
                raise ValueError("mention must be an object")
            start, end = mention.get("start"), mention.get("end")
            start = integer(start, "mention start", minimum=0)
            end = integer(end, "mention end", minimum=1)
            if start >= end or end > len(text) or (start, end) in spans:
                raise ValueError("empty or duplicate mention span")
            spans.add((start, end))
            if "surface" in mention and mention["surface"] != text[start:end]:
                raise ValueError("mention surface differs from original coordinates")
            if "entity_id" not in mention:
                raise ValueError("mention requires explicit entity_id or null")
            if mention["entity_id"] is not None:
                text_field(mention["entity_id"], "external entity ID", cap=1024)
    # Also reject nonfinite/unsupported metadata before any training or writes.
    canonical_json(value)
    return value


def load_learning_corpus(path: str | Path) -> dict[str, Any]:
    return validate_learning_corpus(read_json(Path(path), max_bytes=MAX_CORPUS_BYTES))


def split_documents(corpus: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown corpus split")
    return sorted(
        (doc for doc in corpus["documents"] if doc["split"] == split),
        key=lambda doc: doc["document_id"],
    )


def corpus_summary(corpus: dict[str, Any]) -> dict[str, Any]:
    """Descriptive counts are not a claim of semantic independence."""
    validate_learning_corpus(corpus)
    result: dict[str, Any] = {
        "fingerprint": fingerprint(corpus),
        "annotation_policy": corpus["provenance"]["annotation_policy"],
        "near_duplicate_independence_verified": False,
        "splits": {},
    }
    train_surfaces = {
        doc["text"][mention["start"] : mention["end"]].casefold()
        for doc in split_documents(corpus, "train")
        for mention in doc["mentions"]
    }
    for split in ("train", "validation", "test"):
        documents = split_documents(corpus, split)
        result["splits"][split] = {
            "documents": len(documents),
            "document_ids": [doc["document_id"] for doc in documents],
            "groups": len({doc["group_id"] for doc in documents}),
            "languages": dict(Counter(doc["language"] for doc in documents)),
            "characters": sum(len(doc["text"]) for doc in documents),
            "mentions": sum(len(doc["mentions"]) for doc in documents),
            "unseen_surface_mentions": sum(
                doc["text"][mention["start"] : mention["end"]].casefold()
                not in train_surfaces
                for doc in documents
                for mention in doc["mentions"]
            ),
        }
    return result


def archive_learning_corpus(
    archive: ObservationArchive,
    corpus: dict[str, Any],
    *,
    namespace: str,
) -> dict[str, Any]:
    """Import complete documents; retry resumes at document transaction boundaries.

    The source and its external annotation batch are separate transactions. A
    failed batch does not freeze a corpus; retry reuses the idempotent source.
    """
    validate_learning_corpus(corpus)
    assignments = []
    for document in sorted(corpus["documents"], key=lambda doc: doc["document_id"]):
        source = archive.import_text(
            document["text"],
            namespace=namespace,
            external_key=document["document_id"],
            group_id=document["group_id"],
            metadata={
                "provenance": document["provenance"],
                "language": document["language"],
                "coverage": document["coverage"],
                "annotation_policy": corpus["provenance"]["annotation_policy"],
            },
        )
        entities: dict[str, str] = {}
        for mention in document["mentions"]:
            if mention["entity_id"] is not None:
                entities.setdefault(
                    mention["entity_id"],
                    document["text"][mention["start"] : mention["end"]],
                )
        references = {identity: f"r{index}" for index, identity in enumerate(entities)}
        archive.annotate(
            {
                "schema": "ai2-open-annotations-v1",
                "source_id": source.source_id,
                "source_version": source.version,
                "annotator": document["provenance"]["annotation_origin"],
                "evidence": document["provenance"]["source_url"],
                "instances": [
                    {
                        "ref": references[identity],
                        "external_key": fingerprint(
                            [document["document_id"], identity]
                        ),
                        "label": surface[:512],
                    }
                    for identity, surface in entities.items()
                ],
                "mentions": [
                    {
                        "start": mention["start"],
                        "end": mention["end"],
                        "surface": document["text"][mention["start"] : mention["end"]],
                        "candidates": (
                            []
                            if mention["entity_id"] is None
                            else [references[mention["entity_id"]]]
                        ),
                        "selected": references.get(mention["entity_id"]),
                        "expected_version": 0,
                    }
                    for mention in document["mentions"]
                ],
            }
        )
        assignments.append(
            {
                "source_id": source.source_id,
                "source_version": source.version,
                "split": document["split"],
            }
        )
    return freeze_corpus(archive, assignments)

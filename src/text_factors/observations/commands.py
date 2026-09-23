"""CLI entry points for importing and inspecting open source observations."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..conversation.persistence import _atomic_write, encode_json, read_json
from .archive import ObservationArchive
from .corpus import CORPUS_SCHEMA, freeze_corpus, validate_corpus
from .pilot_corpus import (
    PILOT_ASSIGNMENTS_SCHEMA,
    PILOT_CORPUS_SCHEMA,
    freeze_pilot_corpus,
    validate_pilot_corpus,
)
from .schema import ArchiveLimits, fields


def _print(value: Any) -> None:
    print(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    try:
        with ObservationArchive(
            args.archive,
            create=args.operation == "import",
            limits=ArchiveLimits(
                chunk_chars=getattr(args, "chunk_chars", 2048),
                max_source_bytes=getattr(args, "max_bytes", 64 * 1024 * 1024),
            ),
        ) as archive:
            if args.operation == "import":
                record = archive.import_file(
                    args.input,
                    namespace=args.namespace,
                    external_key=args.source_key,
                    group_id=args.group,
                    metadata={"origin": args.origin, "author": args.author},
                )
                _print(
                    {
                        "status": "archived",
                        "source": record.to_dict(),
                        "semantic_status": "uninterpreted",
                    }
                )
            elif args.operation == "list":
                records = archive.sources(
                    args.namespace, after=args.after, limit=args.limit
                )
                _print(
                    {
                        "sources": [record.to_dict() for record in records],
                        "next_cursor": records[-1].source_id if records else None,
                    }
                )
            elif args.operation == "show":
                record = archive.get_source(args.source, args.version)
                page = archive.observations(
                    record.source_id,
                    record.version,
                    offset=args.offset,
                    limit=args.limit,
                )
                _print(
                    {
                        "source": record.to_dict(),
                        "observations": [item.to_dict() for item in page],
                        "next_offset": page[-1].ordinal + 1 if page else None,
                    }
                )
            elif args.operation == "annotate":
                batch = read_json(Path(args.data), max_bytes=8 * 1024 * 1024)
                _print(archive.annotate(batch))
            elif args.operation == "annotations":
                _print(
                    {
                        "annotations": archive.annotations(
                            args.source,
                            args.version,
                            offset=args.offset,
                            limit=args.limit,
                        )
                    }
                )
            elif args.operation == "verify":
                _print(archive.verify())
            elif args.operation == "freeze":
                specification = fields(
                    read_json(Path(args.data), max_bytes=8 * 1024 * 1024),
                    {"schema", "assignments"},
                )
                if specification["schema"] != "ai2-open-corpus-splits-v1":
                    raise ValueError("unsupported corpus assignment schema")
                manifest = freeze_corpus(archive, specification["assignments"])

                def existing_validator(value: Any) -> None:
                    if type(value) is not dict or value.get("schema") != CORPUS_SCHEMA:
                        raise ValueError("refusing to replace an unrelated file")
                    validate_corpus(archive, value)

                _atomic_write(
                    Path(args.output),
                    encode_json(manifest, max_bytes=8 * 1024 * 1024),
                    overwrite=args.overwrite,
                    existing_validator=existing_validator,
                )
                _print(
                    {
                        "status": "frozen",
                        "fingerprint": manifest["fingerprint"],
                        "sources": len(manifest["payload"]["members"]),
                    }
                )
            elif args.operation == "validate-corpus":
                manifest = read_json(Path(args.data), max_bytes=8 * 1024 * 1024)
                _print(validate_corpus(archive, manifest))
            elif args.operation == "freeze-pilot":
                specification = fields(
                    read_json(Path(args.data), max_bytes=8 * 1024 * 1024),
                    {
                        "schema",
                        "assignments",
                        "restricted_receipts",
                        "near_duplicate_review",
                    },
                )
                if specification["schema"] != PILOT_ASSIGNMENTS_SCHEMA:
                    raise ValueError("unsupported pilot corpus assignment schema")
                manifest = freeze_pilot_corpus(
                    archive,
                    specification["assignments"],
                    restricted_receipts=specification["restricted_receipts"],
                    near_duplicate_review=specification["near_duplicate_review"],
                )

                def existing_pilot_validator(value: Any) -> None:
                    if (
                        type(value) is not dict
                        or value.get("schema") != PILOT_CORPUS_SCHEMA
                    ):
                        raise ValueError("refusing to replace an unrelated file")
                    validate_pilot_corpus(archive, value)

                _atomic_write(
                    Path(args.output),
                    encode_json(manifest, max_bytes=8 * 1024 * 1024),
                    overwrite=args.overwrite,
                    existing_validator=existing_pilot_validator,
                )
                _print(
                    {
                        "status": "frozen",
                        "fingerprint": manifest["fingerprint"],
                        "open_source_versions": len(
                            manifest["payload"]["open_members"]
                        ),
                    }
                )
            elif args.operation == "validate-pilot":
                manifest = read_json(Path(args.data), max_bytes=8 * 1024 * 1024)
                _print(validate_pilot_corpus(archive, manifest))
    except sqlite3.Error as exc:
        raise ValueError(f"observation archive operation failed: {exc}") from exc
    return 0


def add_parsers(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "observations",
        help="archive open UTF-8 input and explicit instance annotations",
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    for name, help_text in (
        ("import", "atomically import a complete UTF-8 source without a word list"),
        ("list", "list source identities within a namespace"),
        ("show", "read a bounded page of original observations"),
        ("annotate", "apply explicit external mention/instance annotations"),
        ("annotations", "inspect current bindings of a source revision"),
        ("verify", "verify original bytes, coordinates and identity references"),
        ("freeze", "pin source/annotation versions and check corpus split leakage"),
        ("validate-corpus", "validate a version-pinned corpus against its archive"),
        ("freeze-pilot", "pin five pilot splits with detached sealed receipts"),
        ("validate-pilot", "validate a five-split pilot corpus manifest"),
    ):
        command = operations.add_parser(name, help=help_text)
        command.add_argument("--archive", required=True, help="SQLite archive path")
        command.set_defaults(handler=run)
        if name == "import":
            command.add_argument("--input", required=True, help="UTF-8 source file")
            command.add_argument("--namespace", required=True)
            command.add_argument("--source-key", required=True)
            command.add_argument(
                "--group", required=True, help="immutable source-family group"
            )
            command.add_argument("--origin", required=True, help="source provenance")
            command.add_argument("--author", default=None)
            command.add_argument("--chunk-chars", type=int, default=2048)
            command.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
        if name == "list":
            command.add_argument("--namespace", required=True)
            command.add_argument("--after", default="")
            command.add_argument("--limit", type=int, default=100)
        if name in {"show", "annotations"}:
            command.add_argument("--source", required=True)
            command.add_argument("--version", type=int, required=name == "annotations")
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--limit", type=int, default=20)
        if name in {
            "annotate",
            "freeze",
            "validate-corpus",
            "freeze-pilot",
            "validate-pilot",
        }:
            command.add_argument("--data", required=True, help="annotation JSON file")
        if name in {"freeze", "freeze-pilot"}:
            command.add_argument("--output", required=True)
            command.add_argument("--overwrite", action="store_true")

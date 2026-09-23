"""Transactional, append-only SQLite archive for source evidence and identity.

Every import is one transaction. Chunking bounds working memory, not meaning.
Namespaces isolate identities; they are not an authentication mechanism.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .encoding import utf8_chunks
from .schema import (
    ArchiveLimits,
    Binding,
    Instance,
    Mention,
    Observation,
    SourceRecord,
    canonical_json,
    fields,
    integer,
    text_field,
)

APPLICATION_ID = 0x4149324F
SCHEMA_VERSION = 2
ANNOTATION_SCHEMA = "ai2-open-annotations-v1"
IDENTITY_PAIRS_SCHEMA = "ai2-open-identity-pairs-v1"

_SCHEMA = """
CREATE TABLE sources (
 source_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, external_key TEXT NOT NULL,
 group_id TEXT NOT NULL, UNIQUE(namespace, external_key)
);
CREATE TABLE revisions (
 source_id TEXT NOT NULL REFERENCES sources(source_id), version INTEGER NOT NULL,
 sha256 TEXT NOT NULL, byte_count INTEGER NOT NULL, char_count INTEGER NOT NULL,
 observation_count INTEGER NOT NULL, received_at TEXT NOT NULL, metadata TEXT NOT NULL,
 PRIMARY KEY(source_id, version), CHECK(version > 0)
);
CREATE TABLE observations (
 observation_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, version INTEGER NOT NULL,
 ordinal INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
 byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL, raw BLOB NOT NULL,
 sha256 TEXT NOT NULL, UNIQUE(source_id, version, ordinal),
 FOREIGN KEY(source_id, version) REFERENCES revisions(source_id, version)
 DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX observation_spans
 ON observations(source_id, version, char_start, char_end);
CREATE TABLE instances (
 instance_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, external_key TEXT NOT NULL,
 label TEXT NOT NULL, UNIQUE(namespace, external_key)
);
CREATE TABLE mentions (
 mention_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_version INTEGER NOT NULL,
 char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, surface TEXT NOT NULL,
 UNIQUE(source_id, source_version, char_start, char_end),
 FOREIGN KEY(source_id, source_version) REFERENCES revisions(source_id, version)
);
CREATE TABLE bindings (
 mention_id TEXT NOT NULL REFERENCES mentions(mention_id), version INTEGER NOT NULL,
 selected_id TEXT REFERENCES instances(instance_id), annotator TEXT NOT NULL,
 evidence TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(mention_id, version), CHECK(version > 0)
);
CREATE TABLE candidates (
 mention_id TEXT NOT NULL, binding_version INTEGER NOT NULL,
 instance_id TEXT NOT NULL REFERENCES instances(instance_id),
 PRIMARY KEY(mention_id, binding_version, instance_id),
 FOREIGN KEY(mention_id, binding_version) REFERENCES bindings(mention_id, version)
);
"""

_PAIR_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_pair_scopes (
 scope_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_version INTEGER NOT NULL,
 char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, policy_reference TEXT NOT NULL,
 UNIQUE(source_id, source_version, char_start, char_end),
 FOREIGN KEY(source_id, source_version) REFERENCES revisions(source_id, version),
 CHECK(char_start >= 0 AND char_end > char_start)
);
CREATE TABLE IF NOT EXISTS identity_scope_versions (
 scope_id TEXT NOT NULL REFERENCES identity_pair_scopes(scope_id),
 version INTEGER NOT NULL, coverage TEXT NOT NULL,
 annotator TEXT NOT NULL, evidence TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(scope_id, version), CHECK(version > 0),
 CHECK(coverage IN ('complete', 'partial', 'unknown'))
);
CREATE TABLE IF NOT EXISTS identity_scope_mentions (
 scope_id TEXT NOT NULL, scope_version INTEGER NOT NULL,
 mention_id TEXT NOT NULL REFERENCES mentions(mention_id),
 PRIMARY KEY(scope_id, scope_version, mention_id),
 FOREIGN KEY(scope_id, scope_version)
 REFERENCES identity_scope_versions(scope_id, version)
);
CREATE TABLE IF NOT EXISTS identity_pair_labels (
 scope_id TEXT NOT NULL, scope_version INTEGER NOT NULL,
 left_mention_id TEXT NOT NULL REFERENCES mentions(mention_id),
 right_mention_id TEXT NOT NULL REFERENCES mentions(mention_id),
 label TEXT NOT NULL,
 PRIMARY KEY(scope_id, scope_version, left_mention_id, right_mention_id),
 FOREIGN KEY(scope_id, scope_version)
 REFERENCES identity_scope_versions(scope_id, version),
 CHECK(left_mention_id < right_mention_id),
 CHECK(label IN ('same', 'different', 'unknown'))
);
CREATE INDEX IF NOT EXISTS identity_scopes_by_source
 ON identity_pair_scopes(source_id, source_version, char_start, char_end);
"""

_PAIR_TABLES = (
    "identity_pair_scopes",
    "identity_scope_versions",
    "identity_scope_mentions",
    "identity_pair_labels",
)


def _immutable_triggers(tables: tuple[str, ...]) -> str:
    return "\n".join(
        f"CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()} "
        f"BEFORE {operation} ON {table} BEGIN "
        "SELECT RAISE(ABORT, 'archive records are immutable'); END;"
        for table in tables
        for operation in ("UPDATE", "DELETE")
    )


def _check_pair_consistency(pairs: list[dict[str, str]]) -> None:
    """A reviewed difference cannot contradict the transitive same relation."""
    parent: dict[str, str] = {}

    def root(mention_id: str) -> str:
        parent.setdefault(mention_id, mention_id)
        while parent[mention_id] != mention_id:
            parent[mention_id] = parent[parent[mention_id]]
            mention_id = parent[mention_id]
        return mention_id

    for pair in pairs:
        if pair["label"] == "same":
            parent[root(pair["left_mention_id"])] = root(pair["right_mention_id"])
    for pair in pairs:
        if pair["label"] == "different" and root(pair["left_mention_id"]) == root(
            pair["right_mention_id"]
        ):
            raise ValueError("different identity pair contradicts reviewed same links")


def _id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _AlreadyImported(Exception):
    def __init__(self, record: SourceRecord):
        self.record = record


class ObservationArchive:
    def __init__(
        self,
        path: str | Path,
        *,
        create: bool = False,
        limits: ArchiveLimits | None = None,
    ) -> None:
        self.path = Path(path).absolute()
        self.limits = limits or ArchiveLimits()
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("archive must be a regular non-symlink file")
        created = False
        if not self.path.exists():
            if not create:
                raise ValueError("archive does not exist; import creates an archive")
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                created = True
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys=ON")
            if created:
                # executescript owns the initialization transaction.
                triggers = _immutable_triggers(
                    (
                        "sources",
                        "revisions",
                        "observations",
                        "instances",
                        "mentions",
                        "bindings",
                        "candidates",
                        *_PAIR_TABLES,
                    )
                )
                self._db.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _SCHEMA
                    + _PAIR_SCHEMA
                    + triggers
                    + f"\nPRAGMA application_id={APPLICATION_ID};"
                    + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                )
            application_id = self._db.execute("PRAGMA application_id").fetchone()[0]
            user_version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if application_id != APPLICATION_ID or user_version not in (
                1,
                SCHEMA_VERSION,
            ):
                raise ValueError("unsupported or unrelated observation archive")
            if user_version == 1:
                # Additive migration keeps existing source and binding IDs pinned.
                self._db.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _PAIR_SCHEMA
                    + _immutable_triggers(_PAIR_TABLES)
                    + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
        except (sqlite3.Error, ValueError) as exc:
            self._db.close()
            raise ValueError(f"cannot open observation archive: {exc}") from exc

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ObservationArchive:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            self._db.execute("BEGIN IMMEDIATE")
            yield
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def import_file(
        self,
        path: str | Path,
        *,
        namespace: str,
        external_key: str,
        group_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        with Path(path).open("rb") as stream:
            return self.import_blocks(
                iter(lambda: stream.read(65_536), b""),
                namespace=namespace,
                external_key=external_key,
                group_id=group_id,
                metadata=metadata,
            )

    def import_text(
        self,
        text: str,
        *,
        namespace: str,
        external_key: str,
        group_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        if type(text) is not str:
            raise ValueError("source text must be a string")
        return self.import_blocks(
            (text[i : i + 16_384].encode("utf-8") for i in range(0, len(text), 16_384)),
            namespace=namespace,
            external_key=external_key,
            group_id=group_id,
            metadata=metadata,
        )

    def import_blocks(
        self,
        blocks: Iterable[bytes],
        *,
        namespace: str,
        external_key: str,
        group_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        for key, value in (
            ("namespace", namespace),
            ("external_key", external_key),
            ("group_id", group_id),
        ):
            text_field(value, key)
        if metadata is not None and type(metadata) is not dict:
            raise ValueError("metadata must be an object")
        encoded_metadata = canonical_json(metadata or {})
        if len(encoded_metadata.encode("utf-8")) > 65_536:
            raise ValueError("source metadata exceeds capacity")
        try:
            with self._transaction():
                source = self._db.execute(
                    "SELECT * FROM sources WHERE namespace=? AND external_key=?",
                    (namespace, external_key),
                ).fetchone()
                previous = None
                if source is None:
                    source_id, version = _id("src"), 1
                    self._db.execute(
                        "INSERT INTO sources VALUES (?,?,?,?)",
                        (source_id, namespace, external_key, group_id),
                    )
                else:
                    if source["group_id"] != group_id:
                        raise ValueError("source group is immutable across revisions")
                    source_id = source["source_id"]
                    previous = self.get_source(source_id)
                    version = previous.version + 1
                digest = hashlib.sha256()
                chars = byte_count = count = 0
                for chunk in utf8_chunks(
                    blocks,
                    chunk_chars=self.limits.chunk_chars,
                    max_bytes=self.limits.max_source_bytes,
                ):
                    raw = chunk.encode("utf-8")
                    digest.update(raw)
                    self._db.execute(
                        "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            _id("obs"),
                            source_id,
                            version,
                            count,
                            chars,
                            chars + len(chunk),
                            byte_count,
                            byte_count + len(raw),
                            raw,
                            hashlib.sha256(raw).hexdigest(),
                        ),
                    )
                    chars += len(chunk)
                    byte_count += len(raw)
                    count += 1
                sha256 = digest.hexdigest()
                if (
                    previous is not None
                    and previous.sha256 == sha256
                    and canonical_json(previous.metadata) == encoded_metadata
                ):
                    raise _AlreadyImported(previous)
                self._db.execute(
                    "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)",
                    (
                        source_id,
                        version,
                        sha256,
                        byte_count,
                        chars,
                        count,
                        _now(),
                        encoded_metadata,
                    ),
                )
                result = self.get_source(source_id, version)
            return result
        except _AlreadyImported as existing:
            return existing.record

    def get_source(self, source_id: str, version: int | None = None) -> SourceRecord:
        text_field(source_id, "source ID")
        if version is not None:
            integer(version, "source version", minimum=1)
        row = self._db.execute(
            "SELECT s.*, r.* FROM sources s JOIN revisions r USING(source_id) "
            "WHERE s.source_id=? AND (? IS NULL OR version=?) "
            "ORDER BY version DESC LIMIT 1",
            (source_id, version, version),
        ).fetchone()
        if row is None:
            raise ValueError("unknown source/version")
        return SourceRecord(
            row["source_id"],
            row["version"],
            row["namespace"],
            row["external_key"],
            row["group_id"],
            row["sha256"],
            row["byte_count"],
            row["char_count"],
            row["observation_count"],
            row["received_at"],
            json.loads(row["metadata"]),
        )

    def sources(
        self,
        namespace: str,
        *,
        after: str = "",
        limit: int = 100,
    ) -> tuple[SourceRecord, ...]:
        text_field(namespace, "namespace")
        text_field(after, "source cursor", empty=True)
        integer(limit, "page limit", minimum=1)
        if limit > 1000:
            raise ValueError("page limit exceeds 1000")
        rows = self._db.execute(
            "SELECT source_id FROM sources WHERE namespace=? AND source_id>? "
            "ORDER BY source_id LIMIT ?",
            (namespace, after, limit),
        )
        return tuple(self.get_source(row[0]) for row in rows)

    @staticmethod
    def _observation(row: sqlite3.Row) -> Observation:
        result = Observation(
            row["observation_id"],
            row["source_id"],
            row["version"],
            row["ordinal"],
            row["char_start"],
            row["char_end"],
            row["byte_start"],
            row["byte_end"],
            row["raw"],
            row["sha256"],
        )
        if (
            hashlib.sha256(result.raw).hexdigest() != result.sha256
            or len(result.raw) != result.byte_end - result.byte_start
            or len(result.text) != result.char_end - result.char_start
        ):
            raise ValueError("observation checksum or coordinate mismatch")
        return result

    def observations(
        self,
        source_id: str,
        version: int,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[Observation, ...]:
        self.get_source(source_id, version)
        integer(offset, "page offset")
        integer(limit, "page limit", minimum=1)
        if limit > 1000:
            raise ValueError("page limit exceeds 1000")
        rows = self._db.execute(
            "SELECT * FROM observations WHERE source_id=? AND version=? "
            "AND ordinal>=? ORDER BY ordinal LIMIT ?",
            (source_id, version, offset, limit),
        )
        return tuple(self._observation(row) for row in rows)

    def iter_observations(self, source_id: str, version: int) -> Iterator[Observation]:
        offset = 0
        while page := self.observations(source_id, version, offset=offset):
            yield from page
            offset = page[-1].ordinal + 1

    def read_span(self, source_id: str, version: int, start: int, end: int) -> str:
        record = self.get_source(source_id, version)
        integer(start, "span start")
        integer(end, "span end", minimum=1)
        if not start < end <= record.char_count:
            raise ValueError("span is outside the source")
        if end - start > self.limits.max_span_chars:
            raise ValueError("span exceeds configured working budget")
        rows = self._db.execute(
            "SELECT * FROM observations WHERE source_id=? AND version=? "
            "AND char_end>? AND char_start<? ORDER BY ordinal",
            (source_id, version, start, end),
        )
        pieces = []
        cursor = start
        for row in rows:
            observation = self._observation(row)
            left, right = (
                max(start, observation.char_start),
                min(end, observation.char_end),
            )
            if left != cursor:
                raise ValueError("source observations have a gap or overlap")
            pieces.append(
                observation.text[
                    left - observation.char_start : right - observation.char_start
                ]
            )
            cursor = right
        if cursor != end:
            raise ValueError("incomplete source span")
        return "".join(pieces)

    def get_instance(self, instance_id: str) -> Instance:
        text_field(instance_id, "instance ID")
        row = self._db.execute(
            "SELECT * FROM instances WHERE instance_id=?",
            (instance_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown instance")
        return Instance(**dict(row))

    def _instance(self, namespace: str, external_key: str, label: str) -> Instance:
        text_field(external_key, "instance key")
        text_field(label, "instance label", empty=True)
        row = self._db.execute(
            "SELECT * FROM instances WHERE namespace=? AND external_key=?",
            (namespace, external_key),
        ).fetchone()
        if row is not None:
            result = Instance(**dict(row))
            if result.label != label:
                raise ValueError("instance key already has a different label")
            return result
        result = Instance(_id("inst"), namespace, external_key, label)
        self._db.execute(
            "INSERT INTO instances VALUES (?,?,?,?)", tuple(result.to_dict().values())
        )
        return result

    def get_mention(self, mention_id: str) -> Mention:
        text_field(mention_id, "mention ID")
        row = self._db.execute(
            "SELECT * FROM mentions WHERE mention_id=?",
            (mention_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown mention")
        result = Mention(**dict(row))
        if (
            self.read_span(
                result.source_id,
                result.source_version,
                result.char_start,
                result.char_end,
            )
            != result.surface
        ):
            raise ValueError("mention does not match original source")
        return result

    def _mention(
        self, source: SourceRecord, start: int, end: int, surface: str
    ) -> Mention:
        actual = self.read_span(source.source_id, source.version, start, end)
        if actual != surface:
            raise ValueError("annotation surface does not match original source")
        row = self._db.execute(
            "SELECT * FROM mentions WHERE source_id=? AND source_version=? "
            "AND char_start=? AND char_end=?",
            (source.source_id, source.version, start, end),
        ).fetchone()
        if row is not None:
            return Mention(**dict(row))
        result = Mention(
            _id("mention"), source.source_id, source.version, start, end, surface
        )
        self._db.execute(
            "INSERT INTO mentions VALUES (?,?,?,?,?,?)",
            tuple(result.to_dict().values()),
        )
        return result

    def get_binding(
        self, mention_id: str, version: int | None = None
    ) -> Binding | None:
        self.get_mention(mention_id)
        if version is not None:
            integer(version, "binding version", minimum=1)
        row = self._db.execute(
            "SELECT * FROM bindings WHERE mention_id=? AND (? IS NULL OR version=?) "
            "ORDER BY version DESC LIMIT 1",
            (mention_id, version, version),
        ).fetchone()
        if row is None:
            return None
        candidate_ids = tuple(
            r[0]
            for r in self._db.execute(
                "SELECT instance_id FROM candidates WHERE mention_id=? "
                "AND binding_version=? "
                "ORDER BY instance_id",
                (mention_id, row["version"]),
            )
        )
        if row["selected_id"] is not None and row["selected_id"] not in candidate_ids:
            raise ValueError("selected instance is not a candidate")
        return Binding(candidate_ids=candidate_ids, **dict(row))

    def _bind(
        self,
        mention: Mention,
        candidate_ids: tuple[str, ...],
        selected_id: str | None,
        *,
        expected_version: int,
        annotator: str,
        evidence: str,
    ) -> Binding:
        integer(expected_version, "expected binding version")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("duplicate binding candidate")
        if selected_id is not None and selected_id not in candidate_ids:
            raise ValueError("selected instance must occur among candidates")
        namespace = self.get_source(mention.source_id, mention.source_version).namespace
        for candidate in candidate_ids:
            if self.get_instance(candidate).namespace != namespace:
                raise ValueError("cross-namespace identity binding")
        previous = self.get_binding(mention.mention_id)
        candidates = tuple(sorted(candidate_ids))
        current_version = previous.version if previous else 0
        if previous and (
            previous.candidate_ids == candidates
            and previous.selected_id == selected_id
            and previous.annotator == annotator
            and previous.evidence == evidence
        ):
            if expected_version not in (current_version, current_version - 1):
                raise ValueError("stale annotation version")
            return previous
        if expected_version != current_version:
            raise ValueError("stale annotation version")
        result = Binding(
            mention.mention_id,
            current_version + 1,
            candidates,
            selected_id,
            annotator,
            evidence,
            _now(),
        )
        self._db.execute(
            "INSERT INTO bindings VALUES (?,?,?,?,?,?)",
            (
                result.mention_id,
                result.version,
                selected_id,
                annotator,
                evidence,
                result.created_at,
            ),
        )
        self._db.executemany(
            "INSERT INTO candidates VALUES (?,?,?)",
            (
                (result.mention_id, result.version, candidate)
                for candidate in candidates
            ),
        )
        return result

    def annotate(self, batch: Any) -> dict[str, Any]:
        """Apply external spans/identities atomically, with explicit provenance.

        The vocabulary and identity links come from this data, never from
        built-in entity/pronoun lists. No automatic coreference is claimed.
        """
        batch = fields(
            batch,
            {
                "schema",
                "source_id",
                "source_version",
                "annotator",
                "evidence",
                "instances",
                "mentions",
            },
        )
        if batch["schema"] != ANNOTATION_SCHEMA:
            raise ValueError("unsupported annotation schema")
        annotator = text_field(batch["annotator"], "annotator")
        evidence = text_field(batch["evidence"], "annotation evidence")
        integer(batch["source_version"], "source version", minimum=1)
        for key in ("instances", "mentions"):
            if (
                type(batch[key]) is not list
                or len(batch[key]) > self.limits.max_annotations
            ):
                raise ValueError("invalid annotation batch size")
        with self._transaction():
            source = self.get_source(batch["source_id"], batch["source_version"])
            instances: dict[str, Instance] = {}
            for row in batch["instances"]:
                row = fields(row, {"ref"}, {"instance_id", "external_key", "label"})
                ref = text_field(row["ref"], "local instance reference")
                if ref in instances:
                    raise ValueError("duplicate local instance reference")
                if "instance_id" in row:
                    fields(row, {"ref", "instance_id"})
                    instance = self.get_instance(row["instance_id"])
                else:
                    fields(row, {"ref", "external_key", "label"})
                    instance = self._instance(
                        source.namespace, row["external_key"], row["label"]
                    )
                if instance.namespace != source.namespace:
                    raise ValueError("cross-namespace instance reference")
                instances[ref] = instance
            results = []
            seen: set[tuple[int, int]] = set()
            for row in batch["mentions"]:
                row = fields(
                    row,
                    {
                        "start",
                        "end",
                        "surface",
                        "candidates",
                        "selected",
                        "expected_version",
                    },
                )
                integer(row["start"], "mention start")
                integer(row["end"], "mention end")
                span = (row["start"], row["end"])
                if span in seen:
                    raise ValueError("duplicate mention span in one batch")
                seen.add(span)
                if (
                    type(row["candidates"]) is not list
                    or len(row["candidates"]) > self.limits.max_annotations
                ):
                    raise ValueError("invalid candidate references")
                for ref in row["candidates"]:
                    if type(ref) is not str or ref not in instances:
                        raise ValueError("unknown candidate reference")
                selected = row["selected"]
                if selected is not None and (
                    type(selected) is not str or selected not in instances
                ):
                    raise ValueError("unknown selected reference")
                mention = self._mention(
                    source, row["start"], row["end"], row["surface"]
                )
                binding = self._bind(
                    mention,
                    tuple(instances[ref].instance_id for ref in row["candidates"]),
                    instances[selected].instance_id if selected is not None else None,
                    expected_version=row["expected_version"],
                    annotator=annotator,
                    evidence=evidence,
                )
                results.append(
                    {"mention": mention.to_dict(), "binding": binding.to_dict()}
                )
        return {
            "source_id": source.source_id,
            "source_version": source.version,
            "instances": {key: value.to_dict() for key, value in instances.items()},
            "annotations": results,
            "semantic_status": "externally_annotated",
        }

    def annotations(
        self,
        source_id: str,
        version: int,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[dict[str, Any], ...]:
        self.get_source(source_id, version)
        integer(offset, "annotation offset")
        integer(limit, "page limit", minimum=1)
        if limit > 1000:
            raise ValueError("page limit exceeds 1000")
        rows = self._db.execute(
            "SELECT mention_id FROM mentions WHERE source_id=? AND source_version=? "
            "ORDER BY char_start, char_end LIMIT ? OFFSET ?",
            (source_id, version, limit, offset),
        )
        result = []
        for row in rows:
            mention = self.get_mention(row[0])
            binding = self.get_binding(row[0])
            result.append(
                {
                    "mention": mention.to_dict(),
                    "binding": binding.to_dict() if binding else None,
                }
            )
        return tuple(result)

    def get_identity_scope(
        self, scope_id: str, version: int | None = None
    ) -> dict[str, Any]:
        """Read one immutable snapshot of an externally reviewed pair scope."""
        text_field(scope_id, "identity scope ID")
        if version is not None:
            integer(version, "scope version", minimum=1)
        row = self._db.execute(
            "SELECT s.*, v.version, v.coverage, v.annotator, v.evidence, "
            "v.created_at FROM identity_pair_scopes s "
            "JOIN identity_scope_versions v USING(scope_id) "
            "WHERE scope_id=? AND (? IS NULL OR v.version=?) "
            "ORDER BY v.version DESC LIMIT 1",
            (scope_id, version, version),
        ).fetchone()
        if row is None:
            raise ValueError("unknown identity scope/version")
        mention_ids = [
            item[0]
            for item in self._db.execute(
                "SELECT mention_id FROM identity_scope_mentions "
                "WHERE scope_id=? AND scope_version=? ORDER BY mention_id",
                (scope_id, row["version"]),
            )
        ]
        pairs = [
            {"left_mention_id": item[0], "right_mention_id": item[1], "label": item[2]}
            for item in self._db.execute(
                "SELECT left_mention_id, right_mention_id, label "
                "FROM identity_pair_labels WHERE scope_id=? AND scope_version=? "
                "ORDER BY left_mention_id, right_mention_id",
                (scope_id, row["version"]),
            )
        ]
        if (
            len(pairs) > len(mention_ids) * (len(mention_ids) - 1) // 2
            or row["coverage"] == "complete"
            and len(pairs) != len(mention_ids) * (len(mention_ids) - 1) // 2
        ):
            raise ValueError("identity scope coverage does not match explicit pairs")
        members = set(mention_ids)
        for mention_id in mention_ids:
            mention = self.get_mention(mention_id)
            if (
                mention.source_id != row["source_id"]
                or mention.source_version != row["source_version"]
                or mention.char_start < row["char_start"]
                or mention.char_end > row["char_end"]
            ):
                raise ValueError("identity scope mention is outside source/window")
        for pair in pairs:
            if (
                pair["left_mention_id"] not in members
                or pair["right_mention_id"] not in members
                or row["coverage"] == "unknown"
                and pair["label"] != "unknown"
            ):
                raise ValueError("invalid identity pair label or membership")
        _check_pair_consistency(pairs)
        return {
            "scope_id": row["scope_id"],
            "source_id": row["source_id"],
            "source_version": row["source_version"],
            "start": row["char_start"],
            "end": row["char_end"],
            "policy_reference": row["policy_reference"],
            "version": row["version"],
            "coverage": row["coverage"],
            "annotator": row["annotator"],
            "evidence": row["evidence"],
            "created_at": row["created_at"],
            "mention_ids": mention_ids,
            "pairs": pairs,
        }

    def identity_scopes(
        self, source_id: str, version: int, *, offset: int = 0, limit: int = 100
    ) -> tuple[dict[str, Any], ...]:
        self.get_source(source_id, version)
        integer(offset, "identity scope offset")
        integer(limit, "page limit", minimum=1)
        if limit > 1000:
            raise ValueError("page limit exceeds 1000")
        rows = self._db.execute(
            "SELECT scope_id FROM identity_pair_scopes "
            "WHERE source_id=? AND source_version=? "
            "ORDER BY char_start, char_end LIMIT ? OFFSET ?",
            (source_id, version, limit, offset),
        )
        return tuple(self.get_identity_scope(row[0]) for row in rows)

    def annotate_identity_pairs(self, batch: Any) -> dict[str, Any]:
        """Append a reviewed snapshot; omissions and unknowns never become negatives.

        Completeness means every pair among the explicitly pinned mentions in
        this source window was reviewed. Existing source mentions in the window
        must all be named when declaring complete coverage.
        """
        batch = fields(
            batch,
            {
                "schema",
                "source_id",
                "source_version",
                "start",
                "end",
                "coverage",
                "policy_reference",
                "annotator",
                "evidence",
                "mention_ids",
                "pairs",
                "expected_version",
            },
        )
        if batch["schema"] != IDENTITY_PAIRS_SCHEMA:
            raise ValueError("unsupported identity pairs schema")
        integer(batch["source_version"], "source version", minimum=1)
        start = integer(batch["start"], "scope start")
        end = integer(batch["end"], "scope end", minimum=1)
        expected = integer(batch["expected_version"], "expected scope version")
        coverage = batch["coverage"]
        if coverage not in ("complete", "partial", "unknown"):
            raise ValueError("invalid identity pair coverage")
        policy = text_field(batch["policy_reference"], "identity annotation policy")
        annotator = text_field(batch["annotator"], "identity annotator")
        evidence = text_field(batch["evidence"], "identity review evidence")
        mention_ids = batch["mention_ids"]
        pairs = batch["pairs"]
        if (
            type(mention_ids) is not list
            or type(pairs) is not list
            or len(mention_ids) > self.limits.max_annotations
            or len(pairs) > self.limits.max_annotations
        ):
            raise ValueError("invalid identity annotation batch size")
        with self._transaction():
            source = self.get_source(batch["source_id"], batch["source_version"])
            if not start < end <= source.char_count:
                raise ValueError("identity scope is outside source")
            ids = set()
            for mention_id in mention_ids:
                mention = self.get_mention(mention_id)
                if (
                    mention_id in ids
                    or mention.source_id != source.source_id
                    or mention.source_version != source.version
                    or mention.char_start < start
                    or mention.char_end > end
                ):
                    raise ValueError("duplicate or out-of-scope identity mention")
                ids.add(mention_id)
            if coverage == "complete":
                current_ids = {
                    row[0]
                    for row in self._db.execute(
                        "SELECT mention_id FROM mentions WHERE source_id=? "
                        "AND source_version=? AND char_start>=? AND char_end<=?",
                        (source.source_id, source.version, start, end),
                    )
                }
                if ids != current_ids:
                    raise ValueError("complete scope must include all current mentions")
            explicit = []
            seen_pairs = set()
            for item in pairs:
                pair = fields(item, {"left_mention_id", "right_mention_id", "label"})
                left, right = pair["left_mention_id"], pair["right_mention_id"]
                if (
                    type(left) is not str
                    or type(right) is not str
                    or left not in ids
                    or right not in ids
                    or left == right
                ):
                    raise ValueError(
                        "identity pair must name two distinct scope mentions"
                    )
                left, right = sorted((left, right))
                if (left, right) in seen_pairs:
                    raise ValueError("duplicate explicit identity pair")
                seen_pairs.add((left, right))
                label = pair["label"]
                if label not in ("same", "different", "unknown") or (
                    coverage == "unknown" and label != "unknown"
                ):
                    raise ValueError("invalid explicit identity pair label")
                explicit.append(
                    {"left_mention_id": left, "right_mention_id": right, "label": label}
                )
            if (
                coverage == "complete"
                and len(explicit) != len(ids) * (len(ids) - 1) // 2
            ):
                raise ValueError(
                    "complete scope requires an explicit label for every pair"
                )
            explicit.sort(
                key=lambda item: (item["left_mention_id"], item["right_mention_id"])
            )
            _check_pair_consistency(explicit)
            row = self._db.execute(
                "SELECT scope_id, policy_reference FROM identity_pair_scopes "
                "WHERE source_id=? AND source_version=? "
                "AND char_start=? AND char_end=?",
                (source.source_id, source.version, start, end),
            ).fetchone()
            if row is None:
                overlapping = self._db.execute(
                    "SELECT 1 FROM identity_pair_scopes WHERE source_id=? "
                    "AND source_version=? AND char_start<? AND char_end>? LIMIT 1",
                    (source.source_id, source.version, end, start),
                ).fetchone()
                if overlapping:
                    raise ValueError("identity scopes cannot overlap")
                scope_id = _id("scope")
                self._db.execute(
                    "INSERT INTO identity_pair_scopes VALUES (?,?,?,?,?,?)",
                    (scope_id, source.source_id, source.version, start, end, policy),
                )
                current_version = 0
                previous = None
            else:
                if row["policy_reference"] != policy:
                    raise ValueError("identity annotation policy is immutable")
                scope_id = row["scope_id"]
                previous = self.get_identity_scope(scope_id)
                current_version = previous["version"]
            if previous and (
                previous["coverage"] == coverage
                and previous["annotator"] == annotator
                and previous["evidence"] == evidence
                and previous["mention_ids"] == sorted(ids)
                and previous["pairs"] == explicit
            ):
                if expected not in (current_version, current_version - 1):
                    raise ValueError("stale identity scope version")
                return previous
            if expected != current_version:
                raise ValueError("stale identity scope version")
            new_version = current_version + 1
            self._db.execute(
                "INSERT INTO identity_scope_versions VALUES (?,?,?,?,?,?)",
                (scope_id, new_version, coverage, annotator, evidence, _now()),
            )
            self._db.executemany(
                "INSERT INTO identity_scope_mentions VALUES (?,?,?)",
                ((scope_id, new_version, mention_id) for mention_id in sorted(ids)),
            )
            self._db.executemany(
                "INSERT INTO identity_pair_labels VALUES (?,?,?,?,?)",
                (
                    (
                        scope_id,
                        new_version,
                        item["left_mention_id"],
                        item["right_mention_id"],
                        item["label"],
                    )
                    for item in explicit
                ),
            )
            return self.get_identity_scope(scope_id, new_version)

    def verify_source(self, source_id: str, version: int) -> SourceRecord:
        source = self.get_source(source_id, version)
        digest = hashlib.sha256()
        chars = byte_count = count = 0
        for observation in self.iter_observations(source.source_id, source.version):
            if (
                observation.ordinal != count
                or observation.char_start != chars
                or observation.byte_start != byte_count
            ):
                raise ValueError("noncontiguous source observations")
            digest.update(observation.raw)
            chars, byte_count = observation.char_end, observation.byte_end
            count += 1
        if (
            digest.hexdigest() != source.sha256
            or chars != source.char_count
            or byte_count != source.byte_count
            or count != source.observation_count
        ):
            raise ValueError("source revision checksum/count mismatch")
        return source

    def verify(self) -> dict[str, Any]:
        """Check source bytes, coordinates, links and namespace integrity."""
        with self._transaction():
            if self._db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            if self._db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ValueError("archive has dangling references")
            versions = observations = bytes_verified = 0
            for row in self._db.execute("SELECT source_id, version FROM revisions"):
                source = self.verify_source(row[0], row[1])
                versions += 1
                observations += source.observation_count
                bytes_verified += source.byte_count
            mentions = bindings = identity_scope_versions = 0
            for row in self._db.execute("SELECT mention_id FROM mentions"):
                self.get_mention(row[0])
                mentions += 1
            for row in self._db.execute("SELECT mention_id, version FROM bindings"):
                binding = self.get_binding(row[0], row[1])
                assert binding is not None
                mention = self.get_mention(binding.mention_id)
                namespace = self.get_source(
                    mention.source_id, mention.source_version
                ).namespace
                for candidate in binding.candidate_ids:
                    if self.get_instance(candidate).namespace != namespace:
                        raise ValueError("cross-namespace binding in archive")
                bindings += 1
            for row in self._db.execute(
                "SELECT scope_id, version FROM identity_scope_versions"
            ):
                self.get_identity_scope(row[0], row[1])
                identity_scope_versions += 1
        return {
            "status": "verified",
            "source_versions": versions,
            "observations": observations,
            "bytes_verified": bytes_verified,
            "mentions": mentions,
            "binding_versions": bindings,
            "identity_scope_versions": identity_scope_versions,
            "semantic_accuracy": None,
        }

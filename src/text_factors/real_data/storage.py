"""Durable task journal and live source-access gate for the experimental pilot.

The access database must live outside the directory containing rollback-able
model bundles. Publishing and revoking serialize through SQLite write locks;
the prepared dialogue draft is never an authorized answer by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observations import ObservationArchive
from .contracts import AnswerReceipt, SourceSlice
from .engine import RealDataEngine

_APPLICATION_ID = 0x41493233
_SCHEMA_VERSION = 3
_MAX_ID = 512
_MAX_RESULT = 1_000_000
_REVISION_SCHEMA = (
    "CREATE TABLE revision_meta (id INTEGER PRIMARY KEY CHECK(id=1), "
    "epoch INTEGER NOT NULL, state_version TEXT);"
    "INSERT INTO revision_meta VALUES (1,0,NULL);"
    "CREATE TABLE revision_jobs (revision_id TEXT PRIMARY KEY, "
    "base_state_version TEXT NOT NULL, new_state_version TEXT, "
    "start_epoch INTEGER NOT NULL, finish_epoch INTEGER, "
    "status TEXT NOT NULL CHECK(status IN ('pending','complete')));"
    "CREATE TABLE pending_claims (revision_id TEXT NOT NULL "
    "REFERENCES revision_jobs(revision_id), claim_id TEXT NOT NULL, "
    "PRIMARY KEY(revision_id,claim_id));"
)
_FAMILY_SCHEMA = (
    "CREATE TABLE revoked_source_families (source_id TEXT PRIMARY KEY NOT NULL);"
)


class AccessDenied(ValueError):
    """The caller or a source lost authorization before publication."""


class StalePublication(ValueError):
    """The draft does not match the current source, state, or request journal."""


def _nonempty(value: str, field: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_ID:
        raise ValueError(f"invalid {field}")
    return value


def _source_key(source_id: str, version: int) -> tuple[str, int]:
    _nonempty(source_id, "source_id")
    if type(version) is not int or version <= 0:
        raise ValueError("invalid source version")
    return source_id, version


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class PublicationQuote:
    source: SourceSlice
    claim_id: str
    text: str

    def __post_init__(self) -> None:
        if type(self.source) is not SourceSlice:
            raise ValueError("quote requires a pinned source slice")
        _nonempty(self.claim_id, "claim_id")
        if type(self.text) is not str or not self.text:
            raise ValueError("quote text must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "claim_id": self.claim_id,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublicationQuote:
        if type(value) is not dict or set(value) != {"source", "claim_id", "text"}:
            raise ValueError("invalid published quote")
        return cls(
            SourceSlice.from_dict(value["source"]), value["claim_id"], value["text"]
        )


@dataclass(frozen=True, slots=True)
class PublishedAnswer:
    receipt: AnswerReceipt
    quotes: tuple[PublicationQuote, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "quotes": [item.to_dict() for item in self.quotes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublishedAnswer:
        if (
            type(value) is not dict
            or set(value) != {"receipt", "quotes"}
            or type(value["quotes"]) is not list
        ):
            raise ValueError("invalid published answer")
        return cls(
            AnswerReceipt.from_dict(value["receipt"]),
            tuple(PublicationQuote.from_dict(item) for item in value["quotes"]),
        )


@dataclass(frozen=True, slots=True)
class TaskStatus:
    request_id: str
    status: str
    fingerprint: str


class OperationalStore:
    """SQLite access authority and idempotent journal, independent of bundles."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).absolute()
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("operational store must be a regular non-symlink file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        with self._connection() as db:
            db.execute("PRAGMA secure_delete=ON")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if (app_id, version) == (0, 0):
                db.executescript(
                    "BEGIN IMMEDIATE;"
                    "CREATE TABLE grants (principal_id TEXT NOT NULL, "
                    "scope TEXT NOT NULL, PRIMARY KEY(principal_id,scope));"
                    "CREATE TABLE revoked_sources (source_id TEXT NOT NULL, "
                    "version INTEGER NOT NULL, PRIMARY KEY(source_id,version));"
                    "CREATE TABLE tasks (request_id TEXT PRIMARY KEY, "
                    "fingerprint TEXT NOT NULL, principal_id TEXT NOT NULL, "
                    "status TEXT NOT NULL CHECK(status IN "
                    "('pending','published','complete','revoked')), result_json TEXT);"
                    "CREATE TABLE publication_sources (request_id TEXT NOT NULL "
                    "REFERENCES tasks(request_id) ON DELETE CASCADE, "
                    "source_id TEXT NOT NULL, version INTEGER NOT NULL, "
                    "scope TEXT NOT NULL, PRIMARY KEY(request_id,source_id,version));"
                    + _REVISION_SCHEMA
                    + _FAMILY_SCHEMA
                    + f"PRAGMA application_id={_APPLICATION_ID};"
                    f"PRAGMA user_version={_SCHEMA_VERSION};"
                    "COMMIT;"
                )
            elif (app_id, version) == (_APPLICATION_ID, 1):
                # v1 stored answer bodies. Retire old results and wipe free
                # pages during an explicit additive schema migration.
                db.executescript(
                    "BEGIN IMMEDIATE;"
                    "UPDATE tasks SET status='revoked', result_json=NULL "
                    "WHERE status IN ('published','complete');"
                    + _REVISION_SCHEMA
                    + _FAMILY_SCHEMA
                    + f"PRAGMA user_version={_SCHEMA_VERSION};COMMIT;"
                )
                db.execute("VACUUM")
            elif (app_id, version) == (_APPLICATION_ID, 2):
                db.executescript(
                    "BEGIN IMMEDIATE;"
                    + _FAMILY_SCHEMA
                    + f"PRAGMA user_version={_SCHEMA_VERSION};COMMIT;"
                )
            elif (app_id, version) != (_APPLICATION_ID, _SCHEMA_VERSION):
                raise ValueError(
                    "unsupported operational store schema; explicit migration required"
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=10000")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA secure_delete=ON")
            yield db
        finally:
            db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                raise

    def grant_scope(self, principal_id: str, scope: str) -> None:
        _nonempty(principal_id, "principal_id")
        _nonempty(scope, "scope")
        with self._write() as db:
            db.execute(
                "INSERT OR IGNORE INTO grants VALUES (?,?)", (principal_id, scope)
            )

    def revoke_scope(self, principal_id: str, scope: str) -> None:
        _nonempty(principal_id, "principal_id")
        _nonempty(scope, "scope")
        with self._write() as db:
            db.execute(
                "DELETE FROM grants WHERE principal_id=? AND scope=?",
                (principal_id, scope),
            )
            changed = db.execute("SELECT changes()").fetchone()[0]
            db.execute(
                "UPDATE tasks SET status='revoked', result_json=NULL "
                "WHERE principal_id=? AND request_id IN "
                "(SELECT request_id FROM publication_sources WHERE scope=?)",
                (principal_id, scope),
            )
            if changed:
                db.execute("UPDATE revision_meta SET epoch=epoch+1 WHERE id=1")

    def revoke_source(self, source_id: str, version: int) -> None:
        _source_key(source_id, version)
        with self._write() as db:
            db.execute(
                "INSERT OR IGNORE INTO revoked_sources VALUES (?,?)",
                (source_id, version),
            )
            changed = db.execute("SELECT changes()").fetchone()[0]
            db.execute(
                "UPDATE tasks SET status='revoked', result_json=NULL "
                "WHERE request_id IN (SELECT request_id FROM publication_sources "
                "WHERE source_id=? AND version=?)",
                (source_id, version),
            )
            if changed:
                db.execute("UPDATE revision_meta SET epoch=epoch+1 WHERE id=1")

    def revoke_source_family(self, source_id: str) -> None:
        """Deny all historical and future revisions of one source ID."""
        _nonempty(source_id, "source_id")
        with self._write() as db:
            db.execute(
                "INSERT OR IGNORE INTO revoked_source_families VALUES (?)",
                (source_id,),
            )
            changed = db.execute("SELECT changes()").fetchone()[0]
            db.execute(
                "UPDATE tasks SET status='revoked',result_json=NULL "
                "WHERE request_id IN (SELECT request_id FROM publication_sources "
                "WHERE source_id=?)",
                (source_id,),
            )
            if changed:
                db.execute("UPDATE revision_meta SET epoch=epoch+1 WHERE id=1")

    def revoked_sources(self) -> tuple[tuple[str, int], ...]:
        with self._connection() as db:
            return tuple(
                (row[0], row[1])
                for row in db.execute(
                    "SELECT source_id,version FROM revoked_sources "
                    "ORDER BY source_id,version"
                )
            )

    def revoked_source_families(self) -> tuple[str, ...]:
        with self._connection() as db:
            return tuple(
                source_id
                for (source_id,) in db.execute(
                    "SELECT source_id FROM revoked_source_families ORDER BY source_id"
                )
            )

    @staticmethod
    def _source_revoked(db: sqlite3.Connection, source: SourceSlice) -> bool:
        return (
            db.execute(
                "SELECT 1 FROM revoked_sources WHERE source_id=? AND version=? "
                "UNION SELECT 1 FROM revoked_source_families WHERE source_id=?",
                (source.source_id, source.source_version, source.source_id),
            ).fetchone()
            is not None
        )

    def revision_epoch(self) -> int:
        with self._connection() as db:
            return int(
                db.execute("SELECT epoch FROM revision_meta WHERE id=1").fetchone()[0]
            )

    def pending_claim_ids(self) -> tuple[str, ...]:
        with self._connection() as db:
            return tuple(
                claim_id
                for (claim_id,) in db.execute(
                    "SELECT DISTINCT claim_id FROM pending_claims ORDER BY claim_id"
                )
            )

    def is_revision_pending(self, revision_id: str) -> bool:
        _nonempty(revision_id, "revision_id")
        with self._connection() as db:
            row = db.execute(
                "SELECT status FROM revision_jobs WHERE revision_id=?", (revision_id,)
            ).fetchone()
        return row is not None and row[0] == "pending"

    def start_revision(
        self,
        revision_id: str,
        affected_claim_ids: tuple[str, ...],
        base_state_version: str,
    ) -> int:
        """Mark a revision pending before any checkpoint or claim mutation."""
        _nonempty(revision_id, "revision_id")
        _nonempty(base_state_version, "base_state_version")
        if (
            type(affected_claim_ids) is not tuple
            or not affected_claim_ids
            or any(type(claim_id) is not str for claim_id in affected_claim_ids)
            or len(set(affected_claim_ids)) != len(affected_claim_ids)
        ):
            raise ValueError("revision needs distinct affected claim IDs")
        for claim_id in affected_claim_ids:
            _nonempty(claim_id, "affected claim ID")
        with self._write() as db:
            row = db.execute(
                "SELECT base_state_version,start_epoch,status "
                "FROM revision_jobs WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row is not None:
                prior = tuple(
                    item[0]
                    for item in db.execute(
                        "SELECT claim_id FROM pending_claims "
                        "WHERE revision_id=? ORDER BY claim_id",
                        (revision_id,),
                    )
                )
                if (
                    row[0] != base_state_version
                    or row[2] != "pending"
                    or prior != tuple(sorted(affected_claim_ids))
                ):
                    raise StalePublication(
                        "revision ID changed or was already completed"
                    )
                return int(row[1])
            if (
                db.execute(
                    "SELECT 1 FROM revision_jobs WHERE status='pending' LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise StalePublication("another revision is pending")
            epoch, state = db.execute(
                "SELECT epoch,state_version FROM revision_meta WHERE id=1"
            ).fetchone()
            if state is not None and state != base_state_version:
                raise StalePublication("revision base state no longer active")
            next_epoch = int(epoch) + 1
            db.execute(
                "UPDATE revision_meta SET epoch=?,state_version=? WHERE id=1",
                (next_epoch, base_state_version),
            )
            db.execute(
                "INSERT INTO revision_jobs"
                "(revision_id,base_state_version,start_epoch,status) "
                "VALUES (?,?,?,'pending')",
                (revision_id, base_state_version, next_epoch),
            )
            db.executemany(
                "INSERT INTO pending_claims VALUES (?,?)",
                ((revision_id, claim_id) for claim_id in sorted(affected_claim_ids)),
            )
            return next_epoch

    def finish_revision(
        self, revision_id: str, new_state_version: str, *, expected_epoch: int
    ) -> int:
        """Finish only the same job and epoch after durable engine commit."""
        _nonempty(revision_id, "revision_id")
        _nonempty(new_state_version, "new_state_version")
        if type(expected_epoch) is not int or expected_epoch < 0:
            raise ValueError("invalid expected_epoch")
        with self._write() as db:
            row = db.execute(
                "SELECT base_state_version,new_state_version,start_epoch,"
                "finish_epoch,status "
                "FROM revision_jobs WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row is None or row[2] != expected_epoch:
                raise StalePublication("unknown or stale revision checkpoint")
            if row[4] == "complete":
                if row[1] != new_state_version:
                    raise StalePublication("completed revision changed outcome")
                return int(row[3])
            epoch, state = db.execute(
                "SELECT epoch,state_version FROM revision_meta WHERE id=1"
            ).fetchone()
            if epoch != expected_epoch or state != row[0]:
                raise StalePublication("revision epoch or base state changed")
            next_epoch = int(epoch) + 1
            db.execute("DELETE FROM pending_claims WHERE revision_id=?", (revision_id,))
            db.execute(
                "UPDATE revision_jobs SET status='complete',"
                "new_state_version=?,finish_epoch=? WHERE revision_id=?",
                (new_state_version, next_epoch, revision_id),
            )
            db.execute(
                "UPDATE revision_meta SET epoch=?,state_version=? WHERE id=1",
                (next_epoch, new_state_version),
            )
            return next_epoch

    def bind_state_version(self, state_version: str, *, expected_epoch: int) -> int:
        """Explicit rollback activation; never changes grants or tombstones."""
        _nonempty(state_version, "state_version")
        if type(expected_epoch) is not int or expected_epoch < 0:
            raise ValueError("invalid expected_epoch")
        with self._write() as db:
            if (
                db.execute(
                    "SELECT 1 FROM revision_jobs WHERE status='pending' LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise StalePublication("cannot activate state during pending revision")
            epoch = db.execute("SELECT epoch FROM revision_meta WHERE id=1").fetchone()[
                0
            ]
            if epoch != expected_epoch:
                raise StalePublication("active epoch changed before rollback")
            db.execute(
                "UPDATE revision_meta SET epoch=epoch+1,state_version=? WHERE id=1",
                (state_version,),
            )
            return int(epoch) + 1

    def task_status(self, request_id: str) -> TaskStatus | None:
        _nonempty(request_id, "request_id")
        with self._connection() as db:
            row = db.execute(
                "SELECT status,fingerprint FROM tasks WHERE request_id=?", (request_id,)
            ).fetchone()
        return None if row is None else TaskStatus(request_id, row[0], row[1])

    def begin_task(
        self, request_id: str, request_fingerprint: str, principal_id: str
    ) -> TaskStatus:
        """Persistent pending state is safe to resume after worker termination."""
        _nonempty(request_id, "request_id")
        _nonempty(request_fingerprint, "request_fingerprint")
        _nonempty(principal_id, "principal_id")
        with self._write() as db:
            row = db.execute(
                "SELECT fingerprint,principal_id,status FROM tasks WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is not None:
                if row[0] != request_fingerprint or row[1] != principal_id:
                    raise StalePublication(
                        "request ID reused with different content or principal"
                    )
                return TaskStatus(request_id, row[2], row[0])
            db.execute(
                "INSERT INTO tasks(request_id,fingerprint,principal_id,status) "
                "VALUES (?,?,?,'pending')",
                (request_id, request_fingerprint, principal_id),
            )
            return TaskStatus(request_id, "pending", request_fingerprint)

    def complete_task(
        self,
        request_id: str,
        request_fingerprint: str,
        principal_id: str,
        result: dict[str, Any],
    ) -> TaskStatus:
        """Journal a result digest exactly once, without retaining result bytes."""
        self.begin_task(request_id, request_fingerprint, principal_id)
        payload = _json(result).encode("utf-8")
        if len(payload) > _MAX_RESULT:
            raise ValueError("task result exceeds budget")
        digest = hashlib.sha256(payload).hexdigest()
        with self._write() as db:
            row = db.execute(
                "SELECT status,result_json FROM tasks WHERE request_id=?", (request_id,)
            ).fetchone()
            assert row is not None
            if row[0] == "pending":
                db.execute(
                    "UPDATE tasks SET status='complete', result_json=? "
                    "WHERE request_id=?",
                    (digest, request_id),
                )
            elif row[0] != "complete" or row[1] != digest:
                raise StalePublication("task already completed with another result")
        return TaskStatus(request_id, "complete", request_fingerprint)

    def publish_answer(
        self,
        *,
        request_id: str,
        request_fingerprint: str,
        principal_id: str,
        engine: RealDataEngine,
        archive: ObservationArchive,
        receipt: AnswerReceipt,
        quotes: tuple[PublicationQuote, ...],
        expected_revision_epoch: int,
        revision_guard: Callable[[], bool] | None = None,
    ) -> PublishedAnswer:
        """Authorize, verify and journal a complete answer at one lock point.

        A revoke which commits before this transaction is always effective; a
        revoke after it takes effect for every later publication or replay.
        """
        for field, value in (
            ("request_id", request_id),
            ("request_fingerprint", request_fingerprint),
            ("principal_id", principal_id),
        ):
            _nonempty(value, field)
        if type(expected_revision_epoch) is not int or expected_revision_epoch < 0:
            raise StalePublication("publication needs its prepared revision epoch")
        if (
            type(receipt) is not AnswerReceipt
            or type(quotes) is not tuple
            or not quotes
            or any(type(q) is not PublicationQuote for q in quotes)
        ):
            raise StalePublication(
                "publication needs a complete receipt and verified quotes"
            )
        if (
            not receipt.complete
            or not receipt.answer_text
            or receipt.answer_text != "\n".join(q.text for q in quotes)
        ):
            raise StalePublication("answer text differs from the verified quotes")
        if not receipt.claim_ids or set(receipt.claim_ids) != {
            q.claim_id for q in quotes
        }:
            raise StalePublication("each claimed fact must have a published quote")
        result = PublishedAnswer(receipt, quotes)
        payload = _json(result.to_dict()).encode("utf-8")
        if len(payload) > _MAX_RESULT:
            raise ValueError("publication exceeds task budget")
        digest = hashlib.sha256(payload).hexdigest()
        with self._write() as db:
            row = db.execute(
                "SELECT fingerprint,principal_id,status,result_json "
                "FROM tasks WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is not None and (
                row[0] != request_fingerprint or row[1] != principal_id
            ):
                raise StalePublication(
                    "request ID reused with different content or principal"
                )
            if row is not None and row[2] == "revoked":
                raise AccessDenied("request was revoked")
            if row is not None and row[2] not in ("pending", "published"):
                raise StalePublication("request ID belongs to another task")
            allowed = tuple(
                scope
                for (scope,) in db.execute(
                    "SELECT scope FROM grants WHERE principal_id=? ORDER BY scope",
                    (principal_id,),
                )
            )
            if not allowed:
                raise AccessDenied("principal has no active scope grant")
            epoch, state_version = db.execute(
                "SELECT epoch,state_version FROM revision_meta WHERE id=1"
            ).fetchone()
            if epoch != expected_revision_epoch:
                raise StalePublication("revision epoch changed after preparation")
            if state_version is not None and state_version != receipt.state_version:
                raise StalePublication("state version no longer active")
            if (
                db.execute(
                    "SELECT 1 FROM pending_claims WHERE claim_id IN ("
                    + ",".join("?" for _ in receipt.claim_ids)
                    + ") LIMIT 1",
                    receipt.claim_ids,
                ).fetchone()
                is not None
            ):
                raise AccessDenied("claimed fact is under revision")
            if revision_guard is not None and not revision_guard():
                raise AccessDenied("dependent revision is pending")
            expected = engine.receipt(
                question_id=receipt.question_id,
                answer_text=receipt.answer_text,
                claim_ids=receipt.claim_ids,
                model_version=receipt.model_version,
                allowed_scopes=allowed,
            )
            if not expected.complete:
                raise AccessDenied(
                    expected.reason or "receipt is not currently grounded"
                )
            if expected != receipt:
                raise StalePublication(
                    "receipt was prepared against another engine state"
                )
            cited: set[tuple[str, int, int, int, str]] = set()
            for root_id in receipt.evidence_roots:
                root = engine.evidence.root(root_id)
                if root.source is None or root.access_scope not in allowed:
                    raise AccessDenied("evidence lacks active source binding or grant")
                source = root.source
                if self._source_revoked(db, source):
                    raise AccessDenied("source or source revision was revoked")
                self._check_archive(archive, source)
                cited.add(
                    (
                        source.source_id,
                        source.source_version,
                        source.start,
                        source.end,
                        source.sha256,
                    )
                )
            for quote in quotes:
                source = quote.source
                if source.access_scope not in allowed:
                    raise AccessDenied("quote source scope is not granted")
                if self._source_revoked(db, source):
                    raise AccessDenied("quote source or revision was revoked")
                if (
                    source.source_id,
                    source.source_version,
                    source.start,
                    source.end,
                    source.sha256,
                ) not in cited:
                    raise StalePublication("quote is not a receipt evidence root")
                if not any(
                    engine.evidence.root(root_id).source == source
                    for root_id in engine.evidence.claim_roots(quote.claim_id)
                ):
                    raise StalePublication("quote does not support its claimed fact")
                if self._check_archive(archive, source) != quote.text:
                    raise StalePublication("quote differs from archive text")
            if row is not None and row[2] == "published":
                if row[3] != digest:
                    raise StalePublication(
                        "idempotent retry changed the published answer"
                    )
            elif row is None:
                db.execute(
                    "INSERT INTO tasks VALUES (?,?,?,'published',?)",
                    (request_id, request_fingerprint, principal_id, digest),
                )
            else:
                db.execute(
                    "UPDATE tasks SET status='published', result_json=? "
                    "WHERE request_id=?",
                    (digest, request_id),
                )
            if state_version is None:
                db.execute(
                    "UPDATE revision_meta SET state_version=? WHERE id=1",
                    (receipt.state_version,),
                )
            for source in {q.source for q in quotes} | {
                engine.evidence.root(r).source for r in receipt.evidence_roots
            }:
                if source is None:
                    raise AccessDenied("unbound root")
                db.execute(
                    "INSERT OR IGNORE INTO publication_sources VALUES (?,?,?,?)",
                    (
                        request_id,
                        source.source_id,
                        source.source_version,
                        source.access_scope,
                    ),
                )
        return result

    @staticmethod
    def _check_archive(archive: ObservationArchive, source: SourceSlice) -> str:
        try:
            record = archive.get_source(source.source_id, source.source_version)
            text = archive.read_span(
                source.source_id, source.source_version, source.start, source.end
            )
        except ValueError as exc:
            raise AccessDenied("source is unavailable in the archive") from exc
        scope = record.metadata.get("access_scope", "default")
        if (
            scope != source.access_scope
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != source.sha256
        ):
            raise AccessDenied("source scope or digest changed")
        return text

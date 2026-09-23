"""Durable, bounded reconsideration of a pinned dependency snapshot.

    The resolver is a pure function of a node ID and detached values for that
    node and its immediate parents. Parent values reflect earlier decisions.
The checkpoint file is authoritative for this *offline* job only. An
application may publish ``materialized_state`` only after ``commit`` has
atomically switched that file from ready to committed. Online answers need a
shared transactional guard around starting the job and publishing responses;
source checking and activation of a new application version belong to the
caller.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .budget import BudgetExceeded, BudgetTracker
from .dependencies import DependencyGraph


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("revision state must contain finite JSON values") from exc


def _digest(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ReevaluationDecision:
    """A proposed new value or a scoped inability to resolve one node."""

    value: Any = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError("decision must contain exactly one of value and reason")
        if self.reason is not None and (
            type(self.reason) is not str or not self.reason or len(self.reason) > 4096
        ):
            raise ValueError("invalid uncertainty reason")
        if self.value is not None:
            _canonical(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"value": deepcopy(self.value), "reason": self.reason}

    @classmethod
    def from_dict(cls, raw: Any) -> ReevaluationDecision:
        if type(raw) is not dict or set(raw) != {"value", "reason"}:
            raise ValueError("invalid reevaluation decision")
        return cls(value=raw["value"], reason=raw["reason"])


@dataclass(frozen=True, slots=True)
class RevisionProgress:
    phase: str
    processed: int
    total: int
    pending_claim_ids: tuple[str, ...]
    state_version: str
    stop_reason: str | None = None


class RevisionBarrier(Protocol):
    """Shared transaction gate implemented by the P16 operational store."""

    def start_revision(
        self,
        revision_id: str,
        affected_claim_ids: tuple[str, ...],
        base_state_version: str,
    ) -> int: ...

    def finish_revision(
        self,
        revision_id: str,
        new_state_version: str,
        *,
        expected_epoch: int,
    ) -> int: ...

    def is_revision_pending(self, revision_id: str) -> bool: ...

    def pending_claim_ids(self) -> tuple[str, ...]: ...


class RevisionJob:
    """One revision over a fixed graph and original state.

    Each batch saves staged results with atomic rename and fsync. A kill in
    the middle of a batch repeats that batch's *pure* resolver calls on resume;
    it never publishes partial values. There is one writer per checkpoint path.
    """

    FORMAT = "real-data-revision-v1"

    def __init__(
        self,
        path: Path,
        graph: DependencyGraph,
        original: Mapping[str, Any],
        payload: dict[str, Any],
        barrier: RevisionBarrier | None = None,
    ) -> None:
        self.path = path
        self.graph = graph
        self.original = deepcopy(dict(original))
        self._payload = payload
        self.barrier = barrier

    @classmethod
    def start(
        cls,
        *,
        path: str | Path,
        graph: DependencyGraph,
        original: Mapping[str, Any],
        changed_ids: tuple[str, ...],
        state_version: str,
        barrier: RevisionBarrier | None = None,
        revision_id: str | None = None,
    ) -> RevisionJob:
        if (
            type(state_version) is not str
            or not state_version
            or len(state_version) > 4096
        ):
            raise ValueError("invalid state version")
        snapshot = cls._snapshot(original)
        queue = graph.ordered_affected(changed_ids)
        if not set(queue) <= set(snapshot):
            raise ValueError("affected node missing from original state")
        if (barrier is None) != (revision_id is None):
            raise ValueError("barrier and revision_id must be supplied together")
        if revision_id is not None and (
            type(revision_id) is not str or not revision_id or len(revision_id) > 4096
        ):
            raise ValueError("invalid revision_id")
        # Shared SQL transaction records the pending claims *before* a job
        # checkpoint is created. If the filesystem write fails, the SQL gate
        # remains pending and online answers fail closed until recovery.
        start_epoch = (
            barrier.start_revision(revision_id, tuple(sorted(queue)), state_version)
            if barrier is not None and revision_id is not None
            else None
        )
        payload = {
            "format": cls.FORMAT,
            "state_version": state_version,
            "graph_digest": graph.fingerprint,
            "original_digest": _digest(snapshot),
            "changed_ids": list(dict.fromkeys(changed_ids)),
            "queue": list(queue),
            "cursor": 0,
            "decisions": {},
            "phase": "pending",
            "revision_id": revision_id,
            "start_epoch": start_epoch,
        }
        job = cls(Path(path), graph, snapshot, payload, barrier)
        job._write(exclusive=True)
        return job

    @classmethod
    def resume(
        cls,
        *,
        path: str | Path,
        graph: DependencyGraph,
        original: Mapping[str, Any],
        state_version: str,
        barrier: RevisionBarrier | None = None,
    ) -> RevisionJob:
        snapshot = cls._snapshot(original)
        try:
            raw = json.loads(Path(path).read_bytes())
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError("invalid or missing revision checkpoint") from exc
        if type(raw) is not dict or set(raw) != {"payload", "sha256"}:
            raise ValueError("invalid revision checkpoint envelope")
        payload = raw["payload"]
        if type(payload) is not dict or raw["sha256"] != _digest(payload):
            raise ValueError("corrupt revision checkpoint")
        expected = {
            "format",
            "state_version",
            "graph_digest",
            "original_digest",
            "changed_ids",
            "queue",
            "cursor",
            "decisions",
            "phase",
            "revision_id",
            "start_epoch",
        }
        if set(payload) != expected or payload["format"] != cls.FORMAT:
            raise ValueError("incompatible revision checkpoint format")
        if (
            payload["state_version"] != state_version
            or payload["graph_digest"] != graph.fingerprint
            or payload["original_digest"] != _digest(snapshot)
        ):
            raise ValueError("revision checkpoint belongs to another state or graph")
        changed = payload["changed_ids"]
        queue = payload["queue"]
        cursor = payload["cursor"]
        decisions = payload["decisions"]
        if (
            type(changed) is not list
            or not changed
            or any(type(item) is not str for item in changed)
            or len(changed) != len(set(changed))
            or type(queue) is not list
            or queue != list(graph.ordered_affected(tuple(changed)))
            or not set(queue) <= set(snapshot)
            or type(cursor) is not int
            or not 0 <= cursor <= len(queue)
            or type(decisions) is not dict
            or set(decisions) != set(queue[:cursor])
        ):
            raise ValueError("invalid revision cursor or affected nodes")
        if payload["phase"] not in ("pending", "ready", "committed") or (
            (cursor == len(queue)) != (payload["phase"] != "pending")
        ):
            raise ValueError("invalid revision phase")
        revision_id = payload["revision_id"]
        start_epoch = payload["start_epoch"]
        if (barrier is None) != (revision_id is None):
            raise ValueError("checkpoint requires its original revision barrier")
        if revision_id is None:
            if start_epoch is not None:
                raise ValueError("offline checkpoint has an online epoch")
        elif (
            type(revision_id) is not str
            or not revision_id
            or type(start_epoch) is not int
            or start_epoch < 1
        ):
            raise ValueError("invalid checkpoint revision barrier")
        elif barrier is not None and barrier.is_revision_pending(revision_id):
            if not set(queue) <= set(barrier.pending_claim_ids()):
                raise ValueError("revision pending barrier does not match checkpoint")
        elif payload["phase"] != "committed":
            raise ValueError("revision barrier is no longer pending")
        for node in queue[:cursor]:
            decision = ReevaluationDecision.from_dict(decisions[node])
            if decision.reason is None and any(
                parent in decisions and decisions[parent]["reason"] is not None
                for parent in graph.parents(node)
            ):
                raise ValueError("resolved node depends on an unresolved parent")
        return cls(Path(path), graph, snapshot, payload, barrier)

    @staticmethod
    def _snapshot(original: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(original, Mapping):
            raise ValueError("original revision state must be a mapping")
        if any(type(key) is not str or not key or len(key) > 4096 for key in original):
            raise ValueError("invalid revision state node ID")
        # Normalize tuples and any custom JSON encoder quirks at the boundary.
        return json.loads(_canonical(dict(original)))

    def _verify_graph(self) -> None:
        if self.graph.fingerprint != self._payload["graph_digest"]:
            raise ValueError("dependency graph changed during revision")

    def _verify_barrier(self) -> None:
        if self.barrier is not None and self.phase != "committed":
            revision_id = self._payload["revision_id"]
            if not self.barrier.is_revision_pending(revision_id) or not set(
                self.affected_ids
            ) <= set(self.barrier.pending_claim_ids()):
                raise ValueError("revision barrier changed during processing")

    @property
    def phase(self) -> str:
        return self._payload["phase"]

    @property
    def state_version(self) -> str:
        return self._payload["state_version"]

    @property
    def affected_ids(self) -> tuple[str, ...]:
        return tuple(self._payload["queue"])

    @property
    def pending_claim_ids(self) -> tuple[str, ...]:
        if self.phase != "committed":
            return tuple(sorted(self.affected_ids))
        return tuple(
            sorted(
                node
                for node, raw in self._payload["decisions"].items()
                if raw["reason"] is not None
            )
        )

    @property
    def progress(self) -> RevisionProgress:
        return RevisionProgress(
            self.phase,
            self._payload["cursor"],
            len(self.affected_ids),
            self.pending_claim_ids,
            self.state_version,
        )

    def _write(self, *, exclusive: bool = False, max_bytes: int | None = None) -> None:
        raw = _canonical({"payload": self._payload, "sha256": _digest(self._payload)})
        if max_bytes is not None and len(raw) > max_bytes:
            raise ValueError("revision checkpoint exceeds artifact budget")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".revision-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            if exclusive:
                os.link(temporary, self.path)
            else:
                os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Serialize workers of the same job, also across process crashes."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            expected = _canonical(
                {
                    "payload": self._payload,
                    "sha256": _digest(self._payload),
                }
            )
            if self.path.read_bytes() != expected:
                raise ValueError("revision checkpoint advanced in another worker")
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def run_batch(
        self,
        evaluator: Callable[[str, Mapping[str, Any]], ReevaluationDecision],
        *,
        batch_size: int,
        budget: BudgetTracker,
    ) -> RevisionProgress:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._exclusive():
            return self._run_batch_locked(
                evaluator, batch_size=batch_size, budget=budget
            )

    def _run_batch_locked(
        self,
        evaluator: Callable[[str, Mapping[str, Any]], ReevaluationDecision],
        *,
        batch_size: int,
        budget: BudgetTracker,
    ) -> RevisionProgress:
        self._verify_graph()
        self._verify_barrier()
        if self.phase != "pending":
            return self.progress
        candidate = deepcopy(self._payload)
        view = deepcopy(self.original)
        for node, raw in candidate["decisions"].items():
            if raw["reason"] is None:
                view[node] = deepcopy(raw["value"])
        reason: str | None = None
        for node in self.affected_ids[
            candidate["cursor"] : candidate["cursor"] + batch_size
        ]:
            try:
                budget.consume(steps=1, items=1)
            except BudgetExceeded as exc:
                reason = exc.reason
                break
            uncertain_parents = sorted(
                parent
                for parent in self.graph.parents(node)
                if parent in candidate["decisions"]
                and candidate["decisions"][parent]["reason"] is not None
            )
            if uncertain_parents:
                decision = ReevaluationDecision(
                    reason=(
                        f"unresolved_parents:{len(uncertain_parents)}:"
                        + uncertain_parents[0][:256]
                    )
                )
            else:
                context = {
                    key: deepcopy(view[key])
                    for key in (node, *self.graph.parents(node))
                    if key in view
                }
                decision = evaluator(node, context)
                if not isinstance(decision, ReevaluationDecision):
                    raise ValueError("evaluator must return ReevaluationDecision")
            raw = decision.to_dict()
            candidate["decisions"][node] = raw
            if decision.reason is None:
                view[node] = deepcopy(raw["value"])
            candidate["cursor"] += 1
        if candidate["cursor"] == len(self.affected_ids):
            candidate["phase"] = "ready"
        previous = self._payload
        self._payload = candidate
        try:
            self._write(max_bytes=budget.budget.max_artifact_bytes)
        except BaseException:
            self._payload = previous
            raise
        result = self.progress
        return RevisionProgress(
            result.phase,
            result.processed,
            result.total,
            result.pending_claim_ids,
            result.state_version,
            reason,
        )

    def commit(
        self,
        *,
        current_state_version: str,
        current_original: Mapping[str, Any],
    ) -> RevisionProgress:
        """Atomically expose the offline result if the source is still current.

        The caller must pass a fresh read of the source immediately before
        commit. A live service additionally needs its own serialized store
        transaction covering this check and response publication.
        """
        with self._exclusive():
            return self._commit_locked(
                current_state_version=current_state_version,
                current_original=current_original,
            )

    def _commit_locked(
        self,
        *,
        current_state_version: str,
        current_original: Mapping[str, Any],
    ) -> RevisionProgress:
        self._verify_graph()
        self._verify_barrier()
        if (
            self.barrier is not None
            and self.pending_claim_ids
            and self.phase != "committed"
            and any(
                raw["reason"] is not None for raw in self._payload["decisions"].values()
            )
        ):
            raise ValueError("online revision has unresolved claims")
        if (
            current_state_version != self.state_version
            or _digest(self._snapshot(current_original))
            != self._payload["original_digest"]
        ):
            raise ValueError("revision source changed before commit")
        if self.phase == "pending":
            raise ValueError("cannot commit an unfinished revision")
        if self.phase == "ready":
            previous = self._payload
            self._payload = {**previous, "phase": "committed"}
            try:
                self._write()
            except BaseException:
                self._payload = previous
                raise
        return self.progress

    def finish_barrier(
        self, *, durable_state_version: str, durable_state: Mapping[str, Any]
    ) -> int:
        """Release the SQL gate *after* the caller durably activates this state.

        A process crash before this call leaves pending claims blocked. The
        shared store checks the original epoch and current state under SQL's
        writer lock; a concurrent revocation prevents an old job finishing.
        """
        if self.phase != "committed" or self.pending_claim_ids:
            raise ValueError("only a fully resolved committed job can release its gate")
        if self.barrier is None:
            raise ValueError("offline revision has no publication barrier")
        if type(durable_state_version) is not str or not durable_state_version:
            raise ValueError("invalid durable state version")
        if _digest(self._snapshot(durable_state)) != _digest(self.materialized_state()):
            raise ValueError("durable state differs from committed revision")
        return self.barrier.finish_revision(
            self._payload["revision_id"],
            durable_state_version,
            expected_epoch=self._payload["start_epoch"],
        )

    def materialized_state(self) -> dict[str, Any]:
        """Return a detached committed view; unresolved nodes stay blocked."""
        if self.phase != "committed":
            raise ValueError("revision must be committed before publishing")
        result = deepcopy(self.original)
        for node, raw in self._payload["decisions"].items():
            if raw["reason"] is None:
                result[node] = deepcopy(raw["value"])
        return result

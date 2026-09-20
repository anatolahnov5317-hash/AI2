"""Dependency-driven bounded revision with resumable checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .budget import BudgetExceeded, BudgetTracker


def _id(value: str, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class RevisionCheckpoint:
    state_version: str
    changed_ids: tuple[str, ...]
    queue: tuple[str, ...]
    cursor: int
    processed_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _id(self.state_version, "state_version")
        for value in (*self.changed_ids, *self.queue, *self.processed_ids):
            _id(value, "revision id")
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.queue):
            raise ValueError("invalid revision cursor")
        if self.processed_ids != self.queue[: self.cursor]:
            raise ValueError("processed_ids must equal the consumed queue prefix")

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.queue)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_ids"] = list(self.changed_ids)
        value["queue"] = list(self.queue)
        value["processed_ids"] = list(self.processed_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RevisionCheckpoint:
        if type(value) is not dict:
            raise ValueError("revision checkpoint must be an object")
        expected = {"state_version", "changed_ids", "queue", "cursor", "processed_ids"}
        if set(value) != expected:
            raise ValueError("invalid revision checkpoint fields")
        return cls(
            state_version=value["state_version"],
            changed_ids=tuple(value["changed_ids"]),
            queue=tuple(value["queue"]),
            cursor=value["cursor"],
            processed_ids=tuple(value["processed_ids"]),
        )


@dataclass(frozen=True, slots=True)
class RevisionResult:
    checkpoint: RevisionCheckpoint
    complete: bool
    stop_reason: str | None = None


class DependencyGraph:
    """Directed provenance/dependency graph.

    An edge A -> B means B depends on A and must be reconsidered when A changes.
    """

    def __init__(self) -> None:
        self._dependents: dict[str, set[str]] = {}

    def add_node(self, node_id: str) -> None:
        _id(node_id, "node_id")
        self._dependents.setdefault(node_id, set())

    def add_dependency(self, parent_id: str, dependent_id: str) -> None:
        _id(parent_id, "parent_id")
        _id(dependent_id, "dependent_id")
        if parent_id == dependent_id:
            raise ValueError("self dependency is not allowed")
        self.add_node(parent_id)
        self.add_node(dependent_id)
        self._dependents[parent_id].add(dependent_id)

    def dependents(self, node_id: str) -> tuple[str, ...]:
        _id(node_id, "node_id")
        return tuple(sorted(self._dependents.get(node_id, set())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependents": {
                node_id: sorted(dependents)
                for node_id, dependents in sorted(self._dependents.items())
            }
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DependencyGraph:
        if (
            type(value) is not dict
            or set(value) != {"dependents"}
            or type(value["dependents"]) is not dict
        ):
            raise ValueError("invalid dependency graph")
        graph = cls()
        for node_id, raw_dependents in value["dependents"].items():
            if (
                type(node_id) is not str
                or type(raw_dependents) is not list
                or any(type(dependent) is not str for dependent in raw_dependents)
            ):
                raise ValueError("invalid dependency entry")
            graph.add_node(node_id)
            for dependent in raw_dependents:
                graph.add_dependency(node_id, dependent)
        return graph

    def affected(self, changed_ids: tuple[str, ...]) -> tuple[str, ...]:
        if not changed_ids:
            raise ValueError("changed_ids cannot be empty")
        queue = list(dict.fromkeys(changed_ids))
        for node_id in queue:
            _id(node_id, "changed_id")
        seen = set(queue)
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for dependent in sorted(self._dependents.get(current, set())):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        return tuple(queue)

    def start_revision(
        self, changed_ids: tuple[str, ...], *, state_version: str
    ) -> RevisionCheckpoint:
        queue = self.affected(changed_ids)
        return RevisionCheckpoint(
            state_version=state_version,
            changed_ids=tuple(dict.fromkeys(changed_ids)),
            queue=queue,
            cursor=0,
            processed_ids=(),
        )

    def process(
        self,
        checkpoint: RevisionCheckpoint,
        *,
        handler,
        budget: BudgetTracker,
        batch_size: int = 64,
    ) -> RevisionResult:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if checkpoint.complete:
            return RevisionResult(checkpoint, True)

        cursor = checkpoint.cursor
        limit = min(len(checkpoint.queue), cursor + batch_size)
        while cursor < limit:
            node_id = checkpoint.queue[cursor]
            try:
                budget.consume(steps=1, items=1)
            except BudgetExceeded as exc:
                current = RevisionCheckpoint(
                    state_version=checkpoint.state_version,
                    changed_ids=checkpoint.changed_ids,
                    queue=checkpoint.queue,
                    cursor=cursor,
                    processed_ids=checkpoint.queue[:cursor],
                )
                return RevisionResult(current, False, exc.reason)
            handler(node_id)
            cursor += 1

        current = RevisionCheckpoint(
            state_version=checkpoint.state_version,
            changed_ids=checkpoint.changed_ids,
            queue=checkpoint.queue,
            cursor=cursor,
            processed_ids=checkpoint.queue[:cursor],
        )
        return RevisionResult(current, current.complete)

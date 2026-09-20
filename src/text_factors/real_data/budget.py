"""Explicit resource budgets and resumable progress for long real-data tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any


def _positive_int(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


def _positive_float(value: float | None, name: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Finite budgets are engineering contracts, not language restrictions."""

    max_steps: int | None = 100_000
    max_items: int | None = 100_000
    max_bytes: int | None = 256 * 1024 * 1024
    max_wall_seconds: float | None = 300.0
    max_artifact_bytes: int | None = 128 * 1024 * 1024
    checkpoint_every_steps: int = 1000

    def __post_init__(self) -> None:
        _positive_int(self.max_steps, "max_steps")
        _positive_int(self.max_items, "max_items")
        _positive_int(self.max_bytes, "max_bytes")
        _positive_float(self.max_wall_seconds, "max_wall_seconds")
        _positive_int(self.max_artifact_bytes, "max_artifact_bytes")
        _positive_int(self.checkpoint_every_steps, "checkpoint_every_steps")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    steps: int
    items: int
    bytes_processed: int
    elapsed_seconds: float
    last_checkpoint_step: int
    complete: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("steps", "items", "bytes_processed", "last_checkpoint_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.elapsed_seconds) not in (int, float) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.last_checkpoint_step > self.steps:
            raise ValueError("checkpoint cannot be ahead of processed steps")
        if self.complete and self.stop_reason is not None:
            raise ValueError("complete snapshot cannot have a stop_reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BudgetSnapshot":
        if type(value) is not dict:
            raise ValueError("budget snapshot must be an object")
        allowed = {
            "steps",
            "items",
            "bytes_processed",
            "elapsed_seconds",
            "last_checkpoint_step",
            "complete",
            "stop_reason",
        }
        if set(value) != allowed:
            raise ValueError("invalid budget snapshot fields")
        return cls(**value)


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, snapshot: BudgetSnapshot):
        super().__init__(reason)
        self.reason = reason
        self.snapshot = snapshot


class BudgetTracker:
    """Mutable counter with deterministic resume and injectable monotonic clock."""

    def __init__(
        self,
        budget: ResourceBudget,
        *,
        snapshot: BudgetSnapshot | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.budget = budget
        self._clock = clock
        now = float(clock())
        if snapshot is None:
            self.steps = 0
            self.items = 0
            self.bytes_processed = 0
            self.last_checkpoint_step = 0
            self._started_at = now
        else:
            if snapshot.complete:
                raise ValueError("cannot resume a completed budget snapshot")
            self.steps = snapshot.steps
            self.items = snapshot.items
            self.bytes_processed = snapshot.bytes_processed
            self.last_checkpoint_step = snapshot.last_checkpoint_step
            self._started_at = now - float(snapshot.elapsed_seconds)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def snapshot(
        self, *, complete: bool = False, stop_reason: str | None = None
    ) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps=self.steps,
            items=self.items,
            bytes_processed=self.bytes_processed,
            elapsed_seconds=self.elapsed_seconds,
            last_checkpoint_step=self.last_checkpoint_step,
            complete=complete,
            stop_reason=stop_reason,
        )

    def _raise(self, reason: str) -> None:
        raise BudgetExceeded(reason, self.snapshot(stop_reason=reason))

    def check(self) -> None:
        if (
            self.budget.max_steps is not None
            and self.steps > self.budget.max_steps
        ):
            self._raise("max_steps")
        if (
            self.budget.max_items is not None
            and self.items > self.budget.max_items
        ):
            self._raise("max_items")
        if (
            self.budget.max_bytes is not None
            and self.bytes_processed > self.budget.max_bytes
        ):
            self._raise("max_bytes")
        if (
            self.budget.max_wall_seconds is not None
            and self.elapsed_seconds > float(self.budget.max_wall_seconds)
        ):
            self._raise("max_wall_seconds")

    def consume(
        self, *, steps: int = 0, items: int = 0, bytes_processed: int = 0
    ) -> None:
        for name, value in (
            ("steps", steps),
            ("items", items),
            ("bytes_processed", bytes_processed),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} increment must be non-negative")
        self.steps += steps
        self.items += items
        self.bytes_processed += bytes_processed
        self.check()

    def should_checkpoint(self) -> bool:
        return (
            self.steps - self.last_checkpoint_step
            >= self.budget.checkpoint_every_steps
        )

    def mark_checkpoint(self) -> BudgetSnapshot:
        self.check()
        self.last_checkpoint_step = self.steps
        return self.snapshot()

    def check_artifact_size(self, size: int) -> None:
        if type(size) is not int or size < 0:
            raise ValueError("artifact size must be a non-negative integer")
        if (
            self.budget.max_artifact_bytes is not None
            and size > self.budget.max_artifact_bytes
        ):
            self._raise("max_artifact_bytes")

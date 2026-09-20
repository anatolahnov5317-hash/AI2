"""Scoped uncertainty state for partially understood or stale knowledge."""

from __future__ import annotations

from dataclasses import replace

from .contracts import Claim, UncertaintyScope


class UncertaintyIndex:
    """Store independent uncertainty scopes and resolve only matching regions."""

    def __init__(self) -> None:
        self._items: dict[str, UncertaintyScope] = {}

    @property
    def items(self) -> tuple[UncertaintyScope, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def add(self, scope: UncertaintyScope) -> None:
        existing = self._items.get(scope.uncertainty_id)
        if existing is not None and existing != scope:
            raise ValueError("uncertainty ID already exists with different content")
        self._items[scope.uncertainty_id] = scope

    def remove(self, uncertainty_id: str) -> None:
        self._items.pop(uncertainty_id, None)

    def affecting(self, claim: Claim) -> tuple[UncertaintyScope, ...]:
        return tuple(item for item in self.items if item.affects_claim(claim))

    def is_uncertain(self, claim: Claim) -> bool:
        return bool(self.affecting(claim))

    def resolve_claim(self, claim_id: str) -> tuple[str, ...]:
        """Remove only an explicit claim from scopes; keep unrelated uncertainty."""

        changed: list[str] = []
        for key, item in list(self._items.items()):
            if claim_id not in item.affected_claim_ids:
                continue
            remaining = tuple(
                value for value in item.affected_claim_ids if value != claim_id
            )
            changed.append(key)
            if (
                not remaining
                and not item.affected_instance_ids
                and item.relation_id is None
            ):
                del self._items[key]
            else:
                self._items[key] = replace(item, affected_claim_ids=remaining)
        return tuple(changed)

    def to_dict(self) -> dict[str, object]:
        return {"items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "UncertaintyIndex":
        if (
            type(value) is not dict
            or set(value) != {"items"}
            or type(value["items"]) is not list
        ):
            raise ValueError("invalid uncertainty index")
        index = cls()
        for raw in value["items"]:
            if type(raw) is not dict:
                raise ValueError("invalid uncertainty entry")
            index.add(UncertaintyScope.from_dict(raw))
        return index

    def resolve_instance(
        self, instance_id: str, *, relation_id: str | None = None
    ) -> tuple[str, ...]:
        """Resolve one instance/relation slice without clearing a whole topic."""

        changed: list[str] = []
        for key, item in list(self._items.items()):
            if instance_id not in item.affected_instance_ids:
                continue
            if relation_id is not None and item.relation_id not in (None, relation_id):
                continue
            remaining = tuple(
                value for value in item.affected_instance_ids if value != instance_id
            )
            changed.append(key)
            if (
                not remaining
                and not item.affected_claim_ids
                and item.relation_id is None
            ):
                del self._items[key]
            else:
                self._items[key] = replace(item, affected_instance_ids=remaining)
        return tuple(changed)

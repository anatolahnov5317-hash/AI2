"""Evidence provenance with deduplication by independent root groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _nonempty(value: str, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceRoot:
    root_id: str
    group_id: str
    source_id: str
    source_version: int

    def __post_init__(self) -> None:
        _nonempty(self.root_id, "root_id")
        _nonempty(self.group_id, "group_id")
        _nonempty(self.source_id, "source_id")
        if type(self.source_version) is not int or self.source_version <= 0:
            raise ValueError("source_version must be positive")


    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "group_id": self.group_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceRoot:
        expected = {"root_id", "group_id", "source_id", "source_version"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid evidence root")
        return cls(
            root_id=value["root_id"],
            group_id=value["group_id"],
            source_id=value["source_id"],
            source_version=value["source_version"],
        )


class EvidenceLedger:
    """Track roots instead of counting derivation paths as fresh evidence."""

    def __init__(self) -> None:
        self._roots: dict[str, EvidenceRoot] = {}
        self._claim_roots: dict[str, set[str]] = {}

    def register_root(self, root: EvidenceRoot) -> None:
        existing = self._roots.get(root.root_id)
        if existing is not None and existing != root:
            raise ValueError("evidence root ID reused with different provenance")
        self._roots[root.root_id] = root

    def attach(self, claim_id: str, root_ids: tuple[str, ...]) -> None:
        _nonempty(claim_id, "claim_id")
        unknown = [root for root in root_ids if root not in self._roots]
        if unknown:
            raise ValueError("unknown evidence root: " + ", ".join(sorted(unknown)))
        self._claim_roots.setdefault(claim_id, set()).update(root_ids)

    def derive(self, claim_id: str, parent_claim_ids: tuple[str, ...]) -> None:
        _nonempty(claim_id, "claim_id")
        roots: set[str] = set()
        for parent in parent_claim_ids:
            _nonempty(parent, "parent_claim_id")
            roots.update(self._claim_roots.get(parent, set()))
        self._claim_roots.setdefault(claim_id, set()).update(roots)

    def claim_roots(self, claim_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._claim_roots.get(claim_id, set())))

    def support_count(self, claim_id: str) -> int:
        return len(self._claim_roots.get(claim_id, set()))

    def independent_support(self, claim_id: str) -> int:
        return len(
            {
                self._roots[root_id].group_id
                for root_id in self._claim_roots.get(claim_id, set())
            }
        )

    def same_evidence(self, left_claim_id: str, right_claim_id: str) -> bool:
        left = self._claim_roots.get(left_claim_id, set())
        right = self._claim_roots.get(right_claim_id, set())
        return bool(left) and left == right

    def root(self, root_id: str) -> EvidenceRoot:
        try:
            return self._roots[root_id]
        except KeyError as exc:
            raise ValueError("unknown evidence root") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": [
                self._roots[root_id].to_dict()
                for root_id in sorted(self._roots)
            ],
            "claim_roots": {
                claim_id: sorted(root_ids)
                for claim_id, root_ids in sorted(self._claim_roots.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceLedger:
        if (
            type(value) is not dict
            or set(value) != {"roots", "claim_roots"}
            or type(value["roots"]) is not list
            or type(value["claim_roots"]) is not dict
        ):
            raise ValueError("invalid evidence ledger")
        ledger = cls()
        for raw in value["roots"]:
            if type(raw) is not dict:
                raise ValueError("invalid evidence root entry")
            ledger.register_root(EvidenceRoot.from_dict(raw))
        for claim_id, raw_roots in value["claim_roots"].items():
            if type(claim_id) is not str or type(raw_roots) is not list:
                raise ValueError("invalid claim evidence entry")
            roots = tuple(raw_roots)
            if any(type(root_id) is not str for root_id in roots):
                raise ValueError("invalid claim evidence root ID")
            ledger.attach(claim_id, roots)
        return ledger

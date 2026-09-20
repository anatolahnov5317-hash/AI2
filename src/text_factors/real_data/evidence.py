"""Evidence provenance with deduplication by independent root groups."""

from __future__ import annotations

from dataclasses import dataclass


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

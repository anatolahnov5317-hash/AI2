"""Scoped relations and bounded exchange of provenance, never vote counting.

Language, memory and historical features of one root remain one row. A support
cycle can move references to that root but cannot manufacture another witness.
These explicit contracts are engineering rules, not a learned truth estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .schema import bounded_text


@dataclass(frozen=True)
class Scope:
    event: str
    time: str = "present"
    speaker: str = "user"
    mode: str = "actual"

    def __post_init__(self) -> None:
        for name in ("event", "time", "speaker", "mode"):
            bounded_text(getattr(self, name), name, empty=False)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    scope: Scope
    slot: str
    value: str
    roots: frozenset[str] = frozenset()
    depends_on: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("claim_id", "slot", "value"):
            bounded_text(getattr(self, name), name, empty=False)
        if not isinstance(self.scope, Scope):
            raise ValueError("invalid claim scope")
        for names in (self.roots, self.depends_on):
            if type(names) is not frozenset or len(names) > 128:
                raise ValueError("invalid claim references")
            for name in names:
                bounded_text(name, "claim reference", empty=False)


def relation(left: Claim, right: Claim) -> str:
    if left.claim_id in right.depends_on or right.claim_id in left.depends_on:
        return "dependency"
    if left.scope != right.scope or left.slot != right.slot:
        return "coexistence"
    if left.value != right.value:
        return "conflict"
    return "duplicate" if left.roots & right.roots else "support"


class EvidenceLedger:
    """A shared ledger across all feature channels for one immutable snapshot."""

    def __init__(self, snapshot_id: str) -> None:
        bounded_text(snapshot_id, "snapshot", empty=False)
        self.snapshot_id = snapshot_id
        self._rows: dict[str, dict[str, float]] = {}

    def observe(
        self, root: str, channel: str, value: float, *, snapshot_id: str, complete: bool
    ) -> None:
        if snapshot_id != self.snapshot_id:
            raise ValueError("stale evidence snapshot")
        bounded_text(root, "evidence root", empty=False)
        bounded_text(channel, "evidence channel", empty=False)
        if type(complete) is not bool:
            raise ValueError("invalid evidence completion")
        if (
            type(value) not in (int, float)
            or not isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("invalid evidence feature")
        if not complete:
            return
        if root not in self._rows and len(self._rows) >= 128:
            raise ValueError("evidence root capacity")
        row = self._rows.setdefault(root, {})
        if channel not in row and len(row) >= 16:
            raise ValueError("evidence channel capacity")
        row[channel] = max(row.get(channel, 0.0), float(value))

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            key: dict(sorted(row.items())) for key, row in sorted(self._rows.items())
        }


def exchange(
    claims: tuple[Claim, ...],
    links: tuple[tuple[str, str], ...],
    *,
    max_rounds: int = 4,
) -> dict:
    """Synchronous monotone union; output is incomplete until a fixed point.

    Links are explicit support/dependency edges, not inferred from similarity.
    Conflicts never become positive links. Coexisting claims remain distinct.
    """
    if type(max_rounds) is not int or not 1 <= max_rounds <= 16:
        raise ValueError("invalid agreement round budget")
    if len(claims) > 64 or len(links) > 256:
        raise ValueError("agreement graph capacity")
    nodes = {claim.claim_id: claim for claim in claims}
    if len(nodes) != len(claims):
        raise ValueError("duplicate claim id")
    edges = sorted(set(links))
    for source, target in edges:
        if source not in nodes or target not in nodes:
            raise ValueError("unknown agreement node")
        if relation(nodes[source], nodes[target]) == "conflict":
            raise ValueError("conflict cannot pass positive evidence")
    roots = {name: set(claim.roots) for name, claim in nodes.items()}
    complete, rounds = False, 0
    for _ in range(max_rounds):
        rounds += 1
        updated = {name: set(values) for name, values in roots.items()}
        for source, target in edges:
            updated[target].update(roots[source])
            if len(updated[target]) > 128:
                raise ValueError("agreement provenance capacity")
        if updated == roots:
            complete = True
            break
        roots = updated
    return {
        "complete": complete,
        "rounds": rounds,
        "roots": {name: sorted(values) for name, values in sorted(roots.items())},
        "reason": "" if complete else "agreement_round_budget",
    }

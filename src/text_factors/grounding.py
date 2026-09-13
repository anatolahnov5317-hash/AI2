"""Optional research matching of explicitly named factor exemplars.

This is an engineering hypothesis, not a claim about a canonical theory or a
learned grammar. An atom is (memory point, matched input bit). The score is the
smaller of the two overlap fractions, not a probability. Output bits, observation
counts and content-key spelling do not contribute to similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

FactorAtoms = frozenset[tuple[int, int]]
MAX_MATCH_ATOMS = 8192
MAX_MATCH_EXEMPLARS = 4096
MAX_ATOM_VISITS = 262144


@dataclass(frozen=True, slots=True)
class GroundingPolicy:
    mode: str = "exact"
    min_atoms: int = 4
    min_points: int = 2
    min_shared_atoms: int = 3
    threshold: float = 0.8
    margin: float = 0.1

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {"exact", "factor"}:
            raise ValueError("grounding mode must be exact or factor")
        for name in ("min_atoms", "min_points", "min_shared_atoms"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_MATCH_ATOMS:
                raise ValueError(f"{name} must be an integer in [1, {MAX_MATCH_ATOMS}]")
        if self.min_points > self.min_atoms:
            raise ValueError("min_points cannot exceed min_atoms")
        for name in ("threshold", "margin"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.threshold <= 1 or not 0 <= self.margin <= 1:
            raise ValueError("threshold must be in (0, 1] and margin in [0, 1]")


@dataclass(frozen=True, slots=True)
class WordResolution:
    words: tuple[str, ...] = ()
    support_event_ids: tuple[int, ...] = ()
    method: str = "unknown"
    score: float = 0.0
    ambiguous: bool = False
    matched_content_keys: tuple[str, ...] = ()
    reason: str = "no_confirmed_match"


@dataclass(frozen=True, slots=True)
class FactorExemplar:
    content_key: str
    words: tuple[str, ...]
    support_event_ids: tuple[int, ...]
    atoms: FactorAtoms


def match_factor_words(
    atoms: FactorAtoms,
    exemplars: tuple[FactorExemplar, ...],
    policy: GroundingPolicy,
) -> WordResolution:
    """Resolve against the complete bounded exemplar set without updating it.

    Large additional contributions lower reciprocal coverage. If a disjoint
    additional contribution itself covers another named exemplar, this function
    asks for clarification instead of naming the whole after its largest part.
    Unknown noise and an unknown extra object cannot always be distinguished.
    """

    if (
        len(atoms) > MAX_MATCH_ATOMS
        or len(exemplars) > MAX_MATCH_EXEMPLARS
        or any(len(item.atoms) > MAX_MATCH_ATOMS for item in exemplars)
        or len(atoms) + sum(len(item.atoms) for item in exemplars) > MAX_ATOM_VISITS
    ):
        return WordResolution(reason="work_limit")
    if len(atoms) < policy.min_atoms or len({p for p, _ in atoms}) < policy.min_points:
        return WordResolution(reason="insufficient_evidence")
    matches: list[tuple[float, FactorExemplar, FactorAtoms]] = []
    for exemplar in exemplars:
        if len(exemplar.atoms) < policy.min_atoms:
            continue
        shared = atoms & exemplar.atoms
        if (
            len(shared) < policy.min_shared_atoms
            or len({point for point, _ in shared}) < policy.min_points
        ):
            continue
        score = len(shared) / max(len(atoms), len(exemplar.atoms))
        matches.append((score, exemplar, shared))
    if not matches:
        return WordResolution(reason="insufficient_shared_evidence")
    matches.sort(key=lambda item: (-item[0], item[1].content_key))
    best_score, best, best_shared = matches[0]
    if best_score < policy.threshold:
        return WordResolution(score=best_score, reason="below_threshold")

    # One compact composition guard: a strong main match plus a disjoint named
    # remainder. Do not infer a new object segmentation or merge their labels.
    if len(best_shared) / len(best.atoms) >= policy.threshold:
        for _, other, shared in matches[1:]:
            if (
                set(other.words) != set(best.words)
                and len(shared) / len(other.atoms) >= policy.threshold
                and best_shared.isdisjoint(shared)
            ):
                return _resolution(
                    (best, other),
                    best_score,
                    ambiguous=True,
                    reason="multiple_named_parts",
                )

    near = tuple(
        exemplar
        for score, exemplar, _ in matches
        if best_score - score <= policy.margin + 1e-12
    )
    ambiguous = any(set(item.words) != set(best.words) for item in near)
    return _resolution(
        near,
        best_score,
        ambiguous=ambiguous,
        reason="competing_names" if ambiguous else "factor_overlap",
    )


def _resolution(
    exemplars: tuple[FactorExemplar, ...],
    score: float,
    *,
    ambiguous: bool,
    reason: str,
) -> WordResolution:
    return WordResolution(
        words=tuple(sorted({word for item in exemplars for word in item.words})),
        support_event_ids=tuple(
            sorted({event for item in exemplars for event in item.support_event_ids})
        ),
        method="ambiguous" if ambiguous else "factor",
        score=score,
        ambiguous=ambiguous,
        matched_content_keys=tuple(sorted({item.content_key for item in exemplars})),
        reason=reason,
    )

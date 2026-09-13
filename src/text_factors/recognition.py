"""Bounded recognition of explicit views against one local factor memory.

This is an additional read API. It keeps all stable contributions above the
activation threshold rather than applying the legacy global quality cutoff.
Views and their source positions are supplied by the observation adapter;
discovering object boundaries or semantic contradictions is not assumed here.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from math import isfinite, log1p
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .memory import ClusterStatus, CombinatorialMemory

RelationKind = Literal["duplicate", "compatible", "conflict", "undetermined"]


@dataclass(frozen=True, slots=True)
class InterpretationClaim:
    """A scoped hypothesis declared by an adapter, not inferred from bit overlap."""

    scope: str
    subject: str
    property: str
    value: str

    def __post_init__(self) -> None:
        if any(
            type(v) is not str or not v or len(v) > 256 for v in asdict(self).values()
        ):
            raise ValueError(
                "claim fields must be nonempty strings of at most 256 characters"
            )

    @builtins.property
    def key(self) -> tuple[str, str, str]:
        return self.scope, self.subject, self.property


@dataclass(frozen=True, slots=True)
class ContextView:
    """One explicitly located interpretation of an observation."""

    view_id: str
    context_id: str
    bits: NDArray[np.bool_]
    source_positions: tuple[int, ...]
    observation_id: str = "input"
    claims: tuple[InterpretationClaim, ...] = ()
    bit_sources: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class ClusterEvidence:
    point_index: int
    signature: tuple[int, ...]
    matched_bits: tuple[int, ...]
    observations: int
    output_bit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecognitionCandidate:
    candidate_id: str
    context_id: str
    content_key: str
    output_bits: tuple[int, ...]
    source_positions: tuple[int, ...]
    evidence: tuple[ClusterEvidence, ...]
    familiarity: float
    quality: int
    active_points: int
    observation_id: str = "input"
    claims: tuple[InterpretationClaim, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateRelation:
    left: str
    right: str
    kind: RelationKind
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    candidates: tuple[RecognitionCandidate, ...]
    relations: tuple[CandidateRelation, ...]
    complete: bool
    examined_views: int
    total_views: int
    suppressed: tuple[RecognitionCandidate, ...] = ()
    stop_reason: str | None = None
    memory_step: int = 0
    encoding_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecognitionLimits:
    max_views: int = 128
    max_candidates: int = 64
    max_evidence: int = 8192
    seconds: float = 5.0

    def __post_init__(self) -> None:
        for key, ceiling in (
            ("max_views", 2048),
            ("max_candidates", 128),
            ("max_evidence", 65536),
        ):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{key} must be an integer in [1, {ceiling}]")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or self.seconds <= 0
        ):
            raise ValueError("seconds must be a positive finite number")


def memory_encoding_id(memory: CombinatorialMemory) -> str:
    """Identify this local receptive/output coordinate system, not a global ontology."""

    digest = hashlib.sha256(
        json.dumps(memory.config.to_dict(), sort_keys=True).encode()
    )
    for array in (memory.receptors, memory.output_map):
        digest.update(array.astype("<i4", copy=False).tobytes())
    return digest.hexdigest()


def relate_candidates(
    left: RecognitionCandidate, right: RecognitionCandidate
) -> CandidateRelation:
    """Compare only available provenance and explicitly scoped claims.

    Disjoint support means these contributions can coexist in this observation,
    not that their meanings are universally compatible. Other overlap remains
    undetermined. Relations are never transitively merged.
    """

    left_claims = {claim.key: claim.value for claim in left.claims}
    right_claims = {claim.key: claim.value for claim in right.claims}
    kind: RelationKind = "undetermined"
    reason = (
        "overlapping or unrelated provenance does not establish a semantic relation"
    )
    if any(
        left_claims[key] != right_claims[key]
        for key in left_claims.keys() & right_claims.keys()
    ):
        kind, reason = (
            "conflict",
            "different values for the same explicit scope, subject and property",
        )
    elif (
        left.observation_id == right.observation_id
        and left.source_positions
        and left.source_positions == right.source_positions
        and left.content_key == right.content_key
        and left_claims == right_claims
        and tuple((e.point_index, e.signature, e.matched_bits) for e in left.evidence)
        == tuple((e.point_index, e.signature, e.matched_bits) for e in right.evidence)
    ):
        kind, reason = "duplicate", "same factor evidence and same located contribution"
    elif (
        left.observation_id == right.observation_id
        and left.source_positions
        and right.source_positions
        and set(left.source_positions).isdisjoint(right.source_positions)
    ):
        kind, reason = (
            "compatible",
            "disjoint contributions to one observation, with no declared contradiction",
        )
    return CandidateRelation(left.candidate_id, right.candidate_id, kind, reason)


def _validate_view(view: ContextView, memory: CombinatorialMemory) -> NDArray[np.bool_]:
    if not isinstance(view, ContextView):
        raise ValueError("views must contain ContextView records")
    for value in (view.view_id, view.context_id, view.observation_id):
        if type(value) is not str or not value or len(value) > 256:
            raise ValueError(
                "view identifiers must be nonempty strings of at most 256 characters"
            )
    active = np.asarray(view.bits)
    if active.dtype != np.bool_ or active.shape != (memory.config.input_bits,):
        raise ValueError("view bits must be a Boolean vector matching input_bits")
    if (
        type(view.source_positions) is not tuple
        or len(view.source_positions) > 4096
        or any(type(p) is not int or p < 0 for p in view.source_positions)
        or tuple(sorted(set(view.source_positions))) != view.source_positions
    ):
        raise ValueError("source positions must be sorted unique non-negative integers")
    if type(view.claims) is not tuple or any(
        not isinstance(c, InterpretationClaim) for c in view.claims
    ):
        raise ValueError("claims must be a tuple of InterpretationClaim records")
    keys = [claim.key for claim in view.claims]
    if len(set(keys)) != len(keys):
        raise ValueError("each view may assert each scoped property only once")
    if type(view.bit_sources) is not tuple or len(view.bit_sources) > 4096:
        raise ValueError("bit_sources must be a tuple")
    seen: set[int] = set()
    covered: set[int] = set()
    for position, bits in view.bit_sources:
        if (
            type(position) is not int
            or position not in view.source_positions
            or position in seen
        ):
            raise ValueError(
                "bit source must identify a unique declared source position"
            )
        seen.add(position)
        if type(bits) is not tuple or any(
            type(b) is not int or not 0 <= b < memory.config.input_bits for b in bits
        ):
            raise ValueError("bit sources must contain valid input bit indices")
        if tuple(sorted(set(bits))) != bits or any(not active[b] for b in bits):
            raise ValueError("bit sources must be sorted unique active indices")
        covered.update(bits)
    if view.bit_sources and covered != set(int(b) for b in np.flatnonzero(active)):
        raise ValueError("bit_sources must account for every active input bit")
    return active


def recognize_views(
    memory: CombinatorialMemory,
    views: Iterable[ContextView],
    *,
    total_views: int,
    limits: RecognitionLimits | None = None,
    encoding_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RecognitionResult:
    """Read a bounded stream of explicit views without learning or global top-1.

    ``total_views`` describes the source stream so truncation stays visible.
    No unseen view is classified as unknown or incompatible. A candidate's
    content key identifies a snapshot of contributing clusters; it is not a
    permanent concept ID and can change after consolidation.
    """

    if type(total_views) is not int or total_views < 0:
        raise ValueError("total_views must be a non-negative integer")
    limits = limits or RecognitionLimits()
    if not isinstance(limits, RecognitionLimits):
        raise ValueError("limits must be RecognitionLimits")
    if encoding_id is not None and (type(encoding_id) is not str or not encoding_id):
        raise ValueError("encoding_id must be a nonempty string")
    namespace = encoding_id or memory_encoding_id(memory)
    started = perf_counter()
    examined = 0
    found: list[RecognitionCandidate] = []
    seen_ids: set[str] = set()
    stop_reason: str | None = None
    evidence_count = 0

    def check_budget() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        if perf_counter() - started >= limits.seconds:
            raise InterruptedError("time_budget")

    iterator = iter(views)
    try:
        while examined < total_views:
            check_budget()
            if examined >= limits.max_views:
                stop_reason = "view_limit"
                break
            try:
                view = next(iterator)
            except StopIteration as error:
                raise ValueError("view stream is shorter than total_views") from error
            active = _validate_view(view, memory)
            if view.view_id in seen_ids:
                raise ValueError("view identifiers must be unique")
            seen_ids.add(view.view_id)
            evidence: list[ClusterEvidence] = []
            score = 0.0
            quality = 0
            for number, (point, cluster) in enumerate(memory.iter_clusters()):
                if number % 64 == 0:
                    check_budget()
                if cluster.status != ClusterStatus.STABLE:
                    continue
                matched = tuple(int(b) for b in cluster.bits if active[b])
                if len(matched) < memory.config.activation_threshold:
                    continue
                if evidence_count >= limits.max_evidence:
                    raise InterruptedError("evidence_limit")
                evidence_count += 1
                evidence.append(
                    ClusterEvidence(
                        point,
                        cluster.signature,
                        matched,
                        cluster.partial_hits,
                        int(memory.output_map[point]),
                    )
                )
                score += (
                    4.0 * len(matched) / len(cluster.bits) * log1p(cluster.partial_hits)
                )
                quality = max(quality, len(matched))
            examined += 1
            points = {e.point_index for e in evidence}
            if len(points) >= memory.config.min_active_points:
                evidence.sort(key=lambda e: (e.point_index, e.signature))
                digest = hashlib.sha256(namespace.encode())
                digest.update(
                    json.dumps(
                        [(e.point_index, e.signature) for e in evidence],
                        separators=(",", ":"),
                    ).encode()
                )
                supported = {b for e in evidence for b in e.matched_bits}
                origins = (
                    tuple(
                        sorted(
                            p
                            for p, bits in view.bit_sources
                            if supported.intersection(bits)
                        )
                    )
                    if view.bit_sources
                    else view.source_positions
                )
                found.append(
                    RecognitionCandidate(
                        candidate_id=view.view_id,
                        context_id=view.context_id,
                        content_key=digest.hexdigest(),
                        output_bits=tuple(sorted({e.output_bit for e in evidence})),
                        source_positions=origins,
                        evidence=tuple(evidence),
                        familiarity=score,
                        quality=quality,
                        active_points=len(points),
                        observation_id=view.observation_id,
                        claims=view.claims,
                    )
                )
            if progress is not None:
                progress(
                    f"recognition: {examined}/{total_views} views; "
                    f"{len(found)} candidates"
                )
    except InterruptedError as error:
        stop_reason = str(error)

    # A deterministic bounded prefix has been evaluated; scores never imply a
    # probability. Suppression requires direct duplicate evidence, not chains.
    found.sort(key=lambda c: (-c.familiarity, -c.quality, c.candidate_id))
    kept: list[RecognitionCandidate] = []
    suppressed: list[RecognitionCandidate] = []
    relations: list[CandidateRelation] = []
    for candidate in found:
        duplicate = next(
            (
                relate_candidates(other, candidate)
                for other in kept
                if relate_candidates(other, candidate).kind == "duplicate"
            ),
            None,
        )
        if duplicate is not None:
            relations.append(duplicate)
            suppressed.append(candidate)
        elif len(kept) < limits.max_candidates:
            kept.append(candidate)
        else:
            stop_reason = stop_reason or "candidate_limit"
    for index, left in enumerate(kept):
        for right in kept[index + 1 :]:
            relations.append(relate_candidates(left, right))
    return RecognitionResult(
        tuple(kept),
        tuple(relations),
        stop_reason is None and examined == total_views,
        examined,
        total_views,
        tuple(suppressed),
        stop_reason,
        memory.step,
        namespace,
    )

"""Read explicitly taught factor portraits from whole transformed scenes.

Portrait registration reads an already trained common memory. It stores cluster
conjunctions, never a raw-input template, a word or an evaluator's part mask.
This is supervised portrait registration, not discovery of objects or contexts.
At query time each portrait receives only its own supported factor evidence;
unrelated scene contributions do not dilute its coverage. Proposals sharing
factor atoms remain visible and unresolved. Neither shared nor disjoint atoms
establish object boundaries or semantic conflict; the supplied source scope is
never subdivided.

The memory is caller-owned and must not be trained concurrently with a read.
Further training is allowed between calls: vanished cluster signatures cease to
support old portraits. Replacing the receptive/output coordinates is rejected.
The caller remains responsible for using the declared input encoding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from math import isfinite, log1p
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .memory import ClusterStatus, CombinatorialMemory
from .recognition import (
    CandidateRelation,
    ClusterEvidence,
    ContextView,
    RecognitionCandidate,
    RecognitionLimits,
    RecognitionResult,
    _validate_view,
    memory_encoding_id,
    relate_candidates,
)


@dataclass(frozen=True, slots=True)
class SceneRecognitionConfig:
    min_atoms: int = 4
    min_points: int = 2
    min_shared_atoms: int = 3
    coverage: float = 0.6
    max_portraits: int = 128
    max_portrait_evidence: int = 8192
    max_atom_visits: int = 262144
    seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("min_atoms", 8192),
            ("min_points", 8192),
            ("min_shared_atoms", 8192),
            ("max_portraits", 2048),
            ("max_portrait_evidence", 65536),
            ("max_atom_visits", 4194304),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
        if self.min_points > self.min_atoms:
            raise ValueError("min_points cannot exceed min_atoms")
        for name in ("coverage", "seconds"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        if self.seconds <= 0:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True, slots=True)
class FactorPortrait:
    portrait_id: str
    evidence: tuple[ClusterEvidence, ...]
    atoms: tuple[tuple[int, int], ...]
    memory_step: int


@dataclass(frozen=True, slots=True)
class SceneProposal:
    candidate_id: str
    portrait_id: str
    support: int
    portrait_atoms: int
    coverage: float
    score: float
    support_bits: tuple[int, ...]
    overlapping_candidates: tuple[str, ...] = ()
    ambiguous: bool = False


@dataclass(frozen=True, slots=True)
class SceneViewTrace:
    view_id: str
    context_id: str
    observation_id: str
    active_bits: tuple[int, ...]
    supported_bits: tuple[int, ...]
    unexplained_bits: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    complete: bool
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SceneRecognitionResult:
    recognition: RecognitionResult
    proposals: tuple[SceneProposal, ...]
    view_traces: tuple[SceneViewTrace, ...]
    complete: bool
    stop_reason: str | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Budget:
    def __init__(
        self,
        seconds: float,
        atoms: int,
        evidence: int,
        cancelled: Callable[[], bool] | None,
        memory: CombinatorialMemory | None = None,
    ) -> None:
        self.started = perf_counter()
        self.seconds = seconds
        self.atoms_left = atoms
        self.evidence_left = evidence
        self.cancelled = cancelled
        self.memory = memory
        self.memory_step = memory.step if memory is not None else None

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("cancelled")
        if self.memory is not None and self.memory.step != self.memory_step:
            raise InterruptedError("memory_changed")
        if perf_counter() - self.started >= self.seconds:
            raise InterruptedError("time_budget")

    def atoms(self, count: int) -> None:
        self.check()
        self.atoms_left -= count
        if self.atoms_left < 0:
            raise InterruptedError("atom_visit_limit")

    def evidence(self, count: int = 1) -> None:
        self.check()
        self.evidence_left -= count
        if self.evidence_left < 0:
            raise InterruptedError("evidence_limit")


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")


def _atoms(evidence: Iterable[ClusterEvidence]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {(item.point_index, bit) for item in evidence for bit in item.matched_bits}
        )
    )


def _content_key(namespace: str, evidence: tuple[ClusterEvidence, ...]) -> str:
    # Including matched bits prevents a partial conjunction from acquiring the
    # exact key of a fully supported portrait. Neither portrait ID nor words
    # contribute; the same evidence has the same key in every context.
    digest = hashlib.sha256(namespace.encode())
    digest.update(b"scene-factor-evidence-v1")
    digest.update(
        json.dumps(
            [
                (item.point_index, item.signature, item.matched_bits)
                for item in evidence
            ],
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


class FactorSceneReader:
    """A bounded, read-only scene reader over explicitly registered portraits.

    Coverage counts distinct (memory point, matched input bit) atoms, but an atom
    can contribute only through the same cluster signature observed during
    portrait registration. Its shared conjunction must itself reach the memory's
    activation threshold. Thus equal raw bits at another point or in a different
    conjunction cannot stand in for learned evidence. Score equals coverage and
    is not a calibrated probability.

    Each interrupted view is discarded atomically; preceding complete views and
    a trace of the interrupted input survive. Residual bits mean unexplained by
    accepted portraits, not a recognized unknown object. Two identical instances
    are distinguishable only when their locations differ in the input encoding.
    ``ambiguous`` marks shared factor atoms, not a proven semantic conflict;
    disjoint atoms also do not prove that different objects were discovered.
    Training detected during a call aborts it with ``memory_changed``. Retained
    complete views report the memory step captured at the start of that call.
    """

    def __init__(
        self,
        memory: CombinatorialMemory,
        *,
        config: SceneRecognitionConfig | None = None,
        encoding_id: str | None = None,
    ) -> None:
        if not isinstance(memory, CombinatorialMemory):
            raise ValueError("memory must be CombinatorialMemory")
        if config is not None and not isinstance(config, SceneRecognitionConfig):
            raise ValueError("config must be SceneRecognitionConfig")
        if encoding_id is not None:
            _identifier(encoding_id, "encoding_id")
        self.memory = memory
        self.config = config or SceneRecognitionConfig()
        self._coordinate_fingerprint = memory_encoding_id(memory)
        self.encoding_id = encoding_id or self._coordinate_fingerprint
        self._portraits: dict[str, FactorPortrait] = {}
        self._stored_evidence = 0

    @property
    def portraits(self) -> tuple[FactorPortrait, ...]:
        """Immutable evidence snapshots, without retained input templates."""
        return tuple(self._portraits[key] for key in sorted(self._portraits))

    def to_dict(self) -> dict[str, Any]:
        """Export registration provenance for audits, not a memory checkpoint.

        The caller must separately retain the common memory. Loading/restoring a
        reader is deliberately not implied by this read-only snapshot API.
        """
        return {
            "schema": "factor-scene-reader-v1",
            "config": asdict(self.config),
            "encoding_id": self.encoding_id,
            "coordinate_fingerprint": self._coordinate_fingerprint,
            "portraits": [asdict(item) for item in self.portraits],
        }

    def _check_encoding(self, budget: _Budget) -> None:
        budget.check()
        if memory_encoding_id(self.memory) != self._coordinate_fingerprint:
            raise ValueError("memory coordinate encoding changed after reader creation")
        budget.check()

    def _read_evidence(
        self, active: NDArray[np.bool_], budget: _Budget
    ) -> tuple[ClusterEvidence, ...]:
        evidence: list[ClusterEvidence] = []
        for point, cluster in self.memory.iter_clusters():
            budget.check()
            if cluster.status != ClusterStatus.STABLE:
                continue
            budget.atoms(len(cluster.bits))
            matched = tuple(int(bit) for bit in cluster.bits if active[bit])
            if len(matched) < self.memory.config.activation_threshold:
                continue
            budget.evidence()
            evidence.append(
                ClusterEvidence(
                    point,
                    cluster.signature,
                    matched,
                    cluster.partial_hits,
                    int(self.memory.output_map[point]),
                )
            )
        return tuple(
            sorted(evidence, key=lambda item: (item.point_index, item.signature))
        )

    def observe_portrait(
        self,
        portrait_id: str,
        bits: NDArray[np.bool_],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Register a taught portrait without training the common memory.

        Return False for insufficient stable evidence. An identical evidence
        registration is idempotent; reusing an ID for different evidence raises
        ValueError. Budget/cancellation raises InterruptedError before any state
        change. Counts and time bound this explicit registration as well as reads.
        """
        _identifier(portrait_id, "portrait_id")
        budget = _Budget(
            self.config.seconds,
            self.config.max_atom_visits,
            self.config.max_portrait_evidence,
            cancelled,
            self.memory,
        )
        self._check_encoding(budget)
        view = ContextView("registration", "registration", bits, ())
        active = _validate_view(view, self.memory).copy()
        active.flags.writeable = False
        evidence = self._read_evidence(active, budget)
        budget.atoms(sum(len(item.matched_bits) for item in evidence))
        atoms = _atoms(evidence)
        if (
            len(atoms) < self.config.min_atoms
            or len({point for point, _ in atoms}) < self.config.min_points
        ):
            return False
        portrait = FactorPortrait(portrait_id, evidence, atoms, self.memory.step)
        existing = self._portraits.get(portrait_id)
        if existing is not None:
            if _content_key(self.encoding_id, evidence) != _content_key(
                self.encoding_id, existing.evidence
            ):
                raise ValueError("portrait_id already has different factor evidence")
            budget.check()
            return True
        if len(self._portraits) >= self.config.max_portraits:
            raise ValueError("portrait capacity exceeded")
        if self._stored_evidence + len(evidence) > self.config.max_portrait_evidence:
            raise ValueError("stored portrait evidence capacity exceeded")
        budget.check()
        self._portraits[portrait_id] = portrait
        self._stored_evidence += len(evidence)
        return True

    def _proposal(
        self,
        view: ContextView,
        portrait: FactorPortrait,
        available: dict[tuple[int, tuple[int, ...]], ClusterEvidence],
        budget: _Budget,
    ) -> tuple[RecognitionCandidate, SceneProposal] | None:
        evidence: list[ClusterEvidence] = []
        for anchor in portrait.evidence:
            budget.atoms(len(anchor.matched_bits))
            current = available.get((anchor.point_index, anchor.signature))
            if current is None:
                continue
            budget.atoms(len(current.matched_bits))
            shared = tuple(sorted(set(anchor.matched_bits) & set(current.matched_bits)))
            if len(shared) < self.memory.config.activation_threshold:
                continue
            evidence.append(replace(current, matched_bits=shared))
        budget.atoms(sum(len(item.matched_bits) for item in evidence))
        atoms = _atoms(evidence)
        coverage = len(atoms) / len(portrait.atoms)
        if (
            len(atoms) < self.config.min_shared_atoms
            or len({point for point, _ in atoms}) < self.config.min_points
            or coverage < self.config.coverage
        ):
            return None
        budget.evidence(len(evidence))
        frozen = tuple(evidence)
        identity = json.dumps(
            [view.view_id, portrait.portrait_id], separators=(",", ":")
        )
        candidate_id = "scene-" + hashlib.sha256(identity.encode()).hexdigest()
        support_bits = tuple(sorted({bit for _, bit in atoms}))
        candidate = RecognitionCandidate(
            candidate_id=candidate_id,
            context_id=view.context_id,
            content_key=_content_key(self.encoding_id, frozen),
            output_bits=tuple(sorted({item.output_bit for item in frozen})),
            # This remains the caller's whole source scope, not a located part.
            source_positions=view.source_positions,
            evidence=frozen,
            familiarity=sum(
                4
                * len(item.matched_bits)
                / len(item.signature)
                * log1p(item.observations)
                for item in frozen
            ),
            quality=max(len(item.matched_bits) for item in frozen),
            active_points=len({point for point, _ in atoms}),
            observation_id=view.observation_id,
            claims=view.claims,
        )
        proposal = SceneProposal(
            candidate_id,
            portrait.portrait_id,
            len(atoms),
            len(portrait.atoms),
            coverage,
            coverage,
            support_bits,
        )
        budget.check()
        return candidate, proposal

    def recognize_views(
        self,
        views: Iterable[ContextView],
        *,
        total_views: int,
        limits: RecognitionLimits | None = None,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> SceneRecognitionResult:
        """Evaluate whole views without query labels, masks or memory updates.

        Limits cover raw stable evidence plus returned candidate evidence. No
        global top-1 or greedy competition removes a weaker supported portrait.
        Overlap of factor atoms is reported within each view; a relation of
        compatibility only asserts coexistence of disjoint factor supports.
        Neither relation proves object boundaries or a universal semantic
        relationship. Missing views remain unexamined.
        """
        if type(total_views) is not int or total_views < 0:
            raise ValueError("total_views must be a nonnegative integer")
        if limits is not None and not isinstance(limits, RecognitionLimits):
            raise ValueError("limits must be RecognitionLimits")
        read_step = self.memory.step
        checked_limits = limits or RecognitionLimits(seconds=self.config.seconds)
        budget = _Budget(
            checked_limits.seconds,
            self.config.max_atom_visits,
            checked_limits.max_evidence,
            cancelled,
            self.memory,
        )
        found: list[RecognitionCandidate] = []
        proposals: list[SceneProposal] = []
        traces: list[SceneViewTrace] = []
        relations: list[CandidateRelation] = []
        seen_ids: set[str] = set()
        examined = 0
        stop_reason: str | None = None
        pending_view: ContextView | None = None
        pending_bits: tuple[int, ...] = ()
        iterator = iter(views)
        try:
            self._check_encoding(budget)
            portraits = self.portraits
            while examined < total_views:
                budget.check()
                if examined >= checked_limits.max_views:
                    raise InterruptedError("view_limit")
                try:
                    view = next(iterator)
                except StopIteration as error:
                    raise ValueError(
                        "view stream is shorter than total_views"
                    ) from error
                active = _validate_view(view, self.memory).copy()
                active.flags.writeable = False
                if view.view_id in seen_ids:
                    raise ValueError("view identifiers must be unique")
                seen_ids.add(view.view_id)
                pending_view = view
                pending_bits = tuple(int(bit) for bit in np.flatnonzero(active))
                current = self._read_evidence(active, budget)
                available = {
                    (item.point_index, item.signature): item for item in current
                }
                local_candidates: list[RecognitionCandidate] = []
                local_proposals: list[SceneProposal] = []
                local_relations: list[CandidateRelation] = []
                for portrait in portraits:
                    budget.check()
                    matched = self._proposal(view, portrait, available, budget)
                    if matched is None:
                        continue
                    if (
                        len(found) + len(local_candidates)
                        >= checked_limits.max_candidates
                    ):
                        raise InterruptedError("candidate_limit")
                    candidate, proposal = matched
                    local_candidates.append(candidate)
                    local_proposals.append(proposal)
                overlaps: dict[str, list[str]] = {
                    item.candidate_id: [] for item in local_proposals
                }
                factor_supports: dict[str, frozenset[tuple[int, int]]] = {}
                for candidate in local_candidates:
                    budget.atoms(
                        sum(len(item.matched_bits) for item in candidate.evidence)
                    )
                    factor_supports[candidate.candidate_id] = frozenset(
                        _atoms(candidate.evidence)
                    )
                for index, left in enumerate(local_candidates):
                    for right_index in range(index + 1, len(local_candidates)):
                        budget.check()
                        right = local_candidates[right_index]
                        left_atoms = factor_supports[left.candidate_id]
                        right_atoms = factor_supports[right.candidate_id]
                        budget.atoms(len(left_atoms) + len(right_atoms))
                        overlap = not left_atoms.isdisjoint(right_atoms)
                        relation = relate_candidates(left, right)
                        if overlap:
                            overlaps[left.candidate_id].append(right.candidate_id)
                            overlaps[right.candidate_id].append(left.candidate_id)
                            if relation.kind != "conflict":
                                relation = CandidateRelation(
                                    left.candidate_id,
                                    right.candidate_id,
                                    "undetermined",
                                    "shared portrait factor atoms; "
                                    "identity or competition unresolved",
                                )
                        elif relation.kind != "conflict":
                            relation = CandidateRelation(
                                left.candidate_id,
                                right.candidate_id,
                                "compatible",
                                "disjoint factor supports in one view can coexist",
                            )
                        local_relations.append(relation)
                    for previous in found:
                        budget.check()
                        local_relations.append(relate_candidates(previous, left))
                local_proposals = [
                    replace(
                        item,
                        overlapping_candidates=tuple(
                            sorted(overlaps[item.candidate_id])
                        ),
                        ambiguous=bool(overlaps[item.candidate_id]),
                    )
                    for item in local_proposals
                ]
                supported = {
                    bit for item in local_proposals for bit in item.support_bits
                }
                trace = SceneViewTrace(
                    view.view_id,
                    view.context_id,
                    view.observation_id,
                    pending_bits,
                    tuple(sorted(supported)),
                    tuple(bit for bit in pending_bits if bit not in supported),
                    tuple(item.candidate_id for item in local_candidates),
                    True,
                )
                budget.check()
                found.extend(local_candidates)
                proposals.extend(local_proposals)
                relations.extend(local_relations)
                traces.append(trace)
                examined += 1
                pending_view = None
                if progress is not None:
                    progress(
                        f"scene-recognition: {examined}/{total_views} views; "
                        f"{len(found)} supported portraits"
                    )
                budget.check()
        except InterruptedError as error:
            stop_reason = str(error)
            if pending_view is not None:
                traces.append(
                    SceneViewTrace(
                        pending_view.view_id,
                        pending_view.context_id,
                        pending_view.observation_id,
                        pending_bits,
                        (),
                        pending_bits,
                        (),
                        False,
                        stop_reason,
                    )
                )
        complete = stop_reason is None and examined == total_views
        recognition = RecognitionResult(
            tuple(found),
            tuple(relations),
            complete,
            examined,
            total_views,
            stop_reason=stop_reason,
            memory_step=read_step,
            encoding_id=self.encoding_id,
        )
        return SceneRecognitionResult(
            recognition,
            tuple(proposals),
            tuple(traces),
            complete,
            stop_reason,
            perf_counter() - budget.started,
        )

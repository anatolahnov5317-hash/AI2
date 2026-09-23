"""Bridge from a versioned semantic hypothesis to an explicitly reviewed claim.

Parsing is a proposal, not a source of asserted facts. The bridge pins every
candidate to the immutable source revision, retains unexplained text in its
clause, and requires a separate reviewer decision before registering evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..observations import ObservationArchive
from .archive_bridge import source_slice
from .contracts import (
    Claim,
    ClaimModality,
    ClaimPolarity,
    ClaimStatus,
    RoleValue,
    SourceSlice,
)
from .engine import RealDataEngine
from .evidence import EvidenceRoot
from .open_semantics import SemanticGraph, Span


def _overlaps(first: Span, second: Span) -> bool:
    return first.start < second.end and second.start < first.end


def _pinned_source(graph: SemanticGraph, archive: ObservationArchive) -> None:
    if graph.schema != "ai2-supervised-event-graph-v1":
        raise ValueError("unknown graph schema")
    if archive.get_source(graph.source_id).version != graph.source_version:
        raise ValueError("semantic graph source revision is no longer current")
    record = archive.get_source(graph.source_id, graph.source_version)
    if record.sha256 != graph.text_sha256:
        raise ValueError("semantic graph does not match archived source revision")


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    """Non-publishable interpretation of one event in an archived source."""

    candidate_id: str
    source: SourceSlice
    source_digest: str
    excerpt: str
    draft_claim: Claim
    trigger: Span
    unexplained: tuple[Span, ...]
    time_label: str | None
    guarded_negation: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or not self.candidate_id
            or type(self.source_digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", self.source_digest)
            or self.draft_claim.status is not ClaimStatus.UNRESOLVED
            or self.draft_claim.source != self.source
            or self.draft_claim.evidence_roots
            or not self.excerpt
            or not (
                self.source.start
                <= self.trigger.start
                < self.trigger.end
                <= self.source.end
            )
            or type(self.guarded_negation) is not bool
        ):
            raise ValueError("invalid unreviewed candidate")


def propose_review_candidates(
    graph: SemanticGraph, archive: ObservationArchive
) -> tuple[ReviewCandidate, ...]:
    """Pin parsed events as proposals without touching the evidence ledger.

    Raw input and externally identified mentions use this same API. A parser's
    positive reading never erases an unexplained negation cue: its clause keeps
    the residual and cannot be promoted until the graph is corrected.
    """

    _pinned_source(graph, archive)
    result = []
    for event in graph.events:
        source = source_slice(
            archive,
            graph.source_id,
            graph.source_version,
            event.source_span.start,
            event.source_span.end,
        )
        excerpt = archive.read_span(
            source.source_id, source.source_version, source.start, source.end
        )
        unexplained = {
            span for span in graph.unexplained if _overlaps(span, event.source_span)
        }
        # Guardrail, not a replacement for a learned negation model: a missed
        # explicit Russian "не" immediately before the predicate cannot be
        # published as a positive event even if the graph falsely claims its
        # clause has no residual. The reviewer must correct the parse first.
        guarded_negation = False
        if not event.negated:
            before = excerpt[: event.trigger.start - source.start]
            marker = re.search(r"(?iu)(?<!\w)не\s+$", before)
            if marker is not None:
                guarded_negation = True
                unexplained.add(
                    Span(
                        source.start + marker.start(),
                        source.start + marker.start() + 2,
                    )
                )
        claim = Claim(
            claim_id=f"{event.event_id}:draft",
            relation_id=event.relation_id,
            arguments=tuple(
                RoleValue(role.role, role.instance_id, mention_id=role.mention_id)
                for role in event.roles
            ),
            status=ClaimStatus.UNRESOLVED,
            source=source,
            model_version=graph.model_fingerprint,
            polarity=(
                ClaimPolarity.NEGATIVE if event.negated else ClaimPolarity.POSITIVE
            ),
            # The event graph has no modality classifier. No source utterance
            # may be promoted to a factual assertion on this unknown value.
            modality=ClaimModality.UNKNOWN,
        )
        result.append(
            ReviewCandidate(
                candidate_id=event.event_id,
                source=source,
                source_digest=graph.text_sha256,
                excerpt=excerpt,
                draft_claim=claim,
                trigger=event.trigger,
                unexplained=tuple(
                    sorted(unexplained, key=lambda span: (span.start, span.end))
                ),
                time_label=event.time_label,
                guarded_negation=guarded_negation,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    """Human/external reviewer attests the exact excerpt and semantic fields.

    Equal strings alone cannot prove the semantic judgement was correct; this
    record keeps who made the decision and what was explicitly checked.
    """

    candidate_id: str
    reviewer_id: str
    claim_id: str
    excerpt: str
    relation_id: str
    arguments: tuple[RoleValue, ...]
    polarity: ClaimPolarity
    modality: ClaimModality
    status: ClaimStatus
    approved: bool
    speaker_id: str | None = None
    resolved_spans: tuple[Span, ...] = ()
    identity_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value or len(value) > 4096
            for value in (
                self.candidate_id,
                self.reviewer_id,
                self.claim_id,
                self.excerpt,
            )
        ):
            raise ValueError("review decision requires reviewer, ID and excerpt")
        if type(self.approved) is not bool:
            raise ValueError("approved must be a boolean")
        if not isinstance(self.polarity, ClaimPolarity) or not isinstance(
            self.modality, ClaimModality
        ):
            raise ValueError("review decision requires explicit semantic fields")
        if not isinstance(self.status, ClaimStatus):
            raise ValueError("review decision requires a valid evidence status")
        if type(self.resolved_spans) is not tuple or any(
            not isinstance(span, Span) for span in self.resolved_spans
        ):
            raise ValueError("resolved spans must be a tuple of exact source spans")
        if type(self.identity_bindings) is not tuple or any(
            type(binding) is not tuple
            or len(binding) != 2
            or any(type(item) is not str or not item for item in binding)
            for binding in self.identity_bindings
        ):
            raise ValueError("identity bindings must name exact mention and entity IDs")


def _reviewed_arguments(draft: Claim, decision: ReviewDecision) -> None:
    """Only a reviewer may bind one source-local mention to an existing entity."""

    if len(draft.arguments) != len(decision.arguments):
        raise ValueError("review changed the event's argument layout")
    expected_bindings: dict[str, str] = {}
    all_targets: dict[str, str] = {}
    for proposed, reviewed in zip(draft.arguments, decision.arguments, strict=True):
        if (
            not isinstance(reviewed, RoleValue)
            or reviewed.role != proposed.role
            or reviewed.value_type != proposed.value_type
            or reviewed.mention_id != proposed.mention_id
        ):
            raise ValueError("review changed a role, mention or value type")
        mention_id = proposed.mention_id
        if mention_id is not None:
            previous = all_targets.setdefault(mention_id, reviewed.value_id)
            if previous != reviewed.value_id:
                raise ValueError("the same mention was bound to different entities")
        if proposed.value_id != reviewed.value_id:
            if mention_id is None:
                raise ValueError("identity binding needs a source mention ID")
            expected_bindings[mention_id] = reviewed.value_id
    if (
        len(decision.identity_bindings) != len(expected_bindings)
        or dict(decision.identity_bindings) != expected_bindings
    ):
        raise ValueError("review must explicitly bind each changed mention ID")


def register_reviewed_candidate(
    candidate: ReviewCandidate,
    decision: ReviewDecision,
    root: EvidenceRoot,
    *,
    archive: ObservationArchive,
    engine: RealDataEngine,
    correction_target_id: str | None = None,
) -> Claim:
    """Register an independently reviewed event; never called by parsing.

    A reviewer must explicitly resolve every otherwise unexplained text span.
    A missed direct negation additionally requires correcting the parser graph.
    A reviewer cannot silently invert polarity, change roles, or turn possible
    wording into an observed fact. Corrections need an explicit old claim ID;
    equal words or matching roles never select the target automatically.
    """

    if not decision.approved or candidate.guarded_negation:
        raise ValueError("candidate denied or positive graph missed a negation")
    if len(decision.resolved_spans) != len(candidate.unexplained) or set(
        decision.resolved_spans
    ) != set(candidate.unexplained):
        raise ValueError("review must explicitly resolve every unexplained source span")
    draft = candidate.draft_claim
    if (
        decision.candidate_id != candidate.candidate_id
        or decision.claim_id == draft.claim_id
        or decision.excerpt != candidate.excerpt
        or decision.relation_id != draft.relation_id
        or decision.polarity is not draft.polarity
        or decision.modality in {ClaimModality.UNKNOWN, ClaimModality.QUESTION}
    ):
        raise ValueError("review does not match the proposed semantic hypothesis")
    _reviewed_arguments(draft, decision)
    if correction_target_id is not None and (
        type(correction_target_id) is not str
        or not correction_target_id
        or correction_target_id not in engine.claims
        or correction_target_id == decision.claim_id
        or decision.status not in {ClaimStatus.ASSERTED, ClaimStatus.OBSERVED}
    ):
        raise ValueError("addressed correction needs a valid old claim and status")
    if decision.modality in {ClaimModality.HYPOTHETICAL, ClaimModality.POSSIBLE}:
        if decision.status is not ClaimStatus.UNRESOLVED:
            raise ValueError("hypothetical or possible event cannot become a fact")
    elif decision.status not in {ClaimStatus.ASSERTED, ClaimStatus.OBSERVED}:
        raise ValueError("approved direct or reported event needs reviewed status")
    record = archive.get_source(
        candidate.source.source_id, candidate.source.source_version
    )
    if archive.get_source(candidate.source.source_id).version != record.version:
        raise ValueError("review candidate source revision is no longer current")
    if record.sha256 != candidate.source_digest:
        raise ValueError("review candidate is pinned to a different source revision")
    current = source_slice(
        archive,
        candidate.source.source_id,
        candidate.source.source_version,
        candidate.source.start,
        candidate.source.end,
    )
    if (
        current != candidate.source
        or archive.read_span(
            current.source_id, current.source_version, current.start, current.end
        )
        != candidate.excerpt
    ):
        raise ValueError("reviewed excerpt or source scope has changed")
    if (
        root.source != current
        or root.source_id != current.source_id
        or root.source_version != current.source_version
        or root.access_scope != current.access_scope
        or root.group_id != record.group_id
    ):
        raise ValueError("evidence root must pin the exact reviewed source slice")
    claim = Claim(
        claim_id=decision.claim_id,
        relation_id=decision.relation_id,
        arguments=decision.arguments,
        status=decision.status,
        speaker_id=decision.speaker_id,
        source=current,
        evidence_roots=(root.root_id,),
        model_version=draft.model_version,
        polarity=decision.polarity,
        modality=decision.modality,
        reviewer_id=decision.reviewer_id,
    )
    engine.register_evidence(root)
    if correction_target_id is None:
        engine.add_claim(claim)
    else:
        engine.correct_claim(correction_target_id, claim)
    return claim

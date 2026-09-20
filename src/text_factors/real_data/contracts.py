"""Domain-independent contracts for the experimental real-data route.

The records in this module intentionally separate provenance, interpretation,
claims, uncertainty and answer receipts. They contain no built-in ontology of
people, objects, verbs or relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


def _text(value: str, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not value and not empty) or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"invalid Unicode in {name}") from exc
    return value


def _optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


class ClaimStatus(str, Enum):
    """Evidence status. Prediction is never promoted to observation implicitly."""

    ASSERTED = "asserted"
    OBSERVED = "observed"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    DISPUTED = "disputed"
    RETRACTED = "retracted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SourceSlice:
    source_id: str
    source_version: int
    start: int
    end: int
    sha256: str
    access_scope: str = "default"

    def __post_init__(self) -> None:
        _text(self.source_id, "source_id")
        _text(self.sha256, "sha256")
        _text(self.access_scope, "access_scope")
        if type(self.source_version) is not int or self.source_version <= 0:
            raise ValueError("source_version must be positive")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("source offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source slice must be a nonempty half-open range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "start": self.start,
            "end": self.end,
            "sha256": self.sha256,
            "access_scope": self.access_scope,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceSlice":
        expected = {
            "source_id",
            "source_version",
            "start",
            "end",
            "sha256",
            "access_scope",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid source slice")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class RoleValue:
    role: str
    value_id: str
    value_type: str = "instance"
    mention_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.role, "role")
        _text(self.value_id, "value_id")
        _text(self.value_type, "value_type")
        _optional_text(self.mention_id, "mention_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "value_id": self.value_id,
            "value_type": self.value_type,
            "mention_id": self.mention_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoleValue":
        expected = {"role", "value_id", "value_type", "mention_id"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid role value")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    relation_id: str
    arguments: tuple[RoleValue, ...]
    status: ClaimStatus
    valid_from: str | None = None
    valid_to: str | None = None
    speaker_id: str | None = None
    source: SourceSlice | None = None
    evidence_roots: tuple[str, ...] = ()
    model_version: str | None = None

    def __post_init__(self) -> None:
        _text(self.claim_id, "claim_id")
        _text(self.relation_id, "relation_id")
        _optional_text(self.valid_from, "valid_from")
        _optional_text(self.valid_to, "valid_to")
        _optional_text(self.speaker_id, "speaker_id")
        _optional_text(self.model_version, "model_version")
        if not isinstance(self.status, ClaimStatus):
            raise ValueError("status must be ClaimStatus")
        roles = [item.role for item in self.arguments]
        if len(roles) != len(set(roles)):
            raise ValueError("claim roles must be unique")
        for root in self.evidence_roots:
            _text(root, "evidence_root")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "relation_id": self.relation_id,
            "arguments": [item.to_dict() for item in self.arguments],
            "status": self.status.value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "speaker_id": self.speaker_id,
            "source": self.source.to_dict() if self.source else None,
            "evidence_roots": list(self.evidence_roots),
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Claim":
        expected = {
            "claim_id",
            "relation_id",
            "arguments",
            "status",
            "valid_from",
            "valid_to",
            "speaker_id",
            "source",
            "evidence_roots",
            "model_version",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid claim")
        raw_arguments = value["arguments"]
        raw_roots = value["evidence_roots"]
        if type(raw_arguments) is not list or type(raw_roots) is not list:
            raise ValueError("invalid claim collections")
        raw_source = value["source"]
        if raw_source is not None and type(raw_source) is not dict:
            raise ValueError("invalid claim source")
        try:
            status = ClaimStatus(value["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid claim status") from exc
        return cls(
            claim_id=value["claim_id"],
            relation_id=value["relation_id"],
            arguments=tuple(RoleValue.from_dict(item) for item in raw_arguments),
            status=status,
            valid_from=value["valid_from"],
            valid_to=value["valid_to"],
            speaker_id=value["speaker_id"],
            source=(
                SourceSlice.from_dict(raw_source)
                if raw_source is not None
                else None
            ),
            evidence_roots=tuple(raw_roots),
            model_version=value["model_version"],
        )


@dataclass(frozen=True, slots=True)
class Interpretation:
    interpretation_id: str
    source: SourceSlice
    context_id: str
    claims: tuple[Claim, ...]
    alternative_ids: tuple[str, ...] = ()
    score: float | None = None
    model_version: str | None = None
    complete: bool = False

    def __post_init__(self) -> None:
        _text(self.interpretation_id, "interpretation_id")
        _text(self.context_id, "context_id")
        _optional_text(self.model_version, "model_version")
        for value in self.alternative_ids:
            _text(value, "alternative_id")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "interpretation_id": self.interpretation_id,
            "source": self.source.to_dict(),
            "context_id": self.context_id,
            "claims": [claim.to_dict() for claim in self.claims],
            "alternative_ids": list(self.alternative_ids),
            "score": self.score,
            "model_version": self.model_version,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyScope:
    """Scoped stale/unknown region; it is not a global session flag."""

    uncertainty_id: str
    reason: str
    affected_claim_ids: tuple[str, ...] = ()
    affected_instance_ids: tuple[str, ...] = ()
    relation_id: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    source_id: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _text(self.uncertainty_id, "uncertainty_id")
        _text(self.reason, "reason")
        _optional_text(self.relation_id, "relation_id")
        _optional_text(self.valid_from, "valid_from")
        _optional_text(self.valid_to, "valid_to")
        _optional_text(self.source_id, "source_id")
        for item in (*self.affected_claim_ids, *self.affected_instance_ids):
            _text(item, "affected_id")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def affects_claim(self, claim: Claim) -> bool:
        if claim.claim_id in self.affected_claim_ids:
            return True
        if self.relation_id is not None and claim.relation_id != self.relation_id:
            return False
        values = {argument.value_id for argument in claim.arguments}
        if self.affected_instance_ids and values.intersection(
            self.affected_instance_ids
        ):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "uncertainty_id": self.uncertainty_id,
            "reason": self.reason,
            "affected_claim_ids": list(self.affected_claim_ids),
            "affected_instance_ids": list(self.affected_instance_ids),
            "relation_id": self.relation_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_id": self.source_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UncertaintyScope":
        expected = {
            "uncertainty_id",
            "reason",
            "affected_claim_ids",
            "affected_instance_ids",
            "relation_id",
            "valid_from",
            "valid_to",
            "source_id",
            "confidence",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid uncertainty scope")
        if (
            type(value["affected_claim_ids"]) is not list
            or type(value["affected_instance_ids"]) is not list
        ):
            raise ValueError("invalid uncertainty ID lists")
        return cls(
            uncertainty_id=value["uncertainty_id"],
            reason=value["reason"],
            affected_claim_ids=tuple(value["affected_claim_ids"]),
            affected_instance_ids=tuple(value["affected_instance_ids"]),
            relation_id=value["relation_id"],
            valid_from=value["valid_from"],
            valid_to=value["valid_to"],
            source_id=value["source_id"],
            confidence=value["confidence"],
        )


@dataclass(frozen=True, slots=True)
class LearningEpisode:
    episode_id: str
    group_id: str
    source_code: tuple[int, ...]
    target_code: tuple[int, ...]
    observed_target_bits: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.episode_id, "episode_id")
        _text(self.group_id, "group_id")
        for code in (self.source_code, self.target_code):
            if any(type(bit) is not int or bit < 0 for bit in code):
                raise ValueError("codes must contain non-negative integer bits")
        if self.observed_target_bits is not None and any(
            type(bit) is not int or bit < 0 for bit in self.observed_target_bits
        ):
            raise ValueError("observed_target_bits must be non-negative integers")


@dataclass(frozen=True, slots=True)
class AnswerReceipt:
    question_id: str
    answer_text: str
    claim_ids: tuple[str, ...]
    evidence_roots: tuple[str, ...]
    model_version: str
    state_version: str
    complete: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.question_id, "question_id")
        _text(self.answer_text, "answer_text", empty=True)
        _text(self.model_version, "model_version")
        _text(self.state_version, "state_version")
        _optional_text(self.reason, "reason")
        for item in (*self.claim_ids, *self.evidence_roots):
            _text(item, "receipt_id")

    @property
    def grounded(self) -> bool:
        return not self.claim_ids or bool(self.evidence_roots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "claim_ids": list(self.claim_ids),
            "evidence_roots": list(self.evidence_roots),
            "model_version": self.model_version,
            "state_version": self.state_version,
            "complete": self.complete,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerReceipt":
        expected = {
            "question_id",
            "answer_text",
            "claim_ids",
            "evidence_roots",
            "model_version",
            "state_version",
            "complete",
            "reason",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid answer receipt")
        if (
            type(value["claim_ids"]) is not list
            or type(value["evidence_roots"]) is not list
        ):
            raise ValueError("invalid answer receipt IDs")
        return cls(
            question_id=value["question_id"],
            answer_text=value["answer_text"],
            claim_ids=tuple(value["claim_ids"]),
            evidence_roots=tuple(value["evidence_roots"]),
            model_version=value["model_version"],
            state_version=value["state_version"],
            complete=value["complete"],
            reason=value["reason"],
        )

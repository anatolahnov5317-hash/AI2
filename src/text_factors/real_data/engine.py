"""Small integration layer for the experimental real-data primitives.

This is deliberately not a language parser. Learned/external interpretation
layers feed claims into this engine; the engine enforces evidence, uncertainty
and dependency contracts without a built-in domain ontology.
"""

from __future__ import annotations

from .contracts import AnswerReceipt, Claim, ClaimStatus, UncertaintyScope
from .dependencies import DependencyGraph
from .evidence import EvidenceLedger, EvidenceRoot
from .uncertainty import UncertaintyIndex


class RealDataEngine:
    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self.evidence = EvidenceLedger()
        self.uncertainty = UncertaintyIndex()
        self.dependencies = DependencyGraph()
        self._revision = 0

    @property
    def state_version(self) -> str:
        return f"real-data-state-{self._revision}"

    def _changed(self) -> None:
        self._revision += 1

    def register_evidence(self, root: EvidenceRoot) -> None:
        self.evidence.register_root(root)

    def add_claim(
        self, claim: Claim, *, parent_claim_ids: tuple[str, ...] = ()
    ) -> None:
        existing = self.claims.get(claim.claim_id)
        if existing is not None:
            if existing != claim:
                raise ValueError("claim ID already exists with different content")
            return
        missing = [value for value in parent_claim_ids if value not in self.claims]
        if missing:
            raise ValueError("unknown parent claim: " + ", ".join(sorted(missing)))
        self.dependencies.add_node(claim.claim_id)
        for parent in parent_claim_ids:
            self.dependencies.add_dependency(parent, claim.claim_id)
        if claim.evidence_roots:
            self.evidence.attach(claim.claim_id, claim.evidence_roots)
        if parent_claim_ids:
            self.evidence.derive(claim.claim_id, parent_claim_ids)
        self.claims[claim.claim_id] = claim
        self._changed()

    def mark_uncertainty(self, scope: UncertaintyScope) -> None:
        self.uncertainty.add(scope)
        self._changed()

    def clear_claim_uncertainty(self, claim_id: str) -> tuple[str, ...]:
        changed = self.uncertainty.resolve_claim(claim_id)
        if changed:
            self._changed()
        return changed

    def to_dict(self) -> dict[str, object]:
        return {
            "revision": self._revision,
            "claims": [
                self.claims[claim_id].to_dict()
                for claim_id in sorted(self.claims)
            ],
            "evidence": self.evidence.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "dependencies": self.dependencies.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RealDataEngine":
        expected = {
            "revision",
            "claims",
            "evidence",
            "uncertainty",
            "dependencies",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid real-data engine state")
        revision = value["revision"]
        raw_claims = value["claims"]
        if type(revision) is not int or revision < 0 or type(raw_claims) is not list:
            raise ValueError("invalid engine revision or claims")
        engine = cls()
        for raw in raw_claims:
            if type(raw) is not dict:
                raise ValueError("invalid persisted claim")
            claim = Claim.from_dict(raw)
            if claim.claim_id in engine.claims:
                raise ValueError("duplicate persisted claim ID")
            engine.claims[claim.claim_id] = claim
        raw_evidence = value["evidence"]
        raw_uncertainty = value["uncertainty"]
        raw_dependencies = value["dependencies"]
        if (
            type(raw_evidence) is not dict
            or type(raw_uncertainty) is not dict
            or type(raw_dependencies) is not dict
        ):
            raise ValueError("invalid persisted engine components")
        engine.evidence = EvidenceLedger.from_dict(raw_evidence)
        engine.uncertainty = UncertaintyIndex.from_dict(raw_uncertainty)
        engine.dependencies = DependencyGraph.from_dict(raw_dependencies)
        for claim in engine.claims.values():
            roots = set(engine.evidence.claim_roots(claim.claim_id))
            if not set(claim.evidence_roots) <= roots:
                raise ValueError("claim evidence does not match persisted ledger")
            engine.dependencies.add_node(claim.claim_id)
        engine._revision = revision
        return engine

    def receipt(
        self,
        *,
        question_id: str,
        answer_text: str,
        claim_ids: tuple[str, ...],
        model_version: str,
    ) -> AnswerReceipt:
        claims: list[Claim] = []
        for claim_id in claim_ids:
            try:
                claims.append(self.claims[claim_id])
            except KeyError as exc:
                raise ValueError("unknown claim in answer") from exc

        blocked = [
            claim.claim_id
            for claim in claims
            if claim.status
            in {
                ClaimStatus.PREDICTED,
                ClaimStatus.UNRESOLVED,
                ClaimStatus.DISPUTED,
                ClaimStatus.RETRACTED,
            }
            or self.uncertainty.is_uncertain(claim)
        ]
        roots = tuple(
            sorted(
                {
                    root
                    for claim in claims
                    for root in self.evidence.claim_roots(claim.claim_id)
                }
            )
        )
        if blocked:
            return AnswerReceipt(
                question_id,
                "",
                claim_ids,
                roots,
                model_version,
                self.state_version,
                False,
                reason="uncertain_or_unconfirmed:" + ",".join(sorted(blocked)),
            )
        if claims and not roots:
            return AnswerReceipt(
                question_id,
                "",
                claim_ids,
                (),
                model_version,
                self.state_version,
                False,
                reason="missing_evidence",
            )
        return AnswerReceipt(
            question_id,
            answer_text,
            claim_ids,
            roots,
            model_version,
            self.state_version,
            True,
        )

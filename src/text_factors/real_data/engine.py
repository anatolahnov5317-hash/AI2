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
        # Addressed corrections are append-only. A dependent of a corrected
        # claim is stale until it is rebuilt from the replacement observation.
        self._corrections: dict[str, str] = {}
        self._corrected_stale: set[str] = set()
        self._stale_claims: set[str] = set()
        self._obsolete_sources: set[tuple[str, int]] = set()
        self._revision = 0

    @property
    def state_version(self) -> str:
        return f"real-data-state-{self._revision}"

    def is_superseded(self, claim_id: str) -> bool:
        """Return whether an addressed correction replaced this historical claim."""
        return claim_id in self._corrections

    def _changed(self) -> None:
        self._revision += 1

    def register_evidence(self, root: EvidenceRoot) -> None:
        if (root.source_id, root.source_version) in self._obsolete_sources:
            raise ValueError("source revision has been invalidated")
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
        if set(parent_claim_ids) & self._stale_claims:
            raise ValueError("cannot derive from a corrected claim")
        # Reject invalid provenance before mutating the dependency graph.
        for root_id in claim.evidence_roots:
            root = self.evidence.root(root_id)
            if (root.source_id, root.source_version) in self._obsolete_sources:
                raise ValueError("claim relies on an invalidated source revision")
            if claim.source is not None and (
                root.source_id != claim.source.source_id
                or root.source_version != claim.source.source_version
                or root.access_scope != claim.source.access_scope
                or root.source != claim.source
            ):
                raise ValueError(
                    "claim evidence points to a different source revision or scope"
                )
        inherited_roots = {
            root_id
            for parent in parent_claim_ids
            for root_id in self.evidence.claim_roots(parent)
        }
        all_roots = inherited_roots | set(claim.evidence_roots)
        if (
            claim.source is not None
            and all_roots
            and not any(
                self.evidence.root(root_id).source == claim.source
                for root_id in all_roots
            )
        ):
            raise ValueError("claim source slice has no matching evidence root")
        self.dependencies.add_node(claim.claim_id)
        for parent in parent_claim_ids:
            self.dependencies.add_dependency(parent, claim.claim_id)
        if claim.evidence_roots:
            self.evidence.attach(claim.claim_id, claim.evidence_roots)
        if parent_claim_ids:
            self.evidence.derive(claim.claim_id, parent_claim_ids)
        self.claims[claim.claim_id] = claim
        self._changed()

    def correct_claim(self, target_id: str, replacement: Claim) -> tuple[str, ...]:
        """Replace one explicitly addressed claim and invalidate its dependents.

        The interpretation layer must supply the target ID. Similar spelling
        or equal relation/arguments never selects a correction target here.
        The replacement has its own registered source evidence and a new ID;
        existing claims and historical receipts retain their original state.
        """
        if target_id not in self.claims:
            raise ValueError("unknown correction target")
        if target_id in self._corrected_stale:
            raise ValueError("correction target was already corrected")
        if replacement.claim_id in self.claims or replacement.claim_id == target_id:
            raise ValueError("correction must use a new claim ID")
        if replacement.status not in {ClaimStatus.ASSERTED, ClaimStatus.OBSERVED}:
            raise ValueError("correction must be asserted or observed")
        if not replacement.evidence_roots:
            raise ValueError("correction requires direct source evidence")
        if replacement.source is None:
            raise ValueError("correction requires a pinned source slice")
        # add_claim verifies every root and source version before any mutation.
        self.add_claim(replacement)
        affected = self.dependencies.affected((target_id,))
        self._corrections[target_id] = replacement.claim_id
        self._corrected_stale.update(affected)
        self._stale_claims.update(affected)
        self._changed()
        return affected

    def mark_uncertainty(self, scope: UncertaintyScope) -> None:
        self.uncertainty.add(scope)
        self._changed()

    def invalidate_source(self, source_id: str, source_version: int) -> tuple[str, ...]:
        """Invalidate one reported source revision and every dependent claim.

        The custodian must call this after a revocation or superseding import;
        this engine has no connection to the archive's version change stream.
        """
        if (
            type(source_id) is not str
            or not source_id
            or type(source_version) is not int
            or source_version < 1
        ):
            raise ValueError("invalid source revision")
        roots = self.evidence.to_dict()["roots"]
        if not any(
            root["source_id"] == source_id and root["source_version"] == source_version
            for root in roots
        ):
            raise ValueError("unknown source revision")
        impacted = {
            claim_id
            for claim_id in self.claims
            if any(
                (
                    self.evidence.root(root_id).source_id,
                    self.evidence.root(root_id).source_version,
                )
                == (source_id, source_version)
                for root_id in self.evidence.claim_roots(claim_id)
            )
        }
        affected = {
            dependent
            for claim_id in impacted
            for dependent in self.dependencies.affected((claim_id,))
        }
        key = (source_id, source_version)
        if key not in self._obsolete_sources:
            self._obsolete_sources.add(key)
            self._stale_claims.update(affected)
            self._changed()
        return tuple(sorted(affected))

    def clear_claim_uncertainty(self, claim_id: str) -> tuple[str, ...]:
        changed = self.uncertainty.resolve_claim(claim_id)
        if changed:
            self._changed()
        return changed

    def to_dict(self) -> dict[str, object]:
        return {
            "revision": self._revision,
            "claims": [
                self.claims[claim_id].to_dict() for claim_id in sorted(self.claims)
            ],
            "evidence": self.evidence.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "corrections": dict(sorted(self._corrections.items())),
            "obsolete_sources": [
                {"source_id": source_id, "source_version": version}
                for source_id, version in sorted(self._obsolete_sources)
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> RealDataEngine:
        expected = {
            "revision",
            "claims",
            "evidence",
            "uncertainty",
            "dependencies",
        }
        # Legacy v1 snapshots had no correction log; their other fields and
        # fingerprint remain unchanged and can still be restored.
        if (
            type(value) is not dict
            or not expected <= set(value)
            or set(value) - expected - {"corrections", "obsolete_sources"}
        ):
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
        corrections = value.get("corrections", {})
        if type(corrections) is not dict:
            raise ValueError("invalid corrections log")
        for claim in engine.claims.values():
            roots = set(engine.evidence.claim_roots(claim.claim_id))
            if not set(claim.evidence_roots) <= roots:
                raise ValueError("claim evidence does not match persisted ledger")
            if (
                claim.source is not None
                and roots
                and any(
                    engine.evidence.root(root_id).source is not None
                    for root_id in roots
                )
                and not any(
                    engine.evidence.root(root_id).source == claim.source
                    for root_id in roots
                )
            ):
                raise ValueError("persisted claim has no matching evidence root")
            for root_id in claim.evidence_roots:
                root = engine.evidence.root(root_id)
                if claim.source is not None and (
                    root.source_id != claim.source.source_id
                    or root.source_version != claim.source.source_version
                    or root.access_scope not in (None, claim.source.access_scope)
                    or (root.source is not None and root.source != claim.source)
                ):
                    raise ValueError("persisted claim source and evidence differ")
            engine.dependencies.add_node(claim.claim_id)
        if len(set(corrections.values())) != len(corrections):
            raise ValueError("corrections must have distinct replacement claims")
        for target_id, replacement_id in corrections.items():
            if (
                type(target_id) is not str
                or type(replacement_id) is not str
                or target_id not in engine.claims
                or replacement_id not in engine.claims
                or target_id == replacement_id
                or not engine.claims[replacement_id].evidence_roots
                or engine.claims[replacement_id].source is None
                or engine.claims[replacement_id].status
                not in {ClaimStatus.ASSERTED, ClaimStatus.OBSERVED}
            ):
                raise ValueError("invalid correction target or replacement")
        # Validate in correction order rather than lexicographic claim-ID
        # order. Chains are valid, cycles and revising stale dependents are not.
        replacement_ids = set(corrections.values())
        pending = set(corrections)
        roots = sorted(pending - replacement_ids)
        while roots:
            target_id = roots.pop(0)
            if target_id in engine._corrected_stale:
                raise ValueError("correction target was already corrected")
            affected = engine.dependencies.affected((target_id,))
            engine._corrected_stale.update(affected)
            engine._stale_claims.update(affected)
            replacement_id = corrections[target_id]
            engine._corrections[target_id] = replacement_id
            pending.remove(target_id)
            if replacement_id in pending:
                roots.append(replacement_id)
        if pending:
            raise ValueError("cyclic correction chain")
        obsolete = value.get("obsolete_sources", [])
        if type(obsolete) is not list:
            raise ValueError("invalid obsolete source revisions")
        for entry in obsolete:
            if type(entry) is not dict or set(entry) != {"source_id", "source_version"}:
                raise ValueError("invalid obsolete source revision")
            engine.invalidate_source(entry["source_id"], entry["source_version"])
        if len(engine._obsolete_sources) != len(obsolete):
            raise ValueError("duplicate obsolete source revision")
        engine._revision = revision
        return engine

    def receipt(
        self,
        *,
        question_id: str,
        answer_text: str,
        claim_ids: tuple[str, ...],
        model_version: str,
        allowed_scopes: tuple[str, ...] = ("default",),
    ) -> AnswerReceipt:
        if (
            type(allowed_scopes) is not tuple
            or not allowed_scopes
            or any(type(scope) is not str or not scope for scope in allowed_scopes)
        ):
            raise ValueError("allowed_scopes must be a nonempty tuple of scopes")
        claims: list[Claim] = []
        for claim_id in claim_ids:
            try:
                claims.append(self.claims[claim_id])
            except KeyError as exc:
                raise ValueError("unknown claim in answer") from exc
        if not claims and answer_text:
            raise ValueError("answer has no grounded claims")

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
            or claim.claim_id in self._stale_claims
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
                (),
                model_version,
                self.state_version,
                False,
                reason="uncertain_or_unconfirmed:" + ",".join(sorted(blocked)),
            )
        missing = [
            claim.claim_id
            for claim in claims
            if not self.evidence.claim_roots(claim.claim_id)
        ]
        if missing:
            return AnswerReceipt(
                question_id,
                "",
                claim_ids,
                (),
                model_version,
                self.state_version,
                False,
                reason="missing_evidence:" + ",".join(sorted(missing)),
            )
        direct_sources: dict[str, list[Claim]] = {}
        for known in self.claims.values():
            for root_id in known.evidence_roots:
                if root_id in roots:
                    direct_sources.setdefault(root_id, []).append(known)
        unbound = {
            claim.claim_id
            for claim in claims
            if claim.source is None
            or not any(
                self.evidence.root(root_id).source == claim.source
                for root_id in self.evidence.claim_roots(claim.claim_id)
            )
        }
        denied = {
            claim.claim_id
            for claim in claims
            if claim.source is not None
            and claim.source.access_scope not in allowed_scopes
        }
        for root_id in roots:
            source_root = self.evidence.root(root_id)
            if source_root.access_scope is None or source_root.source is None:
                unbound.update(claim_ids)
            elif source_root.access_scope not in allowed_scopes:
                denied.update(claim_ids)
            if root_id not in direct_sources:
                unbound.update(claim_ids)
            for known in direct_sources.get(root_id, ()):
                if known.source is None:
                    unbound.update(claim_ids)
                elif known.source.access_scope not in allowed_scopes:
                    denied.update(claim_ids)
        if unbound:
            return AnswerReceipt(
                question_id,
                "",
                claim_ids,
                (),
                model_version,
                self.state_version,
                False,
                reason="source_binding_missing:" + ",".join(sorted(unbound)),
            )
        if denied:
            return AnswerReceipt(
                question_id,
                "",
                claim_ids,
                (),
                model_version,
                self.state_version,
                False,
                reason="source_access_denied:" + ",".join(sorted(denied)),
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

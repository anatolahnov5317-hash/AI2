"""A bounded raw-text dialogue path with explicit, auditable human review.

The language models only propose event and question interpretations. Source
claims enter memory through reviewed decisions, while identity resolution for
questions draws only from accessible, current, reviewed source excerpts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..observations import ObservationArchive
from .archive_bridge import evidence_root
from .contracts import Claim, ClaimStatus
from .engine import RealDataEngine
from .hypothesis_bridge import (
    ReviewCandidate,
    ReviewDecision,
    propose_review_candidates,
    register_reviewed_candidate,
)
from .open_semantics import SemanticGraph
from .question_language import QuestionLanguageModel, ReviewedIdentity
from .raw_language import RawSemanticModel
from .research_checkpoint import load_research_checkpoint, save_research_checkpoint
from .storage import OperationalStore, StalePublication
from .verified_dialogue import (
    DialogueAnswer,
    DialogueQuestion,
    GroundedDialogue,
    QueryVocabulary,
)


@dataclass(frozen=True, slots=True)
class ImportedMessage:
    source_id: str
    source_version: int
    graph: SemanticGraph
    candidates: tuple[ReviewCandidate, ...]


class ConnectedDialogue:
    """Connect raw messages, review, and access-scoped quoted responses.

    A reviewer must decide roles, identities, modality and any parser residual;
    a correction additionally needs the explicit target claim ID. The store
    remains a separate live authority, including after restoring engine state.
    """

    def __init__(
        self,
        archive: ObservationArchive,
        engine: RealDataEngine,
        store: OperationalStore,
        *,
        raw_model: RawSemanticModel,
        question_model: QuestionLanguageModel,
        vocabulary: QueryVocabulary,
        model_version: str,
        checkpoint_dir: str | Path,
    ) -> None:
        self.archive = archive
        self.engine = engine
        self.store = store
        self.raw_model = raw_model
        self.question_model = question_model
        self.checkpoint_dir = Path(checkpoint_dir)
        active_version = store.active_state_version()
        if active_version is not None and active_version != engine.state_version:
            raise ValueError("engine state does not match live access authority")
        self.dialogue = GroundedDialogue(
            archive, engine, vocabulary, model_version=model_version
        )

    def _checkpoint(self) -> None:
        """Persist engine and both language models before publishing a version."""
        save_research_checkpoint(
            self.checkpoint_dir,
            self.engine,
            self.raw_model,
            self.question_model,
            model_version=self.dialogue.model_version,
            store=self.store,
        )

    @classmethod
    def restore(
        cls,
        archive: ObservationArchive,
        store: OperationalStore,
        *,
        checkpoint_dir: str | Path,
        model_version: str,
        vocabulary: QueryVocabulary,
    ) -> ConnectedDialogue:
        """Resume only a complete, active checkpoint with live tombstones."""
        engine, raw_model, question_model = load_research_checkpoint(
            checkpoint_dir, expected_model_version=model_version, store=store
        )
        return cls(
            archive,
            engine,
            store,
            raw_model=raw_model,
            question_model=question_model,
            vocabulary=vocabulary,
            model_version=model_version,
            checkpoint_dir=checkpoint_dir,
        )

    def import_message(
        self,
        text: str,
        *,
        namespace: str,
        external_key: str,
        group_id: str,
        access_scope: str = "default",
    ) -> ImportedMessage:
        """Import a new immutable source revision and expose only proposals."""
        if type(access_scope) is not str or not access_scope:
            raise ValueError("access scope must be nonempty")
        # Reject malformed/oversized text before touching the append-only
        # archive or revoking a prior source version.
        self.raw_model.parse(text, source_id="import-preflight", source_version=1)
        record = self.archive.import_text(
            text,
            namespace=namespace,
            external_key=external_key,
            group_id=group_id,
            metadata={"access_scope": access_scope},
        )
        if record.version > 1:
            old_version = record.version - 1
            affected = tuple(
                claim.claim_id
                for claim in self.engine.claims.values()
                if claim.source is not None
                and claim.source.source_id == record.source_id
                and claim.source.source_version == old_version
            )
            # Revoke the previous source revision first, including in the
            # event of a crash between import and the engine checkpoint.
            self.store.revoke_source(record.source_id, old_version)
            if affected and not self.engine.is_source_obsolete(
                record.source_id, old_version
            ):
                all_affected = self.engine.dependencies.affected(affected)
                epoch = self.store.start_revision(
                    f"import:{record.source_id}:v{record.version}",
                    all_affected,
                    self.engine.state_version,
                )
                self.engine.invalidate_source(record.source_id, old_version)
                self._checkpoint()
                self.store.finish_revision(
                    f"import:{record.source_id}:v{record.version}",
                    self.engine.state_version,
                    expected_epoch=epoch,
                )
        graph = self.raw_model.parse(
            text, source_id=record.source_id, source_version=record.version
        )
        return ImportedMessage(
            record.source_id,
            record.version,
            graph,
            propose_review_candidates(graph, self.archive),
        )

    def review(
        self,
        candidate: ReviewCandidate,
        decision: ReviewDecision,
        *,
        correction_target_id: str | None = None,
    ) -> Claim:
        """Register an exact reviewed proposal, optionally correcting one ID."""
        root = evidence_root(
            self.archive, candidate.source, root_id=f"root:{decision.claim_id}"
        )
        # Validate the complete review on a private copy, so an ordinary
        # rejected decision never leaves a pending revision job behind.
        preview = RealDataEngine.from_dict(self.engine.to_dict())
        register_reviewed_candidate(
            candidate,
            decision,
            root,
            archive=self.archive,
            engine=preview,
            correction_target_id=correction_target_id,
        )
        affected = (decision.claim_id,)
        if correction_target_id is not None:
            affected += self.engine.dependencies.affected((correction_target_id,))
        revision_id = f"review:{decision.claim_id}"
        epoch = self.store.start_revision(
            revision_id, tuple(sorted(set(affected))), self.engine.state_version
        )
        claim = register_reviewed_candidate(
            candidate,
            decision,
            root,
            archive=self.archive,
            engine=self.engine,
            correction_target_id=correction_target_id,
        )
        self._checkpoint()
        self.store.finish_revision(
            revision_id, self.engine.state_version, expected_epoch=epoch
        )
        return claim

    def _accessible_identities(
        self,
        *,
        principal_id: str,
        question_id: str,
        question_text: str,
        allowed_scopes: tuple[str, ...],
    ) -> tuple[ReviewedIdentity, ...] | None:
        # This read creates no task or publication. It limits even the
        # vocabulary seen by the question interpreter to accessible evidence.
        catalog_id = "catalog:" + hashlib.sha256(question_id.encode()).hexdigest()
        catalog_fingerprint = hashlib.sha256(
            (question_text + repr(allowed_scopes)).encode("utf-8")
        ).hexdigest()
        try:
            access = self.store.retrieval_access(
                principal_id,
                allowed_scopes,
                request_id=catalog_id,
                request_fingerprint=catalog_fingerprint,
            )
        except StalePublication:
            return None
        if not access.allowed_scopes:
            return None
        identities: set[tuple[str, str, str]] = set()
        for claim in self.engine.claims.values():
            if (
                claim.reviewer_id is None
                or claim.status not in {ClaimStatus.ASSERTED, ClaimStatus.OBSERVED}
                or claim.source is None
                or self.engine.is_superseded(claim.claim_id)
                or not access.permits_source(claim.source)
                or claim.claim_id in access.pending_claim_ids
            ):
                continue
            roots = self.engine.evidence.claim_roots(claim.claim_id)
            root_sources = tuple(
                self.engine.evidence.root(root_id).source for root_id in roots
            )
            if not root_sources or any(
                source is None or not access.permits_source(source)
                for source in root_sources
            ):
                continue
            if not self.engine.receipt(
                question_id=question_id,
                answer_text="",
                claim_ids=(claim.claim_id,),
                model_version=self.dialogue.model_version,
                allowed_scopes=access.allowed_scopes,
            ).complete:
                continue
            text = self.archive.read_span(
                claim.source.source_id,
                claim.source.source_version,
                0,
                self.archive.get_source(
                    claim.source.source_id, claim.source.source_version
                ).char_count,
            )
            mentions = self.raw_model.infer_mentions(
                text,
                source_id=claim.source.source_id,
                source_version=claim.source.source_version,
            )
            mention_by_id = {mention.mention_id: mention for mention in mentions}
            for role in claim.arguments:
                if role.mention_id is None:
                    continue
                mention = mention_by_id.get(role.mention_id)
                if mention is None:
                    continue
                surface = text[mention.span.start : mention.span.end]
                identities.add((surface, mention.kind, role.value_id))
        # Revocation during construction forces a new request. Dialogue.ask
        # independently checks its own access snapshot again at publication.
        if access.revision_epoch != self.store.revision_epoch():
            return None
        return tuple(ReviewedIdentity(*row) for row in sorted(identities))

    def ask(
        self,
        question_id: str,
        text: str,
        *,
        principal_id: str,
        allowed_scopes: tuple[str, ...] = ("default",),
    ) -> DialogueAnswer:
        identities = self._accessible_identities(
            principal_id=principal_id,
            question_id=question_id,
            question_text=text,
            allowed_scopes=allowed_scopes,
        )
        if identities is None:
            return DialogueAnswer(
                "blocked", "", clarification="Публикация пока недоступна."
            )
        hypothesis = self.question_model.parse(
            text, raw_model=self.raw_model, reviewed_identities=identities
        )
        return self.dialogue.ask_interpreted(
            DialogueQuestion(question_id, text),
            hypothesis,
            principal_id=principal_id,
            store=self.store,
            allowed_scopes=allowed_scopes,
        )

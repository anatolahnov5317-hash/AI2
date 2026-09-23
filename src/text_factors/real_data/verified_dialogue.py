"""Conservative quotation dialogue over reviewed claims and archived spans.

Query words only resolve IDs from a supplied, reviewed vocabulary. There is
no text-to-fact model here: callers import documents and register their own
reviewed claims before asking. A quote is released only by the operational
store, which rechecks current permissions and source revisions at publication.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..observations import ObservationArchive
from .contracts import AnswerReceipt, Claim, ClaimStatus, SourceSlice
from .engine import RealDataEngine

if TYPE_CHECKING:
    from .question_language import QueryHypothesis
    from .storage import OperationalStore, RetrievalAccess


def _id(value: str, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    return value


def _alias_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", value.casefold(), re.UNICODE))


@dataclass(frozen=True, slots=True)
class QueryVocabulary:
    """Reviewed aliases, not discovered language or hard-coded domain rules.

    `subject_roles` selects the role of the queried instance. An empty tuple
    matches any role and can lead to a clarification when candidates conflict.
    """

    entity_aliases: tuple[tuple[str, str], ...]
    relation_aliases: tuple[tuple[str, str], ...]
    subject_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for collection in (self.entity_aliases, self.relation_aliases):
            if type(collection) is not tuple or not collection:
                raise ValueError("query vocabulary requires reviewed aliases")
            for item in collection:
                if type(item) is not tuple or len(item) != 2:
                    raise ValueError("invalid query alias")
                alias, target = item
                _id(alias, "alias")
                _id(target, "alias target")
                if not _alias_tokens(alias):
                    raise ValueError("query alias must contain a word")
        if type(self.subject_roles) is not tuple:
            raise ValueError("subject_roles must be a tuple")
        for role in self.subject_roles:
            _id(role, "subject role")


@dataclass(frozen=True, slots=True)
class DialogueQuestion:
    question_id: str
    text: str
    subject_id: str | None = None
    relation_id: str | None = None

    def __post_init__(self) -> None:
        _id(self.question_id, "question ID")
        _id(self.text, "question text")
        if self.subject_id is not None:
            _id(self.subject_id, "subject ID")
        if self.relation_id is not None:
            _id(self.relation_id, "relation ID")


@dataclass(frozen=True, slots=True)
class VerifiedQuote:
    claim_id: str
    text: str
    source: SourceSlice
    evidence_roots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DialogueAnswer:
    status: str
    state_version: str
    receipt: AnswerReceipt | None = None
    quotes: tuple[VerifiedQuote, ...] = ()
    clarification: str | None = None


@dataclass(frozen=True, slots=True)
class _Prepared:
    receipt: AnswerReceipt
    quotes: tuple[VerifiedQuote, ...]


class GroundedDialogue:
    """Retrieve cited text; answer only after a live publication transaction."""

    def __init__(
        self,
        archive: ObservationArchive,
        engine: RealDataEngine,
        vocabulary: QueryVocabulary,
        *,
        model_version: str,
        max_quotes: int = 3,
    ) -> None:
        _id(model_version, "model version")
        if type(max_quotes) is not int or not 1 <= max_quotes <= 10:
            raise ValueError("max_quotes must be between 1 and 10")
        self.archive = archive
        self.engine = engine
        self.vocabulary = vocabulary
        self.model_version = model_version
        self.max_quotes = max_quotes

    @staticmethod
    def _resolve(text: str, aliases: tuple[tuple[str, str], ...]) -> set[str]:
        words = _alias_tokens(text)
        matches: set[str] = set()
        for alias, target in aliases:
            phrase = _alias_tokens(alias)
            width = len(phrase)
            if any(
                words[offset : offset + width] == phrase
                for offset in range(len(words) - width + 1)
            ):
                matches.add(target)
        return matches

    def _has_unreviewed_words(self, text: str) -> bool:
        """Do not answer a partly understood question as if it were complete."""
        words = _alias_tokens(text)
        covered: set[int] = set()
        for alias, _ in (
            *self.vocabulary.entity_aliases,
            *self.vocabulary.relation_aliases,
        ):
            phrase = _alias_tokens(alias)
            width = len(phrase)
            for offset in range(len(words) - width + 1):
                if words[offset : offset + width] == phrase:
                    covered.update(range(offset, offset + width))
        return len(covered) != len(words)

    def _no_answer(
        self, status: str, clarification: str | None = None
    ) -> DialogueAnswer:
        # A global engine version also changes when unrelated private claims
        # change; no-answer responses have no citeable state to expose.
        return DialogueAnswer(status, "", clarification=clarification)

    def _read_quote(self, claim: Claim) -> VerifiedQuote:
        if claim.source is None:
            raise ValueError("claim has no pinned source")
        source = claim.source
        record = self.archive.get_source(source.source_id, source.source_version)
        if record.metadata.get("access_scope", "default") != source.access_scope:
            raise ValueError("source access scope has changed")
        text = self.archive.read_span(
            source.source_id, source.source_version, source.start, source.end
        )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != source.sha256:
            raise ValueError("source excerpt checksum mismatch")
        roots = self.engine.evidence.claim_roots(claim.claim_id)
        if not claim.evidence_roots or not any(
            self.engine.evidence.root(root_id).source == source
            for root_id in claim.evidence_roots
        ):
            raise ValueError("quote has no direct pinned evidence")
        return VerifiedQuote(claim.claim_id, text, source, roots)

    def _prepare(
        self,
        question: DialogueQuestion,
        *,
        allowed_scopes: tuple[str, ...],
        pending_claim_ids: frozenset[str],
        access: RetrievalAccess | None = None,
        interpreted: QueryHypothesis | None = None,
    ) -> _Prepared | DialogueAnswer:
        if not allowed_scopes or any(
            type(item) is not str or not item for item in allowed_scopes
        ):
            raise ValueError("allowed_scopes must contain at least one scope")
        if not isinstance(pending_claim_ids, frozenset):
            raise ValueError("pending_claim_ids must be a frozenset")
        if interpreted is None and self._has_unreviewed_words(question.text):
            return self._no_answer("clarify", "Уточните объект и тип вопроса.")
        subjects = (
            {interpreted.subject_id}
            if interpreted is not None
            else {question.subject_id}
            if question.subject_id is not None
            else self._resolve(question.text, self.vocabulary.entity_aliases)
        )
        relations = (
            {interpreted.relation_id}
            if interpreted is not None
            else {question.relation_id}
            if question.relation_id is not None
            else self._resolve(question.text, self.vocabulary.relation_aliases)
        )
        if len(subjects) != 1 or len(relations) != 1:
            return self._no_answer("clarify", "Уточните объект и тип вопроса.")
        subject = next(iter(subjects))
        relation = next(iter(relations))
        # Previously corrected statements are historical, never rival current
        # replacements. Do not inspect inaccessible candidates in response
        # selection or clarification, which would reveal their existence.
        candidates = [
            claim
            for claim in self.engine.claims.values()
            if not self.engine.is_superseded(claim.claim_id)
            and claim.relation_id == relation
            and claim.source is not None
            and claim.source.access_scope in allowed_scopes
            and (
                access is None
                or (
                    access.permits_source(claim.source)
                    and all(
                        (root_source := self.engine.evidence.root(root_id).source)
                        is not None
                        and access.permits_source(root_source)
                        for root_id in self.engine.evidence.claim_roots(claim.claim_id)
                    )
                )
            )
            and any(
                argument.value_id == subject
                and (
                    not self.vocabulary.subject_roles
                    or argument.role in self.vocabulary.subject_roles
                )
                for argument in claim.arguments
            )
            and (
                interpreted is None
                or any(arg.role == interpreted.asked_role for arg in claim.arguments)
            )
        ]
        if not candidates:
            return self._no_answer("clarify", "Уточните объект и тип вопроса.")
        if any(claim.claim_id in pending_claim_ids for claim in candidates):
            return self._no_answer(
                "revision_pending", "Пересмотр фактов ещё не завершён."
            )
        confirmed: list[Claim] = []
        for claim in candidates:
            if claim.status not in (ClaimStatus.ASSERTED, ClaimStatus.OBSERVED):
                return self._no_answer("blocked", "Сведения ещё не подтверждены.")
            preview = self.engine.receipt(
                question_id=question.question_id,
                answer_text="",
                claim_ids=(claim.claim_id,),
                model_version=self.model_version,
                allowed_scopes=allowed_scopes,
            )
            if not preview.complete:
                return self._no_answer("blocked", "Сведения ещё не подтверждены.")
            confirmed.append(claim)
        signatures = {
            (
                tuple(
                    sorted(
                        (arg.role, arg.value_id, arg.value_type)
                        for arg in claim.arguments
                    )
                ),
                claim.valid_from,
                claim.valid_to,
                claim.polarity,
                claim.modality,
            )
            for claim in confirmed
        }
        if len(signatures) != 1:
            return self._no_answer("clarify", "Уточните период или событие.")
        quotes: list[VerifiedQuote] = []
        try:
            for claim in sorted(
                confirmed,
                key=lambda row: (
                    row.source.source_id if row.source else "",
                    row.source.source_version if row.source else 0,
                    row.source.start if row.source else 0,
                    row.claim_id,
                ),
            )[: self.max_quotes]:
                quotes.append(self._read_quote(claim))
        except ValueError:
            return self._no_answer("blocked", "Не удалось проверить источник.")
        answer_text = "\n".join(quote.text for quote in quotes)
        if len(answer_text) > 4096:
            return self._no_answer("blocked", "Выдержка превышает допустимый размер.")
        receipt = self.engine.receipt(
            question_id=question.question_id,
            answer_text=answer_text,
            claim_ids=tuple(quote.claim_id for quote in quotes),
            model_version=self.model_version,
            allowed_scopes=allowed_scopes,
        )
        if not receipt.complete:
            return self._no_answer("blocked", "Сведения ещё не подтверждены.")
        return _Prepared(receipt, tuple(quotes))

    def ask(
        self,
        question: DialogueQuestion,
        *,
        principal_id: str,
        store: OperationalStore | None,
        allowed_scopes: tuple[str, ...] = ("default",),
        pending_claim_ids: frozenset[str] = frozenset(),
    ) -> DialogueAnswer:
        """Publish checked quotations; a missing store cannot expose drafts.

        `allowed_scopes` only narrows retrieval. Real grants and revocations are
        checked again by `store.publish_answer` in its publication transaction.
        """
        return self._ask(
            question,
            principal_id=principal_id,
            store=store,
            allowed_scopes=allowed_scopes,
            pending_claim_ids=pending_claim_ids,
            interpreted=None,
        )

    def ask_interpreted(
        self,
        question: DialogueQuestion,
        hypothesis: QueryHypothesis,
        *,
        principal_id: str,
        store: OperationalStore | None,
        allowed_scopes: tuple[str, ...] = ("default",),
        pending_claim_ids: frozenset[str] = frozenset(),
    ) -> DialogueAnswer:
        """Use a fully explained typed question interpretation.

        A raw question without this plan still goes through the reviewed alias
        gate. The interpretation must be produced from the exact question text;
        downstream authority always belongs to the SQLite publication store.
        """
        from .open_semantics import Span
        from .question_language import QueryHypothesis

        if (
            type(hypothesis) is not QueryHypothesis
            or hypothesis.fully_covered is not True
            or not hypothesis.resolved
            or type(hypothesis.subject_id) is not str
            or not hypothesis.subject_id
            or type(hypothesis.relation_id) is not str
            or not hypothesis.relation_id
            or type(hypothesis.asked_role) is not str
            or not hypothesis.asked_role
            or type(hypothesis.subject_span) is not Span
            or hypothesis.subject_span.end > len(question.text)
            or not _alias_tokens(
                question.text[
                    hypothesis.subject_span.start : hypothesis.subject_span.end
                ]
            )
            or hypothesis.text_sha256
            != hashlib.sha256(question.text.encode("utf-8")).hexdigest()
            or question.subject_id not in (None, hypothesis.subject_id)
            or question.relation_id not in (None, hypothesis.relation_id)
            or any(
                token in {"не", "ни", "никогда", "нет", "без"}
                for token in _alias_tokens(question.text)
            )
        ):
            return self._no_answer("clarify", "Уточните объект и тип вопроса.")
        return self._ask(
            question,
            principal_id=principal_id,
            store=store,
            allowed_scopes=allowed_scopes,
            pending_claim_ids=pending_claim_ids,
            interpreted=hypothesis,
        )

    def _ask(
        self,
        question: DialogueQuestion,
        *,
        principal_id: str,
        store: OperationalStore | None,
        allowed_scopes: tuple[str, ...],
        pending_claim_ids: frozenset[str],
        interpreted: QueryHypothesis | None,
    ) -> DialogueAnswer:
        _id(principal_id, "principal ID")
        if store is None:
            return self._no_answer("blocked", "Публикация пока недоступна.")
        if not isinstance(pending_claim_ids, frozenset):
            raise ValueError("pending_claim_ids must be a frozenset")
        # A request ID also scopes the read-only access view. Authorization is
        # evaluated before selection: inaccessible claims cannot affect either
        # ambiguity or the status of an otherwise unanswered question.
        from .storage import AccessDenied, PublicationQuote, StalePublication

        fingerprint_fields: tuple[object, ...] = (
            question.question_id,
            question.text,
            question.subject_id,
            question.relation_id,
            self.model_version,
            allowed_scopes,
        )
        if interpreted is not None:
            fingerprint_fields += (
                "interpreted-v1",
                interpreted.subject_id,
                interpreted.relation_id,
                interpreted.asked_role,
                interpreted.text_sha256,
                interpreted.subject_span.start if interpreted.subject_span else None,
                interpreted.subject_span.end if interpreted.subject_span else None,
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_fields,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            access = store.retrieval_access(
                principal_id,
                allowed_scopes,
                request_id=question.question_id,
                request_fingerprint=fingerprint,
            )
        except StalePublication:
            return self._no_answer("blocked", "Публикация пока недоступна.")
        if not access.allowed_scopes or access.task_status in {"revoked", "complete"}:
            return self._no_answer("blocked", "Публикация пока недоступна.")
        # The epoch is captured with grants and revocations. A change after
        # drafting is checked again, including for non-publication responses.
        expected_revision_epoch = access.revision_epoch
        prepared = self._prepare(
            question,
            allowed_scopes=access.allowed_scopes,
            pending_claim_ids=pending_claim_ids | access.pending_claim_ids,
            access=access,
            interpreted=interpreted,
        )
        if isinstance(prepared, DialogueAnswer):
            if (
                store.revision_epoch() != expected_revision_epoch
                or access.task_status == "published"
            ):
                return self._no_answer("blocked", "Публикация пока недоступна.")
            return prepared
        publication_quotes = tuple(
            PublicationQuote(quote.source, quote.claim_id, quote.text)
            for quote in prepared.quotes
        )
        try:
            published = store.publish_answer(
                request_id=question.question_id,
                request_fingerprint=fingerprint,
                principal_id=principal_id,
                engine=self.engine,
                archive=self.archive,
                receipt=prepared.receipt,
                quotes=publication_quotes,
                expected_revision_epoch=expected_revision_epoch,
            )
        except (AccessDenied, StalePublication):
            return self._no_answer("blocked", "Источник недоступен или изменён.")
        # The exact published values are returned from the access-controlled
        # transaction; do not return a prepared quote in their place.
        final_quotes = tuple(
            VerifiedQuote(
                quote.claim_id,
                quote.text,
                quote.source,
                self.engine.evidence.claim_roots(quote.claim_id),
            )
            for quote in published.quotes
        )
        return DialogueAnswer(
            "answered",
            published.receipt.state_version,
            receipt=published.receipt,
            quotes=final_quotes,
        )

"""Access checks must precede claim selection as well as publication."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from text_factors.observations import ObservationArchive
from text_factors.real_data.archive_bridge import evidence_root, source_slice
from text_factors.real_data.contracts import (
    Claim,
    ClaimModality,
    ClaimPolarity,
    ClaimStatus,
    RoleValue,
)
from text_factors.real_data.engine import RealDataEngine
from text_factors.real_data.open_semantics import Span
from text_factors.real_data.question_language import QueryHypothesis
from text_factors.real_data.storage import OperationalStore
from text_factors.real_data.verified_dialogue import (
    DialogueQuestion,
    GroundedDialogue,
    QueryVocabulary,
)


class PreselectionAccessGateTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.archive = ObservationArchive(root / "archive.db", create=True)
        self.addCleanup(self.archive.close)
        self.engine = RealDataEngine()
        self.store = OperationalStore(root / "access.db")
        self.dialogue = GroundedDialogue(
            self.archive,
            self.engine,
            QueryVocabulary(
                entity_aliases=(("ключ", "key"), ("фонарь", "lamp")),
                relation_aliases=(("где", "location"),),
                subject_roles=("item",),
            ),
            model_version="access-test",
        )

    def add(
        self,
        claim_id: str,
        text: str,
        *,
        scope: str,
        place: str,
        polarity: ClaimPolarity = ClaimPolarity.UNKNOWN,
        modality: ClaimModality = ClaimModality.UNKNOWN,
    ) -> Claim:
        record = self.archive.import_text(
            text,
            namespace="development",
            external_key=claim_id,
            group_id=claim_id,
            metadata={"access_scope": scope},
        )
        source = source_slice(self.archive, record.source_id, 1, 0, record.char_count)
        root_id = "root-" + claim_id
        self.engine.register_evidence(
            evidence_root(self.archive, source, root_id=root_id)
        )
        claim = Claim(
            claim_id,
            "location",
            (RoleValue("item", "key"), RoleValue("place", place)),
            ClaimStatus.ASSERTED,
            source=source,
            evidence_roots=(root_id,),
            polarity=polarity,
            modality=modality,
        )
        self.engine.add_claim(claim)
        return claim

    def ask(self, request_id: str, text: str = "Где ключ?", **kwargs):
        return self.dialogue.ask(
            DialogueQuestion(request_id, text),
            principal_id="reader",
            store=self.store,
            **kwargs,
        )

    def test_no_grant_cannot_distinguish_existing_and_missing_facts(self) -> None:
        self.add("confidential", "Ключ в сейфе.", scope="default", place="сейф")
        existing = self.ask("existing")
        missing = self.ask("missing", "Где фонарь?")
        self.assertEqual(existing.status, "blocked")
        self.assertEqual(missing.status, existing.status)
        self.assertEqual(missing.clarification, existing.clarification)
        self.assertEqual(existing.state_version, "")
        self.assertEqual(missing.state_version, "")
        self.assertFalse(existing.quotes)
        self.assertIsNone(self.store.task_status("existing"))
        self.assertIsNone(self.store.task_status("missing"))

        self.store.grant_scope("reader", "other")
        hidden = self.ask("invisible", allowed_scopes=("default", "other"))
        unknown = self.ask(
            "unknown", "Где фонарь?", allowed_scopes=("default", "other")
        )
        self.assertEqual(hidden.status, "clarify")
        self.assertEqual(unknown.status, hidden.status)
        self.assertEqual(hidden.clarification, unknown.clarification)
        self.assertEqual(hidden.state_version, "")
        self.assertIsNone(self.store.task_status("invisible"))

    def test_revoked_conflict_and_inaccessible_pending_are_invisible(self) -> None:
        active = self.add("active", "Ключ в ящике.", scope="default", place="ящик")
        revoked = self.add("revoked", "Ключ на столе.", scope="default", place="стол")
        hidden = self.add("hidden", "Ключ на полке.", scope="secret", place="полка")
        self.store.grant_scope("reader", "default")
        assert revoked.source is not None
        self.store.revoke_source(revoked.source.source_id, 1)
        epoch = self.store.start_revision(
            "revision-invisible",
            (hidden.claim_id, revoked.claim_id),
            self.engine.state_version,
        )

        answer = self.ask("available", allowed_scopes=("default", "secret"))
        self.assertEqual(answer.status, "answered")
        assert answer.receipt is not None
        self.assertEqual(answer.receipt.claim_ids, (active.claim_id,))
        self.assertEqual(answer.quotes[0].text, "Ключ в ящике.")
        self.assertNotIn("столе", answer.receipt.answer_text)
        self.assertEqual(
            answer, self.ask("available", allowed_scopes=("default", "secret"))
        )

        self.store.finish_revision(
            "revision-invisible", self.engine.state_version, expected_epoch=epoch
        )
        self.store.start_revision(
            "revision-active", (active.claim_id,), self.engine.state_version
        )
        still_pending = self.ask("pending", allowed_scopes=("default", "secret"))
        self.assertEqual(still_pending.status, "revision_pending")
        self.assertFalse(still_pending.quotes)

    def test_family_revoke_and_scope_revoke_remove_candidates(self) -> None:
        first = self.add("first", "Ключ в ящике.", scope="default", place="ящик")
        second = self.add("second", "Ключ на столе.", scope="secret", place="стол")
        self.store.grant_scope("reader", "default")
        self.store.grant_scope("reader", "secret")
        self.assertEqual(
            self.ask("conflict", allowed_scopes=("default", "secret")).status,
            "clarify",
        )
        assert second.source is not None
        self.store.revoke_source_family(second.source.source_id)
        result = self.ask("only-public", allowed_scopes=("default", "secret"))
        assert result.receipt is not None
        self.assertEqual(result.receipt.claim_ids, ("first",))

        self.store.revoke_scope("reader", "default")
        no_candidate = self.ask("all-revoked", allowed_scopes=("default", "secret"))
        self.assertEqual(no_candidate.status, "clarify")
        self.assertEqual(
            no_candidate,
            self.ask("absent", "Где фонарь?", allowed_scopes=("default", "secret")),
        )
        self.assertIsNotNone(first.source)

    def test_revoke_between_selection_and_publication_blocks_quote(self) -> None:
        claim = self.add("active", "Ключ в ящике.", scope="default", place="ящик")
        self.store.grant_scope("reader", "default")

        class RacingStore(OperationalStore):
            def publish_answer(self, **kwargs):
                assert claim.source is not None
                self.revoke_source(claim.source.source_id, claim.source.source_version)
                return super().publish_answer(**kwargs)

        racing = RacingStore(self.store.path)
        result = self.dialogue.ask(
            DialogueQuestion("racing", "Где ключ?"),
            principal_id="reader",
            store=racing,
        )
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.quotes)
        self.assertIsNone(result.receipt)
        self.assertIsNone(self.store.task_status("racing"))

    def test_revoke_after_access_snapshot_cannot_expose_stale_conflict(self) -> None:
        self.add("active", "Ключ в ящике.", scope="default", place="ящик")
        stale = self.add("stale", "Ключ на столе.", scope="default", place="стол")
        self.store.grant_scope("reader", "default")

        class RacingStore(OperationalStore):
            def retrieval_access(self, *args, **kwargs):
                access = super().retrieval_access(*args, **kwargs)
                assert stale.source is not None
                self.revoke_source(stale.source.source_id, stale.source.source_version)
                return access

        racing = RacingStore(self.store.path)
        result = self.dialogue.ask(
            DialogueQuestion("racing-clarify", "Где ключ?"),
            principal_id="reader",
            store=racing,
        )
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.quotes)
        self.assertIsNone(self.store.task_status("racing-clarify"))
        self.assertEqual(self.ask("fresh").status, "answered")

    def test_revoked_published_request_is_blocked_on_retry(self) -> None:
        claim = self.add("active", "Ключ в ящике.", scope="default", place="ящик")
        self.store.grant_scope("reader", "default")
        self.assertEqual(self.ask("same").status, "answered")
        assert claim.source is not None
        self.store.revoke_source(claim.source.source_id, claim.source.source_version)
        retry = self.ask("same")
        self.assertEqual(retry.status, "blocked")
        self.assertFalse(retry.quotes)
        self.assertIsNone(retry.receipt)

    def test_polarity_and_modality_conflict_even_with_identical_roles(self) -> None:
        positive = self.add(
            "positive",
            "Ключ в ящике.",
            scope="default",
            place="ящик",
            polarity=ClaimPolarity.POSITIVE,
            modality=ClaimModality.ASSERTED,
        )
        self.add(
            "negative",
            "Ключ не в ящике.",
            scope="default",
            place="ящик",
            polarity=ClaimPolarity.NEGATIVE,
            modality=ClaimModality.ASSERTED,
        )
        self.store.grant_scope("reader", "default")
        self.assertEqual(self.ask("contradiction").status, "clarify")
        assert positive.source is not None
        self.store.revoke_source(
            positive.source.source_id, positive.source.source_version
        )
        negative = self.ask("negative-only")
        self.assertEqual(negative.status, "answered")
        self.assertEqual(negative.quotes[0].text, "Ключ не в ящике.")
        assert negative.receipt is not None
        self.assertEqual(negative.receipt.claim_ids, ("negative",))

    def test_interpreted_question_requires_exact_complete_text_and_role(self) -> None:
        self.add("reviewed", "Ключ в ящике.", scope="default", place="ящик")
        self.store.grant_scope("reader", "default")
        text = "Подскажите, где ключ?"
        at = text.index("ключ")
        hypothesis = QueryHypothesis(
            "key",
            "location",
            "place",
            sha256(text.encode()).hexdigest(),
            True,
            (),
            (),
            Span(at, at + len("ключ")),
        )

        # Explicit IDs cannot make an unsupported raw wording answerable.
        explicit = DialogueQuestion("raw", text, "key", "location")
        self.assertEqual(self.ask("raw", text).status, "clarify")
        self.assertEqual(
            self.dialogue.ask(explicit, principal_id="reader", store=self.store).status,
            "clarify",
        )
        accepted = self.dialogue.ask_interpreted(
            DialogueQuestion("parsed", text),
            hypothesis,
            principal_id="reader",
            store=self.store,
        )
        self.assertEqual(accepted.status, "answered")
        self.assertEqual(accepted.quotes[0].text, "Ключ в ящике.")
        self.assertEqual(
            self.dialogue.ask_interpreted(
                DialogueQuestion("parsed", text),
                hypothesis,
                principal_id="reader",
                store=self.store,
            ),
            accepted,
        )
        self.assertEqual(self.ask("parsed", text).status, "blocked")

        for changed in (
            replace(hypothesis, fully_covered=False, residual=("подскажите",)),
            replace(hypothesis, text_sha256="0" * 64),
            replace(hypothesis, asked_role="recipient"),
        ):
            result = self.dialogue.ask_interpreted(
                DialogueQuestion(
                    "bad-" + str(changed.asked_role) + changed.text_sha256[:2], text
                ),
                changed,
                principal_id="reader",
                store=self.store,
            )
            self.assertEqual(result.status, "clarify")
            self.assertFalse(result.quotes)
        mismatch = self.dialogue.ask_interpreted(
            DialogueQuestion("mismatch", text, "other", "location"),
            hypothesis,
            principal_id="reader",
            store=self.store,
        )
        self.assertEqual(mismatch.status, "clarify")

        negative_text = "Где не ключ?"
        negative_plan = replace(
            hypothesis,
            text_sha256=sha256(negative_text.encode()).hexdigest(),
            subject_span=Span(negative_text.index("ключ"), len(negative_text) - 1),
        )
        negative = self.dialogue.ask_interpreted(
            DialogueQuestion("negative", negative_text),
            negative_plan,
            principal_id="reader",
            store=self.store,
        )
        self.assertEqual(negative.status, "clarify")
        self.assertFalse(negative.quotes)


if __name__ == "__main__":
    unittest.main()

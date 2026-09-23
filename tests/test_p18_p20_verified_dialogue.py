"""Open development checks for a quotation dialogue with reviewed claims."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any

from text_factors.observations import ObservationArchive
from text_factors.real_data.archive_bridge import evidence_root, source_slice
from text_factors.real_data.contracts import (
    Claim,
    ClaimStatus,
    RoleValue,
    UncertaintyScope,
)
from text_factors.real_data.engine import RealDataEngine
from text_factors.real_data.verified_dialogue import (
    DialogueAnswer,
    DialogueQuestion,
    GroundedDialogue,
    QueryVocabulary,
)


class VerifiedDialogueOpenDevelopmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.archive = ObservationArchive(
            Path(self.directory.name) / "archive.db", create=True
        )
        self.addCleanup(self.archive.close)
        self.engine = RealDataEngine()
        self.dialogue = GroundedDialogue(
            self.archive,
            self.engine,
            QueryVocabulary(
                entity_aliases=(("ключ", "object-1"), ("ключик", "object-1")),
                relation_aliases=(("где", "location"), ("место", "location")),
                subject_roles=("item",),
            ),
            model_version="open-dev-v1",
        )

    def _add(
        self,
        claim_id: str,
        text: str,
        *,
        target: str = "ящик",
        scope: str = "default",
        status: ClaimStatus = ClaimStatus.ASSERTED,
    ) -> Claim:
        record = self.archive.import_text(
            text,
            namespace="development",
            external_key=claim_id,
            group_id=claim_id,
            metadata={"access_scope": scope},
        )
        source = source_slice(self.archive, record.source_id, 1, 0, len(text))
        root_id = f"root-{claim_id}"
        self.engine.register_evidence(
            evidence_root(self.archive, source, root_id=root_id)
        )
        claim = Claim(
            claim_id=claim_id,
            relation_id="location",
            arguments=(RoleValue("item", "object-1"), RoleValue("place", target)),
            status=status,
            source=source,
            evidence_roots=(root_id,),
            model_version="reviewed-annotation-v1",
        )
        self.engine.add_claim(claim)
        return claim

    @staticmethod
    def question(question_id: str = "q1", text: str = "Где ключ?") -> DialogueQuestion:
        return DialogueQuestion(question_id, text)

    def _prepare(self, question: DialogueQuestion | None = None, **kwargs) -> Any:
        return self.dialogue._prepare(
            question or self.question(),
            allowed_scopes=kwargs.get("allowed_scopes", ("default",)),
            pending_claim_ids=kwargs.get("pending_claim_ids", frozenset()),
        )

    def _storage_module(self) -> Any:
        try:
            return import_module("text_factors.real_data.storage")
        except ModuleNotFoundError as exc:
            if exc.name != "text_factors.real_data.storage":
                raise
            self.skipTest("P16 store is integrated in the parent worktree")

    def test_import_question_source_excerpt_and_version_are_pinned(self) -> None:
        claim = self._add("old", "Ключ находится в ящике.")
        draft = self._prepare()
        self.assertNotIsInstance(draft, DialogueAnswer)
        self.assertEqual(draft.quotes[0].text, "Ключ находится в ящике.")
        self.assertEqual(draft.quotes[0].source, claim.source)
        self.assertEqual(draft.receipt.claim_ids, ("old",))
        self.assertEqual(draft.receipt.model_version, "open-dev-v1")
        self.assertEqual(draft.receipt.state_version, self.engine.state_version)
        self.assertEqual(draft.receipt.answer_text, draft.quotes[0].text)
        # A draft is private. Without the operational publication gate nothing
        # containing source text may leave ask().
        public = self.dialogue.ask(self.question(), principal_id="reader", store=None)
        self.assertEqual(public.status, "blocked")
        self.assertFalse(public.quotes)
        self.assertIsNone(public.receipt)

    def test_ambiguous_or_unresolved_question_asks_for_clarification(self) -> None:
        self._add("c1", "Ключ в ящике.")
        for text in ("Где ключ и ключик?", "Ключ?", "Где ключ и книга?"):
            result = self._prepare(self.question(text=text))
            self.assertEqual(result.status, "clarify")
        result = self._prepare(DialogueQuestion("q2", "Что случилось?"))
        self.assertEqual(result.status, "clarify")
        self.assertFalse(result.quotes)
        self.assertEqual(
            self._prepare(self.question(text="Где тессеракт?")).status, "clarify"
        )
        self.assertEqual(
            self._prepare(self.question(text="Где сейчас ключ?")).status, "clarify"
        )
        self.assertEqual(
            self._prepare(self.question(text="Где не ключ?")).status, "clarify"
        )

        ambiguous = GroundedDialogue(
            self.archive,
            self.engine,
            QueryVocabulary(
                (("ключ", "object-1"), ("ключ", "object-2")),
                (("где", "location"),),
                ("item",),
            ),
            model_version="open-dev-v1",
        )
        ambiguity = ambiguous._prepare(
            self.question(),
            allowed_scopes=("default",),
            pending_claim_ids=frozenset(),
        )
        if not isinstance(ambiguity, DialogueAnswer):
            self.fail("ambiguous alias must request clarification")
        self.assertEqual(ambiguity.status, "clarify")

    def test_conflicting_current_claims_clarify_instead_of_choosing_one(self) -> None:
        self._add("c1", "Ключ в ящике.")
        self._add("c2", "Ключ лежит на столе.", target="стол")
        result = self._prepare()
        self.assertEqual(result.status, "clarify")
        self.assertFalse(result.quotes)

    def test_import_correction_revision_only_new_excerpt_is_cited(self) -> None:
        self._add("c1", "Ключ в ящике.")
        self.assertEqual(self._prepare().receipt.claim_ids, ("c1",))
        revision = self.archive.import_text(
            "Исправление: ключ лежит на полке.",
            namespace="development",
            external_key="correction",
            group_id="independent-correction",
        )
        source = source_slice(
            self.archive, revision.source_id, 1, 0, revision.char_count
        )
        self.engine.register_evidence(
            evidence_root(self.archive, source, root_id="root-c2")
        )
        new = Claim(
            "c2",
            "location",
            (RoleValue("item", "object-1"), RoleValue("place", "полка")),
            ClaimStatus.ASSERTED,
            source=source,
            evidence_roots=("root-c2",),
        )
        self.assertEqual(self.engine.correct_claim("c1", new), ("c1",))
        self.assertTrue(self.engine.is_superseded("c1"))
        self.assertFalse(self.engine.is_superseded("c2"))
        result = self._prepare(self.question("q2"))
        self.assertEqual(result.receipt.claim_ids, ("c2",))
        self.assertEqual(result.quotes[0].text, "Исправление: ключ лежит на полке.")
        self.assertNotIn("ящике", result.receipt.answer_text)

    def test_revision_pending_uncertainty_prediction_and_corrupt_span_fail_closed(
        self,
    ) -> None:
        claim = self._add("c1", "Ключ в ящике.")
        self.assertEqual(
            self._prepare(pending_claim_ids=frozenset({"c1"})).status,
            "revision_pending",
        )
        self.engine.mark_uncertainty(
            UncertaintyScope("uncertain-1", "indirect correction", ("c1",))
        )
        self.assertEqual(self._prepare().status, "blocked")
        self.engine.clear_claim_uncertainty("c1")
        self.assertNotIsInstance(self._prepare(), DialogueAnswer)
        # Source pinning is checked again against the archive before use.
        from dataclasses import replace

        assert claim.source is not None
        self.engine.claims["c1"] = replace(
            claim, source=replace(claim.source, sha256="0" * 64)
        )
        self.assertEqual(self._prepare().status, "blocked")

    def test_private_source_is_invisible_and_cannot_trigger_ambiguity(self) -> None:
        self._add("private", "Ключ в ящике.", scope="secret")
        no_grant = self._prepare()
        unknown_name = self._prepare(self.question(text="Где тессеракт?"))
        self.assertEqual(no_grant.status, "clarify")
        self.assertEqual(no_grant.clarification, unknown_name.clarification)
        self.assertEqual(
            self._prepare(allowed_scopes=("secret",)).receipt.claim_ids, ("private",)
        )
        self._add("public", "Ключ лежит на столе.", target="стол")
        self.assertEqual(self._prepare().receipt.claim_ids, ("public",))
        self.assertEqual(
            self._prepare(allowed_scopes=("secret", "default")).status, "clarify"
        )

    def test_unconfirmed_prediction_is_never_quoted(self) -> None:
        self._add(
            "forecast", "Возможно, ключ будет в ящике.", status=ClaimStatus.PREDICTED
        )
        self.assertEqual(self._prepare().status, "blocked")

    def test_integrated_publication_revocation_and_retry(self) -> None:
        storage = self._storage_module()
        self._add("c1", "Ключ в ящике.")
        store = storage.OperationalStore(Path(self.directory.name) / "store.db")
        store.grant_scope("reader", "default")
        first = self.dialogue.ask(self.question(), principal_id="reader", store=store)
        self.assertEqual(first.status, "answered")
        assert first.receipt is not None
        self.assertEqual(first.receipt.answer_text, "Ключ в ящике.")
        self.assertEqual(first.quotes[0].source.source_version, 1)
        self.assertEqual(first.state_version, self.engine.state_version)
        self.assertEqual(
            self.dialogue.ask(self.question(), principal_id="reader", store=store),
            first,
        )
        source = self.engine.claims["c1"].source
        assert source is not None
        store.revoke_source(source.source_id, source.source_version)
        again = self.dialogue.ask(self.question(), principal_id="reader", store=store)
        self.assertEqual(again.status, "blocked")
        self.assertFalse(again.quotes)
        self.assertIsNone(again.receipt)

    def test_published_end_to_end_correction_and_revision(self) -> None:
        storage = self._storage_module()
        self._add("c1", "Ключ в ящике.")
        store = storage.OperationalStore(Path(self.directory.name) / "journey.db")
        store.grant_scope("reader", "default")
        before = self.dialogue.ask(self.question(), principal_id="reader", store=store)
        self.assertEqual(before.status, "answered")
        assert before.receipt is not None
        self.assertEqual(before.receipt.claim_ids, ("c1",))

        record = self.archive.import_text(
            "Исправление: ключ находится на полке.",
            namespace="development",
            external_key="corrected-source",
            group_id="corrected-source",
        )
        span = source_slice(self.archive, record.source_id, 1, 0, record.char_count)
        self.engine.register_evidence(evidence_root(self.archive, span, root_id="r2"))
        epoch = store.start_revision("rev-1", ("c1",), self.engine.state_version)
        replacement = Claim(
            "c2",
            "location",
            (RoleValue("item", "object-1"), RoleValue("place", "полка")),
            ClaimStatus.OBSERVED,
            source=span,
            evidence_roots=("r2",),
        )
        self.engine.correct_claim("c1", replacement)
        pending = self.dialogue.ask(
            self.question("q2"), principal_id="reader", store=store
        )
        self.assertEqual(pending.status, "blocked")
        self.assertFalse(pending.quotes)
        store.finish_revision("rev-1", self.engine.state_version, expected_epoch=epoch)

        after = self.dialogue.ask(
            self.question("q2"), principal_id="reader", store=store
        )
        self.assertEqual(after.status, "answered")
        assert after.receipt is not None
        self.assertEqual(after.receipt.claim_ids, ("c2",))
        self.assertEqual(after.quotes[0].text, "Исправление: ключ находится на полке.")
        self.assertNotEqual(before.state_version, after.state_version)
        # Reusing an already published request ID after a correction cannot
        # resurrect its old answer or replace the journal entry with new text.
        replay = self.dialogue.ask(self.question(), principal_id="reader", store=store)
        self.assertEqual(replay.status, "blocked")
        self.assertFalse(replay.quotes)

    def test_revision_started_between_draft_and_publish_fails_closed(self) -> None:
        storage = self._storage_module()
        self._add("c1", "Ключ в ящике.")
        engine = self.engine

        class RacingStore(storage.OperationalStore):
            def publish_answer(self, **kwargs):
                self.start_revision("rev-1", ("c1",), engine.state_version)
                return super().publish_answer(**kwargs)

        store = RacingStore(Path(self.directory.name) / "racing-store.db")
        store.grant_scope("reader", "default")
        public = self.dialogue.ask(self.question(), principal_id="reader", store=store)
        self.assertEqual(public.status, "blocked")
        self.assertFalse(public.quotes)
        self.assertIsNone(public.receipt)
        self.assertEqual(store.pending_claim_ids(), ("c1",))


if __name__ == "__main__":
    unittest.main()

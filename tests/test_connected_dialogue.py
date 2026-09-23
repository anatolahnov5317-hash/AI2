"""Open end-to-end tests: raw text, review, publication and correction."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    from test_raw_language import _train
except ModuleNotFoundError:
    from tests.test_raw_language import _train
from text_factors.observations import ObservationArchive
from text_factors.real_data.connected_dialogue import ConnectedDialogue
from text_factors.real_data.contracts import ClaimModality, ClaimStatus
from text_factors.real_data.engine import RealDataEngine
from text_factors.real_data.hypothesis_bridge import ReviewDecision
from text_factors.real_data.open_semantics import IdentifiedMention, Span
from text_factors.real_data.question_language import (
    LabeledQuestion,
    QuestionLanguageModel,
)
from text_factors.real_data.raw_language import RawSemanticModel
from text_factors.real_data.research_checkpoint import RecoveryRequiresRevision
from text_factors.real_data.storage import OperationalStore
from text_factors.real_data.verified_dialogue import QueryVocabulary


class ConnectedDialogueTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.archive = ObservationArchive(Path(folder.name) / "archive.db", create=True)
        self.addCleanup(self.archive.close)
        self.store = OperationalStore(Path(folder.name) / "access.db")
        self.store.grant_scope("reader", "default")
        self.engine = RealDataEngine()
        example = "Кому Анна передала?"
        start = example.index("Анна")
        questions = QuestionLanguageModel().fit(
            (
                LabeledQuestion(
                    example,
                    "train-family",
                    IdentifiedMention(
                        "q-anna", "anna", Span(start, start + 4), "person", "nom"
                    ),
                    "transfer",
                    "recipient",
                ),
            ),
            held_out_families=frozenset({"new-family"}),
        )
        self.session = ConnectedDialogue(
            self.archive,
            self.engine,
            self.store,
            raw_model=RawSemanticModel().fit(_train()),
            question_model=questions,
            vocabulary=QueryVocabulary(
                entity_aliases=(("Анна", "anna"),),
                relation_aliases=(("передала", "transfer"),),
                subject_roles=("actor",),
            ),
            model_version="open-research-v1",
            checkpoint_dir=Path(folder.name) / "checkpoints",
        )

    def _review(self, candidate, claim_id: str, *, recipient: str):
        targets = ("nina", "folder", recipient)
        args = tuple(
            replace(arg, value_id=canonical)
            for arg, canonical in zip(
                candidate.draft_claim.arguments, targets, strict=True
            )
        )
        bindings = tuple(
            (old.mention_id, new.value_id)
            for old, new in zip(candidate.draft_claim.arguments, args, strict=True)
            if old.value_id != new.value_id and old.mention_id is not None
        )
        return ReviewDecision(
            candidate_id=candidate.candidate_id,
            reviewer_id="reviewer-1",
            claim_id=claim_id,
            excerpt=candidate.excerpt,
            relation_id=candidate.draft_claim.relation_id,
            arguments=args,
            polarity=candidate.draft_claim.polarity,
            modality=ClaimModality.ASSERTED,
            status=ClaimStatus.ASSERTED,
            approved=True,
            resolved_spans=candidate.unexplained,
            identity_bindings=bindings,
        )

    def test_new_name_requires_review_then_answer_correct_and_revoke(self) -> None:
        initial = self.session.import_message(
            "Нина передала папку Дарье.",
            namespace="research",
            external_key="initial",
            group_id="new-family",
        )
        self.assertEqual(len(initial.candidates), 1)
        self.assertFalse(initial.graph.complete)
        self.assertFalse(self.engine.claims)
        before = self.session.ask("q0", "Кому Нина передала?", principal_id="reader")
        self.assertNotEqual(before.status, "answered")
        self.assertFalse(before.quotes)

        first = self.session.review(
            initial.candidates[0],
            self._review(initial.candidates[0], "first", recipient="daria"),
        )
        self.assertEqual(first.reviewer_id, "reviewer-1")
        answered = self.session.ask("q1", "Кому Нина передала?", principal_id="reader")
        self.assertEqual(answered.status, "answered")
        assert answered.receipt is not None
        self.assertEqual(answered.receipt.claim_ids, ("first",))
        self.assertEqual(answered.quotes[0].text, "Нина передала папку Дарье")
        self.assertEqual(answered.quotes[0].source.source_version, 1)

        revision = self.session.import_message(
            "Нина передала папку Вере.",
            namespace="research",
            external_key="correction",
            group_id="new-family",
        )
        replacement = self.session.review(
            revision.candidates[0],
            self._review(revision.candidates[0], "replacement", recipient="vera"),
            correction_target_id="first",
        )
        self.assertEqual(replacement.claim_id, "replacement")
        after = self.session.ask("q2", "Кому Нина передала?", principal_id="reader")
        self.assertEqual(after.status, "answered")
        assert after.receipt is not None
        self.assertEqual(after.receipt.claim_ids, ("replacement",))
        self.assertEqual(after.quotes[0].text, "Нина передала папку Вере")
        self.assertNotEqual(answered.receipt.state_version, after.receipt.state_version)

        resumed = ConnectedDialogue.restore(
            self.archive,
            self.store,
            checkpoint_dir=self.session.checkpoint_dir,
            model_version="open-research-v1",
            vocabulary=self.session.dialogue.vocabulary,
        )
        self.assertIsNot(resumed.engine, self.engine)
        restored_answer = resumed.ask(
            "q-after-restart", "Кому Нина передала?", principal_id="reader"
        )
        self.assertEqual(restored_answer.status, "answered")
        assert restored_answer.receipt is not None
        self.assertEqual(restored_answer.receipt.claim_ids, ("replacement",))

        self.store.revoke_source_family(revision.source_id)
        with self.assertRaises(RecoveryRequiresRevision):
            ConnectedDialogue.restore(
                self.archive,
                self.store,
                checkpoint_dir=self.session.checkpoint_dir,
                model_version="open-research-v1",
                vocabulary=self.session.dialogue.vocabulary,
            )
        revoked = self.session.ask("q3", "Кому Нина передала?", principal_id="reader")
        self.assertNotEqual(revoked.status, "answered")
        self.assertFalse(revoked.quotes)
        no_grant = self.session.ask(
            "q4", "Кому Нина передала?", principal_id="stranger"
        )
        self.assertEqual(no_grant.status, "blocked")
        self.assertEqual(no_grant.state_version, "")

    def test_wrong_question_word_cannot_bypass_language_guard(self) -> None:
        message = self.session.import_message(
            "Нина передала папку Дарье.",
            namespace="research",
            external_key="only",
            group_id="new-family",
        )
        self.session.review(
            message.candidates[0],
            self._review(message.candidates[0], "reviewed", recipient="daria"),
        )
        for index, question in enumerate(
            ("Кому Нина не передала?", "Кому Нина передала вчера?"), start=1
        ):
            answer = self.session.ask(
                f"unsafe-{index}", question, principal_id="reader"
            )
            self.assertEqual(answer.status, "clarify")
            self.assertFalse(answer.quotes)

    def test_duplicate_revision_keeps_same_source_and_state(self) -> None:
        first = self.session.import_message(
            "Нина передала папку Дарье.",
            namespace="research",
            external_key="same-note",
            group_id="new-family",
        )
        self.session.review(
            first.candidates[0],
            self._review(first.candidates[0], "first", recipient="daria"),
        )
        second = self.session.import_message(
            "Нина передала папку Вере.",
            namespace="research",
            external_key="same-note",
            group_id="new-family",
        )
        version = self.engine.state_version
        repeated = self.session.import_message(
            "Нина передала папку Вере.",
            namespace="research",
            external_key="same-note",
            group_id="new-family",
        )
        self.assertEqual(second.source_id, repeated.source_id)
        self.assertEqual(second.source_version, repeated.source_version)
        self.assertEqual(second.candidates, repeated.candidates)
        self.assertEqual(self.engine.state_version, version)


if __name__ == "__main__":
    unittest.main()

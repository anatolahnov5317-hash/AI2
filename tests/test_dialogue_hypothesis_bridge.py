"""Review boundary from versioned event graphs to factual source claims."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from text_factors.observations import ObservationArchive
from text_factors.real_data.archive_bridge import evidence_root
from text_factors.real_data.contracts import (
    Claim,
    ClaimModality,
    ClaimPolarity,
    ClaimStatus,
    RoleValue,
)
from text_factors.real_data.engine import RealDataEngine
from text_factors.real_data.hypothesis_bridge import (
    ReviewCandidate,
    ReviewDecision,
    propose_review_candidates,
    register_reviewed_candidate,
)
from text_factors.real_data.open_semantics import (
    EventRole,
    SemanticEvent,
    SemanticGraph,
    Span,
)


class SemanticReviewBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.archive = ObservationArchive(Path(folder.name) / "sources.db", create=True)
        self.addCleanup(self.archive.close)
        self.engine = RealDataEngine()
        self.text = "Анна не передала ключ Борису."
        self.record = self.archive.import_text(
            self.text,
            namespace="dialogue-test",
            external_key="source-one",
            group_id="independent-story-one",
            metadata={"access_scope": "team"},
        )

    def _graph(
        self, *, negated: bool = True, residual: tuple[Span, ...] = ()
    ) -> SemanticGraph:
        trigger = self.text.index("передала")
        return SemanticGraph(
            source_id=self.record.source_id,
            source_version=self.record.version,
            text_sha256=self.record.sha256,
            model_fingerprint="a" * 64,
            events=(
                SemanticEvent(
                    event_id="parsed-event-one",
                    relation_id="transfer",
                    trigger=Span(trigger, trigger + len("передала")),
                    source_span=Span(0, len(self.text)),
                    roles=(
                        EventRole("actor", "m-anna", "anna"),
                        EventRole("object", "m-key", "key-one"),
                        EventRole("recipient", "m-boris", "boris"),
                    ),
                    negated=negated,
                    time_label=None,
                ),
            ),
            unexplained=residual,
        )

    def _candidate(self, *, negated: bool = True) -> ReviewCandidate:
        return propose_review_candidates(self._graph(negated=negated), self.archive)[0]

    def _review(
        self, candidate: ReviewCandidate, **overrides: object
    ) -> ReviewDecision:
        values: dict[str, object] = {
            "candidate_id": candidate.candidate_id,
            "reviewer_id": "annotator-one",
            "claim_id": "reviewed-event-one",
            "excerpt": candidate.excerpt,
            "relation_id": candidate.draft_claim.relation_id,
            "arguments": candidate.draft_claim.arguments,
            "polarity": candidate.draft_claim.polarity,
            "modality": ClaimModality.ASSERTED,
            "status": ClaimStatus.ASSERTED,
            "approved": True,
            "resolved_spans": candidate.unexplained,
        }
        values.update(overrides)
        return ReviewDecision(**values)  # type: ignore[arg-type]

    def _root(self, candidate: ReviewCandidate):
        return evidence_root(self.archive, candidate.source, root_id="evidence-one")

    def test_negative_event_stays_negative_and_requires_explicit_review(self) -> None:
        candidate = self._candidate()
        self.assertEqual(candidate.excerpt, self.text)
        self.assertEqual(candidate.source.access_scope, "team")
        self.assertEqual(candidate.draft_claim.polarity, ClaimPolarity.NEGATIVE)
        self.assertEqual(candidate.draft_claim.modality, ClaimModality.UNKNOWN)
        self.assertEqual(candidate.draft_claim.status, ClaimStatus.UNRESOLVED)
        self.assertFalse(self.engine.claims)
        self.assertFalse(self.engine.evidence.to_dict()["roots"])
        root = self._root(candidate)
        with self.assertRaisesRegex(ValueError, "semantic hypothesis"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate, polarity=ClaimPolarity.POSITIVE),
                root,
                archive=self.archive,
                engine=self.engine,
            )
        self.assertFalse(self.engine.claims)
        reviewed = register_reviewed_candidate(
            candidate,
            self._review(candidate),
            root,
            archive=self.archive,
            engine=self.engine,
        )
        self.assertEqual(reviewed.polarity, ClaimPolarity.NEGATIVE)
        self.assertEqual(reviewed.modality, ClaimModality.ASSERTED)
        self.assertEqual(reviewed.reviewer_id, "annotator-one")
        self.assertEqual(reviewed.source, root.source)
        self.assertEqual(
            self.engine.evidence.claim_roots(reviewed.claim_id), (root.root_id,)
        )

    def test_parser_missed_negation_cannot_be_promoted(self) -> None:
        cue = self.text.index("не")
        graph = self._graph(negated=False, residual=(Span(cue, cue + 2),))
        candidate = propose_review_candidates(graph, self.archive)[0]
        self.assertEqual(candidate.draft_claim.polarity, ClaimPolarity.POSITIVE)
        self.assertEqual(candidate.unexplained, (Span(cue, cue + 2),))
        self.assertTrue(candidate.guarded_negation)
        with self.assertRaisesRegex(ValueError, "missed a negation"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate),
                self._root(candidate),
                archive=self.archive,
                engine=self.engine,
            )
        self.assertEqual(self.engine.state_version, "real-data-state-0")

    def test_unexplained_new_object_requires_span_by_span_review(self) -> None:
        graph = self._graph()
        event = graph.events[0]
        new_roles = (
            event.roles[0],
            EventRole("object", "m-unseen", "never-trained-object"),
            event.roles[2],
        )
        unknown_start = self.text.index("ключ")
        unknown = Span(unknown_start, unknown_start + len("ключ"))
        candidate = propose_review_candidates(
            replace(
                graph,
                events=(replace(event, roles=new_roles),),
                unexplained=(unknown,),
            ),
            self.archive,
        )[0]
        self.assertEqual(
            candidate.draft_claim.arguments[1].value_id, "never-trained-object"
        )
        self.assertEqual(candidate.unexplained, (unknown,))
        root = self._root(candidate)
        with self.assertRaisesRegex(ValueError, "every unexplained"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate, resolved_spans=()),
                root,
                archive=self.archive,
                engine=self.engine,
            )
        with self.assertRaisesRegex(ValueError, "every unexplained"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate, resolved_spans=(unknown, unknown)),
                root,
                archive=self.archive,
                engine=self.engine,
            )
        reviewed = register_reviewed_candidate(
            candidate,
            self._review(candidate),
            root,
            archive=self.archive,
            engine=self.engine,
        )
        self.assertEqual(reviewed.arguments[1].value_id, "never-trained-object")
        self.assertEqual(reviewed.reviewer_id, "annotator-one")

    def test_explicit_identity_binding_and_addressed_correction(self) -> None:
        old_candidate = self._candidate()
        old = register_reviewed_candidate(
            old_candidate,
            self._review(old_candidate),
            self._root(old_candidate),
            archive=self.archive,
            engine=self.engine,
        )
        independent = Claim(
            "unrelated-fact",
            "location",
            (RoleValue("item", "another-book"),),
            ClaimStatus.ASSERTED,
        )
        self.engine.add_claim(independent)
        corrected_text = "Анна передала ключ Дарье."
        corrected_source = self.archive.import_text(
            corrected_text,
            namespace="dialogue-test",
            external_key="separate-correction",
            # The correction is another source in the SAME story family, so
            # source support is never miscounted as independent evidence.
            group_id="independent-story-one",
            metadata={"access_scope": "team"},
        )
        trigger = corrected_text.index("передала")
        graph = SemanticGraph(
            corrected_source.source_id,
            corrected_source.version,
            corrected_source.sha256,
            "b" * 64,
            (
                SemanticEvent(
                    "event-correction",
                    "transfer",
                    Span(trigger, trigger + len("передала")),
                    Span(0, len(corrected_text)),
                    (
                        EventRole("actor", "new-anna", "source-local-anna"),
                        EventRole("object", "new-key", "source-local-key"),
                        EventRole("recipient", "new-darya", "darya"),
                    ),
                    False,
                    None,
                ),
            ),
            (),
        )
        candidate = propose_review_candidates(graph, self.archive)[0]
        remapped = (
            replace(candidate.draft_claim.arguments[0], value_id="anna"),
            replace(candidate.draft_claim.arguments[1], value_id="key-one"),
            candidate.draft_claim.arguments[2],
        )
        decision = self._review(
            candidate,
            claim_id="reviewed-event-correction",
            arguments=remapped,
            identity_bindings=(("new-anna", "anna"), ("new-key", "key-one")),
        )
        corrected_root = evidence_root(
            self.archive, candidate.source, root_id="evidence-correction"
        )
        for invalid in (
            replace(decision, identity_bindings=()),
            replace(
                decision,
                identity_bindings=decision.identity_bindings + (("extra", "id"),),
            ),
            replace(
                decision,
                arguments=(replace(remapped[0], mention_id="other"), *remapped[1:]),
            ),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                register_reviewed_candidate(
                    candidate,
                    invalid,
                    corrected_root,
                    archive=self.archive,
                    engine=self.engine,
                    correction_target_id=old.claim_id,
                )
        self.assertFalse(self.engine.is_superseded(old.claim_id))
        corrected = register_reviewed_candidate(
            candidate,
            decision,
            corrected_root,
            archive=self.archive,
            engine=self.engine,
            correction_target_id=old.claim_id,
        )
        self.assertEqual(corrected.arguments, remapped)
        self.assertEqual(corrected.polarity, ClaimPolarity.POSITIVE)
        self.assertTrue(self.engine.is_superseded(old.claim_id))
        self.assertEqual(self.engine.claims[old.claim_id], old)
        self.assertEqual(self.engine.claims[independent.claim_id], independent)

    def test_review_rejects_wrong_roles_excerpt_root_and_modality(self) -> None:
        candidate = self._candidate()
        root = self._root(candidate)
        assert root.source is not None
        wrong_root = replace(
            root,
            source=replace(root.source, sha256="0" * 64),
        )
        decisions = (
            self._review(candidate, approved=False),
            self._review(candidate, excerpt="Анна передала ключ Борису."),
            self._review(
                candidate,
                arguments=tuple(reversed(candidate.draft_claim.arguments)),
            ),
            self._review(candidate, modality=ClaimModality.UNKNOWN),
            self._review(
                candidate,
                modality=ClaimModality.POSSIBLE,
                status=ClaimStatus.ASSERTED,
            ),
        )
        for decision in decisions:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                register_reviewed_candidate(
                    candidate,
                    decision,
                    root,
                    archive=self.archive,
                    engine=self.engine,
                )
        with self.assertRaisesRegex(ValueError, "exact reviewed source slice"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate),
                wrong_root,
                archive=self.archive,
                engine=self.engine,
            )
        self.assertFalse(self.engine.claims)

    def test_archive_revision_and_source_digest_checked_twice(self) -> None:
        graph = self._graph()
        with self.assertRaisesRegex(ValueError, "does not match archived"):
            propose_review_candidates(
                replace(graph, text_sha256="0" * 64), self.archive
            )
        candidate = propose_review_candidates(graph, self.archive)[0]
        self.archive.import_text(
            "Анна передала ключ Борису.",
            namespace="dialogue-test",
            external_key="source-one",
            group_id="independent-story-one",
            metadata={"access_scope": "team"},
        )
        with self.assertRaisesRegex(ValueError, "no longer current"):
            propose_review_candidates(graph, self.archive)
        with self.assertRaisesRegex(ValueError, "no longer current"):
            register_reviewed_candidate(
                candidate,
                self._review(candidate),
                self._root(candidate),
                archive=self.archive,
                engine=self.engine,
            )
        self.assertFalse(self.engine.claims)

    def test_claim_legacy_migration_is_explicitly_unknown(self) -> None:
        candidate = self._candidate()
        modern = candidate.draft_claim.to_dict()
        self.assertEqual(Claim.from_dict(modern), candidate.draft_claim)
        legacy = copy.deepcopy(modern)
        del legacy["polarity"]
        del legacy["modality"]
        del legacy["reviewer_id"]
        migrated = Claim.from_dict(legacy)
        self.assertEqual(migrated.polarity, ClaimPolarity.UNKNOWN)
        self.assertEqual(migrated.modality, ClaimModality.UNKNOWN)
        self.assertEqual(Claim.from_dict(migrated.to_dict()), migrated)
        older_v2 = {key: value for key, value in modern.items() if key != "reviewer_id"}
        self.assertEqual(Claim.from_dict(older_v2), candidate.draft_claim)
        self.assertNotEqual(migrated, candidate.draft_claim)
        for corrupt in (
            {**legacy, "polarity": "positive"},
            {**modern, "polarity": "unrecognized"},
            {**modern, "modality": "unrecognized"},
        ):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                Claim.from_dict(corrupt)


if __name__ == "__main__":
    unittest.main()

"""Open crash/recovery checks for version-pinned research dialogue state."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from text_factors.observations import ObservationArchive
from text_factors.real_data.archive_bridge import evidence_root, source_slice
from text_factors.real_data.contracts import (
    Claim,
    ClaimStatus,
    RoleValue,
    UncertaintyScope,
)
from text_factors.real_data.engine import RealDataEngine
from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledEvent,
    LabeledText,
    Span,
)
from text_factors.real_data.question_language import (
    LabeledQuestion,
    QuestionLanguageModel,
)
from text_factors.real_data.raw_language import RawSemanticModel
from text_factors.real_data.research_checkpoint import (
    RecoveryRequiresRevision,
    load_research_checkpoint,
    save_research_checkpoint,
)
from text_factors.real_data.storage import OperationalStore


def models() -> tuple[RawSemanticModel, QuestionLanguageModel]:
    statement = "Анна передала ключ Борису."
    question = "Что передала Анна?"
    mentions = (
        IdentifiedMention("anna", "person:anna", Span(0, 4), "person", "nom"),
        IdentifiedMention("key", "thing:key", Span(14, 18), "thing"),
        IdentifiedMention("boris", "person:boris", Span(19, 25), "person", "dat"),
    )
    raw = RawSemanticModel().fit(
        (
            LabeledText(
                statement,
                mentions,
                (
                    LabeledEvent(
                        "transfer",
                        Span(5, 13),
                        (("actor", "anna"), ("object", "key"), ("recipient", "boris")),
                    ),
                ),
            ),
        )
    )
    language = QuestionLanguageModel().fit(
        (
            LabeledQuestion(
                question,
                "train-family",
                IdentifiedMention("who", "person:anna", Span(13, 17), "person"),
                "transfer",
                "object",
            ),
        )
    )
    return raw, language


def claim_engine(archive: ObservationArchive) -> tuple[RealDataEngine, str]:
    text = "Анна передала ключ Борису."
    record = archive.import_text(
        text,
        namespace="checkpoint",
        external_key="one",
        group_id="independent-history",
        metadata={"access_scope": "private"},
    )
    source = source_slice(archive, record.source_id, record.version, 0, len(text))
    root = evidence_root(archive, source, root_id="root-1")
    engine = RealDataEngine()
    engine.register_evidence(root)
    engine.add_claim(
        Claim(
            "claim-1",
            "transfer",
            (RoleValue("actor", "person:anna"), RoleValue("object", "thing:key")),
            ClaimStatus.OBSERVED,
            source=source,
            evidence_roots=(root.root_id,),
            model_version="model-v1",
        )
    )
    return engine, record.source_id


class ResearchCheckpointTests(unittest.TestCase):
    def test_roundtrip_state_models_and_question_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            raw, questions = models()
            store = OperationalStore(path / "authority.db")
            engine = RealDataEngine()
            folder = path / "checkpoints"
            manifest = save_research_checkpoint(
                folder, engine, raw, questions, model_version="model-v1", store=store
            )
            self.assertEqual(
                manifest,
                save_research_checkpoint(
                    folder,
                    engine,
                    raw,
                    questions,
                    model_version="model-v1",
                    store=store,
                ),
            )
            with self.assertRaisesRegex(ValueError, "active_state_version_unbound"):
                load_research_checkpoint(
                    folder, expected_model_version="model-v1", store=store
                )
            store.bind_state_version(engine.state_version, expected_epoch=0)
            loaded_engine, loaded_raw, loaded_questions = load_research_checkpoint(
                folder, expected_model_version="model-v1", store=store
            )
            self.assertEqual(loaded_engine.to_dict(), engine.to_dict())
            self.assertEqual(loaded_raw.to_dict(), raw.to_dict())
            self.assertEqual(loaded_questions.to_dict(), questions.to_dict())
            self.assertEqual(
                loaded_questions.parse(
                    "Что передала Анна?",
                    raw_model=loaded_raw,
                    reviewed_identities=(),
                ).ambiguity,
                ("unresolved_identity",),
            )
            with self.assertRaisesRegex(ValueError, "model"):
                load_research_checkpoint(
                    folder, expected_model_version="model-v2", store=store
                )

    def test_pending_revision_uses_previous_active_state_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            raw, questions = models()
            store = OperationalStore(path / "authority.db")
            with ObservationArchive(path / "archive.db", create=True) as archive:
                old_engine, _ = claim_engine(archive)
                folder = path / "checkpoints"
                save_research_checkpoint(
                    folder,
                    old_engine,
                    raw,
                    questions,
                    model_version="model-v1",
                    store=store,
                )
                store.bind_state_version(old_engine.state_version, expected_epoch=0)
                old = old_engine.state_version
                pending_epoch = store.start_revision("rev-1", ("claim-1",), old)
                new_engine = RealDataEngine.from_dict(old_engine.to_dict())
                new_engine.mark_uncertainty(
                    UncertaintyScope(
                        "waiting",
                        "revision under review",
                        affected_claim_ids=("claim-1",),
                    )
                )
                save_research_checkpoint(
                    folder,
                    new_engine,
                    raw,
                    questions,
                    model_version="model-v1",
                    store=store,
                )
                with self.assertRaises(RecoveryRequiresRevision) as failure:
                    load_research_checkpoint(
                        folder, expected_model_version="model-v1", store=store
                    )
                self.assertEqual(failure.exception.engine.state_version, old)
                store.finish_revision(
                    "rev-1", new_engine.state_version, expected_epoch=pending_epoch
                )
                restored, _, _ = load_research_checkpoint(
                    folder, expected_model_version="model-v1", store=store
                )
                self.assertEqual(restored.state_version, new_engine.state_version)

    def test_revoked_source_overlay_requires_explicit_revision(self) -> None:
        for family in (False, True):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                raw, questions = models()
                store = OperationalStore(path / "authority.db")
                with ObservationArchive(path / "archive.db", create=True) as archive:
                    engine, source_id = claim_engine(archive)
                    folder = path / "checkpoints"
                    save_research_checkpoint(
                        folder,
                        engine,
                        raw,
                        questions,
                        model_version="model-v1",
                        store=store,
                    )
                    store.bind_state_version(engine.state_version, expected_epoch=0)
                    if family:
                        store.revoke_source_family(source_id)
                    else:
                        store.revoke_source(source_id, 1)
                    with self.assertRaisesRegex(
                        RecoveryRequiresRevision, "revoked sources"
                    ) as failure:
                        load_research_checkpoint(
                            folder, expected_model_version="model-v1", store=store
                        )
                    recovered = failure.exception.engine
                    self.assertNotEqual(recovered.state_version, engine.state_version)
                    self.assertFalse(
                        recovered.receipt(
                            question_id="q",
                            answer_text="",
                            claim_ids=("claim-1",),
                            model_version="model-v1",
                            allowed_scopes=("private",),
                        ).complete
                    )

    def test_wrong_active_version_and_corrupt_files_never_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            raw, questions = models()
            store = OperationalStore(path / "authority.db")
            engine = RealDataEngine()
            folder = path / "checkpoints"
            manifest = save_research_checkpoint(
                folder, engine, raw, questions, model_version="model-v1", store=store
            )
            store.bind_state_version("different-active-version", expected_epoch=0)
            with self.assertRaises(ValueError):
                load_research_checkpoint(
                    folder, expected_model_version="model-v1", store=store
                )
            store.bind_state_version(engine.state_version, expected_epoch=1)
            document = json.loads(manifest.read_text())
            state_path = folder / document["payload"]["state_file"]
            state_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_research_checkpoint(
                    folder, expected_model_version="model-v1", store=store
                )
            with self.assertRaises(ValueError):
                save_research_checkpoint(
                    folder,
                    engine,
                    raw,
                    questions,
                    model_version="model-v1",
                    store=store,
                )

    def test_private_location_and_canonical_question_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            raw, questions = models()
            encoded = questions.to_dict()
            self.assertEqual(
                QuestionLanguageModel.from_dict(encoded).to_dict(), encoded
            )
            tampered = json.loads(json.dumps(encoded))
            tampered["patterns"].append(tampered["patterns"][0])
            with self.assertRaises(ValueError):
                QuestionLanguageModel.from_dict(tampered)
            malformed = json.loads(json.dumps(encoded))
            malformed["patterns"][0]["tokens"][0] = " "
            with self.assertRaises(ValueError):
                QuestionLanguageModel.from_dict(malformed)
            store = OperationalStore(path / "authority.db")
            with self.assertRaisesRegex(ValueError, "outside checkpoints"):
                save_research_checkpoint(
                    path,
                    RealDataEngine(),
                    raw,
                    questions,
                    model_version="model-v1",
                    store=store,
                )

    def test_divergent_same_version_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            store = OperationalStore(path / "authority.db")
            raw, questions = models()
            engine = RealDataEngine()
            folder = path / "checkpoints"
            save_research_checkpoint(
                folder, engine, raw, questions, model_version="model-v1", store=store
            )
            changed_questions = QuestionLanguageModel().fit(
                (
                    LabeledQuestion(
                        "Что передала Анна?",
                        "train-family",
                        IdentifiedMention("who", "person:anna", Span(13, 17), "person"),
                        "transfer",
                        "recipient",
                    ),
                )
            )
            with self.assertRaisesRegex(ValueError, "other content"):
                save_research_checkpoint(
                    folder,
                    engine,
                    raw,
                    changed_questions,
                    model_version="model-v1",
                    store=store,
                )
            store.bind_state_version(engine.state_version, expected_epoch=0)
            link = path / "redirect"
            link.symlink_to(folder, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                load_research_checkpoint(
                    link, expected_model_version="model-v1", store=store
                )


if __name__ == "__main__":
    unittest.main()

"""Open synthetic checks of P16–P17 contracts, never pilot acceptance data."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from text_factors.observations import ObservationArchive
from text_factors.real_data.archive_bridge import evidence_root, source_slice
from text_factors.real_data.bundle import load_bundle, migrate_v1_state, save_bundle
from text_factors.real_data.composition import CompositionalEncoder
from text_factors.real_data.contexts import ContextRegistry
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
    OpenSemanticModel,
    Span,
)
from text_factors.real_data.persistence import save_state
from text_factors.real_data.storage import (
    AccessDenied,
    OperationalStore,
    PublicationQuote,
    StalePublication,
)


def semantic_model() -> OpenSemanticModel:
    text = "Анна дала ключ."
    return OpenSemanticModel().fit(
        (
            LabeledText(
                text,
                (
                    IdentifiedMention("person", "anna", Span(0, 4)),
                    IdentifiedMention("thing", "key", Span(10, 14)),
                ),
                (
                    LabeledEvent(
                        "transfer",
                        Span(5, 9),
                        (("actor", "person"), ("object", "thing")),
                    ),
                ),
            ),
        )
    )


def fixture(archive: ObservationArchive, scope: str = "private"):
    record = archive.import_text(
        "Ключ в шкафу. Компас на столе.",
        namespace="dev",
        external_key="example",
        group_id="independent-1",
        metadata={"access_scope": scope},
    )
    span = source_slice(archive, record.source_id, record.version, 0, 13)
    root = evidence_root(archive, span, root_id="root-1")
    engine = RealDataEngine()
    engine.register_evidence(root)
    engine.add_claim(
        Claim(
            "claim-1",
            "location",
            (RoleValue("object", "key"), RoleValue("place", "wardrobe")),
            ClaimStatus.OBSERVED,
            source=span,
            evidence_roots=(root.root_id,),
            model_version="model-v1",
        )
    )
    quote = PublicationQuote(
        span, "claim-1", archive.read_span(record.source_id, 1, 0, 13)
    )
    receipt = engine.receipt(
        question_id="where",
        answer_text=quote.text,
        claim_ids=("claim-1",),
        model_version="model-v1",
        allowed_scopes=(scope,),
    )
    assert receipt.complete
    return engine, record, quote, receipt


class OperationalStorageTests(unittest.TestCase):
    def test_source_family_tombstone_covers_old_and_future_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            with ObservationArchive(
                directory_path / "archive.db", create=True
            ) as archive:
                first_engine, first_record, first_quote, first_receipt = fixture(
                    archive
                )
                authority = OperationalStore(directory_path / "authority.db")
                authority.grant_scope("user", "private")
                old_bundle = directory_path / "old-bundle.json"
                encoder = CompositionalEncoder(width=128)
                semantics = semantic_model()
                save_bundle(
                    old_bundle,
                    first_engine,
                    contexts=ContextRegistry(width=128),
                    semantics=semantics,
                    encoder=encoder,
                    model_version="model-v1",
                    corpus_version="corpus-v1",
                    policy_version="policy-v1",
                )
                authority.publish_answer(
                    request_id="old",
                    request_fingerprint="old-v1",
                    principal_id="user",
                    engine=first_engine,
                    archive=archive,
                    receipt=first_receipt,
                    quotes=(first_quote,),
                    expected_revision_epoch=0,
                )
                authority.revoke_source_family(first_record.source_id)
                self.assertEqual(
                    authority.revoked_source_families(), (first_record.source_id,)
                )
                old_status = authority.task_status("old")
                assert old_status is not None
                self.assertEqual(old_status.status, "revoked")
                with self.assertRaises(AccessDenied):
                    authority.publish_answer(
                        request_id="old",
                        request_fingerprint="old-v1",
                        principal_id="user",
                        engine=first_engine,
                        archive=archive,
                        receipt=first_receipt,
                        quotes=(first_quote,),
                        expected_revision_epoch=authority.revision_epoch(),
                    )
                old_restored = load_bundle(
                    old_bundle,
                    expected_model_version="model-v1",
                    expected_corpus_version="corpus-v1",
                    expected_policy_version="policy-v1",
                    authority=authority,
                )
                self.assertFalse(
                    old_restored.engine.receipt(
                        question_id="q",
                        answer_text=first_quote.text,
                        claim_ids=("claim-1",),
                        model_version="model-v1",
                        allowed_scopes=("private",),
                    ).complete
                )
                second_record = archive.import_text(
                    "Ключ на столе.",
                    namespace="dev",
                    external_key="example",
                    group_id="independent-1",
                    metadata={"access_scope": "private"},
                )
                self.assertEqual(second_record.source_id, first_record.source_id)
                self.assertEqual(second_record.version, 2)
                second_source = source_slice(
                    archive,
                    second_record.source_id,
                    second_record.version,
                    0,
                    len("Ключ на столе."),
                )
                future_engine = RealDataEngine()
                future_engine.register_evidence(
                    evidence_root(archive, second_source, root_id="root-next")
                )
                future_engine.add_claim(
                    Claim(
                        "next",
                        "location",
                        (RoleValue("object", "key"), RoleValue("place", "table")),
                        ClaimStatus.OBSERVED,
                        source=second_source,
                        evidence_roots=("root-next",),
                        model_version="model-v1",
                    )
                )
                future_quote = PublicationQuote(second_source, "next", "Ключ на столе.")
                future_receipt = future_engine.receipt(
                    question_id="q",
                    answer_text=future_quote.text,
                    claim_ids=("next",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                )
                self.assertTrue(future_receipt.complete)
                future_bundle = directory_path / "future-bundle.json"
                save_bundle(
                    future_bundle,
                    future_engine,
                    contexts=ContextRegistry(width=128),
                    semantics=semantics,
                    encoder=encoder,
                    model_version="model-v1",
                    corpus_version="corpus-v1",
                    policy_version="policy-v1",
                )
                authority.bind_state_version(
                    future_engine.state_version,
                    expected_epoch=authority.revision_epoch(),
                )
                with self.assertRaises(AccessDenied):
                    authority.publish_answer(
                        request_id="future",
                        request_fingerprint="future-v2",
                        principal_id="user",
                        engine=future_engine,
                        archive=archive,
                        receipt=future_receipt,
                        quotes=(future_quote,),
                        expected_revision_epoch=authority.revision_epoch(),
                    )
                future_restored = load_bundle(
                    future_bundle,
                    expected_model_version="model-v1",
                    expected_corpus_version="corpus-v1",
                    expected_policy_version="policy-v1",
                    authority=authority,
                )
                self.assertFalse(
                    future_restored.engine.receipt(
                        question_id="q",
                        answer_text=future_quote.text,
                        claim_ids=("next",),
                        model_version="model-v1",
                        allowed_scopes=("private",),
                    ).complete
                )

    def test_revision_start_blocks_stale_draft_and_keeps_unrelated_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, _, quote, receipt = fixture(archive)
                engine.add_claim(
                    Claim(
                        "unrelated",
                        "location",
                        (RoleValue("object", "compass"), RoleValue("place", "table")),
                        ClaimStatus.OBSERVED,
                        source=quote.source,
                        evidence_roots=("root-1",),
                        model_version="model-v1",
                    )
                )
                old_receipt = engine.receipt(
                    question_id="where",
                    answer_text=quote.text,
                    claim_ids=("claim-1",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                )
                store = OperationalStore(root / "authority.db")
                store.grant_scope("user", "private")
                prepared = store.revision_epoch()
                start = store.start_revision(
                    "revision-1", ("claim-1",), engine.state_version
                )
                self.assertGreater(start, prepared)
                self.assertEqual(store.pending_claim_ids(), ("claim-1",))
                self.assertEqual(
                    store.start_revision(
                        "revision-1", ("claim-1",), engine.state_version
                    ),
                    start,
                )
                with self.assertRaises(StalePublication):
                    store.publish_answer(
                        request_id="old",
                        request_fingerprint="old-input",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=old_receipt,
                        quotes=(quote,),
                        expected_revision_epoch=prepared,
                    )
                with self.assertRaises(AccessDenied):
                    store.publish_answer(
                        request_id="affected",
                        request_fingerprint="new-input",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=old_receipt,
                        quotes=(quote,),
                        expected_revision_epoch=start,
                    )
                unaffected = engine.receipt(
                    question_id="where",
                    answer_text=quote.text,
                    claim_ids=("unrelated",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                )
                allowed_quote = PublicationQuote(quote.source, "unrelated", quote.text)
                self.assertTrue(
                    store.publish_answer(
                        request_id="safe",
                        request_fingerprint="safe-input",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=unaffected,
                        quotes=(allowed_quote,),
                        expected_revision_epoch=start,
                    ).receipt.complete
                )
                restarted = OperationalStore(root / "authority.db")
                self.assertTrue(restarted.is_revision_pending("revision-1"))
                engine.mark_uncertainty(
                    UncertaintyScope("u1", "corrected", affected_claim_ids=("claim-1",))
                )
                finished = restarted.finish_revision(
                    "revision-1", engine.state_version, expected_epoch=start
                )
                self.assertEqual(restarted.pending_claim_ids(), ())
                self.assertEqual(
                    restarted.finish_revision(
                        "revision-1", engine.state_version, expected_epoch=start
                    ),
                    finished,
                )
                with self.assertRaises(StalePublication):
                    store.publish_answer(
                        request_id="safe-late",
                        request_fingerprint="old-state",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=unaffected,
                        quotes=(allowed_quote,),
                        expected_revision_epoch=start,
                    )

    def test_grant_revoke_during_processing_and_replay_never_leaks_old_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, source, quote, receipt = fixture(archive)
                store = OperationalStore(root / "authority.db")
                store.grant_scope("user", "private")
                task = store.begin_task("q1", "request-v1", "user")
                self.assertEqual(task.status, "pending")
                store.revoke_source(source.source_id, 1)
                with self.assertRaises((AccessDenied, StalePublication)):
                    store.publish_answer(
                        request_id="q1",
                        request_fingerprint="request-v1",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=receipt,
                        quotes=(quote,),
                        expected_revision_epoch=0,
                    )
                status = store.task_status("q1")
                self.assertIsNotNone(status)
                assert status is not None
                self.assertEqual(status.status, "pending")

    def test_idempotence_scope_and_revocation_purge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, source, quote, receipt = fixture(archive)
                store = OperationalStore(root / "authority.db")
                with self.assertRaises(AccessDenied):
                    store.publish_answer(
                        request_id="q1",
                        request_fingerprint="request-v1",
                        principal_id="user",
                        engine=engine,
                        archive=archive,
                        receipt=receipt,
                        quotes=(quote,),
                        expected_revision_epoch=0,
                    )
                store.grant_scope("user", "private")
                args: dict[str, Any] = dict(
                    request_id="q1",
                    request_fingerprint="request-v1",
                    principal_id="user",
                    engine=engine,
                    archive=archive,
                    receipt=receipt,
                    quotes=(quote,),
                    expected_revision_epoch=store.revision_epoch(),
                )
                publish = cast(Any, store.publish_answer)
                first = publish(**args)
                self.assertEqual(first, publish(**args))
                status = store.task_status("q1")
                assert status is not None
                self.assertEqual(status.status, "published")
                with self.assertRaises(StalePublication):
                    publish(**(args | {"request_fingerprint": "modified"}))
                store.revoke_source(source.source_id, 1)
                status = store.task_status("q1")
                assert status is not None
                self.assertEqual(status.status, "revoked")
                with self.assertRaises((AccessDenied, StalePublication)):
                    publish(**args)
                # Sensitive quote is redacted from the persisted task result.
                self.assertNotIn(
                    quote.text,
                    (root / "authority.db")
                    .read_bytes()
                    .decode("utf-8", errors="ignore"),
                )

    def test_stale_and_forged_source_or_receipt_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, _, quote, receipt = fixture(archive)
                store = OperationalStore(root / "authority.db")
                store.grant_scope("user", "private")
                args: dict[str, Any] = dict(
                    request_id="q1",
                    request_fingerprint="request-v1",
                    principal_id="user",
                    engine=engine,
                    archive=archive,
                    receipt=receipt,
                    quotes=(quote,),
                    expected_revision_epoch=store.revision_epoch(),
                )
                publish = cast(Any, store.publish_answer)
                with self.assertRaises(StalePublication):
                    publish(
                        **(
                            args
                            | {
                                "quotes": (
                                    PublicationQuote(
                                        quote.source, quote.claim_id, "другой"
                                    ),
                                )
                            }
                        )
                    )
                forged = replace(quote.source, sha256="0" * 64)
                with self.assertRaises(StalePublication):
                    publish(
                        **(
                            args
                            | {
                                "quotes": (
                                    PublicationQuote(
                                        forged, quote.claim_id, quote.text
                                    ),
                                )
                            }
                        )
                    )
                engine.mark_uncertainty(
                    UncertaintyScope(
                        "u1", "correction pending", affected_claim_ids=("claim-1",)
                    )
                )
                with self.assertRaises(AccessDenied):
                    publish(**args)
                self.assertIsNone(store.task_status("q1"))

    def test_task_completion_idempotence_and_exclusive_request_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            store = OperationalStore(Path(directory) / "authority.db")
            self.assertEqual(
                store.begin_task("task", "fingerprint", "user").status, "pending"
            )
            self.assertEqual(
                store.begin_task("task", "fingerprint", "user").status, "pending"
            )
            self.assertEqual(
                store.complete_task("task", "fingerprint", "user", {"done": 1}).status,
                "complete",
            )
            store.complete_task("task", "fingerprint", "user", {"done": 1})
            with self.assertRaises(StalePublication):
                store.complete_task("task", "fingerprint", "user", {"done": 2})
            with self.assertRaises(StalePublication):
                store.begin_task("task", "different", "user")

    def test_concurrent_revoke_before_final_publish_is_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, source, quote, receipt = fixture(archive)
                store = OperationalStore(root / "authority.db")
                store.grant_scope("user", "private")
                prepared = threading.Event()
                revoked = threading.Event()
                outcome: list[str] = []

                def worker() -> None:
                    store.begin_task("q1", "request-v1", "user")
                    prepared.set()
                    if not revoked.wait(5):
                        outcome.append("timeout")
                        return
                    try:
                        store.publish_answer(
                            request_id="q1",
                            request_fingerprint="request-v1",
                            principal_id="user",
                            engine=engine,
                            archive=archive,
                            receipt=receipt,
                            quotes=(quote,),
                            expected_revision_epoch=0,
                        )
                    except (AccessDenied, StalePublication):
                        outcome.append("denied")

                thread = threading.Thread(target=worker)
                thread.start()
                self.assertTrue(prepared.wait(5))
                store.revoke_source(source.source_id, 1)
                revoked.set()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(outcome, ["denied"])

    def test_bundle_migration_pin_integrity_and_revocation_survive_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, source, _, _ = fixture(archive)
                authority = OperationalStore(root / "authority.db")
                encoder = CompositionalEncoder(width=128)
                contexts = ContextRegistry(width=128)
                semantics = semantic_model()
                old = root / "legacy.json"
                snapshot = root / "bundle.json"
                save_state(old, engine, contexts=contexts, model_version="model-v1")
                migrate_v1_state(
                    old,
                    snapshot,
                    semantics=semantics,
                    encoder=encoder,
                    expected_model_version="model-v1",
                    corpus_version="corpus-v1",
                    policy_version="policy-v1",
                )
                loaded = load_bundle(
                    snapshot,
                    expected_model_version="model-v1",
                    expected_corpus_version="corpus-v1",
                    expected_policy_version="policy-v1",
                    authority=authority,
                )
                self.assertEqual(loaded.engine.to_dict(), engine.to_dict())
                self.assertEqual(loaded.semantics.to_dict(), semantics.to_dict())
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_bundle(
                        snapshot,
                        expected_model_version="model-v2",
                        expected_corpus_version="corpus-v1",
                        expected_policy_version="policy-v1",
                        authority=authority,
                    )
                authority.revoke_source(source.source_id, source.version)
                rolled_back = load_bundle(
                    snapshot,
                    expected_model_version="model-v1",
                    expected_corpus_version="corpus-v1",
                    expected_policy_version="policy-v1",
                    authority=authority,
                )
                self.assertFalse(
                    rolled_back.engine.receipt(
                        question_id="where",
                        answer_text="Ключ в шкафу.",
                        claim_ids=("claim-1",),
                        model_version="model-v1",
                        allowed_scopes=("private",),
                    ).complete
                )
                self.assertEqual(authority.revoked_sources(), ((source.source_id, 1),))
                modified = json.loads(snapshot.read_text(encoding="utf-8"))
                modified["body"]["encoder"]["width"] = 256
                snapshot.write_text(
                    json.dumps(modified, ensure_ascii=False), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    load_bundle(
                        snapshot,
                        expected_model_version="model-v1",
                        expected_corpus_version="corpus-v1",
                        expected_policy_version="policy-v1",
                        authority=authority,
                    )

    def test_rollback_activation_keeps_independent_answer_and_denies_revoked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ObservationArchive(root / "archive.db", create=True) as archive:
                engine, revoked, _, _ = fixture(archive)
                independent = archive.import_text(
                    "Компас на столе.",
                    namespace="dev",
                    external_key="second",
                    group_id="independent-2",
                    metadata={"access_scope": "private"},
                )
                span = source_slice(
                    archive,
                    independent.source_id,
                    independent.version,
                    0,
                    len("Компас на столе."),
                )
                engine.register_evidence(evidence_root(archive, span, root_id="root-2"))
                engine.add_claim(
                    Claim(
                        "independent",
                        "location",
                        (RoleValue("object", "compass"), RoleValue("place", "table")),
                        ClaimStatus.OBSERVED,
                        source=span,
                        evidence_roots=("root-2",),
                        model_version="model-v1",
                    )
                )
                authority = OperationalStore(root / "authority.db")
                authority.grant_scope("user", "private")
                snapshot = root / "old-bundle.json"
                save_bundle(
                    snapshot,
                    engine,
                    contexts=ContextRegistry(width=128),
                    semantics=semantic_model(),
                    encoder=CompositionalEncoder(width=128),
                    model_version="model-v1",
                    corpus_version="corpus-v1",
                    policy_version="policy-v1",
                )
                authority.bind_state_version(engine.state_version, expected_epoch=0)
                authority.revoke_source(revoked.source_id, revoked.version)
                loaded = load_bundle(
                    snapshot,
                    expected_model_version="model-v1",
                    expected_corpus_version="corpus-v1",
                    expected_policy_version="policy-v1",
                    authority=authority,
                )
                with self.assertRaises(StalePublication):
                    authority.publish_answer(
                        request_id="stale",
                        request_fingerprint="q0",
                        principal_id="user",
                        engine=loaded.engine,
                        archive=archive,
                        receipt=loaded.engine.receipt(
                            question_id="q",
                            answer_text="Компас на столе.",
                            claim_ids=("independent",),
                            model_version="model-v1",
                            allowed_scopes=("private",),
                        ),
                        quotes=(
                            PublicationQuote(span, "independent", "Компас на столе."),
                        ),
                        expected_revision_epoch=authority.revision_epoch(),
                    )
                new_epoch = authority.bind_state_version(
                    loaded.engine.state_version,
                    expected_epoch=authority.revision_epoch(),
                )
                quote = PublicationQuote(span, "independent", "Компас на столе.")
                receipt = loaded.engine.receipt(
                    question_id="q",
                    answer_text=quote.text,
                    claim_ids=("independent",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                )
                self.assertTrue(
                    authority.publish_answer(
                        request_id="safe",
                        request_fingerprint="q1",
                        principal_id="user",
                        engine=loaded.engine,
                        archive=archive,
                        receipt=receipt,
                        quotes=(quote,),
                        expected_revision_epoch=new_epoch,
                    ).receipt.complete
                )
                denied = loaded.engine.receipt(
                    question_id="q",
                    answer_text="Ключ в шкафу.",
                    claim_ids=("claim-1",),
                    model_version="model-v1",
                    allowed_scopes=("private",),
                )
                self.assertFalse(denied.complete)

    def test_v1_operational_store_migration_retains_revocation_and_erases_body(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.db"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE grants (principal_id TEXT, scope TEXT)")
                db.execute(
                    "CREATE TABLE revoked_sources (source_id TEXT, version INTEGER)"
                )
                db.execute(
                    "CREATE TABLE tasks (request_id TEXT, fingerprint TEXT, "
                    "principal_id TEXT, status TEXT, result_json TEXT)"
                )
                db.execute(
                    "CREATE TABLE publication_sources (request_id TEXT, "
                    "source_id TEXT, version INTEGER, scope TEXT)"
                )
                db.execute("INSERT INTO revoked_sources VALUES ('s1',1)")
                db.execute(
                    "INSERT INTO tasks VALUES ('q','f','u','published',?)",
                    ("{" + '"answer":"PRIVATE SOURCE PAYLOAD"}',),
                )
                db.execute("PRAGMA application_id=1095316019")
                db.execute("PRAGMA user_version=1")
            authority = OperationalStore(path)
            self.assertEqual(authority.revoked_sources(), (("s1", 1),))
            status = authority.task_status("q")
            assert status is not None
            self.assertEqual(status.status, "revoked")
            self.assertNotIn(b"PRIVATE SOURCE PAYLOAD", path.read_bytes())

    def test_v2_operational_schema_migrates_without_losing_existing_tombstones(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.db"
            authority = OperationalStore(path)
            authority.grant_scope("user", "private")
            authority.revoke_source("old-source", 1)
            del authority
            with sqlite3.connect(path) as db:
                db.execute("DROP TABLE revoked_source_families")
                db.execute("PRAGMA user_version=2")
            migrated = OperationalStore(path)
            self.assertEqual(migrated.revoked_sources(), (("old-source", 1),))
            migrated.revoke_source_family("all-versions")
            self.assertEqual(migrated.revoked_source_families(), ("all-versions",))

    def test_incompatible_width_or_legacy_without_contexts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = OperationalStore(root / "authority.db")
            engine = RealDataEngine()
            semantics = semantic_model()
            with self.assertRaisesRegex(ValueError, "width mismatch"):
                save_bundle(
                    root / "mismatch.json",
                    engine,
                    contexts=ContextRegistry(width=64),
                    semantics=semantics,
                    encoder=CompositionalEncoder(width=128),
                    model_version="m1",
                    corpus_version="c1",
                    policy_version="p1",
                )
            save_state(root / "no-contexts.json", engine, model_version="m1")
            with self.assertRaisesRegex(ValueError, "contexts mismatch"):
                migrate_v1_state(
                    root / "no-contexts.json",
                    root / "target.json",
                    semantics=semantics,
                    encoder=CompositionalEncoder(width=128),
                    expected_model_version="m1",
                    corpus_version="c1",
                    policy_version="p1",
                )
            self.assertEqual(authority.revoked_sources(), ())


if __name__ == "__main__":
    unittest.main()

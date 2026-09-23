"""Open development examples for grounded answers and addressed corrections."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from text_factors.observations import ObservationArchive
from text_factors.real_data import (
    Claim,
    ClaimStatus,
    EvidenceRoot,
    RealDataEngine,
    RoleValue,
    SourceSlice,
    evidence_root,
    source_slice,
)


def claim(
    claim_id: str,
    *,
    roots: tuple[str, ...] = (),
    subject: str = "ключ",
    place: str = "ящик",
    source: SourceSlice | None = None,
    status: ClaimStatus = ClaimStatus.ASSERTED,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        relation_id="место",
        arguments=(RoleValue("предмет", subject), RoleValue("место", place)),
        source=(
            source
            if source is not None
            else SourceSlice("s" + roots[0][1:], 1, 0, 1, "0" * 64)
            if roots
            else None
        ),
        status=status,
        evidence_roots=roots,
    )


def stub_root(
    root_id: str, group_id: str, source_id: str, version: int = 1
) -> EvidenceRoot:
    """Only for logic checks: manual fake hashes are not archive verification."""
    return EvidenceRoot(
        root_id,
        group_id,
        source_id,
        version,
        "default",
        SourceSlice(source_id, version, 0, 1, "0" * 64),
    )


def receipt(engine: RealDataEngine, *ids: str, **kwargs):
    return engine.receipt(
        question_id="где?",
        answer_text="текущий ответ",
        claim_ids=ids,
        model_version="open-dev",
        **kwargs,
    )


class GroundedRevisionOpenDevelopmentTests(unittest.TestCase):
    def test_derived_claim_cannot_attribute_an_unrelated_source_slice(self) -> None:
        engine = RealDataEngine()
        engine.register_evidence(stub_root("r1", "family", "s1"))
        engine.add_claim(claim("base", roots=("r1",)))
        forged = SourceSlice("never-imported", 999, 100, 120, "f" * 64)
        before = engine.to_dict()
        with self.assertRaisesRegex(ValueError, "no matching evidence root"):
            engine.add_claim(
                claim("derived", source=forged), parent_claim_ids=("base",)
            )
        self.assertEqual(engine.to_dict(), before)

        source = engine.claims["base"].source
        self.assertIsNotNone(source)
        engine.add_claim(claim("derived", source=source), parent_claim_ids=("base",))
        self.assertTrue(receipt(engine, "derived").complete)

        tampered = deepcopy(engine.to_dict())
        for row in cast(list[dict[str, Any]], tampered["claims"]):
            if row["claim_id"] == "derived":
                row["source"] = forged.to_dict()
        with self.assertRaisesRegex(ValueError, "no matching evidence root"):
            RealDataEngine.from_dict(tampered)
        engine.claims["derived"] = replace(engine.claims["derived"], source=forged)
        self.assertEqual(
            receipt(engine, "derived").reason, "source_binding_missing:derived"
        )

    def test_all_claims_in_a_composite_answer_need_their_own_evidence(self) -> None:
        engine = RealDataEngine()
        engine.register_evidence(stub_root("r1", "g1", "s1"))
        engine.add_claim(claim("supported", roots=("r1",)))
        engine.add_claim(claim("unsourced", subject="шар"))

        response = receipt(engine, "supported", "unsourced")
        self.assertFalse(response.complete)
        self.assertFalse(response.grounded)
        self.assertEqual(response.answer_text, "")
        self.assertEqual(response.reason, "missing_evidence:unsourced")
        self.assertTrue(receipt(engine, "supported").complete)
        with self.assertRaisesRegex(ValueError, "no grounded claims"):
            receipt(engine)

    def test_explicit_correction_targets_one_claim_and_its_dependents(self) -> None:
        engine = RealDataEngine()
        for number in range(4):
            engine.register_evidence(
                stub_root(f"r{number}", f"g{number}", f"s{number}")
            )
        engine.add_claim(claim("z-old", roots=("r0",)))
        engine.add_claim(claim("x-unrelated", roots=("r1",), subject="шар"))
        engine.add_claim(claim("derived"), parent_claim_ids=("z-old",))
        before = receipt(engine, "z-old")
        legacy = engine.to_dict()
        legacy.pop("corrections")
        legacy.pop("obsolete_sources")

        self.assertEqual(
            set(
                engine.correct_claim(
                    "z-old", claim("a-new", roots=("r2",), place="полка")
                )
            ),
            {"z-old", "derived"},
        )
        self.assertFalse(receipt(engine, "z-old").complete)
        self.assertFalse(receipt(engine, "derived").complete)
        self.assertTrue(receipt(engine, "x-unrelated").complete)
        self.assertTrue(receipt(engine, "a-new").complete)
        self.assertTrue(before.complete)
        with self.assertRaisesRegex(ValueError, "corrected"):
            engine.add_claim(claim("cannot-reuse"), parent_claim_ids=("derived",))

        engine.correct_claim("a-new", claim("b-newer", roots=("r3",), place="сумка"))
        restored = RealDataEngine.from_dict(engine.to_dict())
        self.assertEqual(restored.to_dict(), engine.to_dict())
        self.assertTrue(receipt(restored, "b-newer").complete)
        self.assertFalse(receipt(restored, "z-old").complete)
        self.assertFalse(receipt(restored, "a-new").complete)
        self.assertTrue(receipt(restored, "x-unrelated").complete)

        # Existing snapshots without a correction field still restore.
        self.assertTrue(
            RealDataEngine.from_dict(legacy)
            .receipt(
                question_id="q",
                answer_text="old snapshot",
                claim_ids=("z-old",),
                model_version="open-dev",
            )
            .complete
        )

    def test_invalid_correction_does_not_partially_change_engine(self) -> None:
        engine = RealDataEngine()
        engine.register_evidence(stub_root("r1", "g1", "s1"))
        engine.add_claim(claim("old", roots=("r1",)))
        before = engine.to_dict()
        with self.assertRaisesRegex(ValueError, "unknown correction target"):
            engine.correct_claim("wrong", claim("new", roots=("r1",)))
        with self.assertRaisesRegex(ValueError, "direct source evidence"):
            engine.correct_claim("old", claim("new"))
        with self.assertRaisesRegex(ValueError, "unknown evidence root"):
            engine.correct_claim("old", claim("new", roots=("missing",)))
        wrong_revision = SourceSlice("s1", 2, 0, 1, "0" * 64)
        with self.assertRaisesRegex(ValueError, "different source revision"):
            engine.correct_claim(
                "old", claim("new", roots=("r1",), source=wrong_revision)
            )
        self.assertEqual(before, engine.to_dict())

    def test_scope_and_slice_must_match_archive_before_the_answer(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            ObservationArchive(Path(directory) / "source.db", create=True) as archive,
        ):
            record = archive.import_text(
                "Ключ теперь на полке.",
                namespace="dev",
                external_key="task-1",
                group_id="family-1",
                metadata={"access_scope": "restricted"},
            )
            span = source_slice(archive, record.source_id, 1, 0, 4)
            root = evidence_root(archive, span, root_id="r1")
            forged = SourceSlice(
                span.source_id,
                span.source_version,
                span.start,
                span.end,
                "0" * 64,
                access_scope=span.access_scope,
            )
            with self.assertRaisesRegex(ValueError, "hash or access scope"):
                evidence_root(archive, forged, root_id="forged")
            wrong_scope = SourceSlice(
                span.source_id,
                span.source_version,
                span.start,
                span.end,
                span.sha256,
                access_scope="default",
            )
            with self.assertRaisesRegex(ValueError, "hash or access scope"):
                evidence_root(archive, wrong_scope, root_id="forged")

        engine = RealDataEngine()
        engine.register_evidence(root)
        engine.add_claim(claim("private", roots=("r1",), source=span))
        forged_same_version = replace(span, sha256="0" * 64)
        before_forgery = engine.to_dict()
        with self.assertRaisesRegex(ValueError, "source revision or scope"):
            engine.add_claim(
                claim("forged-slice", roots=("r1",), source=forged_same_version)
            )
        self.assertEqual(engine.to_dict(), before_forgery)
        forged_access = replace(span, access_scope="default")
        with self.assertRaisesRegex(ValueError, "source revision or scope"):
            engine.add_claim(
                claim("forged-access", roots=("r1",), source=forged_access)
            )
        self.assertEqual(
            receipt(engine, "private").reason, "source_access_denied:private"
        )
        # A root from the same private archive must not authorize a claim
        # whose own pinned SourceSlice has been omitted.
        unbound = RealDataEngine()
        unbound.register_evidence(root)
        unbound.add_claim(replace(claim("hidden", roots=("r1",)), source=None))
        self.assertEqual(
            receipt(unbound, "hidden").reason,
            "source_binding_missing:hidden",
        )
        self.assertTrue(
            receipt(engine, "private", allowed_scopes=("restricted",)).complete
        )
        with self.assertRaisesRegex(ValueError, "allowed_scopes"):
            receipt(engine, "private", allowed_scopes=())

        legacy = cast(dict[str, Any], engine.to_dict())
        legacy.pop("obsolete_sources")
        legacy.pop("corrections")
        legacy_root = legacy["evidence"]["roots"][0]
        del legacy_root["access_scope"]
        del legacy_root["source"]
        restored_old = RealDataEngine.from_dict(legacy)
        self.assertEqual(
            receipt(restored_old, "private", allowed_scopes=("restricted",)).reason,
            "source_binding_missing:private",
        )

        engine.add_claim(
            claim("derived", source=span),
            parent_claim_ids=("private",),
        )
        self.assertFalse(receipt(engine, "derived").complete)
        self.assertTrue(
            receipt(
                engine, "derived", allowed_scopes=("restricted", "default")
            ).complete
        )

    def test_source_revocation_invalidates_only_dependent_claims(self) -> None:
        engine = RealDataEngine()
        engine.register_evidence(stub_root("r-old", "family", "doc", 1))
        engine.register_evidence(stub_root("r-new", "family", "doc", 2))
        engine.register_evidence(stub_root("r-other", "other", "elsewhere"))
        old = Claim(
            "old",
            "place",
            (),
            ClaimStatus.OBSERVED,
            source=SourceSlice("doc", 1, 0, 1, "0" * 64),
            evidence_roots=("r-old",),
        )
        new = Claim(
            "new",
            "place",
            (),
            ClaimStatus.OBSERVED,
            source=SourceSlice("doc", 2, 0, 1, "0" * 64),
            evidence_roots=("r-new",),
        )
        other = Claim(
            "other",
            "place",
            (),
            ClaimStatus.OBSERVED,
            source=SourceSlice("elsewhere", 1, 0, 1, "0" * 64),
            evidence_roots=("r-other",),
        )
        engine.add_claim(old)
        engine.add_claim(new)
        engine.add_claim(other)
        historical = receipt(engine, "old")
        self.assertEqual(engine.invalidate_source("doc", 1), ("old",))
        self.assertTrue(historical.complete)
        self.assertFalse(receipt(engine, "old").complete)
        self.assertTrue(receipt(engine, "new").complete)
        self.assertTrue(receipt(engine, "other").complete)
        with self.assertRaisesRegex(ValueError, "invalidated"):
            engine.add_claim(claim("stale", roots=("r-old",)))
        engine.correct_claim(
            "old",
            Claim(
                "correction-after-revocation",
                "place",
                (),
                ClaimStatus.OBSERVED,
                source=SourceSlice("doc", 2, 0, 1, "0" * 64),
                evidence_roots=("r-new",),
            ),
        )
        self.assertTrue(receipt(engine, "correction-after-revocation").complete)
        restored = RealDataEngine.from_dict(engine.to_dict())
        self.assertEqual(restored.to_dict(), engine.to_dict())
        self.assertFalse(receipt(restored, "old").complete)

    def test_forecast_is_not_counted_as_grounded_fact(self) -> None:
        engine = RealDataEngine()
        engine.register_evidence(stub_root("r1", "g1", "s1"))
        engine.add_claim(claim("observed", roots=("r1",)))
        engine.add_claim(claim("forecast", roots=("r1",), status=ClaimStatus.PREDICTED))
        self.assertTrue(receipt(engine, "observed").complete)
        self.assertFalse(receipt(engine, "forecast").complete)
        self.assertFalse(receipt(engine, "forecast").grounded)


if __name__ == "__main__":
    unittest.main()

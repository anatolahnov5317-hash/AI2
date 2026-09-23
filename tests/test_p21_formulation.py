"""Open engineering checks; no Russian pilot/semantic faithfulness claim."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from text_factors.observations import ObservationArchive
from text_factors.real_data import (
    Claim,
    ClaimStatus,
    RealDataEngine,
    RoleValue,
    evidence_root,
    source_slice,
)
from text_factors.real_data.formulation import (
    CalibrationCase,
    GuardedFormulator,
    PlanQuote,
    StyleExample,
    SurfaceStyle,
    VerifiedContentPlan,
    compare_to_excerpts,
    plan_from_dialogue,
)


class FormulationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = ObservationArchive(
            Path(self.temp.name) / "quotes.sqlite", create=True
        )
        self.addCleanup(self.archive.close)
        self.engine = RealDataEngine()
        self.quotes: list[PlanQuote] = []
        for index, statement in enumerate(
            ("Ключ у Нины, код 49", "Шар у Олега, код 72")
        ):
            source = self.archive.import_text(
                statement,
                namespace="open-development",
                external_key=f"doc-{index}",
                group_id=f"family-{index}",
                metadata={"access_scope": "default"},
            )
            pin = source_slice(
                self.archive, source.source_id, source.version, 0, len(statement)
            )
            root_id = f"root-{index}"
            self.engine.register_evidence(
                evidence_root(self.archive, pin, root_id=root_id)
            )
            claim_id = f"claim-{index}"
            self.engine.add_claim(
                Claim(
                    claim_id=claim_id,
                    relation_id="owner",
                    arguments=(RoleValue("entity", f"item-{index}"),),
                    status=ClaimStatus.ASSERTED,
                    source=pin,
                    evidence_roots=(root_id,),
                )
            )
            self.quotes.append(PlanQuote(claim_id, statement, pin, (root_id,)))
        self.receipt = self.engine.receipt(
            question_id="q1",
            answer_text="checked",
            claim_ids=tuple(q.claim_id for q in self.quotes),
            model_version="v-open",
        )
        self.assertTrue(self.receipt.complete)
        self.plan = VerifiedContentPlan(self.receipt, tuple(self.quotes))
        self.allowed = True

    def reader(self, quote: PlanQuote) -> str:
        if not self.allowed:
            raise PermissionError("revoked")
        return self.archive.read_span(
            quote.source.source_id,
            quote.source.source_version,
            quote.source.start,
            quote.source.end,
        )

    def fitted(self) -> GuardedFormulator:
        learner = GuardedFormulator()
        style = SurfaceStyle("По источнику: ", ". Также указано: ", ".")
        learner.fit(
            (
                StyleExample("train-a", style, True, 0.7),
                StyleExample("train-b", style, True, 0.9),
            )
        )
        self.assertEqual(learner.style, style)
        return learner

    def test_fit_requires_independent_positive_groups_and_known_labels(self) -> None:
        learner = GuardedFormulator()
        style = SurfaceStyle("В материалах указано: ", "; ", ".")
        learner.fit(
            tuple(StyleExample("one-family", style, True, 1.0) for _ in range(50))
        )
        self.assertIsNone(learner.style)
        learner.fit(
            (StyleExample("one", style, True), StyleExample("two", style, None))
        )
        self.assertIsNone(learner.style)
        learner.fit(
            (
                StyleExample("one", style, True),
                StyleExample("two", style, True),
                StyleExample("three", style, False),
            )
        )
        self.assertIsNone(learner.style)

    def test_no_free_answer_before_disjoint_calibration(self) -> None:
        learner = self.fitted()
        before = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual(before.mode, "excerpt")
        self.assertTrue(before.text.startswith("«Ключ у Нины"))
        assert before.receipt is not None
        self.assertEqual(before.receipt.evidence_roots, self.receipt.evidence_roots)
        with self.assertRaisesRegex(ValueError, "overlap"):
            learner.calibrate((CalibrationCase("train-a", ("x",), "x", False, False),))

    def test_safe_learned_frame_copies_each_quote_and_keeps_receipt(self) -> None:
        learner = self.fitted()
        result = learner.calibrate(
            (
                CalibrationCase(
                    "cal-1",
                    ("Ключ у Васи 1",),
                    "По источнику: Ключ у Васи 1.",
                    False,
                    False,
                ),
                CalibrationCase(
                    "cal-2",
                    ("Шар у Лены 2",),
                    "По источнику: Шар у Лены 2.",
                    False,
                    False,
                ),
            )
        )
        self.assertEqual(result.paired_groups, 2)
        self.assertTrue(result.free_enabled)
        draft = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual(draft.mode, "free_frame")
        self.assertEqual(
            draft.text,
            "По источнику: Ключ у Нины, код 49. Также указано: Шар у Олега, код 72.",
        )
        assert draft.receipt is not None
        self.assertTrue(draft.receipt.complete)
        self.assertEqual(draft.receipt.state_version, self.receipt.state_version)
        self.assertEqual(draft.receipt.claim_ids, self.receipt.claim_ids)
        self.assertEqual(draft.receipt.evidence_roots, self.receipt.evidence_roots)

    def test_unseen_names_numbers_and_polarity_are_reviewed_separately(self) -> None:
        result = compare_to_excerpts(
            (
                CalibrationCase(
                    "new-1", ("Нина держит 49",), "Ира держит 50", False, True
                ),
                CalibrationCase(
                    "new-1", ("Нина держит 49",), "Ира держит 50", False, True
                ),
                CalibrationCase(
                    "new-2", ("Нина держит 49",), "Нина не держит 49", False, True
                ),
                CalibrationCase(
                    "unknown", ("Нина держит 49",), "Нина держит 49", None, None
                ),
            )
        )
        self.assertEqual(result.paired_groups, 2)
        self.assertEqual(result.free_unsupported_groups, 2)
        self.assertEqual(result.new_name_groups, 1)
        self.assertEqual(result.new_number_groups, 1)
        self.assertFalse(result.free_enabled)

    def test_degradation_switches_to_excerpts_without_reenabling(self) -> None:
        learner = self.fitted()
        okay = (
            CalibrationCase(
                "cal-1", ("Нина 49",), "По источнику: Нина 49.", False, False
            ),
            CalibrationCase(
                "cal-2", ("Олег 72",), "По источнику: Олег 72.", False, False
            ),
        )
        self.assertTrue(learner.calibrate(okay).free_enabled)
        bad = okay + (
            CalibrationCase("cal-3", ("Ира 28",), "По источнику: Ира 28.", False, True),
        )
        self.assertFalse(learner.calibrate(bad).free_enabled)
        self.assertTrue(learner.calibrate(okay).free_enabled)
        draft = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual(draft.mode, "excerpt")

    def test_calibration_cannot_measure_a_different_candidate(self) -> None:
        learner = self.fitted()
        with self.assertRaisesRegex(ValueError, "differs from trained style"):
            learner.calibrate(
                (
                    CalibrationCase("cal-1", ("Нина 49",), "Нина не 49", False, True),
                    CalibrationCase("cal-2", ("Олег 72",), "Олег не 72", False, True),
                )
            )

    def test_quote_plan_missing_or_mismatched_claim_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "every receipt claim"):
            VerifiedContentPlan(self.receipt, tuple(self.quotes[:1]))
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            PlanQuote("x", "test", self.quotes[0].source, ("other-root",))
            VerifiedContentPlan(
                self.receipt,
                (
                    PlanQuote(
                        "claim-0",
                        self.quotes[0].text,
                        self.quotes[0].source,
                        ("other-root",),
                    ),
                    self.quotes[1],
                ),
            )
        with self.assertRaisesRegex(ValueError, "unverified factual text"):
            SurfaceStyle("Не находится: ", "; ", ".")

    def test_dialogue_adapter_requires_a_published_answer(self) -> None:
        answer = SimpleNamespace(
            status="answered",
            state_version=self.receipt.state_version,
            receipt=self.receipt,
            quotes=tuple(self.quotes),
        )
        self.assertEqual(plan_from_dialogue(answer), self.plan)
        answer.status = "clarify"
        with self.assertRaisesRegex(ValueError, "published grounded"):
            plan_from_dialogue(answer)
        answer.status = "answered"
        answer.state_version = "superseded"
        with self.assertRaisesRegex(ValueError, "published grounded"):
            plan_from_dialogue(answer)

    def test_revocation_or_correction_blocks_old_plan(self) -> None:
        learner = self.fitted()
        self.allowed = False
        result = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual(result.mode, "blocked")
        self.assertEqual(result.text, "")
        self.allowed = True
        source = self.quotes[0].source
        self.engine.invalidate_source(source.source_id, source.source_version)
        result = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual(result.mode, "blocked")
        self.assertEqual(result.text, "")

    def test_changed_text_or_lost_scope_blocks_and_does_not_publish(self) -> None:
        learner = self.fitted()
        draft = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("secret",),
            read_authorized_quote=self.reader,
        )
        self.assertEqual((draft.mode, draft.text), ("blocked", ""))
        draft = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=lambda quote: quote.text + " 71",
        )
        self.assertEqual((draft.mode, draft.text), ("blocked", ""))

    def test_change_during_render_discards_intermediate_answer(self) -> None:
        learner = self.fitted()
        fired = False

        def reader(quote: PlanQuote) -> str:
            nonlocal fired
            if not fired:
                fired = True
                self.engine.invalidate_source(
                    quote.source.source_id, quote.source.source_version
                )
            return self.reader(quote)

        draft = learner.draft(
            self.plan,
            self.engine,
            allowed_scopes=("default",),
            read_authorized_quote=reader,
        )
        self.assertEqual((draft.mode, draft.text), ("blocked", ""))


if __name__ == "__main__":
    unittest.main()

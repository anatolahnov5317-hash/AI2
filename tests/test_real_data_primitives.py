"""Tests for the isolated real-data integration branch primitives."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from text_factors.observations import ObservationArchive
from text_factors.real_data import (
    BudgetExceeded,
    BudgetTracker,
    Claim,
    ClaimStatus,
    CompositionalEncoder,
    ContextRegistry,
    DependencyGraph,
    EvidenceLedger,
    EvidenceRoot,
    HashedAttentionRanker,
    LearningEpisode,
    PilotSample,
    RealDataEngine,
    ResourceBudget,
    RoleValue,
    SourceSlice,
    UncertaintyIndex,
    UncertaintyScope,
    evaluate_pilot,
    evidence_root,
    load_state,
    save_state,
    source_slice,
)


def _claim(
    claim_id: str,
    *,
    relation: str = "transfer",
    agent: str = "a",
    target: str = "b",
    status: ClaimStatus = ClaimStatus.ASSERTED,
    roots: tuple[str, ...] = (),
) -> Claim:
    return Claim(
        claim_id=claim_id,
        relation_id=relation,
        arguments=(
            RoleValue("agent", agent),
            RoleValue("target", target),
        ),
        status=status,
        evidence_roots=roots,
    )


class RealDataPrimitiveTests(unittest.TestCase):
    def test_compositional_encoding_preserves_role_direction(self):
        encoder = CompositionalEncoder(width=512, active_bits_per_atom=6, seed=7)
        left = encoder.encode_claim(_claim("c1", agent="alice", target="book"))
        right = encoder.encode_claim(_claim("c2", agent="book", target="alice"))
        self.assertNotEqual(left.exact_hash, right.exact_hash)
        self.assertNotEqual(left.lexical.bits, right.lexical.bits)

    def test_new_identifiers_share_structural_representation(self):
        encoder = CompositionalEncoder(width=512, active_bits_per_atom=6, seed=7)
        first = encoder.encode_claim(_claim("c1", agent="alice", target="book"))
        second = encoder.encode_claim(_claim("c2", agent="ira", target="key"))
        self.assertNotEqual(first.lexical.bits, second.lexical.bits)
        self.assertEqual(first.structural.bits, second.structural.bits)

    def test_budget_stops_and_snapshot_resumes(self):
        now = [100.0]

        def clock() -> float:
            return now[0]

        tracker = BudgetTracker(
            ResourceBudget(
                max_steps=1,
                max_items=10,
                max_bytes=10,
                max_wall_seconds=10.0,
                checkpoint_every_steps=1,
            ),
            clock=clock,
        )
        tracker.consume(steps=1)
        self.assertTrue(tracker.should_checkpoint())
        checkpoint = tracker.mark_checkpoint()
        self.assertEqual(checkpoint.last_checkpoint_step, 1)
        with self.assertRaises(BudgetExceeded) as raised:
            tracker.consume(steps=1)
        self.assertEqual(raised.exception.reason, "max_steps")
        resumed = BudgetTracker(
            ResourceBudget(max_steps=4, checkpoint_every_steps=1),
            snapshot=raised.exception.snapshot,
            clock=clock,
        )
        resumed.consume(steps=1)
        self.assertEqual(resumed.steps, 2)

    def test_budget_wall_clock_is_checked(self):
        now = [1.0]
        tracker = BudgetTracker(
            ResourceBudget(max_wall_seconds=2.0),
            clock=lambda: now[0],
        )
        now[0] = 4.0
        with self.assertRaises(BudgetExceeded) as raised:
            tracker.check()
        self.assertEqual(raised.exception.reason, "max_wall_seconds")

    def test_open_attention_learns_without_domain_feature_names(self):
        ranker = HashedAttentionRanker(dimension=128, seed=3)
        ranker.fit(
            [
                AttentionExample("g1", "a", {"signal-x": 1.0}, 1),
                AttentionExample("g1", "b", {"signal-y": 1.0}, 0),
                AttentionExample("g2", "c", {"signal-x": 1.0}, 1),
                AttentionExample("g2", "d", {"signal-y": 1.0}, 0),
            ],
            steps=200,
        )
        ranked = ranker.rank(
            [
                AttentionCandidate("positive", {"signal-x": 1.0}),
                AttentionCandidate("negative", {"signal-y": 1.0}),
            ]
        )
        self.assertEqual(ranked[0][0], "positive")
        restored = HashedAttentionRanker.from_dict(ranker.to_dict())
        self.assertEqual(restored.to_dict(), ranker.to_dict())

    def test_context_registry_learns_reusable_and_distinct_transforms(self):
        registry = ContextRegistry(width=64, assignment_threshold=0.5)
        first = LearningEpisode("e1", "g1", (1, 2), (10, 11))
        second = LearningEpisode("e2", "g2", (1, 2), (10, 11))
        third = LearningEpisode("e3", "g3", (1, 2), (20, 21))
        ctx1, created1, _ = registry.learn(first)
        ctx1b, created2, score2 = registry.learn(second)
        ctx2, created3, score3 = registry.learn(third)
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(ctx1, ctx1b)
        self.assertGreaterEqual(score2, 0.5)
        self.assertTrue(created3)
        self.assertNotEqual(ctx1, ctx2)
        self.assertLess(score3, 0.5)
        self.assertEqual(registry.reliable_context_ids(), (ctx1,))

    def test_context_affinity_uses_active_responses_not_shared_inactivity(self):
        registry = ContextRegistry(width=64, assignment_threshold=0.5)
        ctx1, _, _ = registry.learn(LearningEpisode("e1", "g1", (1,), (10,)))
        ctx2, _, _ = registry.learn(LearningEpisode("e2", "g2", (1,), (20,)))
        self.assertEqual(registry.affinity(ctx1, ctx2), 0.0)
        registry.record_response(ctx1, "probe", 0.9)
        registry.record_response(ctx2, "probe", 0.8)
        self.assertEqual(registry.affinity(ctx1, ctx2), 1.0)
        maxima = registry.local_maxima(
            (1,),
            scorer=lambda _context, predicted: 1.0 if predicted else 0.0,
            suppression_affinity=0.8,
        )
        self.assertEqual(len(maxima), 1)

    def test_evidence_derivations_do_not_create_independent_support(self):
        ledger = EvidenceLedger()
        ledger.register_root(EvidenceRoot("r1", "group-a", "s1", 1))
        ledger.register_root(EvidenceRoot("r2", "group-a", "s2", 1))
        ledger.register_root(EvidenceRoot("r3", "group-b", "s3", 1))
        ledger.attach("c1", ("r1", "r2"))
        ledger.derive("c2", ("c1",))
        self.assertEqual(ledger.support_count("c2"), 2)
        self.assertEqual(ledger.independent_support("c2"), 1)
        ledger.attach("c2", ("r3",))
        self.assertEqual(ledger.independent_support("c2"), 2)

    def test_uncertainty_is_scoped_and_resolved_narrowly(self):
        index = UncertaintyIndex()
        claim1 = _claim("c1", relation="location", agent="item", target="shelf")
        claim2 = _claim("c2", relation="owner", agent="item", target="ira")
        index.add(
            UncertaintyScope(
                "u1",
                "reported reshuffle",
                affected_claim_ids=("c1",),
            )
        )
        self.assertTrue(index.is_uncertain(claim1))
        self.assertFalse(index.is_uncertain(claim2))
        self.assertEqual(index.resolve_claim("c1"), ("u1",))
        self.assertFalse(index.is_uncertain(claim1))

    def test_dependency_revision_is_deterministic_and_resumable(self):
        graph = DependencyGraph()
        graph.add_dependency("a", "b")
        graph.add_dependency("a", "d")
        graph.add_dependency("b", "c")
        checkpoint = graph.start_revision(("a",), state_version="v1")
        self.assertEqual(checkpoint.queue, ("a", "b", "d", "c"))
        handled: list[str] = []
        first = graph.process(
            checkpoint,
            handler=handled.append,
            budget=BudgetTracker(ResourceBudget(max_steps=1)),
            batch_size=4,
        )
        self.assertFalse(first.complete)
        self.assertEqual(first.stop_reason, "max_steps")
        self.assertEqual(handled, ["a"])
        second = graph.process(
            first.checkpoint,
            handler=handled.append,
            budget=BudgetTracker(ResourceBudget(max_steps=10)),
            batch_size=8,
        )
        self.assertTrue(second.complete)
        self.assertEqual(handled, ["a", "b", "d", "c"])

    def test_engine_never_promotes_prediction_to_grounded_fact(self):
        engine = RealDataEngine()
        engine.register_evidence(EvidenceRoot("r1", "g1", "s1", 1))
        engine.add_claim(_claim("observed", roots=("r1",)))
        engine.add_claim(
            _claim(
                "prediction",
                status=ClaimStatus.PREDICTED,
                roots=("r1",),
            )
        )
        grounded = engine.receipt(
            question_id="q1",
            answer_text="supported",
            claim_ids=("observed",),
            model_version="m1",
        )
        predicted = engine.receipt(
            question_id="q2",
            answer_text="should not escape",
            claim_ids=("prediction",),
            model_version="m1",
        )
        self.assertTrue(grounded.complete)
        self.assertTrue(grounded.grounded)
        self.assertFalse(predicted.complete)
        self.assertEqual(predicted.answer_text, "")

    def test_engine_blocks_only_the_uncertain_claim(self):
        engine = RealDataEngine()
        engine.register_evidence(EvidenceRoot("r1", "g1", "s1", 1))
        engine.register_evidence(EvidenceRoot("r2", "g2", "s2", 1))
        engine.add_claim(_claim("c1", relation="location", roots=("r1",)))
        engine.add_claim(_claim("c2", relation="owner", roots=("r2",)))
        engine.mark_uncertainty(
            UncertaintyScope("u1", "possible move", affected_claim_ids=("c1",))
        )
        first = engine.receipt(
            question_id="q1",
            answer_text="x",
            claim_ids=("c1",),
            model_version="m1",
        )
        second = engine.receipt(
            question_id="q2",
            answer_text="y",
            claim_ids=("c2",),
            model_version="m1",
        )
        self.assertFalse(first.complete)
        self.assertTrue(second.complete)

    def test_archive_bridge_keeps_source_group_and_exact_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obs.sqlite"
            with ObservationArchive(path, create=True) as archive:
                record = archive.import_text(
                    "Первый ключ. Второй ключ.",
                    namespace="test",
                    external_key="doc",
                    group_id="family-1",
                    metadata={"access_scope": "private-test"},
                )
                start = len("Первый ")
                end = start + len("ключ")
                span = source_slice(
                    archive, record.source_id, record.version, start, end
                )
                root = evidence_root(archive, span, root_id="root-1")
                self.assertEqual(
                    archive.read_span(record.source_id, record.version, start, end),
                    "ключ",
                )
                self.assertEqual(span.access_scope, "private-test")
                self.assertEqual(root.group_id, "family-1")

    def test_pilot_gate_matches_zero_error_300_case(self):
        samples = tuple(
            PilotSample(resolvable=True, answered=True, grounded=True)
            for _ in range(300)
        )
        report = evaluate_pilot(samples)
        self.assertLess(report["ungrounded_upper_bound"], 0.01)
        self.assertEqual(report["resolvable_coverage"], 1.0)
        self.assertTrue(report["passed"])

    def test_real_data_state_roundtrip_preserves_contexts_and_evidence(self):
        engine = RealDataEngine()
        engine.register_evidence(EvidenceRoot("r1", "g1", "s1", 1))
        engine.add_claim(_claim("c1", roots=("r1",)))
        contexts = ContextRegistry(width=64)
        contexts.learn(LearningEpisode("e1", "g1", (1,), (10,)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_state(path, engine, contexts=contexts, model_version="m1")
            restored, restored_contexts, version = load_state(path)
        self.assertEqual(version, "m1")
        self.assertEqual(restored.to_dict(), engine.to_dict())
        self.assertIsNotNone(restored_contexts)
        assert restored_contexts is not None
        self.assertEqual(restored_contexts.to_dict(), contexts.to_dict())

    def test_real_data_state_detects_corruption(self):
        engine = RealDataEngine()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            save_state(path, engine, model_version="m1")
            value = path.read_text(encoding="utf-8").replace(
                '"model_version":"m1"',
                '"model_version":"m2"',
            )
            path.write_text(value, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_state(path)

    def test_source_slice_rejects_invalid_offsets(self):
        with self.assertRaises(ValueError):
            SourceSlice("s1", 1, 3, 3, "abc")


if __name__ == "__main__":
    unittest.main()

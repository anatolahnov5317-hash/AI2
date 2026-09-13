import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from text_factors import ModelConfig, TextFactorModel
from text_factors.memory import CombinatorialMemory
from text_factors.recognition import (
    ClusterEvidence,
    ContextView,
    InterpretationClaim,
    RecognitionCandidate,
    RecognitionLimits,
    recognize_views,
    relate_candidates,
)


def memory_config(**overrides: Any) -> ModelConfig:
    values: dict[str, Any] = {
        "input_bits": 8,
        "active_bits_per_symbol": 2,
        "positions": 3,
        "frame_size": 2,
        "context_count": 2,
        "receptive_bits": 3,
        "point_count": 3,
        "output_bits": 2,
        "create_threshold": 2,
        "activation_threshold": 2,
        "min_active_points": 1,
        "probation_after": 2,
        "stable_after": 3,
        "max_clusters_per_point": 4,
        "seed": 17,
    }
    values.update(overrides)
    return ModelConfig(**values)


def trained_memory() -> CombinatorialMemory:
    memory = CombinatorialMemory(
        memory_config(),
        receptors=np.asarray([[0, 1, 2], [3, 4, 5], [5, 6, 7]], dtype=np.int32),
        output_map=np.asarray([0, 0, 1], dtype=np.int32),
    )
    strong = np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.bool_)
    weak = np.asarray([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.bool_)
    for _ in range(3):
        memory.observe(strong)
        memory.observe(weak)
    return memory


def bits(*active: int) -> np.ndarray:
    value = np.zeros(8, dtype=np.bool_)
    value[list(active)] = True
    return value


def memory_snapshot(memory: CombinatorialMemory) -> tuple[Any, ...]:
    clusters = tuple(
        (
            point,
            cluster.signature,
            tuple(int(value) for value in cluster.bit_hits),
            cluster.created_at,
            cluster.last_seen,
            int(cluster.status),
            cluster.partial_hits,
            cluster.exact_hits,
            cluster.partial_errors,
            cluster.complete_errors,
            tuple(
                tuple(bool(value) for value in row)
                for row in cluster.activation_history
            ),
        )
        for point, cluster in memory.iter_clusters()
    )
    return memory.step, memory.stats(), clusters


def candidate(
    candidate_id: str,
    *,
    signature: tuple[int, ...] = (0, 1),
    positions: tuple[int, ...] = (0,),
    claims: tuple[InterpretationClaim, ...] = (),
) -> RecognitionCandidate:
    evidence = ClusterEvidence(0, signature, signature, 3, 0)
    return RecognitionCandidate(
        candidate_id=candidate_id,
        context_id="ctx",
        content_key="content:" + ",".join(map(str, signature)),
        output_bits=(0,),
        source_positions=positions,
        evidence=(evidence,),
        familiarity=1.0,
        quality=len(signature),
        active_points=1,
        observation_id="observation",
        claims=claims,
    )


class RecognitionTests(unittest.TestCase):
    def test_recognition_does_not_mutate_memory_and_preserves_legacy_result(
        self,
    ) -> None:
        memory = trained_memory()
        model = TextFactorModel(memory.config, alphabet="ab", memory=memory)
        active = bits(0, 1, 2, 3, 4)
        legacy_before = memory.read(active)
        transform_before = model.transform_window("ab").to_dict()
        before = memory_snapshot(memory)

        result = recognize_views(
            memory,
            (ContextView("all", "ctx", active, (0, 1)),),
            total_views=1,
        )

        self.assertTrue(result.complete)
        self.assertEqual(memory_snapshot(memory), before)
        legacy_after = memory.read(active)
        np.testing.assert_array_equal(legacy_after.output, legacy_before.output)
        np.testing.assert_array_equal(
            legacy_after.point_indices, legacy_before.point_indices
        )
        self.assertEqual(legacy_after.quality, legacy_before.quality)
        self.assertEqual(model.transform_window("ab").to_dict(), transform_before)
        self.assertEqual(model.memory, memory)

    def test_views_keep_weaker_point_dropped_by_legacy_global_quality(self) -> None:
        memory = trained_memory()
        active = bits(0, 1, 2, 3, 4)

        legacy = memory.read(active)
        result = recognize_views(
            memory,
            (ContextView("all", "ctx", active, (0, 1)),),
            total_views=1,
        )

        self.assertEqual(tuple(int(p) for p in legacy.point_indices), (0,))
        self.assertEqual({e.point_index for e in result.candidates[0].evidence}, {0, 1})
        self.assertEqual(result.candidates[0].active_points, 2)

    def test_same_output_bit_with_different_signatures_is_not_duplicate(self) -> None:
        left = candidate("left", signature=(0, 1))
        right = candidate("right", signature=(3, 4))

        relation = relate_candidates(left, right)

        self.assertEqual(left.output_bits, right.output_bits)
        self.assertEqual(relation.kind, "undetermined")

    def test_same_evidence_and_same_origin_is_duplicate(self) -> None:
        memory = trained_memory()
        shared = bits(0, 1, 2)
        views = (
            ContextView("first", "a", shared, (4,), observation_id="obs"),
            ContextView("second", "b", shared, (4,), observation_id="obs"),
        )

        result = recognize_views(memory, views, total_views=2)

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.suppressed), 1)
        self.assertEqual(result.relations[0].kind, "duplicate")

    def test_same_content_at_disjoint_locations_is_compatible(self) -> None:
        left = candidate("left", positions=(1,))
        right = candidate("right", positions=(7,))

        relation = relate_candidates(left, right)

        self.assertEqual(left.content_key, right.content_key)
        self.assertEqual(relation.kind, "compatible")

    def test_conflict_requires_same_explicit_claim_key(self) -> None:
        opened = InterpretationClaim("world", "box", "state", "open")
        closed = InterpretationClaim("world", "box", "state", "closed")
        other_box = InterpretationClaim("world", "other-box", "state", "closed")

        self.assertEqual(
            relate_candidates(
                candidate("open", claims=(opened,)),
                candidate("closed", claims=(closed,)),
            ).kind,
            "conflict",
        )
        self.assertEqual(
            relate_candidates(
                candidate("plain-a"), candidate("plain-b", signature=(3, 4))
            ).kind,
            "undetermined",
        )
        self.assertEqual(
            relate_candidates(
                candidate("one", claims=(opened,)),
                candidate("other", claims=(other_box,)),
            ).kind,
            "undetermined",
        )

    def test_pairwise_relations_are_not_transitively_merged(self) -> None:
        open_claim = InterpretationClaim("world", "box", "state", "open")
        closed_claim = InterpretationClaim("world", "box", "state", "closed")
        a = candidate("a", positions=(0,), claims=(open_claim,))
        b = candidate("b", positions=(1,))
        c = candidate("c", positions=(2,), claims=(closed_claim,))

        relations = {
            (left.candidate_id, right.candidate_id): relate_candidates(left, right).kind
            for left, right in ((a, b), (b, c), (a, c))
        }

        self.assertEqual(relations[("a", "b")], "compatible")
        self.assertEqual(relations[("b", "c")], "compatible")
        self.assertEqual(relations[("a", "c")], "conflict")

    def test_limits_and_cancellation_are_visible(self) -> None:
        memory = trained_memory()
        strong = ContextView("strong", "ctx", bits(0, 1, 2), (0,))
        weak = ContextView("weak", "ctx", bits(3, 4), (1,))

        cancelled = recognize_views(
            memory,
            (strong,),
            total_views=1,
            cancelled=lambda: True,
        )
        self.assertFalse(cancelled.complete)
        self.assertEqual(cancelled.stop_reason, "cancelled")
        self.assertEqual(cancelled.examined_views, 0)

        view_limited = recognize_views(
            memory,
            (strong, weak),
            total_views=2,
            limits=RecognitionLimits(max_views=1),
        )
        self.assertFalse(view_limited.complete)
        self.assertEqual(view_limited.stop_reason, "view_limit")
        self.assertEqual(view_limited.examined_views, 1)

        evidence_limited = recognize_views(
            memory,
            (ContextView("both", "ctx", bits(0, 1, 2, 3, 4), (0, 1)),),
            total_views=1,
            limits=RecognitionLimits(max_evidence=1),
        )
        self.assertFalse(evidence_limited.complete)
        self.assertEqual(evidence_limited.stop_reason, "evidence_limit")

        candidate_limited = recognize_views(
            memory,
            (strong, weak),
            total_views=2,
            limits=RecognitionLimits(max_candidates=1),
        )
        self.assertFalse(candidate_limited.complete)
        self.assertEqual(candidate_limited.stop_reason, "candidate_limit")
        self.assertEqual(len(candidate_limited.candidates), 1)

    def test_save_load_preserves_recognition_content_keys(self) -> None:
        config = memory_config(
            input_bits=32,
            receptive_bits=8,
            point_count=64,
            output_bits=8,
            active_bits_per_symbol=4,
        )
        model = TextFactorModel(config, alphabet="ab")
        for _ in range(3):
            model.partial_fit_window("ab")
        expected = tuple(
            candidate.content_key
            for candidate in model.recognize_window("ab").candidates
        )
        self.assertTrue(expected)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = TextFactorModel.load(path)

        actual = tuple(
            candidate.content_key
            for candidate in loaded.recognize_window("ab").candidates
        )
        self.assertEqual(actual, expected)

    def test_recognize_text_reports_fixed_window_adapter_positions(self) -> None:
        config = memory_config(
            input_bits=32,
            receptive_bits=8,
            point_count=64,
            output_bits=8,
            active_bits_per_symbol=4,
        )
        model = TextFactorModel(config, alphabet="ab")
        for _ in range(3):
            model.partial_fit_window("ab")

        result = model.recognize_text("abab", stride=2)

        self.assertEqual(result.total_views, 4)
        self.assertEqual(result.examined_views, 4)
        self.assertTrue(
            all(
                set(item.source_positions) <= {0, 1, 2, 3} for item in result.candidates
            )
        )


if __name__ == "__main__":
    unittest.main()

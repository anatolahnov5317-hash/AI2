import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from text_factors.context_affinity import ContextAffinity, ContextAffinityConfig
from text_factors.recognition import (
    CandidateRelation,
    ClusterEvidence,
    InterpretationClaim,
    RecognitionCandidate,
    RecognitionResult,
)


def candidate(
    name: str,
    context: str,
    *,
    bits: tuple[int, ...] = (0, 1, 2),
    positions: tuple[int, ...] = (0,),
    point: int = 0,
    score: float = 1.0,
) -> RecognitionCandidate:
    return RecognitionCandidate(
        candidate_id=name,
        context_id=context,
        content_key="content-" + name,
        output_bits=(0,),
        source_positions=positions,
        evidence=(ClusterEvidence(point, bits, bits, 3, 0),),
        familiarity=score,
        quality=len(bits),
        active_points=1,
        observation_id="scene",
    )


def result(*candidates: RecognitionCandidate) -> RecognitionResult:
    return RecognitionResult(
        candidates=tuple(candidates),
        relations=(),
        complete=True,
        examined_views=len(candidates),
        total_views=len(candidates),
        encoding_id="local-encoding",
        memory_step=7,
    )


def learned() -> tuple[ContextAffinity, RecognitionResult]:
    affinity = ContextAffinity("local-encoding")
    sample = result(
        candidate("a", "first", bits=(0, 1, 2, 3), score=3.0),
        candidate("b", "second", bits=(0, 1, 2, 4), score=2.0),
    )
    for number in range(3):
        affinity.observe(f"event-{number}", sample)
    return affinity, sample


class ContextAffinityTests(unittest.TestCase):
    def test_only_learned_near_responses_are_suppressed(self) -> None:
        trained, sample = learned()
        fresh = ContextAffinity("local-encoding")
        self.assertEqual(len(fresh.select(sample).candidates), 2)
        selected = trained.select(sample)
        self.assertEqual([item.candidate_id for item in selected.candidates], ["a"])
        self.assertEqual([item.candidate_id for item in selected.suppressed], ["b"])
        self.assertEqual(selected.encoding_id, sample.encoding_id)
        self.assertEqual(selected.memory_step, sample.memory_step)
        self.assertEqual(selected.examined_views, sample.examined_views)
        self.assertTrue(selected.complete)

    def test_second_instance_is_never_suppressed(self) -> None:
        affinity, sample = learned()
        other = replace(
            sample.candidates[1], candidate_id="another", source_positions=(1,)
        )
        scene = result(*sample.candidates, other)
        selected = affinity.select(scene)
        self.assertEqual(
            {item.candidate_id for item in selected.candidates}, {"a", "another"}
        )
        for positions in ((1,), (0, 1), (1, 2), ()):
            with self.subTest(positions=positions):
                changed = replace(sample.candidates[1], source_positions=positions)
                self.assertEqual(
                    len(
                        affinity.select(
                            result(sample.candidates[0], changed)
                        ).candidates
                    ),
                    2,
                )
        other_scene = replace(sample.candidates[1], observation_id="another-scene")
        self.assertEqual(
            len(affinity.select(result(sample.candidates[0], other_scene)).candidates),
            2,
        )

    def test_separate_parts_do_not_train_affinity(self) -> None:
        affinity = ContextAffinity("local-encoding")
        scene = result(
            candidate("a", "same-1"), candidate("b", "same-2", positions=(1,))
        )
        for number in range(3):
            affinity.observe(str(number), scene)
        self.assertEqual(affinity.to_dict()["pairs"], [])
        self.assertEqual(len(affinity.select(scene).candidates), 2)

    def test_one_event_gives_one_exposure_even_with_multiple_locations(self) -> None:
        affinity = ContextAffinity("local-encoding")
        scene = result(
            candidate("a", "first"),
            candidate("b", "second"),
            candidate("c", "first", positions=(1,)),
        )
        scene = replace(
            scene,
            suppressed=(candidate("d", "second", positions=(1,)),),
            examined_views=4,
            total_views=4,
        )
        affinity.observe("one-observation", scene)
        self.assertEqual(len(affinity.to_dict()["pairs"]), 1)
        counts = affinity.to_dict()["pairs"][0]
        self.assertEqual((counts["exposures"], counts["positive"]), (1, 1))

    def test_current_factor_support_is_required(self) -> None:
        affinity, sample = learned()
        for changed in (
            candidate("b", "second", bits=(6, 7, 8)),
            candidate("b", "second", bits=(0, 1, 2, 3), point=1),
        ):
            self.assertEqual(
                len(affinity.select(result(sample.candidates[0], changed)).candidates),
                2,
            )

    def test_conflicts_from_claims_and_relations_survive(self) -> None:
        affinity, sample = learned()
        left, right = sample.candidates
        declared = replace(
            sample,
            relations=(CandidateRelation("a", "b", "conflict", "explicit conflict"),),
        )
        selected = affinity.select(declared)
        self.assertEqual(len(selected.candidates), 2)
        self.assertEqual(selected.relations, declared.relations)
        claim_left = InterpretationClaim("scene", "item", "shape", "round")
        claim_right = replace(claim_left, value="square")
        claimed = result(
            replace(left, claims=(claim_left,)), replace(right, claims=(claim_right,))
        )
        self.assertEqual(len(affinity.select(claimed).candidates), 2)
        untrained = ContextAffinity("local-encoding")
        for number in range(3):
            untrained.observe(str(number), declared)
        self.assertEqual(untrained.to_dict()["pairs"][0]["positive"], 0)

    def test_exposures_and_minimum_observations_gate_edges(self) -> None:
        affinity, sample = learned()
        negative = result(
            sample.candidates[0], candidate("b", "second", bits=(7, 8, 9))
        )
        affinity.observe("negative", negative)
        counts = affinity.to_dict()["pairs"][0]
        self.assertEqual((counts["positive"], counts["exposures"]), (3, 4))
        self.assertEqual(len(affinity.select(sample).candidates), 2)
        insufficient = ContextAffinity("local-encoding")
        for number in range(2):
            insufficient.observe(str(number), sample)
        self.assertEqual(len(insufficient.select(sample).candidates), 2)

    def test_greedy_selection_does_not_merge_transitive_components(self) -> None:
        affinity = ContextAffinity("local-encoding")
        a = candidate("a", "A", score=3.0)
        b = candidate("b", "B", score=2.0)
        c = candidate("c", "C", score=1.0)
        for number in range(3):
            affinity.observe(f"ab-{number}", result(a, b))
            affinity.observe(f"bc-{number}", result(b, c))
        selected = affinity.select(result(a, b, c))
        self.assertEqual(
            [item.candidate_id for item in selected.candidates], ["a", "c"]
        )
        self.assertEqual([item.candidate_id for item in selected.suppressed], ["b"])

    def test_partial_search_never_trains_and_remains_partial(self) -> None:
        affinity, sample = learned()
        partial = replace(
            sample, complete=False, total_views=3, stop_reason="view_limit"
        )
        before = affinity.to_dict()
        self.assertFalse(affinity.observe("partial", partial))
        selected = affinity.select(partial)
        self.assertEqual(selected.candidates, partial.candidates)
        self.assertEqual(selected.suppressed, partial.suppressed)
        self.assertEqual(selected.relations, partial.relations)
        self.assertFalse(selected.complete)
        self.assertEqual(selected.stop_reason, "view_limit")
        self.assertEqual(selected.total_views, 3)
        self.assertEqual(affinity.to_dict(), before)

    def test_event_retries_are_idempotent_and_conflicting_reuse_rejected(self) -> None:
        affinity, sample = learned()
        before = affinity.to_dict()
        self.assertFalse(affinity.observe("event-0", sample))
        self.assertEqual(affinity.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "different evidence"):
            affinity.observe("event-0", replace(sample, memory_step=8))
        self.assertEqual(affinity.to_dict(), before)

    def test_roundtrip_preserves_readout_and_selection_never_mutates(self) -> None:
        affinity, sample = learned()
        before = affinity.to_dict()
        input_before = sample.to_dict()
        restored = ContextAffinity.from_dict(json.loads(json.dumps(before)))
        self.assertEqual(restored.to_dict(), before)
        self.assertEqual(restored.select(sample), affinity.select(sample))
        self.assertEqual(affinity.to_dict(), before)
        self.assertEqual(sample.to_dict(), input_before)
        self.assertFalse(restored.observe("event-0", sample))

    def test_existing_suppression_is_preserved_and_relations_filtered(self) -> None:
        affinity, sample = learned()
        prior = candidate("prior", "prior-context")
        scene = replace(
            sample,
            suppressed=(prior,),
            relations=(
                CandidateRelation("a", "b", "undetermined", "overlap"),
                CandidateRelation("a", "prior", "duplicate", "exact repeat"),
            ),
        )
        selected = affinity.select(scene)
        self.assertEqual(
            [item.candidate_id for item in selected.suppressed], ["prior", "b"]
        )
        self.assertEqual(selected.relations, ())

    def test_cancel_and_time_budget_do_not_commit_partial_training(self) -> None:
        affinity, sample = learned()
        before = affinity.to_dict()
        with self.assertRaises(InterruptedError):
            affinity.observe("cancelled", sample, cancelled=lambda: True)
        selected = affinity.select(sample, cancelled=lambda: True)
        self.assertFalse(selected.complete)
        self.assertEqual(selected.candidates, sample.candidates)
        self.assertEqual(selected.stop_reason, "context_affinity_cancelled")
        with patch(
            "text_factors.context_affinity.perf_counter", side_effect=(0.0, 3.0)
        ):
            timed = affinity.select(sample)
        self.assertEqual(timed.stop_reason, "context_affinity_time_budget")
        self.assertEqual(affinity.to_dict(), before)

    def test_expensive_event_digest_checks_budget_before_idempotent_return(
        self,
    ) -> None:
        affinity, sample = learned()
        before = affinity.to_dict()
        clock = [0.0]
        dumps = json.dumps

        def delayed_dump(
            value: object, *, sort_keys: bool, separators: tuple[str, str]
        ) -> str:
            clock[0] = 3.0
            return dumps(value, sort_keys=sort_keys, separators=separators)

        with (
            patch(
                "text_factors.context_affinity.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch("text_factors.context_affinity.json.dumps", side_effect=delayed_dump),
            self.assertRaises(InterruptedError),
        ):
            affinity.observe("event-0", sample)
        self.assertEqual(affinity.to_dict(), before)

    def test_capacity_rejection_is_atomic(self) -> None:
        _, sample = learned()
        for config, initial, next_result in (
            (ContextAffinityConfig(min_observations=1, max_events=1), sample, sample),
            (
                ContextAffinityConfig(max_pairs=1),
                sample,
                result(sample.candidates[0], candidate("c", "third")),
            ),
        ):
            affinity = ContextAffinity("local-encoding", config)
            affinity.observe("first", initial)
            before = affinity.to_dict()
            with self.assertRaises(ValueError):
                affinity.observe("overflow", next_result)
            self.assertEqual(affinity.to_dict(), before)
        small = ContextAffinity(
            "local-encoding", ContextAffinityConfig(max_candidates=1)
        )
        with self.assertRaises(ValueError):
            small.observe("too-many", sample)
        self.assertEqual(small.to_dict()["events"], [])

    def test_restored_state_rejects_invalid_counts_and_duplicates(self) -> None:
        affinity, _ = learned()
        good = affinity.to_dict()
        for field, value in (("positive", 4), ("exposures", 4), ("positive", True)):
            bad = copy.deepcopy(good)
            bad["pairs"][0][field] = value
            with self.assertRaises(ValueError):
                ContextAffinity.from_dict(bad)
        for field in ("events", "pairs"):
            bad = copy.deepcopy(good)
            bad[field].append(bad[field][0])
            with self.assertRaises(ValueError):
                ContextAffinity.from_dict(bad)

    def test_encoding_and_malformed_numeric_inputs_are_rejected(self) -> None:
        affinity, sample = learned()
        with self.assertRaisesRegex(ValueError, "encoding"):
            affinity.select(replace(sample, encoding_id="other"))
        for bad_score in (float("nan"), float("inf"), -1.0, True):
            with self.assertRaises(ValueError):
                affinity.select(
                    result(replace(sample.candidates[0], familiarity=bad_score))
                )
        with self.assertRaises(ValueError):
            ContextAffinityConfig(seconds=float("inf"))
        with self.assertRaises(ValueError):
            ContextAffinityConfig(evidence_threshold=0)


if __name__ == "__main__":
    unittest.main()

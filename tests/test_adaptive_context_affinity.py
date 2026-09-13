import copy
import hashlib
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from text_factors.context_affinity import (
    AdaptiveContextAffinity,
    AdaptiveContextAffinityConfig,
    ContextAffinity,
)
from text_factors.recognition import (
    CandidateRelation,
    ClusterEvidence,
    ContextResponse,
    RecognitionCandidate,
    RecognitionResult,
)


def sample(
    pattern: int = 0,
    activities: tuple[bool, ...] = (True, True),
    *,
    positions: tuple[int, ...] = (0,),
) -> RecognitionResult:
    """Two nearby views of a varying three-bit core plus one local feature."""

    candidates = []
    responses = []
    for index, active in enumerate(activities):
        bits = (0, 1, 2, 3 + pattern + index)
        name = f"view-{index}"
        context = f"context-{index}"
        responses.append(
            ContextResponse(
                context,
                name,
                "scene",
                positions,
                2.0 - index if active else 0.0,
                active,
                hashlib.sha256(json.dumps(bits).encode()).hexdigest(),
            )
        )
        if active:
            candidates.append(
                RecognitionCandidate(
                    name,
                    context,
                    f"content-{index}",
                    (0,),
                    positions,
                    (ClusterEvidence(0, bits, bits, 3, 0),),
                    2.0 - index,
                    4,
                    1,
                    "scene",
                )
            )
    return RecognitionResult(
        tuple(candidates),
        (),
        True,
        len(responses),
        len(responses),
        encoding_id="local",
        responses=tuple(responses),
    )


def trained(
    config: AdaptiveContextAffinityConfig | None = None,
) -> AdaptiveContextAffinity:
    affinity = AdaptiveContextAffinity("local", config)
    for index in range(3):
        affinity.observe(f"positive-{index}", sample(index))
    return affinity


class AdaptiveContextAffinityTests(unittest.TestCase):
    def test_diverse_joint_events_learn_a_direct_edge(self) -> None:
        affinity = trained()
        row = affinity.pair_statistics()[0]
        self.assertEqual((row["n11"], row["distinct_joint_patterns"]), (3, 3))
        self.assertEqual(row["activation_agreement"], 1.0)
        self.assertTrue(row["edge"])
        selected = affinity.select(sample(4))
        self.assertEqual(len(selected.candidates), 1)
        self.assertEqual(selected.suppressed[0].context_id, "context-1")

    def test_new_event_and_view_names_do_not_manufacture_diversity(self) -> None:
        affinity = AdaptiveContextAffinity("local")
        initial = sample()
        for index in range(4):
            renamed = replace(
                initial,
                candidates=tuple(
                    replace(c, candidate_id=f"{index}-{c.candidate_id}")
                    for c in initial.candidates
                ),
                responses=tuple(
                    replace(r, view_id=f"{index}-{r.view_id}")
                    for r in initial.responses
                ),
                memory_step=index,
            )
            affinity.observe(str(index), renamed)
        row = affinity.pair_statistics()[0]
        self.assertEqual(row["n11"], 4)
        self.assertEqual(row["distinct_joint_patterns"], 1)
        self.assertFalse(row["edge"])

    def test_one_sided_response_weakens_a_previous_edge(self) -> None:
        affinity = trained()
        affinity.observe("one-sided", sample(5, (True, False)))
        row = affinity.pair_statistics()[0]
        self.assertEqual((row["n11"], row["n10"], row["n01"]), (3, 1, 0))
        self.assertEqual(row["activation_agreement"], 0.75)
        self.assertFalse(row["edge"])
        self.assertEqual(len(affinity.select(sample()).candidates), 2)

    def test_inactivity_is_recorded_and_unobserved_context_is_not_negative(
        self,
    ) -> None:
        affinity = trained()
        affinity.observe("silent", sample(5, (False, False)))
        affinity.observe("one-context-available", sample(6, (True,)))
        row = affinity.pair_statistics()[0]
        self.assertEqual((row["n00"], row["n10"], row["n01"]), (1, 0, 0))
        self.assertEqual(row["observations"], 4)
        self.assertEqual(row["activation_agreement"], 1.0)
        self.assertTrue(row["edge"])
        silent = AdaptiveContextAffinity("local")
        silent.observe("silent-only", sample(0, (False, False)))
        self.assertIsNone(silent.pair_statistics()[0]["activation_agreement"])

    def test_recent_window_forgets_and_can_relearn_after_drift(self) -> None:
        affinity = trained(AdaptiveContextAffinityConfig(window_events=3))
        for index in range(3):
            affinity.observe(f"drift-{index}", sample(index + 5, (True, False)))
        row = affinity.pair_statistics()[0]
        self.assertEqual((row["n11"], row["n10"]), (0, 3))
        self.assertFalse(row["edge"])
        for index in range(3):
            affinity.observe(f"return-{index}", sample(index + 9))
        self.assertTrue(affinity.pair_statistics()[0]["edge"])
        self.assertEqual(len(affinity.to_dict()["history"]), 3)
        self.assertEqual(len(affinity.to_dict()["events"]), 9)
        before = affinity.to_dict()
        self.assertFalse(affinity.observe("positive-0", sample(0)))
        self.assertEqual(affinity.to_dict(), before)

    def test_many_views_give_only_one_pair_exposure(self) -> None:
        affinity = AdaptiveContextAffinity("local")
        original = sample()
        extra = replace(
            original.candidates[0], candidate_id="extra", source_positions=(9,)
        )
        response = replace(
            original.responses[0], view_id="extra", source_positions=(9,)
        )
        many = replace(
            original,
            candidates=(*original.candidates, extra),
            responses=(*original.responses, response),
            total_views=3,
            examined_views=3,
        )
        affinity.observe("multiple-views", many)
        self.assertEqual(affinity.pair_statistics()[0]["n11"], 1)

    def test_unknown_scope_and_another_instance_are_never_suppressed(self) -> None:
        affinity = trained()
        self.assertEqual(len(affinity.select(sample(positions=())).candidates), 2)
        original = sample()
        other = replace(
            original.candidates[1],
            candidate_id="second-instance",
            source_positions=(9,),
        )
        scene = replace(original, candidates=(*original.candidates, other))
        self.assertEqual(
            {c.candidate_id for c in affinity.select(scene).candidates},
            {"view-0", "second-instance"},
        )

    def test_current_evidence_and_conflicts_still_gate_selection(self) -> None:
        affinity = trained()
        original = sample()
        incompatible = replace(
            original,
            relations=(CandidateRelation("view-0", "view-1", "conflict", "given"),),
        )
        self.assertEqual(len(affinity.select(incompatible).candidates), 2)
        right = original.candidates[1]
        unrelated = replace(
            right, evidence=(ClusterEvidence(4, (7, 8, 9), (7, 8, 9), 3, 0),)
        )
        self.assertEqual(
            len(
                affinity.select(
                    replace(original, candidates=(original.candidates[0], unrelated))
                ).candidates
            ),
            2,
        )

    def test_incomplete_inputs_never_train_or_lose_candidates(self) -> None:
        affinity = trained()
        partial = replace(
            sample(), complete=False, stop_reason="view_limit", total_views=3
        )
        before = affinity.to_dict()
        self.assertFalse(affinity.observe("partial", partial))
        self.assertEqual(affinity.select(partial), partial)
        self.assertEqual(affinity.to_dict(), before)

    def test_complete_learning_requires_all_response_records(self) -> None:
        affinity = AdaptiveContextAffinity("local")
        original = sample()
        with self.assertRaisesRegex(ValueError, "completed view"):
            affinity.observe(
                "missing", replace(original, responses=original.responses[:1])
            )
        wrong = replace(original.responses[0], active=False)
        with self.assertRaisesRegex(ValueError, "active completed"):
            affinity.observe(
                "wrong-active",
                replace(original, responses=(wrong, original.responses[1])),
            )
        with self.assertRaisesRegex(ValueError, "one observation"):
            affinity.observe(
                "mixed",
                replace(
                    original,
                    responses=(
                        original.responses[0],
                        replace(original.responses[1], observation_id="different"),
                    ),
                ),
            )

    def test_capacity_and_cancellation_are_atomic(self) -> None:
        affinity = trained(AdaptiveContextAffinityConfig(max_events=3, window_events=3))
        before = affinity.to_dict()
        with self.assertRaisesRegex(ValueError, "capacity"):
            affinity.observe("overflow", sample(7))
        with self.assertRaises(InterruptedError):
            affinity.observe("cancel", sample(), cancelled=lambda: True)
        selected = affinity.select(sample(), cancelled=lambda: True)
        self.assertFalse(selected.complete)
        self.assertEqual(selected.candidates, sample().candidates)
        self.assertEqual(selected.suppressed, sample().suppressed)
        self.assertEqual(affinity.to_dict(), before)
        small = AdaptiveContextAffinity(
            "local", AdaptiveContextAffinityConfig(max_pairs=1)
        )
        with self.assertRaisesRegex(ValueError, "pair capacity"):
            small.observe("three", sample(0, (True, True, True)))
        self.assertEqual(small.to_dict()["events"], [])

    def test_cancellation_after_work_preserves_whole_input_and_old_state(self) -> None:
        affinity = trained()
        initial = sample()
        original = affinity._validator._supported
        flag = [False]

        def slow_support(*args):
            flag[0] = True
            return original(*args)

        with patch.object(affinity._validator, "_supported", side_effect=slow_support):
            selected = affinity.select(initial, cancelled=lambda: flag[0])
        self.assertFalse(selected.complete)
        self.assertEqual(selected.candidates, initial.candidates)
        self.assertEqual(selected.suppressed, initial.suppressed)

    def test_digest_deadline_is_checked_before_retry_or_commit(self) -> None:
        affinity = trained()
        before = affinity.to_dict()
        original_dump = json.dumps
        initial = sample()
        for event_id in ("positive-0", "new-event"):
            clock = [0.0]

            def delayed_dump(value, *, observed_clock=clock, **kwargs):
                observed_clock[0] = 3.0
                return original_dump(value, **kwargs)

            with (
                patch(
                    "text_factors.context_affinity.perf_counter",
                    side_effect=lambda observed_clock=clock: observed_clock[0],
                ),
                patch(
                    "text_factors.context_affinity.json.dumps", side_effect=delayed_dump
                ),
                self.assertRaises(InterruptedError),
            ):
                affinity.observe(event_id, initial)
            self.assertEqual(affinity.to_dict(), before)

    def test_roundtrip_validates_history_and_preserves_selection(self) -> None:
        affinity = trained()
        state = affinity.to_dict()
        restored = AdaptiveContextAffinity.from_dict(json.loads(json.dumps(state)))
        self.assertEqual(restored.to_dict(), state)
        self.assertEqual(restored.pair_statistics(), affinity.pair_statistics())
        self.assertEqual(restored.select(sample()), affinity.select(sample()))
        for field, value in (("state", True), ("state", 4), ("pattern", "not-a-hash")):
            bad = copy.deepcopy(state)
            bad["history"][0]["pairs"][0][field] = value
            with self.assertRaises(ValueError):
                AdaptiveContextAffinity.from_dict(bad)
        bad = copy.deepcopy(state)
        bad["history"].reverse()
        with self.assertRaises(ValueError):
            AdaptiveContextAffinity.from_dict(bad)

    def test_idempotence_and_legacy_store_are_separate(self) -> None:
        affinity = trained()
        before = affinity.to_dict()
        self.assertFalse(affinity.observe("positive-0", sample()))
        with self.assertRaisesRegex(ValueError, "different evidence"):
            affinity.observe("positive-0", replace(sample(), memory_step=1))
        self.assertEqual(affinity.to_dict(), before)
        old = ContextAffinity("local")
        legacy_payload = sample().to_dict()
        legacy_payload.pop("responses")
        old_digest = hashlib.sha256(
            json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        old.observe("historical", sample())
        self.assertEqual(old.to_dict()["events"][0]["digest"], old_digest)
        state = old.to_dict()
        self.assertEqual(ContextAffinity.from_dict(state).to_dict(), state)
        with self.assertRaises(ValueError):
            AdaptiveContextAffinity.from_dict(state)


if __name__ == "__main__":
    unittest.main()

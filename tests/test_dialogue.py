import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from text_factors.dialogue import (
    MAX_STATE_BYTES,
    DialogueLimits,
    GroundedCandidate,
    GroundedDialogue,
    GroundedRelation,
    GroundingEvidence,
    LabelEvent,
    SceneReference,
)
from text_factors.recognition import (
    CandidateRelation,
    ClusterEvidence,
    RecognitionCandidate,
    RecognitionResult,
)


def candidate(
    name: str = "a",
    content: str = "content-a",
    *,
    context: str = "shift-0",
    position: int = 0,
) -> GroundedCandidate:
    return GroundedCandidate(
        name,
        context,
        content,
        "observation-1",
        (7,),
        (position,),
        (GroundingEvidence(0, (1, 2), (1, 2), 3, 7),),
    )


def scene(reference: str = "scene-1", *, relation: str | None = None) -> SceneReference:
    if relation is None:
        return SceneReference(reference, (candidate(),), ())
    return SceneReference(
        reference,
        (candidate(), candidate("b", "content-b", position=1)),
        (GroundedRelation("a", "b", relation, "declared test provenance"),),
    )


class GroundedDialogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dialogue = GroundedDialogue("memory-v1", output_width=16)

    def test_word_requires_confirmation_and_then_transfers_to_another_context(
        self,
    ) -> None:
        self.dialogue.remember(scene())
        self.assertEqual(self.dialogue.describe().kind, "clarification")
        self.assertIn("объект 1", self.dialogue.describe().text)
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "Куб"))
        other = SceneReference(
            "scene-2", (candidate("other", context="shift-3", position=3),), ()
        )
        self.dialogue.remember(other)

        reply = self.dialogue.describe()
        self.assertEqual(reply.kind, "description")
        self.assertEqual(reply.text, "Вижу: куб.")
        self.assertEqual(reply.candidate_ids, ("other",))
        self.assertEqual(reply.support_event_ids, (0,))

    def test_colliding_output_bits_and_changed_clusters_do_not_inherit_words(
        self,
    ) -> None:
        self.dialogue.remember(scene())
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        other = SceneReference(
            "scene-2", (candidate("other", "different-clusters"),), ()
        )
        self.assertEqual(
            other.candidates[0].output_bits, scene().candidates[0].output_bits
        )
        self.dialogue.remember(other)
        self.assertEqual(self.dialogue.describe().kind, "clarification")
        self.assertNotIn("куб", self.dialogue.describe().text)

    def test_only_explicitly_compatible_parts_form_a_composite_phrase(self) -> None:
        self.dialogue.remember(scene(relation="compatible"))
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        self.dialogue.confirm(LabelEvent(1, "scene-1", "b", "шар"))
        self.assertEqual(self.dialogue.describe().text, "Вижу: куб и шар.")

        for kind in ("conflict", "undetermined", "duplicate"):
            with self.subTest(kind=kind):
                reference = scene(kind, relation=kind)
                self.dialogue.remember(reference)
                reply = self.dialogue.describe()
                self.assertEqual(reply.kind, "clarification")
                self.assertIn("совместимость не подтверждена", reply.text)

    def test_missing_relation_and_incomplete_search_are_not_confident_composition(
        self,
    ) -> None:
        original = scene(relation="compatible")
        self.dialogue.remember(replace(original, relations=()))
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        self.dialogue.confirm(LabelEvent(1, "scene-1", "b", "шар"))
        self.assertEqual(self.dialogue.describe().kind, "clarification")
        self.dialogue.remember(
            replace(original, reference_id="partial", complete=False)
        )
        reply = self.dialogue.describe()
        self.assertEqual(reply.kind, "clarification")
        self.assertIn("неполный результат", reply.text)
        self.dialogue.remember(SceneReference("empty-partial", (), (), False))
        self.assertIn("неполный результат", self.dialogue.describe().text)

    def test_unknown_word_and_one_word_for_distinct_contents_require_clarification(
        self,
    ) -> None:
        self.assertEqual(self.dialogue.lookup("куб").kind, "clarification")
        self.dialogue.remember(scene(relation="compatible"))
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "предмет"))
        self.dialogue.confirm(LabelEvent(1, "scene-1", "b", "предмет"))
        reply = self.dialogue.lookup("предмет")
        self.assertEqual(reply.kind, "clarification")
        self.assertEqual(reply.candidate_ids, ("a", "b"))
        self.assertEqual(reply.support_event_ids, (0, 1))

    def test_identical_retry_is_idempotent_and_conflicting_retry_is_rejected(
        self,
    ) -> None:
        self.dialogue.remember(scene())
        event = LabelEvent(10, "scene-1", "a", "куб")
        self.assertTrue(self.dialogue.confirm(event))
        before = self.dialogue.to_dict()
        for _ in range(4):
            self.assertFalse(self.dialogue.confirm(event))
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.dialogue.confirm(replace(event, word="шар"))
        with self.assertRaisesRegex(ValueError, "increase"):
            self.dialogue.confirm(replace(event, event_id=9))
        self.assertEqual(self.dialogue.to_dict(), before)

    def test_correction_is_an_explicit_event_bound_to_the_original_reference(
        self,
    ) -> None:
        self.dialogue.remember(scene())
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        self.dialogue.remember(scene("scene-2", relation="compatible"))
        before = self.dialogue.to_dict()
        for reference, name in (("scene-2", "a"), ("scene-2", "b")):
            with self.assertRaisesRegex(ValueError, "same reference and candidate"):
                self.dialogue.confirm(LabelEvent(1, reference, name, "шар", 0))
        self.assertEqual(self.dialogue.to_dict(), before)
        correction = LabelEvent(1, "scene-1", "a", "шар", 0)
        self.assertTrue(self.dialogue.confirm(correction))
        self.assertFalse(self.dialogue.confirm(correction))
        self.assertEqual(self.dialogue.lookup("куб").kind, "clarification")
        self.assertEqual(self.dialogue.lookup("шар").support_event_ids, (1,))
        with self.assertRaisesRegex(ValueError, "active label"):
            self.dialogue.confirm(LabelEvent(2, "scene-1", "a", "мяч", 0))

    def test_correction_does_not_erase_other_independent_confirmations(self) -> None:
        self.dialogue.remember(scene())
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        self.dialogue.remember(scene("scene-2"))
        self.dialogue.confirm(LabelEvent(1, "scene-2", "a", "куб"))
        self.dialogue.confirm(LabelEvent(2, "scene-1", "a", "шар", 0))
        self.assertEqual(self.dialogue.lookup("куб").support_event_ids, (1,))
        self.assertEqual(self.dialogue.lookup("шар").support_event_ids, (2,))

    def test_reading_and_replies_never_teach_themselves(self) -> None:
        self.dialogue.remember(scene())
        self.dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        before = self.dialogue.to_dict()
        for _ in range(5):
            reply = self.dialogue.handle("что видишь?")
            self.dialogue.handle("что значит куб")
            self.dialogue.handle("прочитай и сам себя обучи")
        with self.assertRaisesRegex(TypeError, "LabelEvent"):
            self.dialogue.confirm(cast(Any, reply))
        self.assertEqual(self.dialogue.to_dict(), before)

    def test_simple_chat_uses_the_current_reference_and_explicit_object_number(
        self,
    ) -> None:
        self.dialogue.remember(scene(relation="compatible"))
        self.assertEqual(self.dialogue.handle("назови 1 куб").kind, "confirmation")
        self.assertEqual(self.dialogue.handle("назови 2 шар").kind, "confirmation")
        self.assertEqual(self.dialogue.handle("что видишь").text, "Вижу: куб и шар.")
        self.assertEqual(self.dialogue.handle("исправь 1 блок").kind, "confirmation")
        self.assertEqual(self.dialogue.handle("что видишь").text, "Вижу: блок и шар.")
        self.dialogue.remember(scene("next"))
        before = self.dialogue.to_dict()
        self.assertEqual(self.dialogue.handle("исправь 1 конус").kind, "clarification")
        self.assertEqual(self.dialogue.to_dict(), before)
        self.assertEqual(self.dialogue.handle("назови 999 конус").kind, "clarification")

    def test_reference_ids_and_candidate_ownership_are_immutable(self) -> None:
        reference = scene()
        self.assertTrue(self.dialogue.remember(reference))
        self.assertFalse(self.dialogue.remember(reference))
        before = self.dialogue.to_dict()
        with self.assertRaisesRegex(ValueError, "different contents"):
            self.dialogue.remember(replace(reference, candidates=(candidate("b"),)))
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.dialogue.confirm(LabelEvent(0, "scene-1", "foreign-candidate", "куб"))
        self.assertEqual(self.dialogue.to_dict(), before)

    def test_capacity_refusal_does_not_destroy_existing_knowledge(self) -> None:
        limits = DialogueLimits(max_references=1, max_events=1, max_words=1)
        dialogue = GroundedDialogue("memory-v1", output_width=16, limits=limits)
        dialogue.remember(scene())
        dialogue.confirm(LabelEvent(0, "scene-1", "a", "куб"))
        before = dialogue.to_dict()
        with self.assertRaisesRegex(ValueError, "reference capacity"):
            dialogue.remember(scene("other"))
        with self.assertRaisesRegex(ValueError, "event capacity"):
            dialogue.confirm(LabelEvent(1, "scene-1", "a", "шар"))
        with self.assertRaisesRegex(ValueError, "length limit"):
            dialogue.handle("x" * (limits.max_message_chars + 1))
        self.assertEqual(dialogue.to_dict(), before)

    def test_recognition_adapter_preserves_cluster_provenance_and_encoding_scope(
        self,
    ) -> None:
        first = RecognitionCandidate(
            "a",
            "shift-0",
            "content-a",
            (7,),
            (0,),
            (ClusterEvidence(0, (1, 2), (1, 2), 3, 7),),
            2.0,
            2,
            1,
        )
        second = replace(
            first, candidate_id="b", content_key="content-b", source_positions=(1,)
        )
        result = RecognitionResult(
            (first, second),
            (CandidateRelation("a", "b", "compatible", "disjoint source positions"),),
            True,
            2,
            2,
            encoding_id="memory-v1",
        )
        self.assertTrue(
            self.dialogue.remember_recognition(
                "from-core", result, encoding_id="memory-v1"
            )
        )
        data = self.dialogue.to_dict()
        self.assertEqual(
            data["references"][0]["candidates"][0]["evidence"][0]["signature"], [1, 2]
        )
        with self.assertRaisesRegex(ValueError, "encoding_id"):
            self.dialogue.remember_recognition(
                "foreign", replace(result, encoding_id="other"), encoding_id="memory-v1"
            )
        self.assertEqual(self.dialogue.to_dict(), data)


class DialoguePersistenceTests(unittest.TestCase):
    def learned_dialogue(self) -> GroundedDialogue:
        dialogue = GroundedDialogue("memory-v1", output_width=16)
        dialogue.remember(scene())
        dialogue.confirm(LabelEvent(10, "scene-1", "a", "куб"))
        dialogue.confirm(LabelEvent(11, "scene-1", "a", "шар", 10))
        dialogue.remember(scene("latest"))
        return dialogue

    def test_complete_round_trip_keeps_corrections_current_reference_and_retry_ledger(
        self,
    ) -> None:
        dialogue = self.learned_dialogue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dialogue.json"
            dialogue.save(path)
            restored = GroundedDialogue.load(path)
        self.assertEqual(restored.to_dict(), dialogue.to_dict())
        self.assertEqual(restored.current_reference, "latest")
        self.assertEqual(restored.describe(), dialogue.describe())
        self.assertFalse(restored.confirm(LabelEvent(11, "scene-1", "a", "шар", 10)))
        self.assertEqual(restored.lookup("куб").kind, "clarification")
        restored.confirm(LabelEvent(12, "latest", "a", "мяч"))
        self.assertEqual(dialogue.stats()["events"], 2)

    def test_exported_state_is_independent_and_can_be_validated_directly(self) -> None:
        dialogue = self.learned_dialogue()
        data = dialogue.to_dict()
        self.assertEqual(GroundedDialogue.from_dict(data).to_dict(), data)
        data["references"][0]["candidates"][0]["output_bits"].append(15)
        self.assertEqual(
            dialogue.to_dict()["references"][0]["candidates"][0]["output_bits"], [7]
        )

    def test_semantically_corrupt_json_is_rejected(self) -> None:
        data = self.learned_dialogue().to_dict()
        mutations = (
            lambda d: d.update(format_version=True),
            lambda d: d.update(unexpected="field"),
            lambda d: d.update(current_reference="missing"),
            lambda d: d["limits"].update(max_events=10**9),
            lambda d: d["references"].append(d["references"][0]),
            lambda d: d["events"].append(d["events"][0]),
            lambda d: d["events"][1].update(reference_id="latest"),
            lambda d: d["events"][0].update(candidate_id="missing"),
            lambda d: d["references"][0]["candidates"][0].update(output_bits=[True]),
            lambda d: d["references"][0]["candidates"][0].update(source_positions=[-1]),
            lambda d: d["references"][0]["candidates"][0]["evidence"][0].update(
                matched_bits=[99]
            ),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                damaged = json.loads(json.dumps(data))
                mutate(damaged)
                with self.assertRaises(ValueError):
                    GroundedDialogue.from_dict(damaged)

    def test_file_limit_duplicate_keys_nan_and_deep_nesting_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            for payload in (
                b'{"format_version":1,"format_version":1}',
                b'{"value":NaN}',
                b"[" * 2000 + b"]" * 2000,
                b"\xff\xff",
            ):
                with self.subTest(payload=payload[:50]):
                    path.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        GroundedDialogue.load(path)
            with path.open("wb") as stream:
                stream.truncate(MAX_STATE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "file size"):
                GroundedDialogue.load(path)


if __name__ == "__main__":
    unittest.main()

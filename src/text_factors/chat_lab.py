"""Small end-to-end laboratory: local factors, explicit parts, taught names.

The demo's sensor strings and their boundaries are supplied scaffolding. Labels
are learned only from confirmations. Parsing and answer templates are supplied;
neither free-language understanding nor autonomous object discovery is claimed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from .config import ModelConfig
from .dialogue import DialogueReply, GroundedDialogue
from .model import TextFactorModel
from .recognition import RecognitionLimits, RecognitionResult


def make_chat_demo_model(seed: int = 7) -> TextFactorModel:
    """Train two unnamed sensor patterns; names are not in the factor memory."""

    config = ModelConfig(
        input_bits=128,
        output_bits=64,
        active_bits_per_symbol=8,
        positions=4,
        frame_size=2,
        context_count=4,
        receptive_bits=24,
        point_count=128,
        create_threshold=4,
        activation_threshold=4,
        min_active_points=1,
        probation_after=2,
        stable_after=3,
        max_clusters_per_point=16,
        seed=seed,
    )
    model = TextFactorModel(config, alphabet="abcd")
    for _ in range(4):
        for pattern in ("ab", "cd"):
            model.partial_fit_window(pattern)
    return model


class ContextChatSession:
    """Read a fixed model and teach its observed contents explicit word labels."""

    def __init__(
        self,
        model: TextFactorModel,
        dialogue: GroundedDialogue | None = None,
        *,
        limits: RecognitionLimits | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.dialogue = dialogue or GroundedDialogue(
            model.recognition_encoding_id, output_width=model.config.output_bits
        )
        if (
            self.dialogue.encoding_id != model.recognition_encoding_id
            or self.dialogue.output_width != model.config.output_bits
        ):
            raise ValueError("saved vocabulary belongs to another model encoding")
        self.limits = limits or RecognitionLimits(max_views=64, max_candidates=16)
        self.progress = progress
        self.last_recognition: RecognitionResult | None = None

    def show(
        self, parts: Sequence[str], *, reference_id: str | None = None
    ) -> DialogueReply:
        reference_id = reference_id or "scene-" + uuid4().hex
        result = self.model.recognize_parts(
            parts,
            observation_id=reference_id,
            limits=self.limits,
            progress=self.progress,
        )
        self.dialogue.remember_recognition(
            reference_id, result, encoding_id=self.model.recognition_encoding_id
        )
        self.last_recognition = result
        return self.dialogue.describe(reference_id)

    def handle(self, message: str) -> DialogueReply:
        if (
            type(message) is not str
            or len(message) > self.dialogue.limits.max_message_chars
        ):
            raise ValueError("message exceeds the configured length limit")
        text = message.strip()
        if text.casefold().startswith("покажи "):
            parts = [part.strip() for part in text[7:].split("|")]
            return self.show(parts)
        if text.casefold() in ("помощь", "help"):
            reply = self.dialogue.handle("помощь")
            return DialogueReply(
                "help",
                "Покажите данные: «покажи ab | cd». Разделитель | задаёт границы "
                "частей наблюдения. " + reply.text,
            )
        return self.dialogue.handle(text)


def run_dialogue_demo(
    *, seed: int = 7, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """A finite integration example; its success is not a language benchmark."""

    if progress is not None:
        progress("dialogue-demo: train two unnamed sensor patterns")
    model = make_chat_demo_model(seed)
    session = ContextChatSession(model)
    trace: list[dict[str, Any]] = []

    def show(parts: tuple[str, ...], reference: str) -> DialogueReply:
        reply = session.show(parts, reference_id=reference)
        assert session.last_recognition is not None
        trace.append(
            {
                "show": list(parts),
                "recognition": session.last_recognition.to_dict(),
                "reply": asdict(reply),
            }
        )
        return reply

    def say(message: str) -> DialogueReply:
        reply = session.handle(message)
        trace.append({"message": message, "reply": asdict(reply)})
        return reply

    before = model.memory.stats()
    if progress is not None:
        progress("dialogue-demo: confirm names from separately observed contents")
    show(("ab",), "teaching-a")
    say("назови 1 куб")
    show(("cd",), "teaching-b")
    say("назови 1 шар")
    if progress is not None:
        progress("dialogue-demo: new arrangement, unknown content, reload")
    combined = show(("cd", "ab"), "new-arrangement")
    names = say("что значит куб")
    unknown = show(("zz",), "unseen-content")
    snapshot = session.dialogue.to_dict()
    restored = GroundedDialogue.from_dict(snapshot)
    restored_reply = restored.describe("new-arrangement")
    unchanged = model.memory.stats() == before
    trace.append({"restored_reply": asdict(restored_reply)})
    result = {
        "schema_version": 1,
        "status": "complete",
        "seed": seed,
        "given": [
            "sensor alphabet and part boundaries",
            "two unnamed sensor patterns",
            "explicit human label confirmations",
            "command grammar and answer templates",
        ],
        "learned": [
            "local clusters from repeated sensor presentations",
            "word bindings to recognized cluster content",
        ],
        "not_tested": [
            "free Russian grammar",
            "learned segmentation",
            "autonomous context discovery",
            "arbitrary semantic relations",
        ],
        "metrics": {
            "combined_description": combined.kind == "description",
            "both_taught_names_present": "куб" in combined.text
            and "шар" in combined.text,
            "unknown_requests_clarification": unknown.kind == "clarification",
            "lookup_reply_kind": names.kind,
            "reload_preserves_description": restored_reply == combined,
            "recognition_did_not_train_memory": unchanged,
            "factor_training_presentations": before.get("step", model.memory.step),
            "label_events": session.dialogue.stats()["events"],
        },
        "model_config": model.config.to_dict(),
        "trace": trace,
    }
    if progress is not None:
        progress("dialogue-demo: complete")
    return result

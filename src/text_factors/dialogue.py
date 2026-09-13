"""A bounded, explicitly taught language interface to contextual recognition.

This is a small chat laboratory, not a learned Russian grammar or a second
recognition engine. Commands and response templates are supplied by this module.
Only caller-confirmed word/content pairs are learned. A recognition result is a
memory readout, not independently verified truth; a LabelEvent is a trust boundary.
Exact word associations use (encoding_id, content_key), never output bits alone.
An optional factor matcher compares unchanged, explicitly confirmed exemplars.
Replies do not add evidence. Complete state is stored as validated JSON, without
pickle; the caller must save the recognition model separately.
"""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .grounding import (
    MAX_MATCH_ATOMS,
    FactorAtoms,
    FactorExemplar,
    GroundingPolicy,
    WordResolution,
    match_factor_words,
)

if TYPE_CHECKING:
    from .recognition import RecognitionResult

FORMAT_VERSION = 2
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_CODE_WIDTH = 1_000_000
MAX_TEXT = 128
MAX_WORD = 64


def _integer(value: int, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def _identifier(value: str, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_TEXT
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be a short identifier without whitespace")


def _word(value: str) -> str:
    if type(value) is not str or not 0 < len(value) <= MAX_WORD:
        raise ValueError("word must contain 1 to 64 characters")
    if any(ord(char) < 32 for char in value):
        raise ValueError("word cannot contain control characters")
    normalized = " ".join(unicodedata.normalize("NFC", value).casefold().split())
    if not normalized or not any(char.isalnum() for char in normalized):
        raise ValueError("word must contain a letter or digit")
    if len(normalized) > MAX_WORD:
        raise ValueError("normalized word is too long")
    return normalized


def _indices(value: tuple[int, ...], name: str, upper: int) -> None:
    if type(value) is not tuple or len(value) > 4096:
        raise ValueError(f"{name} must be a bounded tuple of integer indices")
    if any(type(item) is not int or not 0 <= item < upper for item in value):
        raise ValueError(f"{name} contains an invalid index")
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{name} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DialogueLimits:
    max_references: int = 16
    max_candidates: int = 16
    max_events: int = 512
    max_words: int = 256
    max_total_bits: int = 32768
    max_message_chars: int = 512

    def __post_init__(self) -> None:
        for name, upper in (
            ("max_references", 64),
            ("max_candidates", 32),
            ("max_events", 4096),
            ("max_words", 512),
            ("max_total_bits", 131072),
            ("max_message_chars", 2048),
        ):
            _integer(getattr(self, name), name, 1, upper)


@dataclass(frozen=True, slots=True)
class GroundingEvidence:
    """Immutable provenance copied from a stable cluster readout."""

    point_index: int
    signature: tuple[int, ...]
    matched_bits: tuple[int, ...]
    observations: int
    output_bit: int

    def __post_init__(self) -> None:
        _integer(self.point_index, "point_index", 0, MAX_CODE_WIDTH - 1)
        _indices(self.signature, "signature", MAX_CODE_WIDTH)
        _indices(self.matched_bits, "matched_bits", MAX_CODE_WIDTH)
        _integer(self.observations, "observations", 1, 2**63 - 1)
        _integer(self.output_bit, "output_bit", 0, MAX_CODE_WIDTH - 1)
        if (
            not self.signature
            or not self.matched_bits
            or not set(self.matched_bits).issubset(self.signature)
        ):
            raise ValueError("evidence must match a nonempty cluster signature")


@dataclass(frozen=True, slots=True)
class GroundedCandidate:
    candidate_id: str
    context_id: str
    content_key: str
    observation_id: str
    output_bits: tuple[int, ...]
    source_positions: tuple[int, ...]
    evidence: tuple[GroundingEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("candidate_id", "context_id", "content_key", "observation_id"):
            _identifier(getattr(self, name), name)
        _indices(self.output_bits, "output_bits", MAX_CODE_WIDTH)
        _indices(self.source_positions, "source_positions", MAX_CODE_WIDTH)
        if (
            type(self.evidence) is not tuple
            or not 0 < len(self.evidence) <= 1024
            or any(type(item) is not GroundingEvidence for item in self.evidence)
        ):
            raise ValueError("candidate needs a bounded tuple of cluster evidence")
        if not self.output_bits:
            raise ValueError("candidate needs a nonempty memory readout")
        if any(item.output_bit not in self.output_bits for item in self.evidence):
            raise ValueError("evidence output bits must belong to the readout")


@dataclass(frozen=True, slots=True)
class GroundedRelation:
    left: str
    right: str
    kind: str
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.left, "left")
        _identifier(self.right, "right")
        if self.left == self.right:
            raise ValueError("relation needs two distinct candidates")
        if self.kind not in {"compatible", "conflict", "undetermined", "duplicate"}:
            raise ValueError("invalid relation kind")
        if type(self.reason) is not str or not 0 < len(self.reason) <= 512:
            raise ValueError("relation needs a short reason")


@dataclass(frozen=True, slots=True)
class SceneReference:
    reference_id: str
    candidates: tuple[GroundedCandidate, ...]
    relations: tuple[GroundedRelation, ...]
    complete: bool = True

    def __post_init__(self) -> None:
        _identifier(self.reference_id, "reference_id")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) > 32
            or any(
                type(candidate) is not GroundedCandidate
                for candidate in self.candidates
            )
        ):
            raise TypeError("candidates must be a tuple of GroundedCandidate values")
        ids = {item.candidate_id for item in self.candidates}
        if len(ids) != len(self.candidates):
            raise ValueError("candidate IDs must be unique within a reference")
        if (
            type(self.relations) is not tuple
            or len(self.relations) > 1024
            or any(
                type(relation) is not GroundedRelation for relation in self.relations
            )
        ):
            raise TypeError("relations must be a tuple of GroundedRelation values")
        pairs: set[tuple[str, ...]] = set()
        for relation in self.relations:
            if relation.left not in ids or relation.right not in ids:
                raise ValueError("relation refers to an unknown candidate")
            key = tuple(sorted((relation.left, relation.right)))
            if key in pairs:
                raise ValueError("only one relation per pair is allowed")
            pairs.add(key)
        if type(self.complete) is not bool:
            raise ValueError("complete must be a boolean")


@dataclass(frozen=True, slots=True)
class LabelEvent:
    """An explicit caller confirmation, or correction of one exact prior event."""

    event_id: int
    reference_id: str
    candidate_id: str
    word: str
    corrects_event_id: int | None = None

    def __post_init__(self) -> None:
        _integer(self.event_id, "event_id", 0, 2**63 - 1)
        _identifier(self.reference_id, "reference_id")
        _identifier(self.candidate_id, "candidate_id")
        object.__setattr__(self, "word", _word(self.word))
        if self.corrects_event_id is not None:
            _integer(self.corrects_event_id, "corrects_event_id", 0, 2**63 - 1)
            if self.corrects_event_id >= self.event_id:
                raise ValueError("a correction must follow the corrected event")


@dataclass(frozen=True, slots=True)
class DialogueReply:
    kind: str
    text: str
    reference_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    support_event_ids: tuple[int, ...] = ()


class GroundedDialogue:
    """Single-threaded, bounded vocabulary and explicit reference tracking.

    Capacities reject additions before mutation; nothing silently evicts evidence.
    Exact matching remains the default. Optional factor matching can propose a
    name after cluster changes, using only previously confirmed exemplars.
    Different content keys retain their identity and provenance. This interface
    does not certify a universal semantic identity or a calibrated probability.
    """

    def __init__(
        self,
        encoding_id: str,
        *,
        output_width: int,
        limits: DialogueLimits | None = None,
        grounding_policy: GroundingPolicy | None = None,
    ) -> None:
        _identifier(encoding_id, "encoding_id")
        _integer(output_width, "output_width", 1, MAX_CODE_WIDTH)
        if limits is not None and type(limits) is not DialogueLimits:
            raise TypeError("limits must be DialogueLimits")
        if (
            grounding_policy is not None
            and type(grounding_policy) is not GroundingPolicy
        ):
            raise TypeError("grounding_policy must be GroundingPolicy")
        self.encoding_id = encoding_id
        self.output_width = output_width
        self.limits = limits or DialogueLimits()
        self.grounding_policy = grounding_policy or GroundingPolicy()
        self._references: dict[str, SceneReference] = {}
        self._events: dict[int, LabelEvent] = {}
        self._active_events: dict[int, LabelEvent] = {}
        self._latest_event_id = -1
        self._current_reference: str | None = None

    @property
    def current_reference(self) -> str | None:
        return self._current_reference

    @staticmethod
    def _bit_cost(reference: SceneReference) -> int:
        return sum(
            len(candidate.output_bits)
            + len(candidate.source_positions)
            + sum(
                len(item.signature) + len(item.matched_bits)
                for item in candidate.evidence
            )
            for candidate in reference.candidates
        )

    def remember(self, reference: SceneReference) -> bool:
        """Archive recognition provenance; this operation teaches no words."""

        if type(reference) is not SceneReference:
            raise TypeError("remember accepts only SceneReference")
        retained = self._references.get(reference.reference_id)
        if retained is not None:
            if retained != reference:
                raise ValueError("reference_id already refers to different contents")
            self._current_reference = reference.reference_id
            return False
        if len(self._references) >= self.limits.max_references:
            raise ValueError("reference capacity reached")
        if len(reference.candidates) > self.limits.max_candidates:
            raise ValueError("candidate capacity exceeded")
        if any(
            bit >= self.output_width
            for candidate in reference.candidates
            for bit in candidate.output_bits
        ):
            raise ValueError("candidate readout exceeds configured output_width")
        total_bits = self._bit_cost(reference) + sum(
            self._bit_cost(item) for item in self._references.values()
        )
        if total_bits > self.limits.max_total_bits:
            raise ValueError("stored provenance bit capacity exceeded")
        self._references[reference.reference_id] = reference
        self._current_reference = reference.reference_id
        return True

    def remember_recognition(
        self,
        reference_id: str,
        result: RecognitionResult,
        *,
        encoding_id: str,
    ) -> bool:
        """Copy bounded recognition data; raw outputs are never lexical keys."""

        if encoding_id != self.encoding_id:
            raise ValueError("recognition encoding_id does not match this dialogue")
        if result.encoding_id != encoding_id:
            raise ValueError("result encoding_id does not match the supplied encoding")
        if len(result.candidates) > self.limits.max_candidates:
            raise ValueError("candidate capacity exceeded")
        if len(result.relations) > self.limits.max_candidates**2:
            raise ValueError("relation capacity exceeded")
        if any(len(item.evidence) > 1024 for item in result.candidates):
            raise ValueError("candidate evidence capacity exceeded")
        candidates = tuple(
            GroundedCandidate(
                candidate.candidate_id,
                candidate.context_id,
                candidate.content_key,
                candidate.observation_id,
                candidate.output_bits,
                candidate.source_positions,
                tuple(
                    GroundingEvidence(
                        item.point_index,
                        item.signature,
                        item.matched_bits,
                        item.observations,
                        item.output_bit,
                    )
                    for item in candidate.evidence
                ),
            )
            for candidate in result.candidates
        )
        ids = {candidate.candidate_id for candidate in candidates}
        relations = tuple(
            GroundedRelation(item.left, item.right, item.kind, item.reason)
            for item in result.relations
            if item.left in ids and item.right in ids
        )
        return self.remember(
            SceneReference(reference_id, candidates, relations, result.complete)
        )

    def _candidate(self, reference_id: str, candidate_id: str) -> GroundedCandidate:
        reference = self._references.get(reference_id)
        if reference is None:
            raise ValueError("unknown reference_id")
        for candidate in reference.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ValueError("candidate_id does not belong to this reference")

    def confirm(self, event: LabelEvent) -> bool:
        """Apply one explicit label event; an identical retry is a no-op."""

        if type(event) is not LabelEvent:
            raise TypeError("confirm accepts only LabelEvent, never a reply")
        retained = self._events.get(event.event_id)
        if retained is not None:
            if retained != event:
                raise ValueError("event_id already has different contents")
            return False
        if event.event_id <= self._latest_event_id:
            raise ValueError("new event IDs must increase")
        self._candidate(event.reference_id, event.candidate_id)
        if len(self._events) >= self.limits.max_events:
            raise ValueError("event capacity reached")
        corrected = None
        if event.corrects_event_id is not None:
            corrected = self._active_events.get(event.corrects_event_id)
            if corrected is None:
                raise ValueError("correction must reference an active label event")
            if (corrected.reference_id, corrected.candidate_id) != (
                event.reference_id,
                event.candidate_id,
            ):
                raise ValueError(
                    "correction must keep the same reference and candidate"
                )
        words = {
            item.word
            for item in self._active_events.values()
            if item.event_id != event.corrects_event_id
        }
        words.add(event.word)
        if len(words) > self.limits.max_words:
            raise ValueError("word capacity reached")
        if corrected is not None:
            del self._active_events[corrected.event_id]
        self._events[event.event_id] = event
        self._active_events[event.event_id] = event
        self._latest_event_id = event.event_id
        return True

    def _lexicon(self) -> dict[str, dict[str, tuple[int, ...]]]:
        entries: dict[str, dict[str, list[int]]] = {}
        for event in self._active_events.values():
            candidate = self._candidate(event.reference_id, event.candidate_id)
            entries.setdefault(candidate.content_key, {}).setdefault(
                event.word, []
            ).append(event.event_id)
        return {
            key: {word: tuple(ids) for word, ids in words.items()}
            for key, words in entries.items()
        }

    @staticmethod
    def _atoms(candidate: GroundedCandidate) -> FactorAtoms | None:
        if sum(len(item.matched_bits) for item in candidate.evidence) > MAX_MATCH_ATOMS:
            return None
        return frozenset(
            (item.point_index, bit)
            for item in candidate.evidence
            for bit in item.matched_bits
        )

    def resolve_candidate(
        self, candidate: GroundedCandidate, *, encoding_id: str | None = None
    ) -> WordResolution:
        """Read a name without learning or replacing its exact content key.

        Archived candidates have already passed the recognition encoding guard.
        A caller resolving an unarchived candidate must explicitly supply its
        encoding_id, because GroundedCandidate itself has no encoding field.
        """

        if type(candidate) is not GroundedCandidate:
            raise TypeError("resolve_candidate requires GroundedCandidate")
        if encoding_id is not None and encoding_id != self.encoding_id:
            raise ValueError("candidate encoding_id does not match this dialogue")
        if encoding_id is None and not any(
            candidate in reference.candidates for reference in self._references.values()
        ):
            raise ValueError("unarchived candidate requires an explicit encoding_id")
        if any(bit >= self.output_width for bit in candidate.output_bits):
            raise ValueError("candidate readout exceeds configured output_width")
        lexicon = self._lexicon()
        exact = lexicon.get(candidate.content_key)
        if exact:
            return WordResolution(
                words=tuple(exact),
                support_event_ids=tuple(
                    sorted(event for events in exact.values() for event in events)
                ),
                method="exact",
                score=1.0,
                matched_content_keys=(candidate.content_key,),
                reason="exact_content_key",
            )
        if self.grounding_policy.mode == "exact":
            return WordResolution()
        atoms = self._atoms(candidate)
        if atoms is None:
            return WordResolution(reason="work_limit")
        exemplars: list[FactorExemplar] = []
        seen: set[tuple[str, str]] = set()
        for event in self._active_events.values():
            reference_key = event.reference_id, event.candidate_id
            if reference_key in seen:
                continue
            seen.add(reference_key)
            confirmed = self._candidate(*reference_key)
            confirmed_atoms = self._atoms(confirmed)
            if confirmed_atoms is None:
                # Omitting a competing exemplar could create false confidence.
                return WordResolution(reason="work_limit")
            names = lexicon[confirmed.content_key]
            exemplars.append(
                FactorExemplar(
                    confirmed.content_key,
                    tuple(names),
                    tuple(sorted(e for events in names.values() for e in events)),
                    confirmed_atoms,
                )
            )
        return match_factor_words(atoms, tuple(exemplars), self.grounding_policy)

    @staticmethod
    def _reply(
        kind: str,
        text: str,
        reference_id: str | None = None,
        candidate_ids: tuple[str, ...] = (),
        support_event_ids: tuple[int, ...] = (),
    ) -> DialogueReply:
        if len(text) > 4096:
            kind = "clarification"
            text = "Слишком много вариантов названия. Уточните один объект по номеру."
        return DialogueReply(
            kind,
            text,
            reference_id,
            candidate_ids,
            tuple(sorted(set(support_event_ids))),
        )

    def describe(self, reference_id: str | None = None) -> DialogueReply:
        if reference_id is None:
            reference_id = self._current_reference
        reference = self._references.get(reference_id or "")
        if reference is None:
            return self._reply("clarification", "Сначала покажите наблюдение.")
        ids = tuple(item.candidate_id for item in reference.candidates)
        if not ids:
            text = "Пока не узнаю содержание этого наблюдения."
            if not reference.complete:
                text += " Проверена только часть контекстов; это неполный результат."
            return self._reply(
                "clarification",
                text,
                reference_id,
            )
        resolutions = tuple(
            self.resolve_candidate(item) for item in reference.candidates
        )
        names: list[str] = []
        support: list[int] = []
        unknown: list[int] = []
        ambiguous: list[int] = []
        for number, resolution in enumerate(resolutions, 1):
            if resolution.words:
                names.append(f"{number}: " + " / ".join(resolution.words))
                support.extend(resolution.support_event_ids)
                if resolution.ambiguous:
                    ambiguous.append(number)
            else:
                names.append(f"{number}: содержание без названия")
                unknown.append(number)
        pairs = {
            frozenset((item.left, item.right)): item.kind
            for item in reference.relations
        }
        compatible = all(
            pairs.get(frozenset((left, right))) == "compatible"
            for index, left in enumerate(ids)
            for right in ids[index + 1 :]
        )
        listing = "; ".join(names)
        if not compatible:
            text = (
                f"Есть несколько трактовок: {listing}. "
                "Их совместимость не подтверждена. Уточните, какую имеете в виду."
            )
        elif ambiguous:
            text = (
                f"Возможные названия: {listing}. "
                f"Уточните название всего содержания {ambiguous[0]}."
            )
        elif unknown:
            text = f"Узнанные части: {listing}. Как назвать объект {unknown[0]}?"
        else:
            labels = [" / ".join(resolution.words) for resolution in resolutions]
            prefix = (
                "По сходству признаков: "
                if any(item.method == "factor" for item in resolutions)
                else "Вижу: "
            )
            text = prefix + " и ".join(labels) + "."
        if not reference.complete:
            text += " Проверена только часть контекстов; это неполный результат."
        kind = (
            "description"
            if compatible and not unknown and not ambiguous and reference.complete
            else "clarification"
        )
        return self._reply(kind, text, reference_id, ids, tuple(support))

    def lookup(self, word: str) -> DialogueReply:
        word = _word(word)
        lexicon = self._lexicon()
        contents = {
            key: labels[word] for key, labels in lexicon.items() if word in labels
        }
        if not contents:
            return self._reply(
                "clarification",
                f"Я ещё не знаю слово «{word}». "
                "Покажите содержание и подтвердите название.",
            )
        support = tuple(event for events in contents.values() for event in events)
        reference = self._references.get(self._current_reference or "")
        resolved = tuple(
            (candidate, self.resolve_candidate(candidate))
            for candidate in (reference.candidates if reference else ())
        )
        matching = tuple(
            candidate for candidate, result in resolved if word in result.words
        )
        uncertain = any(
            result.ambiguous for _, result in resolved if word in result.words
        )
        if uncertain:
            return self._reply(
                "clarification",
                f"Название «{word}» подходит к одной из трактовок. "
                "Уточните содержание.",
                self._current_reference,
                tuple(candidate.candidate_id for candidate in matching),
                support,
            )
        if len(contents) > 1:
            return self._reply(
                "clarification",
                f"Словом «{word}» названы разные содержания. Уточните нужный объект.",
                self._current_reference,
                tuple(candidate.candidate_id for candidate in matching),
                support,
            )
        if matching:
            approximate = any(
                result.method == "factor"
                for _, result in resolved
                if word in result.words
            )
            text = (
                f"«{word}» — название сходного содержания по подтверждённым примерам."
                if approximate
                else f"«{word}» — подтверждённое вами название узнанного содержания."
            )
        else:
            text = f"Название «{word}» сохранено; в текущем наблюдении оно не узнано."
        return self._reply(
            "definition",
            text,
            self._current_reference,
            tuple(candidate.candidate_id for candidate in matching),
            support,
        )

    def _label_number(
        self, number: str, word: str, *, correction: bool
    ) -> DialogueReply:
        reference = self._references.get(self._current_reference or "")
        if reference is None:
            raise ValueError("сначала покажите наблюдение")
        if not number.isascii() or not number.isdecimal():
            raise ValueError("номер объекта должен быть целым положительным числом")
        index = int(number) - 1
        if not 0 <= index < len(reference.candidates):
            raise ValueError("в текущем наблюдении нет объекта с таким номером")
        candidate = reference.candidates[index]
        corrected_id = None
        if correction:
            previous = [
                item.event_id
                for item in self._active_events.values()
                if item.reference_id == reference.reference_id
                and item.candidate_id == candidate.candidate_id
            ]
            if not previous:
                raise ValueError(
                    "для этого объекта ещё нет подтверждения для исправления"
                )
            corrected_id = max(previous)
        event = LabelEvent(
            self._latest_event_id + 1,
            reference.reference_id,
            candidate.candidate_id,
            word,
            corrected_id,
        )
        self.confirm(event)
        text = (
            f"Исправлено название объекта {index + 1}: «{event.word}»."
            if correction
            else f"Название объекта {index + 1} подтверждено: «{event.word}»."
        )
        return self._reply(
            "confirmation",
            text,
            reference.reference_id,
            (candidate.candidate_id,),
            (event.event_id,),
        )

    def handle(self, text: str) -> DialogueReply:
        """Parse supplied commands, including explicit teaching by object number.

        Repeating a plain-language teaching command is a new user event. Network
        retry clients must instead use confirm with a stable LabelEvent.event_id.
        """

        if type(text) is not str or len(text) > self.limits.max_message_chars:
            raise ValueError("message exceeds the configured length limit")
        command = text.strip()
        lower = command.casefold().rstrip("?")
        if lower == "что видишь":
            return self.describe()
        if lower.startswith("что значит "):
            return self.lookup(command[len("что значит ") :].rstrip("?"))
        if lower.startswith("опиши "):
            return self.describe(command[len("опиши ") :])
        if lower.startswith(("назови ", "исправь ")):
            pieces = command.split(maxsplit=2)
            if len(pieces) != 3:
                return self._reply("clarification", "Например: «назови 1 куб».")
            try:
                return self._label_number(
                    pieces[1], pieces[2], correction=lower.startswith("исправь ")
                )
            except ValueError as exc:
                return self._reply("clarification", f"Не удалось подтвердить: {exc}.")
        return self._reply(
            "help",
            "Команды: «что видишь», «назови 1 куб», «исправь 1 шар», "
            "«что значит куб», «опиши <ссылка>». Названия учатся из ваших "
            "подтверждений; грамматика команд и фразы ответов заданы программой.",
        )

    def stats(self) -> dict[str, int]:
        return {
            "references": len(self._references),
            "events": len(self._events),
            "active_events": len(self._active_events),
            "words": len({event.word for event in self._active_events.values()}),
            "contents": len(self._lexicon()),
            "latest_event_id": self._latest_event_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON data, including provenance and correction history."""

        return _json_data(
            {
                "format_version": FORMAT_VERSION,
                "encoding_id": self.encoding_id,
                "output_width": self.output_width,
                "limits": asdict(self.limits),
                "grounding_policy": asdict(self.grounding_policy),
                "current_reference": self._current_reference,
                "references": [asdict(item) for item in self._references.values()],
                "events": [asdict(item) for item in self._events.values()],
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> GroundedDialogue:
        """Validate all counts, types and references before returning new state."""

        if type(value) is not dict:
            raise ValueError("invalid dialogue state")
        version = value.get("format_version")
        if type(version) is not int or version not in (1, FORMAT_VERSION):
            raise ValueError("unsupported dialogue format version")
        fields = {
            "format_version",
            "encoding_id",
            "output_width",
            "limits",
            "current_reference",
            "references",
            "events",
        }
        if version == FORMAT_VERSION:
            fields.add("grounding_policy")
        data = _object(value, fields)
        policy = GroundingPolicy()
        if version == FORMAT_VERSION:
            policy = GroundingPolicy(
                **_object(
                    data["grounding_policy"], set(GroundingPolicy.__dataclass_fields__)
                )
            )
        limits_data = _object(data["limits"], set(DialogueLimits.__dataclass_fields__))
        limits = DialogueLimits(**limits_data)
        dialogue = cls(
            data["encoding_id"],
            output_width=data["output_width"],
            limits=limits,
            grounding_policy=policy,
        )
        references = _list(data["references"], limits.max_references)
        events = _list(data["events"], limits.max_events)
        for raw_reference in references:
            item = _object(raw_reference, set(SceneReference.__dataclass_fields__))
            candidates = []
            for raw_candidate in _list(item["candidates"], limits.max_candidates):
                candidate = _object(
                    raw_candidate, set(GroundedCandidate.__dataclass_fields__)
                )
                evidence = []
                for raw_evidence in _list(candidate["evidence"], 1024):
                    detail = _object(
                        raw_evidence, set(GroundingEvidence.__dataclass_fields__)
                    )
                    detail["signature"] = tuple(_list(detail["signature"], 4096))
                    detail["matched_bits"] = tuple(_list(detail["matched_bits"], 4096))
                    evidence.append(GroundingEvidence(**detail))
                candidate["output_bits"] = tuple(_list(candidate["output_bits"], 4096))
                candidate["source_positions"] = tuple(
                    _list(candidate["source_positions"], 4096)
                )
                candidate["evidence"] = tuple(evidence)
                candidates.append(GroundedCandidate(**candidate))
            relations = tuple(
                GroundedRelation(
                    **_object(raw, set(GroundedRelation.__dataclass_fields__))
                )
                for raw in _list(item["relations"], limits.max_candidates**2)
            )
            reference = SceneReference(
                item["reference_id"], tuple(candidates), relations, item["complete"]
            )
            if not dialogue.remember(reference):
                raise ValueError("duplicate reference_id in saved state")
        for raw_event in events:
            event = LabelEvent(
                **_object(raw_event, set(LabelEvent.__dataclass_fields__))
            )
            if not dialogue.confirm(event):
                raise ValueError("duplicate event_id in saved state")
        current = data["current_reference"]
        if current is not None:
            _identifier(current, "current_reference")
            if current not in dialogue._references:
                raise ValueError("current_reference is not archived")
        elif references:
            raise ValueError("nonempty state must have a current reference")
        dialogue._current_reference = current
        return dialogue

    def save(self, path: str | Path) -> None:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(payload) > MAX_STATE_BYTES:
            raise ValueError("dialogue state exceeds the file size limit")
        target = Path(path)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> GroundedDialogue:
        with Path(path).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > MAX_STATE_BYTES:
                raise ValueError("dialogue file exceeds the file size limit")
            payload = stream.read(MAX_STATE_BYTES + 1)
        if len(payload) > MAX_STATE_BYTES:
            raise ValueError("dialogue file exceeds the file size limit")
        try:
            data = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
            return cls.from_dict(data)
        except (RecursionError, UnicodeError, TypeError, KeyError) as exc:
            raise ValueError("invalid dialogue JSON state") from exc


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid object fields in dialogue state")
    return dict(value)


def _json_data(value: Any) -> Any:
    if type(value) is dict:
        return {key: _json_data(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_data(item) for item in value]
    return value


def _list(value: Any, limit: int) -> list[Any]:
    if type(value) is not list or len(value) > limit:
        raise ValueError("invalid or oversized list in dialogue state")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")

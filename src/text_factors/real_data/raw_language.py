"""Small, conservative text-to-graph adapter trained on annotated examples.

Unlike :class:`OpenSemanticModel`, inference accepts just the source text. It
learns mention surfaces and *orthographic* kind/case signatures from training
annotations; it does not receive gold spans, cases, or instance IDs at parse
time. A new mention has a fresh source-local identity. In particular, two
identically spelled names are never silently merged into one person.

This is a bounded research baseline, not a general Russian NER or coreference
model. Novel words used as roles stay in the unexplained region even when a
layout matches: their interpretation still needs review before publication.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from text_factors.real_data.open_semantics import (
    IdentifiedMention,
    LabeledText,
    OpenSemanticModel,
    SemanticGraph,
    Span,
    _matches,
    _validate_text,
    _words,
)

_VOWELS = frozenset("аеёиоуыэюя")
_MAX_SIGNATURES = 16384


def _shape(surface: str) -> str:
    """A deliberately weak spelling feature, never a grammatical oracle."""
    letter = surface[-1].casefold()
    if letter in _VOWELS:
        ending = f"vowel:{letter}"
    elif letter in "йьъ":
        ending = f"other:{letter}"
    elif letter.isalpha():
        ending = "consonant"
    else:
        ending = "other"
    return ("capital:" if surface[0].isupper() else "lower:") + ending


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


class RawSemanticModel:
    """Generate cautious source-local mentions, then run the trained graph model.

    ``fit`` may inspect annotated mentions. ``parse`` never accepts them.
    Orthography-only generalization is explicitly represented as residual
    uncertainty even if the graph has a structurally valid event.
    """

    def __init__(self) -> None:
        self._semantic: OpenSemanticModel | None = None
        self._surfaces: dict[str, set[tuple[str, str | None]]] = {}
        self._shapes: dict[str, set[tuple[str, str | None]]] = {}
        self._cues: tuple[str, ...] = ()
        self.model_fingerprint: str | None = None

    def fit(self, examples: tuple[LabeledText, ...]) -> RawSemanticModel:
        semantic = OpenSemanticModel().fit(examples)
        surfaces: dict[str, set[tuple[str, str | None]]] = {}
        shapes: dict[str, set[tuple[str, str | None]]] = {}
        for example in examples:
            _validate_text(example.text, example.mentions)
            for mention in example.mentions:
                words = _words(example.text, mention.span)
                if (
                    not words
                    or words[0][1].start != mention.span.start
                    or words[-1][1].end != mention.span.end
                ):
                    raise ValueError("annotated mention needs word boundaries")
                surface = " ".join(word for word, _ in words)
                signature = mention.kind, mention.morphology
                surfaces.setdefault(surface, set()).add(signature)
                if len(words) == 1:
                    raw = example.text[mention.span.start : mention.span.end]
                    shapes.setdefault(_shape(raw), set()).add(signature)
        spec = semantic.to_dict()
        cues = (
            *spec["relations"],
            *spec["negation_cues"],
            *spec["time_cues"],
            *spec["event_link_cues"],
        )
        self._semantic = semantic
        self._surfaces = surfaces
        self._shapes = shapes
        self._cues = tuple(sorted(set(cues)))
        self.model_fingerprint = _digest(self.to_dict())
        return self

    def to_dict(self) -> dict[str, Any]:
        if self._semantic is None:
            raise RuntimeError("fit the model first")

        def rows(
            values: dict[str, set[tuple[str, str | None]]],
        ) -> list[dict[str, Any]]:
            return [
                {
                    "key": key,
                    "signatures": [
                        [kind, morphology]
                        for kind, morphology in sorted(options, key=repr)
                    ],
                }
                for key, options in sorted(values.items())
            ]

        return {
            "schema": "ai2-raw-semantic-model-v1",
            "semantic_model": self._semantic.to_dict(),
            "surfaces": rows(self._surfaces),
            "shapes": rows(self._shapes),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RawSemanticModel:
        if (
            type(value) is not dict
            or set(value)
            != {
                "schema",
                "semantic_model",
                "surfaces",
                "shapes",
            }
            or value["schema"] != "ai2-raw-semantic-model-v1"
        ):
            raise ValueError("invalid raw model schema")
        semantic = OpenSemanticModel.from_dict(value["semantic_model"])

        def read_rows(rows: Any) -> dict[str, set[tuple[str, str | None]]]:
            if type(rows) is not list or len(rows) > _MAX_SIGNATURES:
                raise ValueError("invalid mention signatures")
            result: dict[str, set[tuple[str, str | None]]] = {}
            for row in rows:
                if (
                    type(row) is not dict
                    or set(row) != {"key", "signatures"}
                    or type(row["key"]) is not str
                    or not row["key"]
                    or len(row["key"]) > 2048
                    or row["key"] in result
                    or type(row["signatures"]) is not list
                    or not row["signatures"]
                    or len(row["signatures"]) > 64
                ):
                    raise ValueError("invalid mention signatures")
                options: set[tuple[str, str | None]] = set()
                for item in row["signatures"]:
                    if (
                        type(item) is not list
                        or len(item) != 2
                        or type(item[0]) is not str
                        or not item[0]
                        or len(item[0]) > 128
                        or (
                            item[1] is not None
                            and (
                                type(item[1]) is not str
                                or not item[1]
                                or len(item[1]) > 128
                            )
                        )
                    ):
                        raise ValueError("invalid mention signature")
                    options.add((item[0], item[1]))
                result[row["key"]] = options
            return result

        model = cls()
        model._semantic = semantic
        model._surfaces = read_rows(value["surfaces"])
        model._shapes = read_rows(value["shapes"])
        spec = semantic.to_dict()
        model._cues = tuple(
            sorted(
                set(
                    (
                        *spec["relations"],
                        *spec["negation_cues"],
                        *spec["time_cues"],
                        *spec["event_link_cues"],
                    )
                )
            )
        )
        if model.to_dict() != value:
            raise ValueError("noncanonical raw model encoding")
        model.model_fingerprint = _digest(value)
        return model

    def _infer(
        self, text: str, *, source_id: str, source_version: int
    ) -> tuple[tuple[IdentifiedMention, ...], tuple[Span, ...]]:
        if self._semantic is None:
            raise RuntimeError("fit the model before parsing")
        parts = _validate_text(text, ())
        candidates: list[IdentifiedMention] = []
        novel: list[Span] = []
        for part in parts:
            words = _words(text, part)
            occupied: set[int] = set()
            for cue in self._cues:
                for span in _matches(words, cue):
                    occupied.update(
                        at
                        for at, (_, word_span) in enumerate(words)
                        if span.start <= word_span.start and word_span.end <= span.end
                    )
            # Exact annotated multiword mentions take precedence over
            # orthographic guesses about either constituent.
            possibilities: list[tuple[int, int, str, Span]] = []
            for surface in self._surfaces:
                if " " not in surface:
                    continue
                for span in _matches(words, surface):
                    first = next(
                        at
                        for at, (_, word_span) in enumerate(words)
                        if word_span.start == span.start
                    )
                    last = next(
                        at
                        for at, (_, word_span) in enumerate(words)
                        if word_span.end == span.end
                    )
                    possibilities.append((first, last, surface, span))
            for first, last, surface, span in sorted(
                possibilities, key=lambda row: (row[0], -(row[1] - row[0]))
            ):
                if any(at in occupied for at in range(first, last + 1)):
                    continue
                occupied.update(range(first, last + 1))
                signatures = self._surfaces[surface]
                if len(signatures) == 1:
                    kind, morphology = next(iter(signatures))
                    candidates.append(
                        self._mention(source_id, source_version, span, kind, morphology)
                    )
            for at, (word, span) in enumerate(words):
                if at in occupied:
                    continue
                exact = self._surfaces.get(word)
                signatures = (
                    exact
                    if exact is not None
                    else self._shapes.get(_shape(text[span.start : span.end]), set())
                )
                if len(signatures) != 1:
                    continue
                kind, morphology = next(iter(signatures))
                candidates.append(
                    self._mention(source_id, source_version, span, kind, morphology)
                )
                if exact is None:
                    novel.append(span)
        return tuple(sorted(candidates, key=lambda item: item.span.start)), tuple(novel)

    @staticmethod
    def _mention(
        source_id: str,
        source_version: int,
        span: Span,
        kind: str,
        morphology: str | None,
    ) -> IdentifiedMention:
        local_id = f"{source_id}:v{source_version}:mention:{span.start}:{span.end}"
        return IdentifiedMention(local_id, local_id, span, kind, morphology)

    def infer_mentions(
        self, text: str, *, source_id: str, source_version: int
    ) -> tuple[IdentifiedMention, ...]:
        """Inspectable inferred spans; IDs are deliberately not global coref."""
        if type(source_id) is not str or not source_id:
            raise ValueError("source ID required")
        if type(source_version) is not int or source_version <= 0:
            raise ValueError("positive source version required")
        return self._infer(text, source_id=source_id, source_version=source_version)[0]

    def parse(self, text: str, *, source_id: str, source_version: int) -> SemanticGraph:
        """Infer a graph from raw text, keeping untrained role words unresolved."""
        if self._semantic is None:
            raise RuntimeError("fit the model before parsing")
        if type(source_id) is not str or not source_id:
            raise ValueError("source ID required")
        if type(source_version) is not int or source_version <= 0:
            raise ValueError("positive source version required")
        mentions, novel = self._infer(
            text, source_id=source_id, source_version=source_version
        )
        graph = self._semantic.parse(
            text,
            mentions,
            source_id=source_id,
            source_version=source_version,
        )
        # A guessed mention can fill a role, but is never treated as a fully
        # explained/verified piece of language. Keep its exact source span.
        used_novel = tuple(
            span
            for span in novel
            if any(
                role.mention_id
                == f"{source_id}:v{source_version}:mention:{span.start}:{span.end}"
                for event in graph.events
                for role in event.roles
            )
        )
        return replace(
            graph,
            model_fingerprint=self.model_fingerprint or "",
            unexplained=tuple(
                sorted(
                    set((*graph.unexplained, *used_novel)),
                    key=lambda span: (span.start, span.end),
                )
            ),
        )

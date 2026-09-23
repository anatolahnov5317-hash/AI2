"""Conservative supervised question templates over reviewed, accessible identities.

The model learns the words around one annotated subject, the relation ID, and
the requested role from question examples. At inference it receives only raw
text; a separate raw mention model proposes spans, while a caller-supplied
catalog resolves a span to an *accessible, reviewed* instance. Source-local
mention IDs are never mistaken for cross-document identities.

This is a narrow, inspectable language baseline. A novel wording, multiple
subjects, an unresolved name, or an unsupported negated question abstains.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .open_semantics import IdentifiedMention, Span

if TYPE_CHECKING:
    from .raw_language import RawSemanticModel

_TOKENS = re.compile(r"[^\W_]+(?:-[^\W_]+)*|[^\s]", re.UNICODE)
_SUBJECT = "\x00subject"
_NEGATION = frozenset({"не", "ни", "никогда", "нет", "без"})
_MAX_EXAMPLES = 512


def _check_text(value: str, field: str, limit: int = 2048) -> None:
    if type(value) is not str or not value or len(value) > limit:
        raise ValueError(f"invalid {field}")
    value.encode("utf-8", errors="strict")


def _tokenize(text: str) -> tuple[tuple[str, Span], ...]:
    return tuple(
        (match.group().casefold(), Span(match.start(), match.end()))
        for match in _TOKENS.finditer(text)
    )


def _valid_saved_token(token: Any) -> bool:
    if type(token) is not str or not token or len(token) > 2048:
        return False
    try:
        token.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    if token == _SUBJECT:
        return True
    parsed = _tokenize(token)
    return (
        len(parsed) == 1
        and parsed[0][0] == token
        and parsed[0][1] == Span(0, len(token))
    )


def _template(
    tokens: tuple[tuple[str, Span], ...], span: Span
) -> tuple[str, ...] | None:
    """Replace exactly one token-aligned mention with a structural slot."""
    inside = [
        i
        for i, (_, item) in enumerate(tokens)
        if span.start <= item.start and item.end <= span.end
    ]
    if (
        not inside
        or tokens[inside[0]][1].start != span.start
        or tokens[inside[-1]][1].end != span.end
        or inside != list(range(inside[0], inside[-1] + 1))
    ):
        return None
    return (
        *(word for word, _ in tokens[: inside[0]]),
        _SUBJECT,
        *(word for word, _ in tokens[inside[-1] + 1 :]),
    )


@dataclass(frozen=True, slots=True)
class LabeledQuestion:
    """Open training item. ``family_id`` groups related document histories."""

    text: str
    family_id: str
    subject: IdentifiedMention
    relation_id: str
    asked_role: str

    def __post_init__(self) -> None:
        for field in ("text", "family_id", "relation_id", "asked_role"):
            _check_text(getattr(self, field), field)
        if not isinstance(self.subject, IdentifiedMention):
            raise ValueError("subject must be an annotated mention")
        if self.subject.span.end > len(self.text):
            raise ValueError("subject outside the question")
        tokens = _tokenize(self.text)
        if not tokens or tokens[-1][0] != "?":
            raise ValueError("only explicit questions are supported")
        if _template(tokens, self.subject.span) is None:
            raise ValueError("subject must have whole token boundaries")
        if any(word in _NEGATION for word, _ in tokens):
            raise ValueError("negated questions need a separate polarity model")


@dataclass(frozen=True, slots=True)
class ReviewedIdentity:
    """A surface-to-ID link drawn from the caller's authorized source view."""

    surface: str
    kind: str
    instance_id: str

    def __post_init__(self) -> None:
        for field in ("surface", "kind", "instance_id"):
            _check_text(getattr(self, field), field)
        if not self.surface.strip() or len(_tokenize(self.surface)) == 0:
            raise ValueError("identity surface needs a word")


@dataclass(frozen=True, slots=True)
class QueryHypothesis:
    subject_id: str | None
    relation_id: str | None
    asked_role: str | None
    text_sha256: str
    fully_covered: bool
    ambiguity: tuple[str, ...]
    residual: tuple[str, ...]
    subject_span: Span | None = None

    @property
    def resolved(self) -> bool:
        return (
            self.fully_covered
            and bool(self.subject_id and self.relation_id and self.asked_role)
            and not self.ambiguity
            and not self.residual
        )


class QuestionLanguageModel:
    """Learn a single-slot question grammar; never accept a partial match."""

    def __init__(self) -> None:
        self._patterns: dict[tuple[str, ...], set[tuple[str, str, str]]] = {}

    def fit(
        self,
        examples: tuple[LabeledQuestion, ...],
        *,
        held_out_families: frozenset[str] = frozenset(),
    ) -> QuestionLanguageModel:
        if type(examples) is not tuple or not 0 < len(examples) <= _MAX_EXAMPLES:
            raise ValueError("question training examples must be a bounded tuple")
        if type(held_out_families) is not frozenset or any(
            type(item) is not str or not item for item in held_out_families
        ):
            raise ValueError("held_out_families must be a set of family IDs")
        patterns: dict[tuple[str, ...], set[tuple[str, str, str]]] = {}
        for example in examples:
            if not isinstance(example, LabeledQuestion):
                raise ValueError("invalid annotated question")
            if example.family_id in held_out_families:
                raise ValueError("a held-out family cannot appear in training")
            pattern = _template(_tokenize(example.text), example.subject.span)
            assert pattern is not None  # validated when LabeledQuestion was made
            patterns.setdefault(pattern, set()).add(
                (example.subject.kind, example.relation_id, example.asked_role)
            )
        self._patterns = patterns
        return self

    def to_dict(self) -> dict[str, Any]:
        """Persist learned templates only, never annotated training questions."""
        if not self._patterns:
            raise RuntimeError("fit question examples before saving")
        return {
            "schema": "ai2-question-language-v1",
            "patterns": [
                {
                    "tokens": list(pattern),
                    "labels": [list(label) for label in sorted(labels)],
                }
                for pattern, labels in sorted(self._patterns.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> QuestionLanguageModel:
        if (
            type(value) is not dict
            or set(value) != {"schema", "patterns"}
            or value["schema"] != "ai2-question-language-v1"
            or type(value["patterns"]) is not list
            or not 0 < len(value["patterns"]) <= _MAX_EXAMPLES
        ):
            raise ValueError("invalid persisted question language model")
        patterns: dict[tuple[str, ...], set[tuple[str, str, str]]] = {}
        for row in value["patterns"]:
            if (
                type(row) is not dict
                or set(row) != {"tokens", "labels"}
                or type(row["tokens"]) is not list
                or not 1 < len(row["tokens"]) <= 128
                or type(row["labels"]) is not list
                or not 0 < len(row["labels"]) <= _MAX_EXAMPLES
            ):
                raise ValueError("invalid persisted question pattern")
            tokens = row["tokens"]
            if (
                any(not _valid_saved_token(token) for token in tokens)
                or tokens.count(_SUBJECT) != 1
                or tokens[-1] != "?"
            ):
                raise ValueError("invalid persisted question tokens")
            pattern = tuple(tokens)
            if pattern in patterns:
                raise ValueError("duplicate persisted question pattern")
            labels: set[tuple[str, str, str]] = set()
            for raw in row["labels"]:
                if type(raw) is not list or len(raw) != 3:
                    raise ValueError("invalid persisted question label")
                for label in raw:
                    _check_text(label, "question label")
                labels.add((raw[0], raw[1], raw[2]))
            if len(labels) != len(row["labels"]):
                raise ValueError("duplicate persisted question label")
            patterns[pattern] = labels
        model = cls()
        model._patterns = patterns
        if model.to_dict() != value:
            raise ValueError("noncanonical persisted question model")
        return model

    def parse(
        self,
        text: str,
        *,
        raw_model: RawSemanticModel,
        reviewed_identities: tuple[ReviewedIdentity, ...],
    ) -> QueryHypothesis:
        _check_text(text, "question")
        if not self._patterns:
            raise RuntimeError("fit question examples before parsing")
        if type(reviewed_identities) is not tuple or any(
            not isinstance(item, ReviewedIdentity) for item in reviewed_identities
        ):
            raise ValueError("reviewed_identities must be a tuple of identities")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

        def unresolved(reason: str, residual: tuple[str, ...]) -> QueryHypothesis:
            return QueryHypothesis(None, None, None, digest, False, (reason,), residual)

        tokens = _tokenize(text)
        if any(word in _NEGATION for word, _ in tokens):
            return unresolved("negative_question", ("неподдержанное отрицание",))
        if not tokens or tokens[-1][0] != "?":
            return unresolved("unknown_language", (text.strip(),))
        # The raw detector's IDs are source-local. Even exact seen surfaces
        # require identity resolution from the accessible catalog below.
        mentions = raw_model.infer_mentions(
            text, source_id=f"question:{digest}", source_version=1
        )
        catalog: dict[tuple[str, str], set[str]] = {}
        for entry in reviewed_identities:
            catalog.setdefault(
                (entry.surface.casefold().strip(), entry.kind), set()
            ).add(entry.instance_id)
        eligible = [
            mention
            for mention in mentions
            if (
                text[mention.span.start : mention.span.end].casefold().strip(),
                mention.kind,
            )
            in catalog
        ]
        if len(eligible) > 1:
            return unresolved(
                "multiple_subjects",
                tuple(text[item.span.start : item.span.end] for item in eligible),
            )
        matches: list[tuple[IdentifiedMention, tuple[str, str, str]]] = []
        for mention in mentions:
            pattern = _template(tokens, mention.span)
            if pattern is None:
                continue
            matches.extend(
                (mention, label)
                for label in self._patterns.get(pattern, ())
                if label[0] == mention.kind
            )
        if not matches:
            literals = {
                word
                for pattern in self._patterns
                for word in pattern
                if word != _SUBJECT
            }
            residual = tuple(
                text[span.start : span.end]
                for word, span in tokens
                if word not in literals and word != "?"
            )
            return unresolved("unknown_language", residual or (text.strip(),))
        if len(matches) != 1:
            return unresolved("ambiguous_template", (text.strip(),))
        mention, (_, relation_id, asked_role) = matches[0]
        surface = text[mention.span.start : mention.span.end]
        identities = catalog.get((surface.casefold().strip(), mention.kind), set())
        if len(identities) != 1:
            reason = "ambiguous_identity" if identities else "unresolved_identity"
            return QueryHypothesis(
                None,
                relation_id,
                asked_role,
                digest,
                False,
                (reason,),
                (surface,),
                mention.span,
            )
        return QueryHypothesis(
            next(iter(identities)),
            relation_id,
            asked_role,
            digest,
            True,
            (),
            (),
            mention.span,
        )

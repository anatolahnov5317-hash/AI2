"""Fitted, bounded structured understanding without inference-time templates.

Three numerical ridge models learn (1) complete-utterance semantic heads,
(2) typed entity-span role scores, and (3) contextual reference candidate scores.
Entity identity is masked from language/role features. A disclosed finite alias
table supplies entity spelling/case annotations, not predicate interpretations.
The ontology, caps, abstention checks and tree construction are scaffolding.

This is a small supervised research model, not unrestricted Russian dialogue.
Unknown vocabulary and multiple explicit sentences are rejected in full.
Learned unknown labels and score margins support abstention for missing roles,
ambiguous references and incoherent predictions; these are not a universal
completeness or safety certificate for novel language.
The fixed role ontology and finite alias morphology are not discovered concepts.
Reference generalization remains limited: unfamiliar combinations of context
entities can be misresolved even when their individual aliases are familiar.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from .language_data import (
    ENTITY_ALIASES,
    PRONOUN_FORMS,
    TRAIN_DATA_VERSION,
    TrainingUtterance,
    data_fingerprint,
    reference_examples,
    tokenize,
    training_examples,
)
from .schema import (
    DialogueContext,
    Entity,
    Event,
    Interpretation,
    Meaning,
    Query,
    bounded_text,
    exact_fields,
)

if TYPE_CHECKING:
    from .candidate_search import SearchLimits
    from .hypotheses import CandidateSet, Observation

HEAD_DIM = 384
ROLE_DIM = 512
REFERENCE_DIM = 256
MAX_EXAMPLES = 2048
MAX_TOKENS = 96
MAX_SPANS = 16
MAX_CHARS = 2048
_VERSION = "ai2-learned-understanding-v1"
_EMPTY_CONTEXT = DialogueContext()
_ROLES = (
    "actor",
    "object",
    "recipient",
    "place",
    "outer_actor",
    "inner_actor",
    "query_subject",
    "query_value",
)
_HEAD_OPTIONS = {
    "act": (
        "inform",
        "ask",
        "correct",
        "retract",
        "greet",
        "thanks",
        "help",
        "unknown",
    ),
    "predicate": ("", "locate", "move", "give", "have"),
    "outer": ("", "promise", "report", "conditional"),
    "inner": ("", "promise", "report", "conditional"),
    "negated": ("false", "true"),
    "outer_negated": ("false", "true"),
    "inner_negated": ("false", "true"),
    "modality": ("actual", "possible", "intended", "reported", "conditional"),
    "outer_modality": ("actual", "possible", "intended", "reported", "conditional"),
    "inner_modality": ("actual", "possible", "intended", "reported", "conditional"),
    "time": ("past", "present", "future", "unspecified"),
    "outer_time": ("past", "present", "future", "unspecified"),
    "inner_time": ("past", "present", "future", "unspecified"),
    "spatial": ("in", "on"),
    "query": ("", "where", "who_has", "what_has", "verify", "why"),
    "relation": ("", "location", "holder"),
    "query_time": ("past", "present", "future", "unspecified"),
}
_CONFIG = {
    "head_dim": HEAD_DIM,
    "role_dim": ROLE_DIM,
    "reference_dim": REFERENCE_DIM,
    "max_chars": MAX_CHARS,
    "max_tokens": MAX_TOKENS,
    "max_spans": MAX_SPANS,
    "ridge": 0.15,
    "head_margin": 0.035,
    "role_margin": 0.025,
    "reference_margin": 0.065,
}


def _finite_real(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


class _Deadline:
    def __init__(self, seconds: float) -> None:
        if not _finite_real(seconds) or not 0 < seconds <= 120:
            raise ValueError("seconds must be finite in (0, 120]")
        self.started = perf_counter()
        self.end = self.started + seconds

    def check(self) -> None:
        if perf_counter() >= self.end:
            raise TimeoutError("understanding training deadline exceeded")


@dataclass(frozen=True, slots=True)
class _Span:
    index: int
    entity: Entity | None = None
    case: str = ""
    pronoun: str = ""


def _spans(tokens: tuple[str, ...], context: DialogueContext) -> list[_Span]:
    context_names = {
        entity.name.casefold().replace("ё", "е"): entity for entity in context.entities
    }
    found: list[_Span] = []
    for index, word in enumerate(tokens):
        if word in ENTITY_ALIASES:
            entity, case = ENTITY_ALIASES[word]
            found.append(_Span(index, entity, case))
        elif word in PRONOUN_FORMS:
            found.append(_Span(index, None, PRONOUN_FORMS[word][1], word))
        elif word in context_names:
            found.append(_Span(index, context_names[word], "literal"))
    if len(found) > MAX_SPANS:
        raise ValueError("entity span limit")
    return found + [_Span(-1), _Span(-2, pronoun="<implicit>")]


def _masked(tokens: tuple[str, ...], spans: list[_Span]) -> tuple[str, ...]:
    annotations = {span.index: span for span in spans if span.index >= 0}
    result = []
    for index, word in enumerate(tokens):
        span = annotations.get(index)
        if span is None:
            result.append(word)
        elif span.entity is not None:
            result.append(f"E:{span.entity.kind}:{span.case}")
        else:
            result.append(f"P:{span.case}")
    return tuple(result)


def _bag(tokens: tuple[str, ...]) -> list[tuple[str, float]]:
    features: list[tuple[str, float]] = [(f"length:{min(len(tokens) // 3, 10)}", 0.5)]
    for i, word in enumerate(tokens):
        features.append((f"u:{word}", 1.0))
        features.append((f"p:{min(i, 5)}:{word}", 0.35))
        if i:
            features.append((f"b:{tokens[i - 1]}|{word}", 1.25))
        if i >= 2:
            features.append((f"t:{tokens[i - 2]}|{tokens[i - 1]}|{word}", 0.5))
    return features


def _vector(features: list[tuple[str, float]], dimension: int, seed: int) -> np.ndarray:
    result = np.zeros(dimension, dtype=np.float64)
    result[0] = 1.0
    key = seed.to_bytes(4, "little")
    for feature, value in features:
        digest = hashlib.blake2b(feature.encode(), key=key, digest_size=8).digest()
        at = 1 + int.from_bytes(digest, "little") % (dimension - 1)
        result[at] += value
    result /= max(float(np.linalg.norm(result)), 1.0)
    return result


def _head_vector(
    tokens: tuple[str, ...], spans: list[_Span], context: DialogueContext, seed: int
) -> np.ndarray:
    features = _bag(_masked(tokens, spans))
    features.append((f"mentions:{len(spans) - 2}", 1.0))
    features.append((f"pending:{context.pending}", 0.35))
    for kind in ("person", "thing", "place", "unknown"):
        count = sum(entity.kind == kind for entity in context.entities)
        features.append((f"context:{kind}:{min(count, 3)}", 0.2))
    # Previous text is observable input, not hidden gold state.
    if context.turns:
        prior = tokenize(context.turns[-1])[:MAX_TOKENS]
        prior_spans = _spans(prior, DialogueContext())
        for word in _masked(prior, prior_spans):
            features.append((f"prior:{word}", 0.12))
    return _vector(features, HEAD_DIM, seed)


def _role_vectors(tokens: tuple[str, ...], spans: list[_Span], seed: int) -> np.ndarray:
    masked = _masked(tokens, spans)
    bag = _bag(masked)
    values = []
    count = len(spans) - 2
    for ordinal, span in enumerate(spans):
        if span.index < 0:
            marker = "NULL" if span.index == -1 else "IMPLICIT"
            features = [(marker, 2.0)] + [
                (f"{marker}:{key}", value) for key, value in bag
            ]
        else:
            kind = span.entity.kind if span.entity else "reference"
            features = [
                (f"kind:{kind}", 1.5),
                (f"case:{span.case}", 1.0),
                (f"ordinal:{ordinal}", 1.0),
                (f"reverse:{count - ordinal}", 0.7),
                (f"ordinal_kind:{ordinal}:{kind}", 1.25),
                (f"case_kind:{span.case}:{kind}", 1.0),
                (f"count_kind:{count}:{kind}", 0.65),
            ]
            features.extend((f"global:{key}", value * 0.18) for key, value in bag)
            features.extend(
                (f"global_kind:{kind}:{key}", value * 0.4) for key, value in bag
            )
            for offset in (-3, -2, -1, 1, 2, 3):
                at = span.index + offset
                word = masked[at] if 0 <= at < len(masked) else "<edge>"
                features.append(
                    (f"window:{offset}:{word}", 1.2 if abs(offset) == 1 else 0.6)
                )
                features.append((f"window_kind:{offset}:{word}:{kind}", 0.6))
        values.append(_vector(features, ROLE_DIM, seed))
    return np.stack(values)


def _reference_vectors(
    pronoun: str, role: str, context: DialogueContext, seed: int
) -> tuple[list[Entity | None], np.ndarray]:
    candidates: list[Entity | None] = [*context.entities, None]
    values = []
    counts: dict[tuple[str, str], int] = {}
    for entity in context.entities:
        key = (entity.kind, entity.gender)
        counts[key] = counts.get(key, 0) + 1
    for candidate in candidates:
        kind = candidate.kind if candidate else "NULL"
        gender = candidate.gender if candidate else "NULL"
        features = [
            (f"kind:{kind}", 1.0),
            (f"role_kind:{role}:{kind}", 1.5),
            (f"pron_gender:{pronoun}:{gender}", 1.5),
            (f"pron_role_gender:{pronoun}:{role}:{gender}", 1.5),
            (f"pron_role_kind_gender:{pronoun}:{role}:{kind}:{gender}", 1.5),
            (f"pron_role_kind:{pronoun}:{role}:{kind}", 0.8),
            (f"total:{min(len(context.entities), 5)}:{kind}", 0.7),
        ]
        for (other_kind, other_gender), count in counts.items():
            features.append(
                (
                    f"counts:{role}:{pronoun}:{kind}:{other_kind}:"
                    f"{other_gender}:{min(count, 3)}",
                    0.75,
                )
            )
        if candidate:
            rank = (
                context.focus.index(candidate.name)
                if candidate.name in context.focus
                else 16
            )
            features.extend(
                [
                    (f"focus:{pronoun}:{min(rank, 3)}", 0.8),
                    (f"role_focus:{role}:{min(rank, 3)}", 0.4),
                    (f"same_kind_gender:{role}:{min(counts[(kind, gender)], 3)}", 1.0),
                ]
            )
        values.append(_vector(features, REFERENCE_DIM, seed))
    return candidates, np.stack(values)


def _targets(meaning: Meaning) -> tuple[dict[str, str], dict[str, str]]:
    heads: dict[str, str] = {
        name: options[0] for name, options in _HEAD_OPTIONS.items()
    }
    heads.update(
        act=meaning.act,
        predicate="",
        outer="",
        inner="",
        query="",
        relation="",
        time="past",
        query_time="present",
    )
    roles: dict[str, str] = dict.fromkeys(_ROLES, "")
    if meaning.query:
        q = meaning.query
        heads.update(
            query=q.kind,
            relation=q.relation,
            query_time=q.time,
            spatial=q.spatial,
            negated=str(q.negated).lower(),
        )
        roles.update(query_subject=q.subject, query_value=q.value)
    if meaning.event:
        node = meaning.event
        depth = 0
        while node.content is not None:
            if node.condition is not None or depth >= 2:
                raise ValueError("training tree outside supported scope depth")
            prefix = "outer" if depth == 0 else "inner"
            heads[prefix] = node.predicate
            heads[f"{prefix}_negated"] = str(node.negated).lower()
            heads[f"{prefix}_modality"] = node.modality
            heads[f"{prefix}_time"] = node.time
            roles[f"{prefix}_actor"] = node.actor
            if node.object or node.recipient or node.place:
                raise ValueError("unsupported non-actor wrapper role")
            node = node.content
            depth += 1
        heads.update(
            predicate=node.predicate,
            negated=str(node.negated).lower(),
            modality=node.modality,
            time=node.time,
            spatial=node.spatial,
        )
        roles.update(
            actor=node.actor,
            object=node.object,
            recipient=node.recipient,
            place=node.place,
        )
    return heads, roles


def _ridge(x: np.ndarray, y: np.ndarray, deadline: _Deadline) -> np.ndarray:
    deadline.check()
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += _CONFIG["ridge"]
    result = np.linalg.solve(gram, x.T @ y)
    deadline.check()
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite learned parameters")
    return result


def _best(scores: np.ndarray) -> tuple[int, float]:
    if not np.all(np.isfinite(scores)) or scores.ndim != 1 or len(scores) == 0:
        raise ValueError("invalid numeric prediction")
    order = np.argsort(scores)
    winner = int(order[-1])
    margin = float(scores[winner] - scores[int(order[-2])]) if len(order) > 1 else 1.0
    return winner, margin


class LearnedUnderstanding:
    """Structured numerical learner; fitting creates a new independent model."""

    def __init__(self, seed: int = 42) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("invalid model seed")
        self.seed = seed
        self.fingerprint = ""
        self.trained = False
        self.vocabulary: frozenset[str] = frozenset()
        self._labels: dict[str, tuple[str, ...]] = {}
        self._weights: dict[str, np.ndarray] = {}
        self._role_weights = np.zeros((ROLE_DIM, len(_ROLES)))
        self._reference_weights = np.zeros((REFERENCE_DIM, 1))
        self.training_metrics: dict[str, Any] = {}

    @classmethod
    def fit(
        cls,
        examples: Sequence[TrainingUtterance] | None = None,
        *,
        seconds: float = 30.0,
        seed: int = 42,
        shuffle_targets: bool = False,
    ) -> LearnedUnderstanding:
        """Fit utterances plus the disclosed bundled reference curriculum.

        Custom ``examples`` replace only utterance supervision. Reference
        episodes and the fixed entity annotations remain bundled scaffolding;
        the fingerprint covers the supplied utterances and reference episodes.
        """
        deadline = _Deadline(seconds)
        if type(shuffle_targets) is not bool:
            raise ValueError("shuffle_targets must be boolean")
        model = cls(seed)
        examples = training_examples(seed) if examples is None else examples
        if (
            not isinstance(examples, (list, tuple))
            or not 1 <= len(examples) <= MAX_EXAMPLES
        ):
            raise ValueError("training examples must be bounded list/tuple")
        if any(not isinstance(example, TrainingUtterance) for example in examples):
            raise ValueError("expected annotated TrainingUtterance examples")
        rng = np.random.default_rng(seed)
        ref_samples = reference_examples()
        reference_payload = [
            {
                "pronoun": sample.pronoun,
                "role": sample.role,
                "context": sample.context.to_dict(),
                "target": sample.target,
            }
            for sample in ref_samples
        ]
        fingerprint = hashlib.sha256(
            (
                data_fingerprint(examples)
                + ":"
                + TRAIN_DATA_VERSION
                + ":"
                + json.dumps(reference_payload, sort_keys=True, ensure_ascii=False)
            ).encode()
        ).hexdigest()
        if shuffle_targets:
            fingerprint = hashlib.sha256(
                (fingerprint + ":shuffled").encode()
            ).hexdigest()
        x_rows, targets, role_x, role_y, role_groups = [], [], [], [], []
        vocabulary: set[str] = set()
        for example in examples:
            deadline.check()
            tokens = tokenize(example.text)
            if not 1 <= len(tokens) <= MAX_TOKENS or any(
                len(word) > 128 for word in tokens
            ):
                raise ValueError("training token capacity")
            spans = _spans(tokens, example.context)
            vocabulary.update(
                word
                for word in tokens
                if word not in ENTITY_ALIASES and word not in PRONOUN_FORMS
            )
            head_y, roles = _targets(example.meaning)
            targets.append(head_y)
            x_rows.append(_head_vector(tokens, spans, example.context, seed))
            values = _role_vectors(tokens, spans, seed)
            gold = np.zeros((len(spans), len(_ROLES)))
            links = dict(example.links)
            for column, role in enumerate(_ROLES):
                target = roles[role]
                matches = [
                    i
                    for i, span in enumerate(spans)
                    if target
                    and (
                        (span.entity is not None and span.entity.name == target)
                        or links.get(span.index) == target
                    )
                ]
                if not target:
                    matches = [len(spans) - 2]
                elif (
                    not matches
                    and role == "query_subject"
                    and any(
                        entity.name == target for entity in example.context.entities
                    )
                ):
                    matches = [len(spans) - 1]
                if not matches:
                    raise ValueError(
                        f"training role {role} has no annotated surface span"
                    )
                gold[matches, column] = 1.0
            role_groups.append((len(role_x), len(spans)))
            role_x.extend(values)
            role_y.extend(gold)
        if len(vocabulary) > 1024:
            raise ValueError("training vocabulary capacity")
        x = np.stack(x_rows)
        all_y, offsets = [], {}
        offset = 0
        for name, allowed in _HEAD_OPTIONS.items():
            labels = tuple(
                label for label in allowed if any(row[name] == label for row in targets)
            )
            if not labels:
                raise ValueError("unrepresented semantic head")
            model._labels[name] = labels
            columns = np.zeros((len(examples), len(labels)))
            for i, row in enumerate(targets):
                columns[i, labels.index(row[name])] = 1.0
            all_y.append(columns)
            offsets[name] = (offset, offset + len(labels))
            offset += len(labels)
        y = np.concatenate(all_y, axis=1)
        role_x_array, role_y_array = np.stack(role_x), np.stack(role_y)
        if shuffle_targets:
            y = y[rng.permutation(len(y))]
            role_y_array = role_y_array[rng.permutation(len(role_y_array))]
        learned = _ridge(x, y, deadline)
        for name, (start, end) in offsets.items():
            model._weights[name] = learned[:, start:end].copy()
        model._role_weights = _ridge(role_x_array, role_y_array, deadline)

        ref_x, ref_y, ref_groups = [], [], []
        for sample in ref_samples:
            deadline.check()
            candidates, values = _reference_vectors(
                sample.pronoun, sample.role, sample.context, seed
            )
            target = [
                1.0 if (candidate.name if candidate else "") == sample.target else 0.0
                for candidate in candidates
            ]
            ref_groups.append((len(ref_x), len(candidates)))
            ref_x.extend(values)
            ref_y.extend(target)
        ref_x_array = np.stack(ref_x)
        ref_y_array = np.asarray(ref_y).reshape(-1, 1)
        if shuffle_targets:
            ref_y_array = ref_y_array[rng.permutation(len(ref_y_array))]
        model._reference_weights = _ridge(ref_x_array, ref_y_array, deadline)
        predictions = x @ learned
        head_success = sum(
            int(np.argmax(predictions[i, start:end]) == np.argmax(y[i, start:end]))
            for start, end in offsets.values()
            for i in range(len(examples))
        )
        role_predictions = role_x_array @ model._role_weights
        role_success = sum(
            int(
                role_y_array[
                    start
                    + int(np.argmax(role_predictions[start : start + count, col])),
                    col,
                ]
                == 1
            )
            for start, count in role_groups
            for col in range(len(_ROLES))
        )
        ref_predictions = (ref_x_array @ model._reference_weights).ravel()
        ref_success = sum(
            int(
                ref_y_array[
                    start + int(np.argmax(ref_predictions[start : start + count])), 0
                ]
                == 1
            )
            for start, count in ref_groups
        )
        model.training_metrics = {
            "examples": len(examples),
            "role_rows": len(role_x),
            "reference_rows": len(ref_x),
            "head_accuracy": head_success / (len(examples) * len(_HEAD_OPTIONS)),
            "role_training_accuracy": role_success / (len(role_groups) * len(_ROLES)),
            "reference_training_accuracy": ref_success / len(ref_groups),
            "seconds": perf_counter() - deadline.started,
            "shuffle_targets": shuffle_targets,
        }
        model.vocabulary = frozenset(vocabulary)
        model.fingerprint = fingerprint
        model.trained = True
        deadline.check()
        return model

    def _reference(
        self, pronoun: str, role: str, context: DialogueContext
    ) -> tuple[Entity | None, float]:
        candidates, values = _reference_vectors(pronoun, role, context, self.seed)
        scores = (values @ self._reference_weights).ravel()
        best, margin = _best(scores)
        if margin < _CONFIG["reference_margin"]:
            return None, margin
        return candidates[best], margin

    def interpret(
        self, text: str, context: DialogueContext = _EMPTY_CONTEXT
    ) -> Interpretation:
        if not self.trained:
            return Interpretation(None, reason="untrained_understanding")
        if type(text) is not str or not 0 < len(text) <= MAX_CHARS:
            return Interpretation(None, reason="input_capacity")
        try:
            text.encode("utf-8")
        except UnicodeError:
            return Interpretation(None, reason="invalid_unicode")
        if not isinstance(context, DialogueContext):
            return Interpretation(None, reason="invalid_context")
        if any(ord(char) < 32 or ord(char) == 127 for char in text):
            return Interpretation(
                None, reason="unsupported_multiple_or_controlled_input"
            )
        tokens = tokenize(text)
        if not 1 <= len(tokens) <= MAX_TOKENS or any(
            len(word) > 128 for word in tokens
        ):
            return Interpretation(None, reason="token_capacity")
        # Sentence delimiters are structural capacity checks, not verb rules.
        if any(word in {".", "?", "!", ";"} for word in tokens[:-1]):
            return Interpretation(None, reason="multiple_events_unsupported")
        try:
            spans = _spans(tokens, context)
        except ValueError:
            return Interpretation(None, reason="entity_capacity")
        covered_indices = {span.index for span in spans if span.index >= 0}
        unknown = tuple(
            sorted(
                {
                    word
                    for i, word in enumerate(tokens)
                    if i not in covered_indices and word not in self.vocabulary
                }
            )
        )
        if unknown:
            return Interpretation(
                None,
                reason="unknown_lexeme",
                diagnostics={"unknown": list(unknown[:8])},
            )
        if len({entity.name for entity in context.entities}) != len(context.entities):
            return Interpretation(None, reason="ambiguous_context_entity_identity")
        try:
            x = _head_vector(tokens, spans, context, self.seed)
        except ValueError:
            return Interpretation(None, reason="context_capacity")
        predictions, margins = {}, {}
        for name in _HEAD_OPTIONS:
            best, margin = _best(x @ self._weights[name])
            predictions[name] = self._labels[name][best]
            margins[name] = margin
        diagnostics: dict[str, Any] = {
            "model": _VERSION,
            "training_version": TRAIN_DATA_VERSION,
            "fingerprint": self.fingerprint,
            "heads": predictions,
            "head_margins": margins,
            "score_is_calibrated_probability": False,
        }
        minimum = min(margins.values())
        if minimum < _CONFIG["head_margin"]:
            return Interpretation(
                None,
                max(0.0, minimum),
                reason="uncertain_semantic_head",
                diagnostics=diagnostics,
            )
        if predictions["act"] == "unknown":
            return Interpretation(
                None,
                max(0.0, minimum),
                reason="unsupported_learned_utterance",
                diagnostics=diagnostics,
            )
        try:
            required = self._required_roles(predictions)
        except (ValueError, KeyError):
            return Interpretation(
                None,
                max(0.0, minimum),
                reason="incoherent_learned_structure",
                diagnostics=diagnostics,
            )
        role_scores = _role_vectors(tokens, spans, self.seed) @ self._role_weights
        roles: dict[str, str] = dict.fromkeys(_ROLES, "")
        entities = {span.entity.name: span.entity for span in spans if span.entity}
        entities.update({entity.name: entity for entity in context.entities})
        consumed: set[int] = set()
        role_margins, reference_margins = {}, {}
        for role in required:
            column = _ROLES.index(role)
            best, margin = _best(role_scores[:, column])
            role_margins[role] = margin
            span = spans[best]
            minimum = min(minimum, margin)
            # The ontology permits explanation of an entire preceding answer.
            # Its empty subject is selected by learned NULL role scores, never
            # by checking a surface phrase or choosing a context entity.
            optional_subject = (
                predictions["act"] == "ask"
                and predictions["query"] == "why"
                and role == "query_subject"
            )
            if margin < _CONFIG["role_margin"] or (
                span.index == -1 and not optional_subject
            ):
                diagnostics["role_margins"] = role_margins
                return Interpretation(
                    None,
                    max(0.0, minimum),
                    reason="uncertain_or_missing_role",
                    diagnostics=diagnostics,
                )
            if span.index == -1:
                continue
            if span.entity is not None:
                entity = span.entity
            else:
                entity, reference_margin = self._reference(span.pronoun, role, context)
                reference_margins[role] = reference_margin
                minimum = min(minimum, reference_margin)
                if entity is None:
                    diagnostics.update(
                        role_margins=role_margins, reference_margins=reference_margins
                    )
                    return Interpretation(
                        None,
                        max(0.0, minimum),
                        reason="ambiguous_or_missing_reference",
                        diagnostics=diagnostics,
                    )
            roles[role] = entity.name
            entities[entity.name] = entity
            if span.index >= 0:
                consumed.add(span.index)
        diagnostics.update(
            role_margins=role_margins, reference_margins=reference_margins
        )
        if covered_indices != consumed:
            return Interpretation(
                None,
                max(0.0, minimum),
                reason="unconsumed_entity_span",
                diagnostics=diagnostics,
            )
        try:
            meaning = self._build(predictions, roles, entities)
        except (ValueError, KeyError):
            return Interpretation(
                None,
                max(0.0, minimum),
                reason="incoherent_learned_meaning",
                diagnostics=diagnostics,
            )
        return Interpretation(
            meaning, min(1.0, max(0.0, minimum)), diagnostics=diagnostics
        )

    def propose(
        self,
        text: str,
        context: DialogueContext = _EMPTY_CONTEXT,
        *,
        observation: Observation | None = None,
        initial: Interpretation | None = None,
        limits: SearchLimits | None = None,
    ) -> CandidateSet:
        """Retain bounded alternatives; the caller decides before projecting facts."""
        from .candidate_search import propose
        from .hypotheses import Observation

        observation = observation or Observation("input", text)
        if observation.text != text or not isinstance(context, DialogueContext):
            raise ValueError("candidate input does not match its observation")
        return propose(
            self, observation, context, initial or self.interpret(text, context), limits
        )

    @staticmethod
    def _required_roles(heads: dict[str, str]) -> tuple[str, ...]:
        act = heads["act"]
        if act in {"greet", "thanks", "help", "retract"}:
            if heads["predicate"] or heads["query"] or heads["outer"] or heads["inner"]:
                raise ValueError("content in social/retraction act")
            return ()
        if act == "ask":
            if (
                heads["predicate"]
                or heads["outer"]
                or heads["inner"]
                or not heads["query"]
            ):
                raise ValueError("incoherent query")
            if (
                heads["query"] == "verify"
                and heads["relation"] not in {"location", "holder"}
            ) or (heads["query"] != "verify" and heads["relation"]):
                raise ValueError("incoherent query relation")
            return (
                ("query_subject", "query_value")
                if heads["query"] == "verify"
                else ("query_subject",)
            )
        if act not in {"inform", "correct"} or heads["query"]:
            raise ValueError("unsupported act")
        if heads["inner"] and not heads["outer"]:
            raise ValueError("nested wrapper without parent")
        role_schema = {
            "locate": ("object", "place"),
            "move": ("actor", "object", "place"),
            "give": ("actor", "object", "recipient"),
            "have": ("actor", "object"),
        }
        roles = list(role_schema[heads["predicate"]])
        for scope in ("outer", "inner"):
            if heads[scope] and heads[scope] != "conditional":
                roles.append(f"{scope}_actor")
        return tuple(roles)

    @staticmethod
    def _build(
        heads: dict[str, str], roles: dict[str, str], entities: dict[str, Entity]
    ) -> Meaning:
        event, query = None, None
        if heads["act"] in {"inform", "correct"}:
            event = Event(
                heads["predicate"],
                actor=roles["actor"],
                object=roles["object"],
                recipient=roles["recipient"],
                place=roles["place"],
                spatial=heads["spatial"],
                negated=heads["negated"] == "true",
                modality=heads["modality"],
                time=heads["time"],
            )
            for scope in ("inner", "outer"):
                if heads[scope]:
                    event = Event(
                        heads[scope],
                        actor=roles[f"{scope}_actor"],
                        content=event,
                        negated=heads[f"{scope}_negated"] == "true",
                        modality=heads[f"{scope}_modality"],
                        time=heads[f"{scope}_time"],
                    )
        elif heads["act"] == "ask":
            query = Query(
                heads["query"],
                subject=roles["query_subject"],
                value=roles["query_value"],
                relation=heads["relation"],
                time=heads["query_time"],
                spatial=heads["spatial"],
                negated=heads["negated"] == "true",
            )
        names: list[str] = []

        def visit(node: Event) -> None:
            for name in (node.actor, node.object, node.recipient, node.place):
                if name and name not in names:
                    names.append(name)
            if node.content:
                visit(node.content)

        if event:
            visit(event)
        if query:
            names.extend(
                name
                for name in (query.subject, query.value)
                if name and name not in names
            )
        return Meaning(
            heads["act"], event, query, tuple(entities[name] for name in names)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "trained": self.trained,
            "config": dict(_CONFIG),
            "vocabulary": sorted(self.vocabulary),
            "heads": {
                name: {
                    "labels": list(self._labels[name]),
                    "weights": self._weights[name].tolist(),
                }
                for name in self._labels
            },
            "role_weights": self._role_weights.tolist(),
            "reference_weights": self._reference_weights.tolist(),
            "training_metrics": dict(self.training_metrics),
        }

    @classmethod
    def from_dict(cls, value: Any) -> LearnedUnderstanding:
        value = exact_fields(
            value,
            {
                "version",
                "seed",
                "fingerprint",
                "trained",
                "config",
                "vocabulary",
                "heads",
                "role_weights",
                "reference_weights",
                "training_metrics",
            },
            "understanding checkpoint",
        )
        if value["version"] != _VERSION or type(value["trained"]) is not bool:
            raise ValueError("invalid understanding version/trained flag")
        if type(value["config"]) is not dict or value["config"] != _CONFIG:
            raise ValueError("unsupported understanding dimensions/configuration")
        if any(
            type(value["config"][key]) is not type(expected)
            for key, expected in _CONFIG.items()
        ):
            raise ValueError("invalid configuration number type")
        model = cls(value["seed"])
        fingerprint = value["fingerprint"]
        if type(fingerprint) is not str or (
            value["trained"]
            and (
                len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
            )
        ):
            raise ValueError("invalid training fingerprint")
        if not value["trained"] and fingerprint:
            raise ValueError("untrained checkpoint has fingerprint")
        vocabulary = value["vocabulary"]
        if type(vocabulary) is not list or len(vocabulary) > 1024:
            raise ValueError("invalid vocabulary")
        for word in vocabulary:
            bounded_text(word, "vocabulary token", empty=False)
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("duplicate vocabulary token")
        heads = value["heads"]
        if type(heads) is not dict or set(heads) != (
            set(_HEAD_OPTIONS) if value["trained"] else set()
        ):
            raise ValueError("invalid semantic heads")
        # Validate every bounded list/value before conversion to numerical arrays.
        validated: dict[str, tuple[list[str], list[list[float]]]] = {}
        for name, head in heads.items():
            head = exact_fields(head, {"labels", "weights"}, "semantic head")
            labels = head["labels"]
            if (
                type(labels) is not list
                or not 1 <= len(labels) <= len(_HEAD_OPTIONS[name])
                or any(
                    type(label) is not str or label not in _HEAD_OPTIONS[name]
                    for label in labels
                )
                or len(labels) != len(set(labels))
            ):
                raise ValueError("invalid semantic head labels")
            _validate_matrix(head["weights"], HEAD_DIM, len(labels))
            validated[name] = (labels, head["weights"])
        _validate_matrix(value["role_weights"], ROLE_DIM, len(_ROLES))
        _validate_matrix(value["reference_weights"], REFERENCE_DIM, 1)
        _validate_metrics(value["training_metrics"], value["trained"])
        model.trained, model.fingerprint = value["trained"], fingerprint
        model.vocabulary = frozenset(vocabulary)
        for name, (labels, weights) in validated.items():
            model._labels[name] = tuple(labels)
            model._weights[name] = np.asarray(weights, dtype=np.float64)
        model._role_weights = np.asarray(value["role_weights"], dtype=np.float64)
        model._reference_weights = np.asarray(
            value["reference_weights"], dtype=np.float64
        )
        model.training_metrics = dict(value["training_metrics"])
        return model


def _validate_matrix(value: Any, rows: int, columns: int) -> None:
    if type(value) is not list or len(value) != rows:
        raise ValueError("invalid learned matrix dimensions")
    for row in value:
        if type(row) is not list or len(row) != columns:
            raise ValueError("invalid learned matrix dimensions")
        for number in row:
            if not _finite_real(number) or abs(number) > 1e6:
                raise ValueError("invalid learned matrix value")


def _validate_metrics(value: Any, trained: bool) -> None:
    if not trained:
        if type(value) is not dict or value:
            raise ValueError("invalid untrained metrics")
        return
    value = exact_fields(
        value,
        {
            "examples",
            "role_rows",
            "reference_rows",
            "head_accuracy",
            "role_training_accuracy",
            "reference_training_accuracy",
            "seconds",
            "shuffle_targets",
        },
        "training metrics",
    )
    for name, cap in (
        ("examples", MAX_EXAMPLES),
        ("role_rows", MAX_EXAMPLES * (MAX_SPANS + 2)),
        ("reference_rows", 16384),
    ):
        if type(value[name]) is not int or not 1 <= value[name] <= cap:
            raise ValueError("invalid training count")
    for name in (
        "head_accuracy",
        "role_training_accuracy",
        "reference_training_accuracy",
        "seconds",
    ):
        cap = 120 if name == "seconds" else 1
        if not _finite_real(value[name]) or not 0 <= value[name] <= cap:
            raise ValueError("invalid training metric")
    if type(value["shuffle_targets"]) is not bool:
        raise ValueError("invalid shuffled training metadata")


def default_understanding(
    seed: int = 42, *, seconds: float = 30.0
) -> LearnedUnderstanding:
    """Explicit bounded demo fit; production should load an existing checkpoint."""
    return LearnedUnderstanding.fit(seconds=seconds, seed=seed)

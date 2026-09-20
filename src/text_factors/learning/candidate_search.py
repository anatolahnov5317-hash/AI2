"""Limited structural search over learned head, role and reference scores.

The finite ontology, margin window, type/gender checks and search budget are
explicit scaffolding. No phrase templates or independent combinations of all
labels are enumerated. NULL reference scores mean unresolved reference, not
that all real entities are disproved. Such candidates still need selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np

from .hypotheses import (
    MAX_CANDIDATES,
    MAX_EXPANSIONS,
    CandidateSet,
    Hypothesis,
    Observation,
    digest,
)
from .language_data import PRONOUN_FORMS, tokenize
from .schema import DialogueContext, Interpretation

WINDOW = 0.12  # Uncalibrated score difference, fixed before the development run.


@dataclass(frozen=True)
class SearchLimits:
    max_candidates: int = MAX_CANDIDATES
    max_expansions: int = MAX_EXPANSIONS
    seconds: float = 0.4

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_candidates, MAX_CANDIDATES),
            (self.max_expansions, MAX_EXPANSIONS),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid candidate search capacity")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or not 0 < self.seconds <= 60
        ):
            raise ValueError("invalid candidate search time")


def propose(
    model: Any,
    observation: Observation,
    context: DialogueContext,
    initial: Interpretation,
    limits: SearchLimits | None = None,
) -> CandidateSet:
    from .understanding import (
        _HEAD_OPTIONS,
        _ROLES,
        _head_vector,
        _reference_vectors,
        _role_vectors,
        _spans,
    )

    limits = limits or SearchLimits()
    started = perf_counter()
    expansions = 0
    stop_reason = ""
    candidates: dict[str, Hypothesis] = {}
    context_digest = digest(context.to_dict())

    def check(*, expand: bool = False) -> None:
        nonlocal expansions
        if perf_counter() - started >= limits.seconds:
            raise InterruptedError("candidate_time_budget")
        if expand:
            if expansions >= limits.max_expansions:
                raise InterruptedError("candidate_expansion_budget")
            expansions += 1

    def choices(scores: np.ndarray) -> list[tuple[int, float]]:
        check()
        values = np.asarray(scores).ravel()
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("invalid candidate scores")
        top = float(max(values))
        return [
            (i, top - float(values[i]))
            for i in sorted(range(len(values)), key=lambda i: (-values[i], i))
            if top - float(values[i]) <= WINDOW
        ]

    def add(meaning, regret, bindings=(), missing=()):
        candidate = Hypothesis.create(
            observation.observation_id, meaning, regret, tuple(bindings), tuple(missing)
        )
        old = candidates.get(candidate.hypothesis_id)
        if old is None or regret < old.language_regret:
            candidates[candidate.hypothesis_id] = candidate

    # Reject capacity/unknown-vocabulary failures in full. Compatibility with
    # explicitly supplied Interpretation objects also keeps the public API usable.
    details = initial.diagnostics or {}
    if "heads" not in details:
        for meaning in dict.fromkeys((initial.meaning, *initial.alternatives)):
            if meaning is not None:
                add(meaning, 0.0)
        return CandidateSet(observation, context_digest, tuple(candidates.values()))
    recoverable = {
        "",
        "uncertain_semantic_head",
        "unsupported_learned_utterance",
        "uncertain_or_missing_role",
        "ambiguous_or_missing_reference",
        "incoherent_learned_structure",
        "incoherent_learned_meaning",
        "unconsumed_entity_span",
    }
    if initial.reason not in recoverable:
        return CandidateSet(observation, context_digest, ())
    tokens = tokenize(observation.text)
    try:
        check()
        spans = _spans(tokens, context)
        x = _head_vector(tokens, spans, context, model.seed)
        beam: list[tuple[float, dict[str, str]]] = [(0.0, {})]
        for name in _HEAD_OPTIONS:
            options = choices(x @ model._weights[name])
            next_beam = []
            for regret, heads in beam:
                # Unused scope attributes are not additional semantic branches.
                unused = (name.startswith("outer_") and not heads.get("outer")) or (
                    name.startswith("inner_") and not heads.get("inner")
                )
                for at, loss in options[:1] if unused else options:
                    check(expand=True)
                    next_beam.append(
                        (regret + loss, {**heads, name: model._labels[name][at]})
                    )
            next_beam.sort(key=lambda row: (row[0], sorted(row[1].items())))
            if len(next_beam) > limits.max_candidates:
                stop_reason = "candidate_beam_capacity"
            beam = next_beam[: limits.max_candidates]
        all_entities = {span.entity.name: span.entity for span in spans if span.entity}
        all_entities.update({entity.name: entity for entity in context.entities})
        expected_spans = {span.index for span in spans if span.index >= 0}
        role_scores = _role_vectors(tokens, spans, model.seed) @ model._role_weights
        check()
        for head_regret, heads in beam:
            try:
                required = model._required_roles(heads)
            except (ValueError, KeyError):
                continue
            # score, bound roles, evidence bindings, missing roles, used mentions
            states: list[tuple[float, dict[str, str], tuple, tuple, frozenset]] = [
                (head_regret, dict.fromkeys(_ROLES, ""), (), (), frozenset())
            ]
            for role in required:
                options = []
                for at, span_loss in choices(role_scores[:, _ROLES.index(role)]):
                    check()
                    span = spans[at]
                    optional = (
                        heads["act"] == "ask"
                        and heads["query"] == "why"
                        and role == "query_subject"
                    )
                    if span.index == -1:
                        options.append(
                            (span_loss, None, -1, () if optional else (role,))
                        )
                        continue
                    if span.entity is not None:
                        options.append((span_loss, span.entity, span.index, ()))
                        continue
                    refs, vectors = _reference_vectors(
                        span.pronoun, role, context, model.seed
                    )
                    scores = (vectors @ model._reference_weights).ravel()
                    gender = PRONOUN_FORMS.get(span.pronoun, ("unknown", ""))[0]
                    kind = {
                        "actor": "person",
                        "recipient": "person",
                        "outer_actor": "person",
                        "inner_actor": "person",
                        "object": "thing",
                        "place": "place",
                    }.get(role)
                    valid = [
                        (entity, float(scores[i]))
                        for i, entity in enumerate(refs)
                        if entity is not None
                        and (not kind or entity.kind in {kind, "unknown"})
                        and (
                            gender == "unknown" or entity.gender in {gender, "unknown"}
                        )
                    ]
                    if not valid:
                        options.append((span_loss, None, span.index, (role,)))
                        continue
                    best_ref = max(score for _, score in valid)
                    for entity, score in sorted(
                        valid, key=lambda item: (-item[1], item[0].name)
                    ):
                        if best_ref - score <= WINDOW:
                            options.append(
                                (span_loss + best_ref - score, entity, span.index, ())
                            )
                next_states = []
                for regret, roles, bindings, missing, used in states:
                    for loss, entity, index, absent in options:
                        check(expand=True)
                        expected_kind = {
                            "actor": "person",
                            "recipient": "person",
                            "object": "thing",
                            "place": "place",
                            "outer_actor": "person",
                            "inner_actor": "person",
                        }.get(role)
                        if (
                            entity
                            and expected_kind
                            and entity.kind not in {expected_kind, "unknown"}
                        ):
                            continue
                        # Atomic participants cannot consume the same mention in
                        # two distinct roles. Wrapper actors can share a mention.
                        if (
                            index >= 0
                            and index in used
                            and role not in {"outer_actor", "inner_actor"}
                        ):
                            continue
                        next_states.append(
                            (
                                regret + loss,
                                {**roles, role: entity.name if entity else ""},
                                bindings + ((role, index, entity.name),)
                                if entity
                                else bindings,
                                missing + absent,
                                used | {index} if index >= 0 else used,
                            )
                        )
                next_states.sort(key=lambda row: (row[0], sorted(row[1].items())))
                if len(next_states) > limits.max_candidates:
                    stop_reason = "candidate_beam_capacity"
                states = next_states[: limits.max_candidates]
            for regret, roles, bindings, missing, used in states:
                check()
                unresolved = tuple(
                    f"unconsumed_span:{i}" for i in sorted(expected_spans - used)
                )
                try:
                    meaning = model._build(heads, roles, all_entities)
                except (ValueError, KeyError):
                    continue
                add(meaning, regret, bindings, (*missing, *unresolved))
        if not candidates and initial.meaning is None:
            mentions = tuple(
                ("mention", span.index, span.entity.name)
                for span in spans
                if span.entity and span.index >= 0
            )
            add(None, 0.0, mentions, ("unresolved_structure",))
        # An accepted legacy interpretation is a checked candidate too. This
        # prevents beam/type scaffolding silently deleting an existing reading.
        if initial.meaning is not None:
            identity = Hypothesis.identity(
                observation.observation_id, initial.meaning, ()
            )
            if identity not in candidates:
                add(initial.meaning, 0.0)
        check()
    except InterruptedError as exc:
        stop_reason = str(exc)
    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (candidate.language_regret, candidate.hypothesis_id),
    )
    if len(ordered) > limits.max_candidates:
        stop_reason = stop_reason or "candidate_capacity"
    return CandidateSet(
        observation,
        context_digest,
        tuple(ordered[: limits.max_candidates]),
        not bool(stop_reason),
        stop_reason,
        expansions,
    )

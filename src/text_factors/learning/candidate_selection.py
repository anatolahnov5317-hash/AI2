"""Compare plausible structures against common experience before world mutation.

Memory support is a compatibility feature, not independent testimony or truth.
Only close language candidates can compete. A tie remains unresolved; no scores
from repeated contexts are added. The fixed thresholds are engineering choices.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from .hypotheses import CandidateSet, Hypothesis, digest
from .language_data import tokenize
from .schema import Interpretation, Meaning, exact_fields

LANGUAGE_WINDOW = 0.12
LANGUAGE_MARGIN = 0.065
MEMORY_MARGIN = 0.05


def choose(
    batch: CandidateSet, rows: list[dict[str, Any]]
) -> tuple[Hypothesis | None, str]:
    if not batch.complete:
        return None, "ambiguous_incomplete_candidate_search"
    if not batch.candidates:
        return None, "no_candidates"
    best = min(c.language_regret for c in batch.candidates)
    plausible = [
        c for c in batch.candidates if c.language_regret <= best + LANGUAGE_WINDOW
    ]
    if any(
        not c.complete and c.language_regret - best < LANGUAGE_MARGIN for c in plausible
    ):
        return None, "ambiguous_partial_structure"
    plausible = [c for c in plausible if c.complete]
    if not plausible:
        return None, "ambiguous_partial_structure"
    if len(plausible) == 1:
        return plausible[0], "single_complete_candidate"
    scores = {row["hypothesis_id"]: row for row in rows}
    if all(c.meaning is not None and c.meaning.event is not None for c in plausible):
        supported = [c for c in plausible if scores[c.hypothesis_id]["supported"]]
        ranked = sorted(
            supported,
            key=lambda c: (
                -scores[c.hypothesis_id]["score"],
                c.language_regret,
                c.hypothesis_id,
            ),
        )
        if ranked and (
            len(ranked) == 1
            or scores[ranked[0].hypothesis_id]["score"]
            - scores[ranked[1].hypothesis_id]["score"]
            >= MEMORY_MARGIN
        ):
            return ranked[0], "common_experience_disambiguation"
        if not ranked:
            return None, "ambiguous_unsupported_candidates"
    ranked = sorted(plausible, key=lambda c: (c.language_regret, c.hypothesis_id))
    if ranked[1].language_regret - ranked[0].language_regret >= LANGUAGE_MARGIN:
        return ranked[0], "language_margin_after_experience"
    return None, "ambiguous_candidate_tie"


def select(
    batch: CandidateSet,
    initial: Interpretation,
    dynamics: Any,
    before: list[dict[str, Any]],
    *,
    model_fingerprint: str,
    seconds: float,
    dependency_event_ids: tuple[int, ...] = (),
) -> Interpretation:
    events = [
        c for c in batch.candidates if c.complete and c.meaning and c.meaning.event
    ]
    rows = []
    if batch.complete and events:
        scored = dynamics.score_interpretations(
            before,
            [c.meaning.event for c in events if c.meaning is not None],
            seconds=seconds,
            observation_id=batch.observation.observation_id,
            source_positions=tuple(range(len(tokenize(batch.observation.text)))),
        )
        if len(scored) != len(events):
            raise ValueError("candidate comparison returned an incomplete batch")
        for index, (candidate, score) in enumerate(zip(events, scored, strict=True)):
            if score["candidate"] != index:
                raise ValueError("candidate comparison order mismatch")
            if (
                type(score["supported"]) is not bool
                or type(score["score"]) not in (int, float)
                or not isfinite(score["score"])
                or not 0 <= score["score"] <= 1
            ):
                raise ValueError("invalid common experience score")
            rows.append(
                {
                    "hypothesis_id": candidate.hypothesis_id,
                    "score": float(score["score"]),
                    "supported": score["supported"],
                    "contexts": score["contexts"],
                    "reason": score["reason"],
                }
            )
    chosen, reason = choose(batch, rows)
    meaning = chosen.meaning if chosen else None
    snapshot = {
        "model_fingerprint": model_fingerprint,
        "context_digest": batch.context_digest,
        "before_digest": digest(before),
        "observation_digest": digest(batch.observation.to_dict()),
        "dependency_event_ids": list(dependency_event_ids),
    }
    trace = {
        "batch": batch.to_dict(),
        "snapshot": snapshot,
        "snapshot_id": digest(snapshot),
        "reads": rows,
        "selected_id": chosen.hypothesis_id if chosen else None,
        "reason": reason,
        "score_is_probability": False,
    }
    alternatives = tuple(
        c.meaning
        for c in batch.candidates
        if c.complete and c.meaning and c is not chosen
    )
    if not batch.candidates or (
        not events and initial.meaning is None and not alternatives
    ):
        reason = initial.reason
    return replace(
        initial,
        meaning=meaning,
        alternatives=alternatives,
        reason="" if meaning is not None else reason,
        diagnostics={**(initial.diagnostics or {}), "hypotheses": trace},
    )


def validate_trace(value: Any, meaning: Meaning | None) -> CandidateSet:
    trace = exact_fields(
        value,
        {
            "batch",
            "snapshot",
            "snapshot_id",
            "reads",
            "selected_id",
            "reason",
            "score_is_probability",
        },
        "hypothesis trace",
    )
    batch = CandidateSet.from_dict(trace["batch"])
    snapshot = exact_fields(
        trace["snapshot"],
        {
            "model_fingerprint",
            "context_digest",
            "before_digest",
            "observation_digest",
            "dependency_event_ids",
        },
        "candidate snapshot",
    )
    if (
        trace["snapshot_id"] != digest(snapshot)
        or snapshot["context_digest"] != batch.context_digest
        or snapshot["observation_digest"] != digest(batch.observation.to_dict())
        or trace["score_is_probability"] is not False
    ):
        raise ValueError("inconsistent candidate snapshot")
    dependencies = snapshot["dependency_event_ids"]
    if (
        type(dependencies) is not list
        or len(dependencies) > 128
        or any(type(i) is not int or not 1 <= i < 2**53 for i in dependencies)
        or sorted(set(dependencies)) != dependencies
    ):
        raise ValueError("invalid candidate dependencies")
    rows = trace["reads"]
    expected = [
        c.hypothesis_id
        for c in batch.candidates
        if batch.complete and c.complete and c.meaning and c.meaning.event
    ]
    if type(rows) is not list or len(rows) != len(expected):
        raise ValueError("incomplete saved candidate comparison")
    for index, row in enumerate(rows):
        exact_fields(
            row,
            {"hypothesis_id", "score", "supported", "contexts", "reason"},
            "candidate read",
        )
        if (
            row["hypothesis_id"] != expected[index]
            or type(row["supported"]) is not bool
            or type(row["score"]) not in (int, float)
            or not isfinite(row["score"])
            or not 0 <= row["score"] <= 1
            or type(row["contexts"]) is not list
            or len(row["contexts"]) > 2
        ):
            raise ValueError("invalid saved candidate read")
        for context in row["contexts"]:
            if (
                context["complete"] is not True
                or context["observation_id"] != batch.observation.observation_id
                or context["source_positions"]
                != list(range(len(tokenize(batch.observation.text))))
            ):
                raise ValueError("inconsistent candidate evidence provenance")
    namespaces = {
        (ctx["memory_namespace"], ctx["memory_step"])
        for row in rows
        for ctx in row["contexts"]
    }
    if len(namespaces) > 1:
        raise ValueError("mixed common memory versions")
    chosen, reason = choose(batch, rows)
    if (
        trace["selected_id"] != (chosen.hypothesis_id if chosen else None)
        or meaning != (chosen.meaning if chosen else None)
        or trace["reason"] != reason
    ):
        raise ValueError("saved choice does not match candidate evidence")
    return batch

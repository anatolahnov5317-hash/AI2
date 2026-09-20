"""Bounded episode retrieval and reconsideration, without rewriting world facts.

Only an explicit correction (or a structured, targeted ReviewCue) can constrain
an older interpretation. A later ordinary event is not evidence that an earlier
event was misunderstood. Returned candidates are compared on the OLD world's
snapshot, with the current common memory, before a new archive version commits.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Any

from .agreement import Claim, EvidenceLedger, Scope, exchange, relation
from .attention_ranker import AttentionRanker
from .candidate_search import SearchLimits, input_digest
from .candidate_selection import choose, select, validate_trace
from .hypotheses import CandidateSet, Hypothesis, Observation, digest
from .schema import (
    DialogueContext,
    Event,
    Interpretation,
    Meaning,
    bounded_text,
    exact_fields,
)

SCHEMA = "ai2-attention-v1"


@dataclass(frozen=True)
class AttentionLimits:
    max_sources: int = 128
    max_archives: int = 32
    max_scanned: int = 128
    max_retrieved: int = 4
    max_rounds: int = 4
    max_memory_calls: int = 2
    max_expansions: int = 512
    seconds: float = 0.5

    def __post_init__(self) -> None:
        for name, cap in {
            "max_sources": 256,
            "max_archives": 64,
            "max_scanned": 256,
            "max_retrieved": 8,
            "max_rounds": 16,
            "max_memory_calls": 8,
            "max_expansions": 512,
        }.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"invalid attention {name}")
        if self.max_archives > self.max_sources:
            raise ValueError("archive exceeds source capacity")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or not 0 < self.seconds <= 5
        ):
            raise ValueError("invalid attention deadline")


@dataclass(frozen=True)
class ReviewCue:
    target: str
    observation: Observation
    meaning: Meaning
    speaker: str = "user"

    def __post_init__(self) -> None:
        bounded_text(self.target, "review target", empty=False)
        bounded_text(self.speaker, "review speaker", empty=False)
        if (
            not isinstance(self.observation, Observation)
            or not isinstance(self.meaning, Meaning)
            or self.meaning.event is None
        ):
            raise ValueError("review requires observed semantic content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "observation": self.observation.to_dict(),
            "meaning": self.meaning.to_dict(),
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReviewCue:
        v = exact_fields(
            value, {"target", "observation", "meaning", "speaker"}, "review cue"
        )
        return cls(
            v["target"],
            Observation.from_dict(v["observation"]),
            Meaning.from_dict(v["meaning"]),
            v["speaker"],
        )


class Work:
    def __init__(self, limits: AttentionLimits, seconds: float | None = None) -> None:
        self.limits = limits
        self.deadline = perf_counter() + min(
            limits.seconds, seconds if seconds is not None else limits.seconds
        )
        self.counts = {
            "scanned": 0,
            "retrieved": 0,
            "rounds": 0,
            "memory_calls": 0,
            "expansions": 0,
        }
        self.considered: list[dict[str, Any]] = []

    def check(self) -> None:
        if perf_counter() >= self.deadline:
            raise TimeoutError("attention_time_budget")

    def take(self, name: str, count: int = 1) -> None:
        self.check()
        if self.counts[name] + count > getattr(self.limits, "max_" + name):
            raise TimeoutError("attention_" + name + "_budget")
        self.counts[name] += count

    def remaining(self) -> float:
        self.check()
        return self.deadline - perf_counter()


def memory_token(dynamics: Any) -> tuple[Any, Any]:
    bank = dynamics._experience
    return (bank.encoding_id, bank.memory.step) if bank is not None else (None, None)


def same_scope(left: Event, right: Event) -> bool:
    return (
        left.predicate,
        left.object,
        left.time,
        left.modality,
        left.content,
        left.condition,
    ) == (
        right.predicate,
        right.object,
        right.time,
        right.modality,
        right.content,
        right.condition,
    )


def _events(batch: CandidateSet) -> list[dict[str, Any]]:
    values = {
        digest(c.meaning.event.to_dict()): c.meaning.event.to_dict()
        for c in batch.candidates
        if c.meaning and c.meaning.event
    }
    return [values[k] for k in sorted(values)]


def _meaning(batch: CandidateSet, selected: str | None) -> Meaning | None:
    return next(
        (c.meaning for c in batch.candidates if c.hypothesis_id == selected), None
    )


def constrained_choice(
    batch: CandidateSet, cues: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[Hypothesis | None, str, tuple[Hypothesis, ...]]:
    accepted = {digest(c["meaning"]["event"]) for c in cues}
    supported = {row["hypothesis_id"] for row in rows if row["supported"]}
    matching = tuple(
        c
        for c in batch.candidates
        if c.complete
        and c.meaning
        and c.meaning.event
        and c.hypothesis_id in supported
        and all(c.meaning.event.to_dict() == cue["meaning"]["event"] for cue in cues)
    )
    if len(accepted) > 1:
        return None, "ambiguous_conflicting_clarifications", matching
    if not matching or not cues:
        return None, "ambiguous_no_supported_candidate", matching
    selected, reason = choose(
        CandidateSet(batch.observation, batch.context_digest, matching), rows
    )
    return selected, reason, matching


def validate_attention_trace(value: Any, batch: CandidateSet) -> None:
    trace = exact_fields(
        value,
        {"complete", "reason", "snapshot_id", "retrieved", "reviews", "work"},
        "attention trace",
    )
    if (
        trace["complete"] is not True
        or type(trace["snapshot_id"]) is not str
        or len(trace["snapshot_id"]) != 64
    ):
        raise ValueError("incomplete or invalid attention trace")
    bounded_text(trace["reason"], "attention reason")
    counters = exact_fields(
        trace["work"],
        {"scanned", "retrieved", "rounds", "memory_calls", "expansions"},
        "attention counters",
    )
    for key, cap in {
        "scanned": 256,
        "retrieved": 8,
        "rounds": 16,
        "memory_calls": 8,
        "expansions": 512,
    }.items():
        if type(counters[key]) is not int or not 0 <= counters[key] <= cap:
            raise ValueError("invalid attention work count")
    targets = trace["retrieved"]
    if (
        type(targets) is not list
        or len(targets) != counters["retrieved"]
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("duplicate or missing retrieved episode")
    for target in targets:
        bounded_text(target, "retrieved episode", empty=False)
    if (
        type(trace["reviews"]) is not list
        or len(trace["reviews"]) > counters["memory_calls"]
    ):
        raise ValueError("invalid attention review count")
    for r in trace["reviews"]:
        exact_fields(
            r,
            {
                "target",
                "cue_id",
                "snapshot_id",
                "model_fingerprint",
                "memory_version",
                "before_digest",
                "previous_id",
                "selected_id",
                "changed",
                "regenerated",
                "reason",
                "evidence",
                "agreement",
                "review_context",
                "reference_hints",
            },
            "attention review summary",
        )
        if (
            r["target"] not in targets
            or r["cue_id"] != batch.observation.observation_id
            or r["snapshot_id"] != trace["snapshot_id"]
        ):
            raise ValueError("attention review provenance mismatch")
        if (
            type(r["changed"]) is not bool
            or r["changed"] != (r["previous_id"] != r["selected_id"])
            or type(r["regenerated"]) is not bool
        ):
            raise ValueError("invalid attention change status")
        evidence = r["evidence"]
        if (
            type(evidence) is not dict
            or not {r["target"], r["cue_id"]} <= set(evidence)
            or len(evidence) > 9
        ):
            raise ValueError("invalid attention evidence roots")
        for root, row in evidence.items():
            bounded_text(root, "evidence root", empty=False)
            if type(row) is not dict or not 1 <= len(row) <= 16:
                raise ValueError("invalid attention evidence features")
            for key, feature in row.items():
                bounded_text(key, "evidence channel", empty=False)
                if (
                    type(feature) not in (float, int)
                    or not isfinite(feature)
                    or not 0 <= feature <= 1
                ):
                    raise ValueError("invalid attention evidence value")
        agreed = exact_fields(
            r["agreement"],
            {"complete", "rounds", "roots", "reason"},
            "agreement receipt",
        )
        if (
            agreed["complete"] is not True
            or agreed["reason"]
            or type(agreed["rounds"]) is not int
            or not 1 <= agreed["rounds"] <= counters["rounds"]
        ):
            raise ValueError("incomplete saved agreement")


class AttentionState:
    def __init__(
        self, model_fingerprint: str, limits: AttentionLimits | None = None
    ) -> None:
        bounded_text(model_fingerprint, "attention model", empty=False)
        self.model_fingerprint = model_fingerprint
        self.limits = limits or AttentionLimits()
        self.ranker = AttentionRanker.load()
        self.generation = 0
        self.evicted_sources = 0
        self.records: list[dict[str, Any]] = []

    def clone(self) -> AttentionState:
        result = AttentionState(self.model_fingerprint, self.limits)
        result.generation, result.evicted_sources = (
            self.generation,
            self.evicted_sources,
        )
        result.records = deepcopy(self.records)
        return result

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": SCHEMA,
            "model_fingerprint": self.model_fingerprint,
            "ranker_fingerprint": self.ranker.fingerprint,
            "limits": asdict(self.limits),
            "generation": self.generation,
            "evicted_sources": self.evicted_sources,
            "records": deepcopy(self.records),
        }
        return {**value, "digest": digest(value)}

    @property
    def snapshot_id(self) -> str:
        return self.to_dict()["digest"]

    def get(self, observation_id: str) -> dict[str, Any] | None:
        return next(
            (
                record
                for record in self.records
                if record["observation"]["observation_id"] == observation_id
            ),
            None,
        )

    def remember(
        self,
        trace: dict[str, Any],
        context: DialogueContext,
        before: list[dict[str, Any]],
        *,
        speaker: str = "user",
    ) -> None:
        batch = CandidateSet.from_dict(trace["batch"])
        chosen = _meaning(batch, trace["selected_id"])
        validate_trace(trace, chosen)
        if (
            trace["snapshot"]["model_fingerprint"] != self.model_fingerprint
            or batch.context_digest != digest(context.to_dict())
            or trace["snapshot"]["before_digest"] != digest(before)
        ):
            raise ValueError("archive origin snapshot mismatch")
        if self.get(batch.observation.observation_id) is not None:
            raise ValueError("observation already archived")
        bounded_text(speaker, "archive speaker", empty=False)
        record = {
            "observation": batch.observation.to_dict(),
            "speaker": speaker,
            "context": context.to_dict(),
            "before": deepcopy(before),
            "original_snapshot": deepcopy(trace["snapshot"]),
            "batch": batch.to_dict(),
            "index_events": _events(batch),
            "original_selected": trace["selected_id"],
            "original_meaning": chosen.to_dict() if chosen else None,
            "selected_id": trace["selected_id"],
            "selected_meaning": chosen.to_dict() if chosen else None,
            "reason": trace["reason"],
            "revision": 0,
            "cues": [],
            "review": None,
            "suppressed": {
                c.hypothesis_id: ("unresolved" if chosen is None else trace["reason"])
                for c in batch.candidates
                if c.hypothesis_id != trace["selected_id"]
            },
        }
        record["initial_suppressed"] = deepcopy(record["suppressed"])
        self.records.append(record)
        while len(self.records) > self.limits.max_sources:
            self.records.pop(0)
            self.evicted_sources += 1
        archived = [r for r in self.records if r["batch"] is not None]
        for previous in archived[: -self.limits.max_archives]:
            previous["batch"] = None
        self.generation += 1

    def retrieve(
        self, meaning: Meaning, work: Work, *, target: str = ""
    ) -> list[dict[str, Any]]:
        # Bounded coarse index over retained semantic keys; no latest-message bonus.
        event = meaning.event
        candidates = []
        for record in self.records:
            work.take("scanned")
            key = record["observation"]["observation_id"]
            if target:
                applicable_key = key == target
            else:
                applicable_key = event is not None and any(
                    e["object"] and e["object"] == event.object
                    for e in record["index_events"]
                )
            if not applicable_key:
                continue
            work.considered.append(record)
            score = max(
                (
                    self.ranker.score(target, meaning, {**record, "index_event": e})
                    for e in record["index_events"]
                ),
                default=-100.0,
            )
            candidates.append((score, key, record))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        retrieved = []
        for _, _, record in candidates[: self.limits.max_retrieved]:
            work.take("retrieved")
            retrieved.append(record)
        return retrieved

    def prepare_review(
        self,
        cue: ReviewCue,
        dynamics: Any,
        understanding: Any,
        *,
        work: Work | None = None,
    ) -> dict[str, Any]:
        work = work or Work(self.limits)
        base = self.snapshot_id
        version = memory_token(dynamics)
        proposal: dict[str, Any] = {
            "base": base,
            "target": cue.target,
            "complete": False,
            "reason": "",
            "record": None,
            "summary": None,
            "memory_version": list(version),
        }
        try:
            work.check()
            source = self.get(cue.target)
            if source is None:
                proposal.update(complete=True, reason="source_not_retained")
                return proposal
            if cue.observation.turn_id <= source["observation"]["turn_id"]:
                raise ValueError("review must use a later observation")
            event = cue.meaning.event
            assert event is not None
            if cue.speaker != source["speaker"] or not any(
                same_scope(Event.from_dict(e), event) for e in source["index_events"]
            ):
                proposal.update(complete=True, reason="inapplicable_scope")
                return proposal
            record = deepcopy(source)
            prior_batch = (
                CandidateSet.from_dict(record["batch"])
                if record["batch"] is not None
                else None
            )
            regenerated = (
                prior_batch is None
                or not prior_batch.complete
                or not any(
                    c.complete and c.meaning and c.meaning.event == event
                    for c in prior_batch.candidates
                )
            )
            review_context = (
                record["review"]["review_context"]
                if record["review"] is not None and prior_batch is not None
                else record["context"]
            )
            reference_hints = (
                record["review"]["reference_hints"]
                if record["review"] is not None and prior_batch is not None
                else {}
            )
            if regenerated:
                ctx = DialogueContext.from_dict(record["context"])
                entities = {e.name: e for e in (*ctx.entities, *cue.meaning.entities)}
                ctx = DialogueContext(
                    ctx.turns, tuple(entities.values())[-64:], ctx.focus, ctx.pending
                )
                review_context = ctx.to_dict()
                reference_hints = {
                    role: getattr(event, role)
                    for role in ("actor", "recipient", "object", "place")
                    if getattr(event, role)
                }
                batch = understanding.propose(
                    record["observation"]["text"],
                    ctx,
                    observation=Observation.from_dict(record["observation"]),
                    limits=SearchLimits(
                        max_expansions=self.limits.max_expansions,
                        seconds=work.remaining(),
                    ),
                    reference_hints=reference_hints,
                )
                work.take("expansions", batch.expansions)
            else:
                assert prior_batch is not None
                batch = prior_batch
            if not batch.complete:
                proposal["reason"] = "incomplete_candidate_reconstruction"
                return proposal
            cues = {c["observation"]["observation_id"]: c for c in record["cues"]}
            cue_id = cue.observation.observation_id
            if cue_id in cues and cues[cue_id] != cue.to_dict():
                raise ValueError("cue identity reused for different evidence")
            if cue_id not in cues and len(cues) >= 8:
                proposal["reason"] = "attention_cue_capacity"
                return proposal
            cues[cue_id] = cue.to_dict()
            record["cues"] = [cues[key] for key in sorted(cues)]
            work.take("memory_calls")
            compared = select(
                batch,
                Interpretation(None),
                dynamics,
                record["before"],
                model_fingerprint=self.model_fingerprint,
                seconds=work.remaining(),
                dependency_event_ids=tuple(
                    record["original_snapshot"]["dependency_event_ids"]
                ),
            )
            assert compared.diagnostics is not None
            comparison = compared.diagnostics["hypotheses"]
            validate_trace(comparison, compared.meaning)
            if memory_token(dynamics) != version or self.snapshot_id != base:
                raise ValueError("stale review snapshot")
            ledger = EvidenceLedger(base)
            origin = batch.observation.observation_id
            ledger.observe(origin, "language", 1.0, snapshot_id=base, complete=True)
            ledger.observe(
                origin,
                "memory",
                max((r["score"] for r in comparison["reads"]), default=0.0),
                snapshot_id=base,
                complete=True,
            )
            for c in record["cues"]:
                ledger.observe(
                    c["observation"]["observation_id"],
                    "clarification",
                    1.0,
                    snapshot_id=base,
                    complete=True,
                )
            # Claims use the historical event's identity. Different time, speaker
            # or modality cannot silently become contradictory role assignments.
            scope = Scope(cue.target, event.time, cue.speaker, event.modality)
            claims = tuple(
                Claim(
                    c["observation"]["observation_id"],
                    scope,
                    "event",
                    digest(c["meaning"]["event"]),
                    frozenset({c["observation"]["observation_id"]}),
                )
                for c in record["cues"]
            )
            links = tuple(
                (a.claim_id, b.claim_id)
                for a in claims
                for b in claims
                if a != b and relation(a, b) in {"support", "duplicate"}
            )
            agreed = exchange(claims, links, max_rounds=self.limits.max_rounds)
            work.take("rounds", agreed["rounds"])
            if not agreed["complete"]:
                proposal["reason"] = agreed["reason"]
                return proposal
            selected, reason, matching = constrained_choice(
                batch, record["cues"], comparison["reads"]
            )
            previous = record["selected_id"]
            selected_id = selected.hypothesis_id if selected else None
            summary = {
                "target": cue.target,
                "cue_id": cue_id,
                "snapshot_id": base,
                "model_fingerprint": self.model_fingerprint,
                "memory_version": list(version),
                "before_digest": digest(record["before"]),
                "previous_id": previous,
                "selected_id": selected_id,
                "changed": selected_id != previous,
                "regenerated": regenerated,
                "reason": reason,
                "evidence": ledger.to_dict(),
                "agreement": agreed,
                "comparison": comparison,
                "review_context": review_context,
                "reference_hints": reference_hints,
            }
            record.update(
                batch=batch.to_dict(),
                selected_id=selected_id,
                selected_meaning=selected.meaning.to_dict()
                if selected and selected.meaning
                else None,
                reason=reason,
                revision=record["revision"] + 1,
                review=summary,
            )
            record["suppressed"] = {
                c.hypothesis_id: (
                    "cue_incompatible" if c not in matching else "unresolved"
                )
                for c in batch.candidates
                if c.hypothesis_id != selected_id
            }
            work.check()
            proposal.update(
                complete=True, reason=reason, record=record, summary=summary
            )
            return proposal
        except TimeoutError as exc:
            proposal["reason"] = str(exc) or "attention_time_budget"
            return proposal
        finally:
            proposal["work"] = dict(work.counts)

    def commit_reviews(self, proposals: list[dict[str, Any]], dynamics: Any) -> None:
        if not proposals:
            return
        if len(proposals) > self.limits.max_memory_calls:
            raise ValueError("review commit capacity")
        base = self.snapshot_id
        if any(p["base"] != base for p in proposals):
            raise ValueError("stale review commit")
        if any(p["memory_version"] != list(memory_token(dynamics)) for p in proposals):
            raise ValueError("stale memory at review commit")
        if any(p["complete"] is not True for p in proposals):
            raise ValueError("incomplete review commit")
        targets = [p["target"] for p in proposals if p["record"] is not None]
        if len(targets) != len(set(targets)):
            raise ValueError("duplicate review target")
        replacements = {
            p["target"]: p["record"] for p in proposals if p["record"] is not None
        }
        for key, record in replacements.items():
            old = self.get(key)
            if (
                old is None
                or record["observation"] != old["observation"]
                or record["revision"] != old["revision"] + 1
            ):
                raise ValueError("invalid review replacement")
        self.records = [
            deepcopy(replacements.get(r["observation"]["observation_id"], r))
            for r in self.records
        ]
        if replacements:
            self.generation += 1
        archived = [r for r in self.records if r["batch"] is not None]
        for record in archived[: -self.limits.max_archives]:
            record["batch"] = None

    def process(
        self,
        batch: CandidateSet,
        meaning: Meaning | None,
        dynamics: Any,
        understanding: Any,
        *,
        seconds: float,
        excluded_observations: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        work = Work(self.limits, seconds)
        trace: dict[str, Any] = {
            "complete": True,
            "reason": "no_review_cue",
            "snapshot_id": self.snapshot_id,
            "retrieved": [],
            "reviews": [],
            "work": work.counts,
        }
        if meaning is None or meaning.event is None:
            return trace
        try:
            retrieved = self.retrieve(meaning, work)
            trace["retrieved"] = [r["observation"]["observation_id"] for r in retrieved]
            if meaning.act != "correct":
                return trace
            # Check ambiguity across ALL retained keys, not just the top-k. A
            # retrieval budget must not hide another possible correction target.
            possible = [
                r
                for r in work.considered
                if r["speaker"] == "user"
                and r["observation"]["observation_id"] not in excluded_observations
                and any(
                    same_scope(Event.from_dict(e), meaning.event)
                    for e in r["index_events"]
                )
            ]
            if len(possible) != 1:
                trace["reason"] = (
                    "ambiguous_review_target" if possible else "no_applicable_target"
                )
                return trace
            target = possible[0]["observation"]["observation_id"]
            if target not in trace["retrieved"]:
                trace.update(complete=False, reason="attention_retrieval_budget")
                return trace
            cue = ReviewCue(target, batch.observation, meaning)
            proposal = self.prepare_review(cue, dynamics, understanding, work=work)
            if not proposal["complete"]:
                trace.update(complete=False, reason=proposal["reason"])
                return trace
            self.commit_reviews([proposal], dynamics)
            trace["reason"] = proposal["reason"]
            if proposal["summary"] is not None:
                trace["reviews"] = [
                    {
                        key: value
                        for key, value in proposal["summary"].items()
                        if key != "comparison"
                    }
                ]
            return trace
        except TimeoutError as exc:
            trace.update(complete=False, reason=str(exc) or "attention_time_budget")
            return trace

    @classmethod
    def from_dict(cls, value: Any, model_fingerprint: str) -> AttentionState:
        v = exact_fields(
            value,
            {
                "schema",
                "model_fingerprint",
                "ranker_fingerprint",
                "limits",
                "generation",
                "evicted_sources",
                "records",
                "digest",
            },
            "attention state",
        )
        if (
            v["schema"] != SCHEMA
            or v["model_fingerprint"] != model_fingerprint
            or v["digest"] != digest({k: x for k, x in v.items() if k != "digest"})
        ):
            raise ValueError("invalid attention snapshot")
        limits = AttentionLimits(
            **exact_fields(
                v["limits"], set(asdict(AttentionLimits())), "attention limits"
            )
        )
        state = cls(model_fingerprint, limits)
        if v["ranker_fingerprint"] != state.ranker.fingerprint:
            raise ValueError("attention ranker version mismatch")
        for key in ("generation", "evicted_sources"):
            if type(v[key]) is not int or not 0 <= v[key] < 2**53:
                raise ValueError("invalid attention counter")
        if type(v["records"]) is not list or len(v["records"]) > limits.max_sources:
            raise ValueError("attention source capacity")
        seen, turns, archived = set(), [], 0
        for r in v["records"]:
            exact_fields(
                r,
                {
                    "observation",
                    "speaker",
                    "context",
                    "before",
                    "original_snapshot",
                    "batch",
                    "index_events",
                    "original_selected",
                    "original_meaning",
                    "selected_id",
                    "selected_meaning",
                    "reason",
                    "revision",
                    "cues",
                    "review",
                    "suppressed",
                    "initial_suppressed",
                },
                "archived episode",
            )
            observation = Observation.from_dict(r["observation"])
            context = DialogueContext.from_dict(r["context"])
            bounded_text(r["speaker"], "archive speaker", empty=False)
            if observation.observation_id in seen:
                raise ValueError("duplicate archived observation")
            seen.add(observation.observation_id)
            turns.append(observation.turn_id)
            snapshot = r["original_snapshot"]
            exact_fields(
                snapshot,
                {
                    "model_fingerprint",
                    "context_digest",
                    "before_digest",
                    "observation_digest",
                    "dependency_event_ids",
                },
                "archived snapshot",
            )
            if (
                snapshot["model_fingerprint"] != model_fingerprint
                or snapshot["observation_digest"] != digest(r["observation"])
                or snapshot["before_digest"] != digest(r["before"])
                or snapshot["context_digest"] != digest(context.to_dict())
            ):
                raise ValueError("inconsistent archived origin")
            if type(r["before"]) is not list or len(r["before"]) > 128:
                raise ValueError("invalid historical facts")
            for fact in r["before"]:
                exact_fields(
                    fact,
                    {"subject", "relation", "value", "negated", "spatial"},
                    "historical fact",
                )
                if (
                    type(fact["negated"]) is not bool
                    or fact["relation"] not in {"holder", "location"}
                    or fact["spatial"] not in {"in", "on"}
                ):
                    raise ValueError("invalid historical fact fields")
                for field in ("subject", "value"):
                    bounded_text(fact[field], field, empty=False)
            if type(r["index_events"]) is not list or len(r["index_events"]) > 8:
                raise ValueError("invalid archived event keys")
            for e in r["index_events"]:
                Event.from_dict(e)
            if (
                type(r["revision"]) is not int
                or not 0 <= r["revision"] < 2**53
                or type(r["cues"]) is not list
                or len(r["cues"]) > 8
            ):
                raise ValueError("invalid archive revision")
            cue_ids = set()
            for raw in r["cues"]:
                cue = ReviewCue.from_dict(raw)
                cue_event = cue.meaning.event
                assert cue_event is not None
                if (
                    cue.target != observation.observation_id
                    or cue.speaker != r["speaker"]
                    or cue.observation.turn_id <= observation.turn_id
                    or cue.observation.observation_id in cue_ids
                    or not any(
                        same_scope(Event.from_dict(e), cue_event)
                        for e in r["index_events"]
                    )
                ):
                    raise ValueError("invalid archived clarification")
                cue_ids.add(cue.observation.observation_id)
            selected = (
                Meaning.from_dict(r["selected_meaning"])
                if r["selected_meaning"] is not None
                else None
            )
            original = (
                Meaning.from_dict(r["original_meaning"])
                if r["original_meaning"] is not None
                else None
            )
            if r["original_selected"] != (
                Hypothesis.identity(observation.observation_id, original, ())
                if original
                else None
            ):
                raise ValueError("invalid original archive choice")
            if r["selected_id"] != (
                Hypothesis.identity(observation.observation_id, selected, ())
                if selected
                else None
            ):
                raise ValueError("invalid archived choice")
            if r["batch"] is not None:
                batch = CandidateSet.from_dict(r["batch"])
                archived += 1
                if batch.observation != observation or (
                    r["selected_id"] is not None
                    and _meaning(batch, r["selected_id"]) != selected
                ):
                    raise ValueError("archive candidate mismatch")
            if r["review"] is not None:
                review = r["review"]
                compared = review["comparison"]
                comparison_batch = CandidateSet.from_dict(compared["batch"])
                validate_trace(
                    compared, _meaning(comparison_batch, compared["selected_id"])
                )
                expected, reason, _ = constrained_choice(
                    comparison_batch, r["cues"], compared["reads"]
                )
                replay_context = DialogueContext.from_dict(review["review_context"])
                hints = review["reference_hints"]
                if hints and not any(
                    hints
                    == {
                        role: raw["meaning"]["event"][role]
                        for role in ("actor", "recipient", "object", "place")
                        if raw["meaning"]["event"][role]
                    }
                    for raw in r["cues"]
                ):
                    raise ValueError("reference constraints lack observed provenance")
                if (
                    comparison_batch.observation != observation
                    or compared["snapshot"]["before_digest"]
                    != snapshot["before_digest"]
                    or review["selected_id"] != r["selected_id"]
                    or review["model_fingerprint"] != model_fingerprint
                    or review["selected_id"]
                    != (expected.hypothesis_id if expected else None)
                    or review["reason"] != reason
                    or r["reason"] != reason
                    or review["changed"]
                    != (review["previous_id"] != review["selected_id"])
                    or not r["revision"]
                    or not r["cues"]
                    or comparison_batch.context_digest
                    != input_digest(replay_context, review["reference_hints"])
                ):
                    raise ValueError("invalid archived review snapshot")
            elif (
                r["revision"] or r["cues"] or r["selected_id"] != r["original_selected"]
            ):
                raise ValueError("archive revision has no evidence")
        if (
            turns != sorted(set(turns))
            or archived > limits.max_archives
            or v["generation"] < len(turns) + v["evicted_sources"]
        ):
            raise ValueError("inconsistent archive chronology or capacity")
        state.records = deepcopy(v["records"])
        state.generation, state.evicted_sources = v["generation"], v["evicted_sources"]
        return state

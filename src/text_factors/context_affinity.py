"""Bounded learning of near-repeat context responses in one local encoding.

This is an engineering approximation to context affinity, not a Pearson map or
learned semantic equivalence. A training event contributes at most one exposure
to each context pair. It is positive when a pair has sufficiently overlapping
factor evidence at the same explicitly supplied location. Selection additionally
requires that overlap in the current observation. Context names carry no metric.
Independent locations, declared conflicts and unlearned edges are never merged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Any

from .recognition import (
    CandidateRelation,
    ClusterEvidence,
    ContextResponse,
    InterpretationClaim,
    RecognitionCandidate,
    RecognitionResult,
)

FORMAT_VERSION = 1
_MAX_INTEGER = 2**63 - 1
_RELATIONS = {"duplicate", "compatible", "conflict", "undetermined"}
Feature = tuple[int, int]
ContextPair = tuple[str, str]


def _integer(value: Any, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def _identifier(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or not 0 < len(value) <= 256
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be a nonempty identifier of at most 256 chars")


def _indices(value: Any, name: str) -> None:
    if (
        type(value) is not tuple
        or len(value) > 4096
        or any(type(item) is not int or not 0 <= item < 1_000_000 for item in value)
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError(f"{name} must contain sorted unique non-negative indices")


@dataclass(frozen=True, slots=True)
class ContextAffinityConfig:
    """Fixed experimental thresholds; observations count distinct supplied IDs.

    ``evidence_threshold`` is Jaccard overlap of (point, matched input bit).
    ``affinity_threshold`` is positive events / exposed events, not probability.
    ``max_evidence`` bounds the total signature and matched-bit entries per call.
    Different IDs do not by themselves establish statistical independence.
    """

    min_observations: int = 3
    min_shared: int = 2
    evidence_threshold: float = 0.6
    affinity_threshold: float = 0.8
    max_events: int = 512
    max_pairs: int = 2048
    max_candidates: int = 128
    max_evidence: int = 32768
    seconds: float = 2.0

    def __post_init__(self) -> None:
        for name, upper in (
            ("min_observations", 4096),
            ("min_shared", 65536),
            ("max_events", 4096),
            ("max_pairs", 16384),
            ("max_candidates", 128),
            ("max_evidence", 65536),
        ):
            _integer(getattr(self, name), name, 1, upper)
        if self.min_observations > self.max_events:
            raise ValueError("min_observations cannot exceed max_events")
        for name in ("evidence_threshold", "affinity_threshold"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (
            type(self.seconds) not in (int, float)
            or not isfinite(self.seconds)
            or not 0 < self.seconds <= 30
        ):
            raise ValueError("seconds must be finite and in (0, 30]")


@dataclass(frozen=True, slots=True)
class _PairCounts:
    exposures: int
    positive: int


class _Budget:
    def __init__(self, seconds: float, cancelled: Callable[[], bool] | None) -> None:
        self.deadline = perf_counter() + seconds
        self.cancelled = cancelled

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("context_affinity_cancelled")
        if perf_counter() >= self.deadline:
            raise InterruptedError("context_affinity_time_budget")


def _pair(left: str, right: str) -> ContextPair:
    return (left, right) if left < right else (right, left)


def _same_location(left: RecognitionCandidate, right: RecognitionCandidate) -> bool:
    return bool(
        left.observation_id == right.observation_id
        and left.source_positions
        and left.source_positions == right.source_positions
    )


class ContextAffinity:
    """An explicit training store and a pure, greedy local-maximum selector.

    Training may use exact repeats already listed in ``result.suppressed``.
    Incomplete searches are never training events. All mutations commit only
    after validation and budget checks; capacities never evict older evidence.
    A repeated event ID with the same payload is idempotent; a different payload
    with that ID is rejected. Selection never calls ``observe``.
    """

    def __init__(
        self, encoding_id: str, config: ContextAffinityConfig | None = None
    ) -> None:
        _identifier(encoding_id, "encoding_id")
        if config is not None and type(config) is not ContextAffinityConfig:
            raise ValueError("config must be ContextAffinityConfig")
        self.encoding_id = encoding_id
        self.config = config or ContextAffinityConfig()
        self._events: dict[str, str] = {}
        self._pairs: dict[ContextPair, _PairCounts] = {}

    def _validated(
        self, result: RecognitionResult, budget: _Budget
    ) -> tuple[dict[str, frozenset[Feature]], set[ContextPair]]:
        if type(result) is not RecognitionResult:
            raise ValueError("result must be RecognitionResult")
        if result.encoding_id != self.encoding_id:
            raise ValueError("recognition uses a different encoding")
        if type(result.complete) is not bool:
            raise ValueError("complete must be a boolean")
        for name in ("examined_views", "total_views", "memory_step"):
            _integer(getattr(result, name), name, 0, _MAX_INTEGER)
        if result.examined_views > result.total_views:
            raise ValueError("examined_views exceeds total_views")
        if result.complete and result.examined_views != result.total_views:
            raise ValueError("a complete result must examine every view")
        if result.stop_reason is not None:
            _identifier(result.stop_reason, "stop_reason")
            if result.complete:
                raise ValueError("a stopped result cannot be complete")
        if type(result.candidates) is not tuple or type(result.suppressed) is not tuple:
            raise ValueError("candidate collections must be tuples")
        candidates = result.candidates + result.suppressed
        if len(candidates) > self.config.max_candidates:
            raise ValueError("context affinity candidate capacity exceeded")
        features: dict[str, frozenset[Feature]] = {}
        claims: dict[str, dict[tuple[str, str, str], str]] = {}
        evidence_size = 0
        for candidate in candidates:
            budget.check()
            if type(candidate) is not RecognitionCandidate:
                raise ValueError("invalid recognition candidate")
            for name in ("candidate_id", "context_id", "content_key", "observation_id"):
                _identifier(getattr(candidate, name), name)
            if candidate.candidate_id in features:
                raise ValueError("duplicate candidate ID")
            if (
                type(candidate.familiarity) not in (int, float)
                or not isfinite(candidate.familiarity)
                or candidate.familiarity < 0
            ):
                raise ValueError("familiarity must be finite and non-negative")
            for name in ("quality", "active_points"):
                _integer(getattr(candidate, name), name, 0, 1_000_000)
            _indices(candidate.source_positions, "source_positions")
            _indices(candidate.output_bits, "output_bits")
            if (
                type(candidate.evidence) is not tuple
                or not candidate.evidence
                or len(candidate.evidence) > self.config.max_evidence
            ):
                raise ValueError("candidate needs bounded factor evidence")
            matched: set[Feature] = set()
            for entry in candidate.evidence:
                budget.check()
                if type(entry) is not ClusterEvidence:
                    raise ValueError("invalid factor evidence")
                _indices(entry.signature, "signature")
                _indices(entry.matched_bits, "matched_bits")
                evidence_size += len(entry.signature) + len(entry.matched_bits)
                if evidence_size > self.config.max_evidence:
                    raise ValueError("context affinity evidence capacity exceeded")
                _integer(entry.point_index, "point_index", 0, 999999)
                _integer(entry.output_bit, "output_bit", 0, 999999)
                _integer(entry.observations, "observations", 1, _MAX_INTEGER)
                if (
                    not entry.matched_bits
                    or not set(entry.matched_bits).issubset(entry.signature)
                    or entry.output_bit not in candidate.output_bits
                ):
                    raise ValueError("factor evidence does not support the candidate")
                matched.update((entry.point_index, bit) for bit in entry.matched_bits)
            features[candidate.candidate_id] = frozenset(matched)
            if type(candidate.claims) is not tuple or len(candidate.claims) > 128:
                raise ValueError("claims must be a bounded tuple")
            candidate_claims: dict[tuple[str, str, str], str] = {}
            for claim in candidate.claims:
                if type(claim) is not InterpretationClaim:
                    raise ValueError("invalid interpretation claim")
                if claim.key in candidate_claims:
                    raise ValueError("duplicate scoped claim")
                candidate_claims[claim.key] = claim.value
            claims[candidate.candidate_id] = candidate_claims
        max_relations = (
            self.config.max_candidates * (self.config.max_candidates - 1) // 2
        )
        if type(result.relations) is not tuple or len(result.relations) > max_relations:
            raise ValueError("relations must be a bounded tuple")
        declared: dict[ContextPair, str] = {}
        conflicts: set[ContextPair] = set()
        for relation in result.relations:
            budget.check()
            if (
                type(relation) is not CandidateRelation
                or relation.left not in features
                or relation.right not in features
                or relation.left == relation.right
                or type(relation.kind) is not str
                or relation.kind not in _RELATIONS
                or type(relation.reason) is not str
                or not 0 < len(relation.reason) <= 512
            ):
                raise ValueError("invalid candidate relation")
            key = _pair(relation.left, relation.right)
            if key in declared:
                raise ValueError("duplicate candidate relation")
            declared[key] = relation.kind
            if relation.kind == "conflict":
                conflicts.add(key)
        for index, left in enumerate(candidates):
            budget.check()
            left_claims = claims[left.candidate_id]
            for right in candidates[index + 1 :]:
                right_claims = claims[right.candidate_id]
                if any(
                    left_claims[key] != right_claims[key]
                    for key in left_claims.keys() & right_claims.keys()
                ):
                    conflicts.add(_pair(left.candidate_id, right.candidate_id))
        return features, conflicts

    def _supported(
        self,
        left: RecognitionCandidate,
        right: RecognitionCandidate,
        features: dict[str, frozenset[Feature]],
        conflicts: set[ContextPair],
    ) -> bool:
        if (
            not _same_location(left, right)
            or _pair(left.candidate_id, right.candidate_id) in conflicts
        ):
            return False
        left_bits = features[left.candidate_id]
        right_bits = features[right.candidate_id]
        shared = len(left_bits & right_bits)
        return (
            shared >= self.config.min_shared
            and shared / len(left_bits | right_bits) >= self.config.evidence_threshold
        )

    def observe(
        self,
        event_id: str,
        result: RecognitionResult,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Learn a complete supplied event atomically; never train on a query.

        A pair is exposed only when both contexts responded at the same supplied
        location. One event is positive if any such pair has current support.
        Thus counts describe conditional co-responses, not absent context scores.
        Interrupted calls raise ``InterruptedError`` without storing the event.
        """

        _identifier(event_id, "event_id")
        budget = _Budget(self.config.seconds, cancelled)
        features, conflicts = self._validated(result, budget)
        if not result.complete:
            return False
        # This legacy store predates completed-read traces. Keep its historical
        # event digest stable when loading a v1 ledger across the API addition.
        legacy_payload = result.to_dict()
        legacy_payload.pop("responses", None)
        digest = hashlib.sha256(
            json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        budget.check()
        if event_id in self._events:
            if self._events[event_id] != digest:
                raise ValueError("event ID was already used with different evidence")
            return False
        if len(self._events) >= self.config.max_events:
            raise ValueError("context affinity event capacity exceeded")
        candidates = result.candidates + result.suppressed
        observations: dict[ContextPair, bool] = {}
        for index, left in enumerate(candidates):
            budget.check()
            for right in candidates[index + 1 :]:
                if left.context_id == right.context_id or not _same_location(
                    left, right
                ):
                    continue
                key = _pair(left.context_id, right.context_id)
                observations[key] = observations.get(key, False) or self._supported(
                    left, right, features, conflicts
                )
        if len(self._pairs.keys() | observations.keys()) > self.config.max_pairs:
            raise ValueError("context affinity pair capacity exceeded")
        updates: dict[ContextPair, _PairCounts] = {}
        for key, positive in observations.items():
            old = self._pairs.get(key, _PairCounts(0, 0))
            updates[key] = _PairCounts(old.exposures + 1, old.positive + int(positive))
        budget.check()
        self._pairs.update(updates)
        self._events[event_id] = digest
        return True

    def select(
        self,
        result: RecognitionResult,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecognitionResult:
        """Keep local maxima using direct learned edges and current evidence.

        Suppressed candidates are preserved as audit data, never merged into a
        winner or treated as additional observations. Relations are restricted to
        retained candidates. On interruption the original candidates are returned
        with explicit incompleteness, so a partial selection cannot hide a part.
        """

        budget = _Budget(self.config.seconds, cancelled)
        try:
            features, conflicts = self._validated(result, budget)
            budget.check()
            if not result.complete:
                return result
            remaining = sorted(
                result.candidates,
                key=lambda item: (-item.familiarity, -item.quality, item.candidate_id),
            )
            kept: list[RecognitionCandidate] = []
            suppressed = list(result.suppressed)
            while remaining:
                budget.check()
                winner = remaining.pop(0)
                kept.append(winner)
                survivors: list[RecognitionCandidate] = []
                for candidate in remaining:
                    budget.check()
                    counts = self._pairs.get(
                        _pair(winner.context_id, candidate.context_id)
                    )
                    if (
                        counts is not None
                        and counts.positive >= self.config.min_observations
                        and counts.positive / counts.exposures
                        >= self.config.affinity_threshold
                        and self._supported(winner, candidate, features, conflicts)
                    ):
                        suppressed.append(candidate)
                    else:
                        survivors.append(candidate)
                remaining = survivors
            retained = {candidate.candidate_id for candidate in kept}
            relations = tuple(
                relation
                for relation in result.relations
                if relation.left in retained and relation.right in retained
            )
            budget.check()
            return replace(
                result,
                candidates=tuple(kept),
                suppressed=tuple(suppressed),
                relations=relations,
            )
        except InterruptedError as error:
            return replace(
                result, complete=False, stop_reason=result.stop_reason or str(error)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "encoding_id": self.encoding_id,
            "config": asdict(self.config),
            "events": [
                {"event_id": event_id, "digest": digest}
                for event_id, digest in self._events.items()
            ],
            "pairs": [
                {"left": key[0], "right": key[1], **asdict(counts)}
                for key, counts in sorted(self._pairs.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextAffinity:
        """Restore only bounded, validated state; no executable deserialization."""

        if type(data) is not dict or set(data) != {
            "format_version",
            "encoding_id",
            "config",
            "events",
            "pairs",
        }:
            raise ValueError("invalid context affinity state fields")
        if (
            type(data["format_version"]) is not int
            or data["format_version"] != FORMAT_VERSION
        ):
            raise ValueError("unsupported context affinity state version")
        config_data = data["config"]
        if type(config_data) is not dict or set(config_data) != set(
            asdict(ContextAffinityConfig())
        ):
            raise ValueError("invalid context affinity configuration fields")
        config = ContextAffinityConfig(**config_data)
        restored = cls(data["encoding_id"], config)
        budget = _Budget(config.seconds, None)
        events, pairs = data["events"], data["pairs"]
        if type(events) is not list or len(events) > config.max_events:
            raise ValueError("invalid or oversized event list")
        if type(pairs) is not list or len(pairs) > config.max_pairs:
            raise ValueError("invalid or oversized context pair list")
        for event in events:
            budget.check()
            if type(event) is not dict or set(event) != {"event_id", "digest"}:
                raise ValueError("invalid affinity event")
            _identifier(event["event_id"], "event_id")
            digest = event["digest"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or event["event_id"] in restored._events
            ):
                raise ValueError("invalid or duplicate affinity event")
            restored._events[event["event_id"]] = digest
        for pair in pairs:
            budget.check()
            if type(pair) is not dict or set(pair) != {
                "left",
                "right",
                "exposures",
                "positive",
            }:
                raise ValueError("invalid context pair")
            _identifier(pair["left"], "left context")
            _identifier(pair["right"], "right context")
            if pair["left"] >= pair["right"]:
                raise ValueError("context pairs must be distinct and ordered")
            key = (pair["left"], pair["right"])
            if key in restored._pairs:
                raise ValueError("duplicate context pair")
            _integer(pair["exposures"], "exposures", 1, len(events))
            _integer(pair["positive"], "positive", 0, pair["exposures"])
            restored._pairs[key] = _PairCounts(pair["exposures"], pair["positive"])
        return restored


@dataclass(frozen=True, slots=True)
class AdaptiveContextAffinityConfig:
    """A fixed recent-event window, with explicit support and diversity gates.

    Activation agreement is n11/(n11+n10+n01); jointly silent reads are reported
    separately. Neither agreement nor the input-pattern count establishes
    semantic equivalence or statistical independence. A maximum event ledger is
    retained for idempotence; reaching it rejects new events without eviction.
    """

    window_events: int = 32
    min_observations: int = 3
    min_distinct_patterns: int = 3
    min_shared: int = 2
    evidence_threshold: float = 0.6
    affinity_threshold: float = 0.8
    max_events: int = 512
    max_pairs: int = 2048
    max_candidates: int = 128
    max_responses: int = 2048
    max_evidence: int = 32768
    seconds: float = 2.0

    def __post_init__(self) -> None:
        self.legacy_config()
        _integer(self.window_events, "window_events", 1, self.max_events)
        _integer(self.min_distinct_patterns, "min_distinct_patterns", 1, 4096)
        _integer(self.max_responses, "max_responses", 1, 2048)
        if max(self.min_observations, self.min_distinct_patterns) > self.window_events:
            raise ValueError("support and diversity gates cannot exceed the window")

    def legacy_config(self) -> ContextAffinityConfig:
        return ContextAffinityConfig(
            **{name: getattr(self, name) for name in asdict(ContextAffinityConfig())}
        )


@dataclass(frozen=True, slots=True)
class _RecentPair:
    left: str
    right: str
    state: int
    pattern: str


@dataclass(frozen=True, slots=True)
class _RecentEvent:
    event_id: str
    pairs: tuple[_RecentPair, ...]


def _digest(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


class AdaptiveContextAffinity:
    """Learn recent binary context activation agreement from completed reads.

    Each context pair contributes at most once per event, even when there are
    multiple views/candidates. Context activity is the OR of its actual reads.
    Unobserved contexts are not assigned zero activity. All response records must
    refer to one observation. Positive diversity is counted from transformed
    input digests, without event IDs, locations, scores or memory-step counters.

    This deliberately modest approximation is not Pearson correlation or a map
    of equivalent transformations: different transformations may both regularly
    recognize familiar scenes. Selection therefore additionally requires current
    overlapping factor evidence at exactly the same nonempty supplied origin.
    """

    def __init__(
        self, encoding_id: str, config: AdaptiveContextAffinityConfig | None = None
    ) -> None:
        if config is not None and type(config) is not AdaptiveContextAffinityConfig:
            raise ValueError("config must be AdaptiveContextAffinityConfig")
        self.config = config or AdaptiveContextAffinityConfig()
        self._validator = ContextAffinity(encoding_id, self.config.legacy_config())
        self.encoding_id = encoding_id
        self._events: dict[str, str] = {}
        self._history: tuple[_RecentEvent, ...] = ()

    def _responses(
        self, result: RecognitionResult, budget: _Budget
    ) -> dict[str, tuple[bool, tuple[str, ...]]]:
        if (
            type(result.responses) is not tuple
            or len(result.responses) > self.config.max_responses
            or len(result.responses) != result.examined_views
        ):
            raise ValueError("every completed view needs one bounded response record")
        seen: set[str] = set()
        observations: set[str] = set()
        contexts: dict[str, list[ContextResponse]] = {}
        for response in result.responses:
            budget.check()
            if type(response) is not ContextResponse:
                raise ValueError("invalid context response")
            for name in ("context_id", "view_id", "observation_id"):
                _identifier(getattr(response, name), name)
            if response.view_id in seen:
                raise ValueError("duplicate response view ID")
            seen.add(response.view_id)
            observations.add(response.observation_id)
            _indices(response.source_positions, "response source_positions")
            _digest(response.input_digest, "input_digest")
            if (
                type(response.active) is not bool
                or type(response.score) not in (int, float)
                or not isfinite(response.score)
                or response.score < 0
            ):
                raise ValueError("invalid response activity or score")
            contexts.setdefault(response.context_id, []).append(response)
        if len(observations) > 1:
            raise ValueError("one affinity event must refer to one observation")
        for candidate in result.candidates + result.suppressed:
            budget.check()
            if not any(
                response.active and response.observation_id == candidate.observation_id
                for response in contexts.get(candidate.context_id, ())
            ):
                raise ValueError("candidate has no active completed context response")
        return {
            context: (
                any(response.active for response in responses),
                tuple(sorted({response.input_digest for response in responses})),
            )
            for context, responses in contexts.items()
        }

    def _statistics(
        self, history: tuple[_RecentEvent, ...], budget: _Budget
    ) -> tuple[dict[str, Any], ...]:
        counts: dict[ContextPair, list[int]] = {}
        patterns: dict[ContextPair, set[str]] = {}
        joint_patterns: dict[ContextPair, set[str]] = {}
        for event in history:
            budget.check()
            for pair in event.pairs:
                budget.check()
                key = pair.left, pair.right
                counts.setdefault(key, [0, 0, 0, 0])[pair.state] += 1
                if pair.state:
                    patterns.setdefault(key, set()).add(pair.pattern)
                if pair.state == 3:
                    joint_patterns.setdefault(key, set()).add(pair.pattern)
        rows: list[dict[str, Any]] = []
        for key, (n00, n01, n10, n11) in sorted(counts.items()):
            budget.check()
            active_events = n11 + n10 + n01
            agreement = n11 / active_events if active_events else None
            distinct_joint = len(joint_patterns.get(key, ()))
            rows.append(
                {
                    "left": key[0],
                    "right": key[1],
                    "observations": n00 + active_events,
                    "n11": n11,
                    "n10": n10,
                    "n01": n01,
                    "n00": n00,
                    "activation_agreement": agreement,
                    "distinct_active_patterns": len(patterns.get(key, ())),
                    "distinct_joint_patterns": distinct_joint,
                    "edge": (
                        n11 >= self.config.min_observations
                        and distinct_joint >= self.config.min_distinct_patterns
                        and agreement is not None
                        and agreement >= self.config.affinity_threshold
                    ),
                }
            )
        return tuple(rows)

    def pair_statistics(self) -> tuple[dict[str, Any], ...]:
        """Return recent counts; jointly silent reads never inflate agreement."""

        return self._statistics(self._history, _Budget(self.config.seconds, None))

    def observe(
        self,
        event_id: str,
        result: RecognitionResult,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Atomically record a complete event; incomplete inputs never teach."""

        _identifier(event_id, "event_id")
        budget = _Budget(self.config.seconds, cancelled)
        self._validator._validated(result, budget)
        budget.check()
        if not result.complete:
            return False
        contexts = self._responses(result, budget)
        event_digest = hashlib.sha256(
            json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        budget.check()
        if event_id in self._events:
            if self._events[event_id] != event_digest:
                raise ValueError("event ID was already used with different evidence")
            return False
        if len(self._events) >= self.config.max_events:
            raise ValueError("adaptive affinity event capacity exceeded")
        names = sorted(contexts)
        if len(names) * (len(names) - 1) // 2 > self.config.max_pairs:
            raise ValueError("adaptive affinity pair capacity exceeded")
        pairs: list[_RecentPair] = []
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                budget.check()
                left_active, left_patterns = contexts[left]
                right_active, right_patterns = contexts[right]
                pattern = hashlib.sha256(
                    json.dumps(
                        (left_patterns, right_patterns), separators=(",", ":")
                    ).encode()
                ).hexdigest()
                pairs.append(
                    _RecentPair(
                        left, right, 2 * int(left_active) + int(right_active), pattern
                    )
                )
        history = (*self._history, _RecentEvent(event_id, tuple(pairs)))[
            -self.config.window_events :
        ]
        if len(self._statistics(history, budget)) > self.config.max_pairs:
            raise ValueError("adaptive affinity pair capacity exceeded")
        budget.check()
        self._history = history
        self._events[event_id] = event_digest
        return True

    def select(
        self,
        result: RecognitionResult,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecognitionResult:
        """Select direct local maxima without inventing object provenance.

        An interrupted selection returns all original candidates and suppression
        records with explicit incompleteness. Selection never teaches the map.
        """

        budget = _Budget(self.config.seconds, cancelled)
        try:
            features, conflicts = self._validator._validated(result, budget)
            budget.check()
            if not result.complete:
                return result
            edges = {
                (row["left"], row["right"])
                for row in self._statistics(self._history, budget)
                if row["edge"]
            }
            remaining = sorted(
                result.candidates,
                key=lambda item: (-item.familiarity, -item.quality, item.candidate_id),
            )
            kept: list[RecognitionCandidate] = []
            suppressed = list(result.suppressed)
            while remaining:
                budget.check()
                winner = remaining.pop(0)
                kept.append(winner)
                survivors: list[RecognitionCandidate] = []
                for candidate in remaining:
                    budget.check()
                    if _pair(
                        winner.context_id, candidate.context_id
                    ) in edges and self._validator._supported(
                        winner, candidate, features, conflicts
                    ):
                        suppressed.append(candidate)
                    else:
                        survivors.append(candidate)
                remaining = survivors
            retained = {candidate.candidate_id for candidate in kept}
            relations = tuple(
                relation
                for relation in result.relations
                if relation.left in retained and relation.right in retained
            )
            budget.check()
            return replace(
                result,
                candidates=tuple(kept),
                suppressed=tuple(suppressed),
                relations=relations,
            )
        except InterruptedError as error:
            return replace(
                result, complete=False, stop_reason=result.stop_reason or str(error)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "adaptive_context_affinity",
            "format_version": 1,
            "encoding_id": self.encoding_id,
            "config": asdict(self.config),
            "events": [
                {"event_id": event_id, "digest": digest}
                for event_id, digest in self._events.items()
            ],
            "history": [asdict(event) for event in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveContextAffinity:
        """Restore bounded history and derive counts instead of trusting them."""

        if type(data) is not dict or set(data) != {
            "kind",
            "format_version",
            "encoding_id",
            "config",
            "events",
            "history",
        }:
            raise ValueError("invalid adaptive affinity state fields")
        if (
            data["kind"] != "adaptive_context_affinity"
            or type(data["format_version"]) is not int
            or data["format_version"] != 1
        ):
            raise ValueError("unsupported adaptive affinity state version")
        config_data = data["config"]
        if type(config_data) is not dict or set(config_data) != set(
            asdict(AdaptiveContextAffinityConfig())
        ):
            raise ValueError("invalid adaptive affinity configuration fields")
        config = AdaptiveContextAffinityConfig(**config_data)
        restored = cls(data["encoding_id"], config)
        budget = _Budget(config.seconds, None)
        events, history = data["events"], data["history"]
        if type(events) is not list or len(events) > config.max_events:
            raise ValueError("invalid or oversized adaptive event list")
        if type(history) is not list or len(history) > config.window_events:
            raise ValueError("invalid or oversized adaptive history")
        for event in events:
            budget.check()
            if type(event) is not dict or set(event) != {"event_id", "digest"}:
                raise ValueError("invalid adaptive event")
            _identifier(event["event_id"], "event_id")
            _digest(event["digest"], "event digest")
            if event["event_id"] in restored._events:
                raise ValueError("duplicate adaptive event ID")
            restored._events[event["event_id"]] = event["digest"]
        recent: list[_RecentEvent] = []
        for event in history:
            budget.check()
            if type(event) is not dict or set(event) != {"event_id", "pairs"}:
                raise ValueError("invalid adaptive history event")
            _identifier(event["event_id"], "history event_id")
            if (
                type(event["pairs"]) not in (list, tuple)
                or len(event["pairs"]) > config.max_pairs
            ):
                raise ValueError("invalid or oversized history pairs")
            pairs: list[_RecentPair] = []
            seen: set[ContextPair] = set()
            for pair in event["pairs"]:
                budget.check()
                if type(pair) is not dict or set(pair) != {
                    "left",
                    "right",
                    "state",
                    "pattern",
                }:
                    raise ValueError("invalid adaptive history pair")
                _identifier(pair["left"], "left context")
                _identifier(pair["right"], "right context")
                key = pair["left"], pair["right"]
                if key[0] >= key[1] or key in seen:
                    raise ValueError("history pairs must be unique and ordered")
                seen.add(key)
                _integer(pair["state"], "activation state", 0, 3)
                _digest(pair["pattern"], "pattern digest")
                pairs.append(_RecentPair(**pair))
            recent.append(_RecentEvent(event["event_id"], tuple(pairs)))
        if [event.event_id for event in recent] != list(restored._events)[
            -config.window_events :
        ]:
            raise ValueError("history must be the recent ordered event suffix")
        restored._history = tuple(recent)
        if len(restored._statistics(restored._history, budget)) > config.max_pairs:
            raise ValueError("adaptive affinity pair capacity exceeded")
        budget.check()
        return restored

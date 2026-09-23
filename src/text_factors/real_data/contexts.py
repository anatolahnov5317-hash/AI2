"""Learnable sparse context transforms and local-maxima selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import LearningEpisode


def _code(value: tuple[int, ...], width: int, name: str) -> tuple[int, ...]:
    result = tuple(sorted(set(value)))
    if any(type(bit) is not int or not 0 <= bit < width for bit in result):
        raise ValueError(f"{name} contains a bit outside width")
    return result


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


class SparseTransform:
    """Sparse associations with one vote per independent group and bit pair.

    A mask lists *known* target bits. Bits outside it are unknown, while known
    bits missing from ``target_bits`` are observed negatives. Without a group
    ID, historical ``fit`` callers retain their episode-frequency behavior;
    those counts are never reported as independent support.
    """

    MAX_GROUP_VOTES: int = 100_000

    def __init__(self, width: int):
        if type(width) is not int or width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self._counts: dict[int, dict[int, int]] = {}
        self._negative_counts: dict[int, dict[int, int]] = {}
        self._group_votes: dict[str, dict[int, dict[int, int]]] = {}
        self._vote_entries = 0
        self.episodes = 0

    @staticmethod
    def _adjust(
        rows: dict[int, dict[int, int]], source: int, bit: int, delta: int
    ) -> None:
        row = rows.setdefault(source, {})
        count = row.get(bit, 0) + delta
        if count < 0:
            raise ValueError("invalid association count")
        if count:
            row[bit] = count
        else:
            row.pop(bit, None)
            if not row:
                rows.pop(source)

    def fit(
        self,
        source_bits: tuple[int, ...],
        target_bits: tuple[int, ...],
        *,
        observed_target_bits: tuple[int, ...] | None = None,
        group_id: str | None = None,
    ) -> None:
        source = _code(source_bits, self.width, "source")
        target = set(_code(target_bits, self.width, "target"))
        observed: set[int] | None = None
        if observed_target_bits is not None:
            observed = set(_code(observed_target_bits, self.width, "observed target"))
        positive = target if observed is None else target & observed
        negative = set() if observed is None else observed - positive
        if group_id is not None and (
            type(group_id) is not str or not group_id or len(group_id) > 4096
        ):
            raise ValueError("invalid independent group ID")
        if group_id is not None:
            existing = self._group_votes.get(group_id, {})
            new_entries = sum(
                bit not in existing.get(source_bit, {})
                for source_bit in source
                for bit in positive | negative
            )
            if self._vote_entries + new_entries > self.MAX_GROUP_VOTES:
                raise ValueError("independent group vote capacity exceeded")
        for source_bit in source:
            if group_id is None:
                for bit in positive:
                    self._adjust(self._counts, source_bit, bit, 1)
                for bit in negative:
                    self._adjust(self._negative_counts, source_bit, bit, 1)
                continue
            row = self._group_votes.setdefault(group_id, {}).setdefault(source_bit, {})
            for bit, vote in ((bit, 1) for bit in positive):
                previous = row.get(bit)
                if previous is None:
                    row[bit] = vote
                    self._vote_entries += 1
                    self._adjust(self._counts, source_bit, bit, 1)
                elif previous == -1:
                    # Contradictory versions in one group cannot produce two
                    # independent votes or a confident positive/negative.
                    self._adjust(self._negative_counts, source_bit, bit, -1)
                    row[bit] = 0
            for bit, vote in ((bit, -1) for bit in negative):
                previous = row.get(bit)
                if previous is None:
                    row[bit] = vote
                    self._vote_entries += 1
                    self._adjust(self._negative_counts, source_bit, bit, 1)
                elif previous == 1:
                    # Contradictory versions in one group cannot produce two
                    # independent votes or a confident positive/negative.
                    self._adjust(self._counts, source_bit, bit, -1)
                    row[bit] = 0
        self.episodes += 1

    def _ranked(
        self, source: tuple[int, ...], *, observed: set[int] | None = None
    ) -> list[int]:
        scores: dict[int, float] = {}
        for source_bit in source:
            row = self._counts.get(source_bit)
            if not row:
                continue
            total = sum(row.values())
            negatives = self._negative_counts.get(source_bit, {})
            for bit, count in row.items():
                if observed is not None and bit not in observed:
                    continue
                # A known negative weakens the link. Unknown bits do not.
                reliability = count / (count + negatives.get(bit, 0))
                scores[bit] = scores.get(bit, 0.0) + count / total * reliability
        return sorted(scores, key=lambda bit: (-scores[bit], bit))

    def predict(self, source_bits: tuple[int, ...], *, limit: int) -> tuple[int, ...]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be positive")
        source = _code(source_bits, self.width, "source")
        ranked = self._ranked(source)
        return tuple(sorted(ranked[:limit]))

    def support_for(
        self, source_bits: tuple[int, ...], target_bit: int
    ) -> tuple[int, int]:
        """Return positive/negative *independent groups* supporting a link."""

        source = _code(source_bits, self.width, "source")
        bit = _code((target_bit,), self.width, "target")[0]
        positive = negative = 0
        for group in self._group_votes.values():
            votes = {group.get(source_bit, {}).get(bit, 0) for source_bit in source}
            votes.discard(0)
            if votes == {1}:
                positive += 1
            elif votes == {-1}:
                negative += 1
        return positive, negative

    def score(
        self,
        source_bits: tuple[int, ...],
        target_bits: tuple[int, ...],
        *,
        observed_target_bits: tuple[int, ...] | None = None,
    ) -> float:
        target = _code(target_bits, self.width, "target")
        if observed_target_bits is None:
            prediction = self.predict(source_bits, limit=max(1, len(target)))
            return jaccard(prediction, target)
        observed = set(_code(observed_target_bits, self.width, "observed target"))
        if not observed:
            return 0.0  # An empty mask gives no basis for context assignment.
        visible_target = tuple(bit for bit in target if bit in observed)
        ranked = self._ranked(
            _code(source_bits, self.width, "source"), observed=observed
        )
        # Every known absence is informative: a previously predicted bit
        # inside the mask must count against the context if it is now absent.
        prediction = tuple(sorted(ranked))
        if not prediction and not visible_target:
            source = _code(source_bits, self.width, "source")
            if not any(
                self._negative_counts.get(source_bit, {}).get(bit, 0)
                for source_bit in source
                for bit in observed
            ):
                return 0.0  # Shared inactivity is not evidence of similarity.
        return jaccard(prediction, visible_target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "episodes": self.episodes,
            "counts": {
                str(source): {
                    str(target): count for target, count in sorted(row.items())
                }
                for source, row in sorted(self._counts.items())
            },
            "negative_counts": {
                str(source): {
                    str(target): count for target, count in sorted(row.items())
                }
                for source, row in sorted(self._negative_counts.items())
            },
            "group_votes": {
                group: {
                    str(source): {str(bit): vote for bit, vote in sorted(row.items())}
                    for source, row in sorted(sources.items())
                }
                for group, sources in sorted(self._group_votes.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SparseTransform:
        legacy = {"width", "episodes", "counts"}
        current = legacy | {"negative_counts", "group_votes"}
        if type(value) is not dict or set(value) not in (legacy, current):
            raise ValueError("invalid sparse transform")
        transform = cls(value["width"])
        if type(value["episodes"]) is not int or value["episodes"] < 0:
            raise ValueError("invalid transform episode count")
        if type(value["counts"]) is not dict:
            raise ValueError("invalid transform counts")

        def parse_counts(raw: dict[str, Any]) -> dict[int, dict[int, int]]:
            decoded_rows: dict[int, dict[int, int]] = {}
            for source, row in raw.items():
                if (
                    type(source) is not str
                    or not source.isdigit()
                    or type(row) is not dict
                ):
                    raise ValueError("invalid transform row")
                source_bit = int(source)
                if str(source_bit) != source or not 0 <= source_bit < transform.width:
                    raise ValueError("transform source bit outside width")
                decoded: dict[int, int] = {}
                for target, count in row.items():
                    if (
                        type(target) is not str
                        or not target.isdigit()
                        or type(count) is not int
                        or count <= 0
                    ):
                        raise ValueError("invalid transform count")
                    target_bit = int(target)
                    if (
                        str(target_bit) != target
                        or not 0 <= target_bit < transform.width
                    ):
                        raise ValueError("transform target bit outside width")
                    decoded[target_bit] = count
                decoded_rows[source_bit] = decoded
            return decoded_rows

        transform._counts = parse_counts(value["counts"])
        if set(value) == current:
            negatives = value["negative_counts"]
            votes = value["group_votes"]
            if type(negatives) is not dict or type(votes) is not dict:
                raise ValueError("invalid transform independent votes")
            transform._negative_counts = parse_counts(negatives)
            for group, sources in votes.items():
                if (
                    type(group) is not str
                    or not group
                    or len(group) > 4096
                    or type(sources) is not dict
                ):
                    raise ValueError("invalid transform group")
                decoded_sources: dict[int, dict[int, int]] = {}
                for source, row in sources.items():
                    if (
                        type(source) is not str
                        or not source.isdigit()
                        or type(row) is not dict
                    ):
                        raise ValueError("invalid transform group row")
                    source_bit = int(source)
                    if (
                        str(source_bit) != source
                        or not 0 <= source_bit < transform.width
                    ):
                        raise ValueError("invalid transform group source")
                    decoded_row: dict[int, int] = {}
                    for target, vote in row.items():
                        if type(target) is not str or not target.isdigit():
                            raise ValueError("invalid transform group target")
                        bit = int(target)
                        if (
                            str(bit) != target
                            or not 0 <= bit < transform.width
                            or type(vote) is not int
                            or vote not in (-1, 0, 1)
                        ):
                            raise ValueError("invalid transform group vote")
                        decoded_row[bit] = vote
                        transform._vote_entries += 1
                    decoded_sources[source_bit] = decoded_row
                transform._group_votes[group] = decoded_sources
            if transform._vote_entries > transform.MAX_GROUP_VOTES:
                raise ValueError("independent group vote capacity exceeded")
            positive: dict[int, dict[int, int]] = {}
            negative: dict[int, dict[int, int]] = {}
            for sources in transform._group_votes.values():
                for source, row in sources.items():
                    for bit, vote in row.items():
                        if vote:
                            transform._adjust(
                                positive if vote == 1 else negative, source, bit, 1
                            )
            for aggregate, grouped in (
                (transform._counts, positive),
                (transform._negative_counts, negative),
            ):
                for source, row in grouped.items():
                    for bit, count in row.items():
                        if aggregate.get(source, {}).get(bit, 0) < count:
                            raise ValueError("group votes exceed association counts")
        transform.episodes = value["episodes"]
        return transform


@dataclass(slots=True)
class ContextCandidate:
    context_id: str
    transform: SparseTransform
    group_ids: set[str] = field(default_factory=set)
    episode_ids: set[str] = field(default_factory=set)

    @property
    def independent_support(self) -> int:
        return len(self.group_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "group_ids": sorted(self.group_ids),
            "episode_ids": sorted(self.episode_ids),
            "transform": self.transform.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContextResult:
    context_id: str
    predicted_bits: tuple[int, ...]
    score: float
    independent_support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "predicted_bits": list(self.predicted_bits),
            "score": self.score,
            "independent_support": self.independent_support,
        }


class ContextRegistry:
    """Assign episodes to learned transforms and expose non-duplicate maxima."""

    def __init__(
        self,
        *,
        width: int,
        max_contexts: int = 128,
        assignment_threshold: float = 0.35,
        response_activation_threshold: float = 0.5,
    ) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be positive")
        if type(max_contexts) is not int or max_contexts <= 0:
            raise ValueError("max_contexts must be positive")
        for name, value in (
            ("assignment_threshold", assignment_threshold),
            ("response_activation_threshold", response_activation_threshold),
        ):
            if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.width = width
        self.max_contexts = max_contexts
        self.assignment_threshold = float(assignment_threshold)
        self.response_activation_threshold = float(response_activation_threshold)
        self._contexts: dict[str, ContextCandidate] = {}
        self._active_responses: dict[str, set[str]] = {}

    @property
    def contexts(self) -> tuple[ContextCandidate, ...]:
        return tuple(self._contexts[key] for key in sorted(self._contexts))

    def _create(self) -> ContextCandidate:
        if len(self._contexts) >= self.max_contexts:
            raise ValueError("context capacity exceeded")
        context_id = f"ctx_{len(self._contexts) + 1:04d}"
        candidate = ContextCandidate(context_id, SparseTransform(self.width))
        self._contexts[context_id] = candidate
        self._active_responses[context_id] = set()
        return candidate

    def learn(self, episode: LearningEpisode) -> tuple[str, bool, float]:
        source = _code(episode.source_code, self.width, "source")
        target = _code(episode.target_code, self.width, "target")
        observed = (
            episode.observed_target_bits
            if episode.observed_target_bits is not None
            else target
        )
        if not source or not observed:
            raise ValueError(
                "learning episode must contain source and observed target bits"
            )
        ranked = sorted(
            (
                (
                    candidate.transform.score(
                        source,
                        target,
                        observed_target_bits=episode.observed_target_bits,
                    ),
                    candidate.context_id,
                )
                for candidate in self._contexts.values()
                if candidate.transform.episodes
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score = ranked[0][0] if ranked else 0.0
        created = not ranked or best_score < self.assignment_threshold
        candidate = self._create() if created else self._contexts[ranked[0][1]]
        candidate.transform.fit(
            source,
            target,
            observed_target_bits=episode.observed_target_bits,
            group_id=episode.group_id,
        )
        candidate.group_ids.add(episode.group_id)
        candidate.episode_ids.add(episode.episode_id)
        return candidate.context_id, created, best_score

    def predict(self, context_id: str, source_bits: tuple[int, ...], *, limit: int):
        try:
            context = self._contexts[context_id]
        except KeyError as exc:
            raise ValueError("unknown context") from exc
        return context.transform.predict(source_bits, limit=limit)

    def record_response(self, context_id: str, episode_id: str, score: float) -> None:
        if context_id not in self._contexts:
            raise ValueError("unknown context")
        if type(episode_id) is not str or not episode_id:
            raise ValueError("episode_id must be nonempty")
        if type(score) not in (int, float) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("response score must be in [0, 1]")
        if float(score) >= self.response_activation_threshold:
            self._active_responses[context_id].add(episode_id)
        else:
            self._active_responses[context_id].discard(episode_id)

    def affinity(self, left: str, right: str) -> float:
        if left not in self._contexts or right not in self._contexts:
            raise ValueError("unknown context")
        if left == right:
            return 1.0
        left_active = self._active_responses[left]
        right_active = self._active_responses[right]
        union = left_active | right_active
        # Shared inactivity carries no evidence of similarity.
        return 0.0 if not union else len(left_active & right_active) / len(union)

    def local_maxima(
        self,
        source_bits: tuple[int, ...],
        *,
        scorer,
        output_limit: int = 64,
        max_results: int = 4,
        suppression_affinity: float = 0.8,
    ) -> tuple[ContextResult, ...]:
        if type(max_results) is not int or max_results <= 0:
            raise ValueError("max_results must be positive")
        if (
            type(suppression_affinity) not in (int, float)
            or not 0.0 <= float(suppression_affinity) <= 1.0
        ):
            raise ValueError("suppression_affinity must be in [0, 1]")
        ranked: list[ContextResult] = []
        for context in self.contexts:
            predicted = context.transform.predict(source_bits, limit=output_limit)
            score = float(scorer(context.context_id, predicted))
            if not 0.0 <= score <= 1.0:
                raise ValueError("scorer must return a value in [0, 1]")
            ranked.append(
                ContextResult(
                    context.context_id,
                    predicted,
                    score,
                    context.independent_support,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.context_id))
        selected: list[ContextResult] = []
        for item in ranked:
            if item.score <= 0.0:
                continue
            if any(
                self.affinity(item.context_id, prior.context_id)
                >= float(suppression_affinity)
                for prior in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= max_results:
                break
        return tuple(selected)

    def reliable_context_ids(
        self, *, min_independent_groups: int = 2
    ) -> tuple[str, ...]:
        if type(min_independent_groups) is not int or min_independent_groups <= 0:
            raise ValueError("min_independent_groups must be positive")
        return tuple(
            context.context_id
            for context in self.contexts
            if context.independent_support >= min_independent_groups
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "max_contexts": self.max_contexts,
            "assignment_threshold": self.assignment_threshold,
            "response_activation_threshold": self.response_activation_threshold,
            "contexts": [context.to_dict() for context in self.contexts],
            "active_responses": {
                context_id: sorted(values)
                for context_id, values in sorted(self._active_responses.items())
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ContextRegistry:
        expected = {
            "width",
            "max_contexts",
            "assignment_threshold",
            "response_activation_threshold",
            "contexts",
            "active_responses",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("invalid context registry")
        registry = cls(
            width=value["width"],
            max_contexts=value["max_contexts"],
            assignment_threshold=value["assignment_threshold"],
            response_activation_threshold=value["response_activation_threshold"],
        )
        contexts = value["contexts"]
        responses = value["active_responses"]
        if type(contexts) is not list or type(responses) is not dict:
            raise ValueError("invalid context registry collections")
        for raw in contexts:
            required = {"context_id", "group_ids", "episode_ids", "transform"}
            if type(raw) is not dict or set(raw) != required:
                raise ValueError("invalid context candidate")
            context_id = raw["context_id"]
            if type(context_id) is not str or not context_id:
                raise ValueError("invalid context_id")
            if context_id in registry._contexts:
                raise ValueError("duplicate context_id")
            group_ids = raw["group_ids"]
            episode_ids = raw["episode_ids"]
            if (
                type(group_ids) is not list
                or type(episode_ids) is not list
                or any(type(item) is not str or not item for item in group_ids)
                or any(type(item) is not str or not item for item in episode_ids)
            ):
                raise ValueError("invalid context support IDs")
            transform = SparseTransform.from_dict(raw["transform"])
            if transform.width != registry.width:
                raise ValueError("context transform width mismatch")
            registry._contexts[context_id] = ContextCandidate(
                context_id,
                transform,
                set(group_ids),
                set(episode_ids),
            )
        if len(registry._contexts) > registry.max_contexts:
            raise ValueError("persisted contexts exceed max_contexts")
        if set(responses) != set(registry._contexts):
            raise ValueError("active response keys must match context IDs")
        for context_id, raw_ids in responses.items():
            if type(raw_ids) is not list or any(
                type(item) is not str or not item for item in raw_ids
            ):
                raise ValueError("invalid active response IDs")
            registry._active_responses[context_id] = set(raw_ids)
        return registry

"""Sparse local-cluster associative memory."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from math import log1p

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .consolidation import coactivation_weights


class ClusterStatus(IntEnum):
    """Consolidation stages for a local memory cluster."""

    TEMPORARY = 0
    PROBATION = 1
    STABLE = 2


@dataclass(slots=True)
class Cluster:
    """A conjunction of active input bits stored at one receptive point."""

    bits: NDArray[np.int32]
    bit_hits: NDArray[np.int64]
    created_at: int
    last_seen: int
    status: ClusterStatus = ClusterStatus.TEMPORARY
    partial_hits: int = 1
    exact_hits: int = 1
    partial_errors: int = 0
    complete_errors: int = 0
    activation_history: list[NDArray[np.bool_]] = field(default_factory=list)

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(int(bit) for bit in self.bits)


@dataclass(frozen=True, slots=True)
class MemoryReadout:
    output: NDArray[np.bool_]
    quality: int
    active_points: int
    point_indices: NDArray[np.int32]
    output_scores: NDArray[np.float64]

    @property
    def active_output_bits(self) -> list[int]:
        return [int(index) for index in np.flatnonzero(self.output)]


@dataclass(frozen=True, slots=True)
class FactorSummary:
    output_bit: int
    support: int
    cluster_count: int
    point_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "output_bit": self.output_bit,
            "support": self.support,
            "cluster_count": self.cluster_count,
            "point_count": self.point_count,
        }


class CombinatorialMemory:
    """Random receptive fields with local, progressively consolidated clusters."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        receptors: NDArray[np.integer] | None = None,
        output_map: NDArray[np.integer] | None = None,
    ) -> None:
        self.config = config
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))

        if receptors is None:
            generated = [
                np.sort(
                    rng.choice(
                        config.input_bits,
                        size=config.receptive_bits,
                        replace=False,
                    )
                )
                for _ in range(config.point_count)
            ]
            self.receptors = np.asarray(generated, dtype=np.int32)
        else:
            converted = np.asarray(receptors)
            expected = (config.point_count, config.receptive_bits)
            if converted.shape != expected:
                raise ValueError(
                    f"receptors shape must be {expected}, got {converted.shape}"
                )
            if not np.issubdtype(converted.dtype, np.integer):
                raise ValueError("receptors must contain integer bit indices")
            if np.any(converted < 0) or np.any(converted >= config.input_bits):
                raise ValueError("receptors contain an out-of-range bit index")
            if np.any(np.diff(np.sort(converted, axis=1), axis=1) == 0):
                raise ValueError("each receptor must contain unique bit indices")
            self.receptors = converted.astype(np.int32, copy=True)

        if output_map is None:
            self.output_map = rng.integers(
                0,
                config.output_bits,
                size=config.point_count,
                dtype=np.int32,
            )
        else:
            converted = np.asarray(output_map)
            if converted.shape != (config.point_count,):
                raise ValueError(
                    "output_map shape must be "
                    f"{(config.point_count,)}, got {converted.shape}"
                )
            if not np.issubdtype(converted.dtype, np.integer):
                raise ValueError("output_map must contain integer bit indices")
            if np.any(converted < 0) or np.any(converted >= config.output_bits):
                raise ValueError("output_map contains an out-of-range bit index")
            if np.any(converted > np.iinfo(np.int32).max):
                raise ValueError("output_map contains an index too large for int32")
            self.output_map = converted.astype(np.int32, copy=True)

        self.clusters: list[list[Cluster]] = [[] for _ in range(config.point_count)]
        self._signatures: list[set[tuple[int, ...]]] = [
            set() for _ in range(config.output_bits)
        ]
        self._nonempty_points: set[int] = set()
        self._cluster_count = 0
        self.step = 0

    @property
    def cluster_count(self) -> int:
        """Return the exact current cluster count in O(1)."""

        return self._cluster_count

    def _validate_bits(
        self,
        active: NDArray[np.bool_] | Iterable[bool],
        *,
        length: int | None = None,
        name: str = "active",
    ) -> NDArray[np.bool_]:
        converted = np.asarray(active, dtype=np.bool_)
        expected = self.config.input_bits if length is None else length
        if converted.shape != (expected,):
            raise ValueError(
                f"{name} shape must be {(expected,)}, got {converted.shape}"
            )
        return converted

    @staticmethod
    def _match_count(cluster: Cluster, active: NDArray[np.bool_]) -> int:
        return int(np.count_nonzero(active[cluster.bits]))

    def overlap_counts(self, active: NDArray[np.bool_]) -> NDArray[np.int32]:
        checked = self._validate_bits(active)
        # ModelConfig limits bit indices to int32, so int32 also safely holds
        # every possible receptive-field overlap.
        return np.count_nonzero(checked[self.receptors], axis=1).astype(np.int32)

    def familiarity(self, active: NDArray[np.bool_]) -> float:
        """Return prior-memory evidence without mutating the model."""

        checked = self._validate_bits(active)
        score = 0.0
        status_weight = {
            ClusterStatus.TEMPORARY: 1.0,
            ClusterStatus.PROBATION: 2.0,
            ClusterStatus.STABLE: 4.0,
        }
        for point_index in self._nonempty_points:
            for cluster in self.clusters[point_index]:
                matched = self._match_count(cluster, checked)
                if matched >= self.config.activation_threshold:
                    score += (
                        status_weight[cluster.status]
                        * (matched / len(cluster.bits))
                        * log1p(cluster.partial_hits)
                    )
        return score

    def observe(
        self,
        active: NDArray[np.bool_],
        *,
        target: NDArray[np.bool_] | None = None,
    ) -> int:
        """Learn one input online and return the number of created clusters."""

        checked = self._validate_bits(active)
        checked_target: NDArray[np.bool_] | None = None
        if target is not None:
            checked_target = self._validate_bits(
                target,
                length=self.config.output_bits,
                name="target",
            )

        self._update_existing(checked, checked_target)
        self._consolidate()
        created = self._add_new_clusters(checked, checked_target)
        self._consolidate()
        self.step += 1
        return created

    def _update_existing(
        self,
        active: NDArray[np.bool_],
        target: NDArray[np.bool_] | None,
    ) -> None:
        for point_index in tuple(self._nonempty_points):
            output_bit = int(self.output_map[point_index])
            target_is_active = target is None or bool(target[output_bit])
            for cluster in self.clusters[point_index]:
                matched = self._match_count(cluster, active)
                if matched < self.config.activation_threshold:
                    continue

                cluster.partial_hits += 1
                cluster.bit_hits += active[cluster.bits].astype(np.int64)
                cluster.last_seen = self.step
                if self.config.consolidation_method == "coactivation":
                    cluster.activation_history.append(active[cluster.bits].copy())
                    del cluster.activation_history[
                        : -self.config.coactivation_history_size
                    ]
                if target is not None and not target_is_active:
                    cluster.partial_errors += 1

                if matched == len(cluster.bits):
                    cluster.exact_hits += 1
                    if target is not None and not target_is_active:
                        cluster.complete_errors += 1

    def _add_new_clusters(
        self,
        active: NDArray[np.bool_],
        target: NDArray[np.bool_] | None,
    ) -> int:
        overlaps = self.overlap_counts(active)
        candidates = np.flatnonzero(overlaps >= self.config.create_threshold)
        created = 0

        for raw_point_index in candidates:
            point_index = int(raw_point_index)
            output_bit = int(self.output_map[point_index])
            if target is not None and not bool(target[output_bit]):
                continue
            if len(self.clusters[point_index]) >= self.config.max_clusters_per_point:
                continue

            receptor = self.receptors[point_index]
            bits = np.sort(receptor[active[receptor]]).astype(np.int32, copy=True)
            signature = tuple(int(bit) for bit in bits)
            if signature in self._signatures[output_bit]:
                continue

            cluster = Cluster(
                bits=bits,
                bit_hits=np.ones(len(bits), dtype=np.int64),
                created_at=self.step,
                last_seen=self.step,
                activation_history=(
                    [np.ones(len(bits), dtype=np.bool_)]
                    if self.config.consolidation_method == "coactivation"
                    else []
                ),
            )
            self.clusters[point_index].append(cluster)
            self._signatures[output_bit].add(signature)
            self._nonempty_points.add(point_index)
            created += 1
        self._cluster_count += created
        return created

    def _has_excess_error(self, cluster: Cluster) -> bool:
        complete_bad = (
            cluster.exact_hits >= self.config.min_error_observations
            and cluster.complete_errors / cluster.exact_hits
            > self.config.max_complete_error_rate
        )
        partial_bad = (
            cluster.partial_hits >= self.config.min_error_observations
            and cluster.partial_errors / cluster.partial_hits
            > self.config.max_partial_error_rate
        )
        return complete_bad or partial_bad

    def _prune(self, cluster: Cluster) -> bool:
        if self.config.consolidation_method == "coactivation":
            weights = coactivation_weights(
                cluster.activation_history, passes=self.config.coactivation_passes
            )
            keep = weights > self.config.prune_keep_ratio
        else:
            frequencies = cluster.bit_hits / max(cluster.partial_hits, 1)
            keep = frequencies >= self.config.prune_keep_ratio
        if int(np.count_nonzero(keep)) < self.config.activation_threshold:
            return False
        cluster.bits = cluster.bits[keep].copy()
        cluster.bit_hits = cluster.bit_hits[keep].copy()
        cluster.activation_history = [
            row[keep].copy() for row in cluster.activation_history
        ]
        return True

    def replay_consolidation(self) -> dict[str, int]:
        """Refine existing hypotheses from bounded history, without new support.

        No observation is replayed through ``observe``: counters, timestamps,
        stages and ``step`` are unchanged. No new clusters are born. Unsupported
        or duplicate hypotheses can disappear; this is not independent evidence.
        """

        if self.config.consolidation_method != "coactivation":
            raise ValueError("history replay requires coactivation consolidation")
        before = sum(len(clusters) for clusters in self.clusters)
        removed_bits = 0
        for point_index in sorted(self._nonempty_points):
            output_bit = int(self.output_map[point_index])
            retained: list[Cluster] = []
            for cluster in self.clusters[point_index]:
                old_size = len(cluster.bits)
                self._signatures[output_bit].discard(cluster.signature)
                if self._has_excess_error(cluster) or not self._prune(cluster):
                    removed_bits += old_size
                    continue
                if cluster.signature in self._signatures[output_bit]:
                    removed_bits += old_size
                    continue
                self._signatures[output_bit].add(cluster.signature)
                removed_bits += old_size - len(cluster.bits)
                retained.append(cluster)
            self.clusters[point_index] = retained
            if not retained:
                self._nonempty_points.discard(point_index)
        after = sum(len(clusters) for clusters in self.clusters)
        self._cluster_count = after
        return {"removed_clusters": before - after, "removed_bits": removed_bits}

    def _consolidate(self) -> None:
        for point_index in tuple(self._nonempty_points):
            output_bit = int(self.output_map[point_index])
            before_count = len(self.clusters[point_index])
            retained: list[Cluster] = []

            for cluster in self.clusters[point_index]:
                old_signature = cluster.signature
                if self._has_excess_error(cluster):
                    self._signatures[output_bit].discard(old_signature)
                    continue

                next_status: ClusterStatus | None = None
                if (
                    cluster.status == ClusterStatus.TEMPORARY
                    and cluster.partial_hits >= self.config.probation_after
                ):
                    next_status = ClusterStatus.PROBATION
                elif (
                    cluster.status == ClusterStatus.PROBATION
                    and cluster.partial_hits >= self.config.stable_after
                ):
                    next_status = ClusterStatus.STABLE

                if next_status is not None:
                    self._signatures[output_bit].discard(old_signature)
                    if not self._prune(cluster):
                        continue
                    new_signature = cluster.signature
                    if new_signature in self._signatures[output_bit]:
                        continue
                    self._signatures[output_bit].add(new_signature)
                    cluster.status = next_status

                retained.append(cluster)

            self.clusters[point_index] = retained
            self._cluster_count -= before_count - len(retained)
            if not retained:
                self._nonempty_points.discard(point_index)

    def read(
        self,
        active: NDArray[np.bool_],
        *,
        stable_only: bool = True,
    ) -> MemoryReadout:
        """Produce a sparse factor code from matching local clusters."""

        checked = self._validate_bits(active)
        quality = np.zeros(self.config.point_count, dtype=np.int32)
        support = np.zeros(self.config.point_count, dtype=np.float64)

        for point_index in self._nonempty_points:
            for cluster in self.clusters[point_index]:
                if stable_only and cluster.status != ClusterStatus.STABLE:
                    continue
                matched = self._match_count(cluster, checked)
                if matched >= self.config.activation_threshold:
                    quality[point_index] = max(quality[point_index], matched)
                    support[point_index] += float(cluster.partial_hits)

        selected_quality = 0
        selected_points = np.empty(0, dtype=np.int32)
        eligible_qualities = np.unique(
            quality[quality >= self.config.activation_threshold]
        )
        for candidate_quality in sorted(
            (int(value) for value in eligible_qualities), reverse=True
        ):
            candidates = np.flatnonzero(quality >= candidate_quality)
            if len(candidates) >= self.config.min_active_points:
                selected_quality = candidate_quality
                selected_points = candidates.astype(np.int32)
                break

        output_scores = np.zeros(self.config.output_bits, dtype=np.float64)
        if len(selected_points):
            np.add.at(
                output_scores,
                self.output_map[selected_points],
                support[selected_points],
            )
        output = output_scores > 0.0
        return MemoryReadout(
            output=output,
            quality=selected_quality,
            active_points=len(selected_points),
            point_indices=selected_points,
            output_scores=output_scores,
        )

    def predict(self, active: NDArray[np.bool_]) -> MemoryReadout:
        """Produce a supervised output using exact matches of stable clusters."""

        checked = self._validate_bits(active)
        output_scores = np.zeros(self.config.output_bits, dtype=np.float64)
        active_points: list[int] = []
        best_quality = 0

        for point_index in self._nonempty_points:
            point_score = 0.0
            point_quality = 0
            for cluster in self.clusters[point_index]:
                if cluster.status != ClusterStatus.STABLE:
                    continue
                matched = self._match_count(cluster, checked)
                if matched == len(cluster.bits):
                    point_quality = max(point_quality, matched)
                    point_score += matched - self.config.activation_threshold + 1
            if point_score:
                active_points.append(point_index)
                best_quality = max(best_quality, point_quality)
                output_scores[int(self.output_map[point_index])] += point_score

        output = output_scores >= self.config.prediction_vote_threshold
        return MemoryReadout(
            output=output,
            quality=best_quality,
            active_points=len(active_points),
            point_indices=np.asarray(active_points, dtype=np.int32),
            output_scores=output_scores,
        )

    def top_factors(self, limit: int = 10) -> list[FactorSummary]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        cluster_counts = np.zeros(self.config.output_bits, dtype=np.int64)
        supports = np.zeros(self.config.output_bits, dtype=np.int64)
        point_sets: list[set[int]] = [set() for _ in range(self.config.output_bits)]

        for point_index in self._nonempty_points:
            output_bit = int(self.output_map[point_index])
            for cluster in self.clusters[point_index]:
                if cluster.status != ClusterStatus.STABLE:
                    continue
                cluster_counts[output_bit] += 1
                supports[output_bit] += cluster.partial_hits
                point_sets[output_bit].add(point_index)

        factors = [
            FactorSummary(
                output_bit=output_bit,
                support=int(supports[output_bit]),
                cluster_count=int(cluster_counts[output_bit]),
                point_count=len(point_sets[output_bit]),
            )
            for output_bit in range(self.config.output_bits)
            if cluster_counts[output_bit] > 0
        ]
        factors.sort(
            key=lambda factor: (
                -factor.support,
                -factor.cluster_count,
                factor.output_bit,
            )
        )
        return factors[:limit]

    def iter_clusters(self) -> Iterator[tuple[int, Cluster]]:
        for point_index in sorted(self._nonempty_points):
            for cluster in self.clusters[point_index]:
                yield point_index, cluster

    def add_loaded_cluster(self, point_index: int, cluster: Cluster) -> None:
        """Restore a validated cluster from a persisted model."""

        if isinstance(point_index, (bool, np.bool_)) or not isinstance(
            point_index, (int, np.integer)
        ):
            raise ValueError("loaded cluster point index must be an integer")
        if not 0 <= point_index < self.config.point_count:
            raise ValueError("loaded cluster has an invalid point index")
        bits = np.asarray(cluster.bits)
        bit_hits = np.asarray(cluster.bit_hits)
        if bits.ndim != 1 or bit_hits.ndim != 1:
            raise ValueError("loaded cluster bits and bit_hits must be one-dimensional")
        if not np.issubdtype(bits.dtype, np.integer):
            raise ValueError("loaded cluster bits must contain integer indices")
        if not np.issubdtype(bit_hits.dtype, np.integer):
            raise ValueError("loaded cluster bit_hits must contain integers")
        if len(cluster.bits) < self.config.activation_threshold:
            raise ValueError("loaded cluster is shorter than activation_threshold")
        if len(cluster.bits) > self.config.receptive_bits:
            raise ValueError("loaded cluster is longer than its receptor")
        if np.any(bits < 0) or np.any(bits >= self.config.input_bits):
            raise ValueError("loaded cluster contains an out-of-range bit")
        if bits.shape != bit_hits.shape:
            raise ValueError("loaded cluster bits and bit_hits shapes differ")
        if len(bits) > 1 and np.any(bits[1:] <= bits[:-1]):
            raise ValueError("loaded cluster bits must be sorted and unique")
        if not np.all(np.isin(bits, self.receptors[int(point_index)])):
            raise ValueError("loaded cluster contains a bit outside its receptor")

        counter_names = (
            "partial_hits",
            "exact_hits",
            "partial_errors",
            "complete_errors",
            "created_at",
            "last_seen",
        )
        counters: dict[str, int] = {}
        for name in counter_names:
            value = getattr(cluster, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"loaded cluster {name} must be an integer")
            counters[name] = int(value)

        if (
            counters["exact_hits"] < 1
            or counters["partial_hits"] < counters["exact_hits"]
        ):
            raise ValueError(
                "loaded cluster hits must satisfy partial_hits >= exact_hits >= 1"
            )
        if counters["partial_errors"] < 0 or counters["complete_errors"] < 0:
            raise ValueError("loaded cluster error counters cannot be negative")
        if counters["partial_errors"] > counters["partial_hits"]:
            raise ValueError("loaded cluster partial_errors exceed partial_hits")
        if counters["complete_errors"] > counters["exact_hits"]:
            raise ValueError("loaded cluster complete_errors exceed exact_hits")
        if counters["complete_errors"] > counters["partial_errors"]:
            raise ValueError("loaded cluster complete_errors exceed partial_errors")
        if np.any(bit_hits < 1) or np.any(bit_hits > counters["partial_hits"]):
            raise ValueError(
                "loaded cluster bit_hits must be between 1 and partial_hits"
            )
        if not (0 <= counters["created_at"] <= counters["last_seen"] <= self.step):
            raise ValueError(
                "loaded cluster timestamps must satisfy "
                "0 <= created_at <= last_seen <= memory.step"
            )
        if not isinstance(cluster.status, ClusterStatus):
            raise ValueError("loaded cluster has an invalid status")

        history = cluster.activation_history
        if not isinstance(history, list):
            raise ValueError("loaded cluster activation_history must be a list")
        if self.config.consolidation_method == "frequency":
            if history:
                raise ValueError("frequency clusters cannot contain activation history")
        elif len(history) != min(
            self.config.coactivation_history_size, counters["partial_hits"]
        ):
            raise ValueError("loaded cluster activation history count is invalid")
        for row in history:
            if not isinstance(row, np.ndarray) or row.dtype != np.dtype(np.bool_):
                raise ValueError("loaded cluster history must contain boolean arrays")
            if row.shape != bits.shape:
                raise ValueError("loaded cluster history shape differs from bits")
        if history and np.any(np.sum(history, axis=0, dtype=np.int64) > bit_hits):
            raise ValueError("loaded cluster activation history exceeds bit_hits")

        cluster.bits = bits.astype(np.int32, copy=True)
        cluster.bit_hits = bit_hits.astype(np.int64, copy=True)
        cluster.activation_history = [row.copy() for row in history]
        for name, value in counters.items():
            setattr(cluster, name, value)

        output_bit = int(self.output_map[int(point_index)])
        if cluster.signature in self._signatures[output_bit]:
            raise ValueError("persisted model contains a duplicate cluster")
        if len(self.clusters[int(point_index)]) >= self.config.max_clusters_per_point:
            raise ValueError("persisted model exceeds max_clusters_per_point")

        self.clusters[int(point_index)].append(cluster)
        self._cluster_count += 1
        self._signatures[output_bit].add(cluster.signature)
        self._nonempty_points.add(int(point_index))

    def stats(self) -> dict[str, int]:
        status_counts = {status: 0 for status in ClusterStatus}
        total = 0
        for _, cluster in self.iter_clusters():
            status_counts[cluster.status] += 1
            total += 1
        return {
            "step": self.step,
            "points_with_memory": len(self._nonempty_points),
            "clusters": total,
            "temporary_clusters": status_counts[ClusterStatus.TEMPORARY],
            "probation_clusters": status_counts[ClusterStatus.PROBATION],
            "stable_clusters": status_counts[ClusterStatus.STABLE],
        }

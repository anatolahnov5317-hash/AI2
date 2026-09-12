"""High-level text factor model and safe model persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .encoder import DEFAULT_ALPHABET, SparseSymbolEncoder
from .memory import (
    Cluster,
    ClusterStatus,
    CombinatorialMemory,
    FactorSummary,
    MemoryReadout,
)

MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TransformResult:
    window: str
    offset: int
    context: int
    context_scores: tuple[float, ...]
    output_bits: tuple[int, ...]
    quality: int
    active_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "offset": self.offset,
            "context": self.context,
            "context_scores": [round(score, 6) for score in self.context_scores],
            "output_bits": list(self.output_bits),
            "quality": self.quality,
            "active_points": self.active_points,
        }


@dataclass(frozen=True, slots=True)
class ConceptEvidence:
    symbol: str
    position: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "position": self.position,
            "score": round(self.score, 6),
        }


class TextFactorModel:
    """Online text model built from deterministic SDRs and local memories."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        alphabet: str = DEFAULT_ALPHABET,
        encoder: SparseSymbolEncoder | None = None,
        memory: CombinatorialMemory | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.encoder = encoder or SparseSymbolEncoder(self.config, alphabet)
        self.memory = memory or CombinatorialMemory(self.config)
        if self.encoder.config != self.config or self.memory.config != self.config:
            raise ValueError("encoder, memory, and model must use the same config")

    @property
    def alphabet(self) -> str:
        return self.encoder.alphabet

    def _select_context(
        self,
        window: str,
        *,
        offset: int,
    ) -> tuple[int, tuple[float, ...], np.ndarray]:
        interpretations = self.encoder.interpretations(window, offset=offset)
        scores = tuple(
            float(self.memory.familiarity(candidate)) for candidate in interpretations
        )
        # np.argmax and tuple.index both prefer the lowest context on a tie,
        # making cold-start behavior explicit and reproducible.
        context = scores.index(max(scores))
        return context, scores, interpretations[context]

    @staticmethod
    def _result(
        window: str,
        offset: int,
        context: int,
        context_scores: tuple[float, ...],
        readout: MemoryReadout,
    ) -> TransformResult:
        return TransformResult(
            window=window,
            offset=offset,
            context=context,
            context_scores=context_scores,
            output_bits=tuple(readout.active_output_bits),
            quality=readout.quality,
            active_points=readout.active_points,
        )

    def partial_fit_window(self, window: str, *, offset: int = 0) -> TransformResult:
        """Select the most familiar context and learn one window."""

        context, scores, active = self._select_context(window, offset=offset)
        self.memory.observe(active)
        readout = self.memory.read(active)
        return self._result(window, offset, context, scores, readout)

    def transform_window(self, window: str, *, offset: int = 0) -> TransformResult:
        """Analyze one window without modifying the model."""

        context, scores, active = self._select_context(window, offset=offset)
        readout = self.memory.read(active)
        return self._result(window, offset, context, scores, readout)

    def fit_text(
        self,
        text: str,
        *,
        epochs: int = 1,
        stride: int = 1,
    ) -> TextFactorModel:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        for _ in range(epochs):
            for window, offset in self.encoder.iter_windows(text, stride=stride):
                self.partial_fit_window(window, offset=offset)
        return self

    def transform_text(self, text: str, *, stride: int = 1) -> list[TransformResult]:
        return [
            self.transform_window(window, offset=offset)
            for window, offset in self.encoder.iter_windows(text, stride=stride)
        ]

    def learn_context_transform(
        self,
        window: str,
        *,
        source_context: int = 0,
        target_context: int = 1,
        offset: int = 0,
    ) -> MemoryReadout:
        """Learn the source-to-target context mapping from the original demo.

        This mode requires ``output_bits == input_bits`` so each predicted bit
        has the same meaning as an encoder bit.
        """

        if self.config.output_bits != self.config.input_bits:
            raise ValueError(
                "context-transform learning requires output_bits == input_bits"
            )
        source = self.encoder.encode_window(
            window, context=source_context, offset=offset
        )
        target = self.encoder.encode_window(
            window, context=target_context, offset=offset
        )
        self.memory.observe(source, target=target)
        return self.memory.predict(source)

    def predict_context_transform(
        self,
        window: str,
        *,
        source_context: int = 0,
        offset: int = 0,
    ) -> MemoryReadout:
        if self.config.output_bits != self.config.input_bits:
            raise ValueError(
                "context-transform prediction requires output_bits == input_bits"
            )
        source = self.encoder.encode_window(
            window, context=source_context, offset=offset
        )
        return self.memory.predict(source)

    def top_factors(self, limit: int = 10) -> list[FactorSummary]:
        return self.memory.top_factors(limit)

    def explain_factor(
        self,
        output_bit: int,
        *,
        limit: int = 10,
    ) -> list[ConceptEvidence]:
        """Rank symbol-position concepts that overlap a stable output factor."""

        if not 0 <= output_bit < self.config.output_bits:
            raise ValueError(f"output_bit must be in [0, {self.config.output_bits})")
        if limit <= 0:
            raise ValueError("limit must be positive")

        relevant = [
            cluster
            for point_index, cluster in self.memory.iter_clusters()
            if int(self.memory.output_map[point_index]) == output_bit
            and cluster.status == ClusterStatus.STABLE
        ]
        if not relevant:
            return []

        evidence: list[ConceptEvidence] = []
        for symbol, position, concept_bits in self.encoder.concepts():
            score = 0.0
            for cluster in relevant:
                overlap = len(
                    np.intersect1d(cluster.bits, concept_bits, assume_unique=True)
                )
                if overlap:
                    score += cluster.partial_hits * overlap / len(cluster.bits)
            if score:
                evidence.append(
                    ConceptEvidence(symbol=symbol, position=position, score=score)
                )
        evidence.sort(key=lambda item: (-item.score, item.position, item.symbol))
        return evidence[:limit]

    def summary(self, *, factor_limit: int = 10) -> dict[str, Any]:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "alphabet": self.alphabet,
            "memory": self.memory.stats(),
            "top_factors": [
                factor.to_dict() for factor in self.top_factors(factor_limit)
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Save without pickle so loading cannot execute arbitrary code."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        clusters = list(self.memory.iter_clusters())

        offsets = [0]
        flat_bits: list[int] = []
        flat_bit_hits: list[int] = []
        point_indices: list[int] = []
        statuses: list[int] = []
        created_at: list[int] = []
        last_seen: list[int] = []
        partial_hits: list[int] = []
        exact_hits: list[int] = []
        partial_errors: list[int] = []
        complete_errors: list[int] = []

        for point_index, cluster in clusters:
            point_indices.append(point_index)
            flat_bits.extend(int(value) for value in cluster.bits)
            flat_bit_hits.extend(int(value) for value in cluster.bit_hits)
            offsets.append(len(flat_bits))
            statuses.append(int(cluster.status))
            created_at.append(cluster.created_at)
            last_seen.append(cluster.last_seen)
            partial_hits.append(cluster.partial_hits)
            exact_hits.append(cluster.exact_hits)
            partial_errors.append(cluster.partial_errors)
            complete_errors.append(cluster.complete_errors)

        metadata = json.dumps(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "config": self.config.to_dict(),
                "alphabet": self.alphabet,
                "step": self.memory.step,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(metadata),
                codebook=self.encoder.codebook,
                receptors=self.memory.receptors,
                output_map=self.memory.output_map,
                cluster_offsets=np.asarray(offsets, dtype=np.int64),
                cluster_bits=np.asarray(flat_bits, dtype=np.int32),
                cluster_bit_hits=np.asarray(flat_bit_hits, dtype=np.int64),
                cluster_point_indices=np.asarray(point_indices, dtype=np.int32),
                cluster_statuses=np.asarray(statuses, dtype=np.int8),
                cluster_created_at=np.asarray(created_at, dtype=np.int64),
                cluster_last_seen=np.asarray(last_seen, dtype=np.int64),
                cluster_partial_hits=np.asarray(partial_hits, dtype=np.int64),
                cluster_exact_hits=np.asarray(exact_hits, dtype=np.int64),
                cluster_partial_errors=np.asarray(partial_errors, dtype=np.int64),
                cluster_complete_errors=np.asarray(complete_errors, dtype=np.int64),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TextFactorModel:
        source = Path(path)
        with np.load(source, allow_pickle=False) as state:
            metadata = json.loads(str(state["metadata"].item()))
            if metadata.get("format_version") != MODEL_FORMAT_VERSION:
                raise ValueError(
                    "unsupported model format version: "
                    f"{metadata.get('format_version')}"
                )

            config = ModelConfig.from_dict(metadata["config"])
            encoder = SparseSymbolEncoder(
                config,
                metadata["alphabet"],
                codebook=state["codebook"],
            )
            memory = CombinatorialMemory(
                config,
                receptors=state["receptors"],
                output_map=state["output_map"],
            )

            offsets = state["cluster_offsets"]
            bits = state["cluster_bits"]
            bit_hits = state["cluster_bit_hits"]
            point_indices = state["cluster_point_indices"]
            statuses = state["cluster_statuses"]
            created_at = state["cluster_created_at"]
            last_seen = state["cluster_last_seen"]
            partial_hits = state["cluster_partial_hits"]
            exact_hits = state["cluster_exact_hits"]
            partial_errors = state["cluster_partial_errors"]
            complete_errors = state["cluster_complete_errors"]

            cluster_count = len(point_indices)
            arrays = [
                statuses,
                created_at,
                last_seen,
                partial_hits,
                exact_hits,
                partial_errors,
                complete_errors,
            ]
            if len(offsets) != cluster_count + 1 or any(
                len(values) != cluster_count for values in arrays
            ):
                raise ValueError("persisted cluster arrays have inconsistent lengths")
            if int(offsets[-1]) != len(bits) or len(bits) != len(bit_hits):
                raise ValueError("persisted cluster bit arrays are inconsistent")

            for index in range(cluster_count):
                start = int(offsets[index])
                end = int(offsets[index + 1])
                cluster = Cluster(
                    bits=np.asarray(bits[start:end], dtype=np.int32).copy(),
                    bit_hits=np.asarray(bit_hits[start:end], dtype=np.int64).copy(),
                    created_at=int(created_at[index]),
                    last_seen=int(last_seen[index]),
                    status=ClusterStatus(int(statuses[index])),
                    partial_hits=int(partial_hits[index]),
                    exact_hits=int(exact_hits[index]),
                    partial_errors=int(partial_errors[index]),
                    complete_errors=int(complete_errors[index]),
                )
                memory.add_loaded_cluster(int(point_indices[index]), cluster)
            memory.step = int(metadata["step"])

        return cls(config, encoder=encoder, memory=memory)

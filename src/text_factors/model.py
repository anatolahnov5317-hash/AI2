"""High-level text factor model and safe model persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
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
from .recognition import (
    ContextView,
    RecognitionLimits,
    RecognitionResult,
    memory_encoding_id,
    recognize_views,
)

MODEL_FORMAT_VERSION = 3

# These defaults accommodate the theoretical maximum persistence payload of
# historical frequency models using the shipped 20,000-point/150-cluster
# configuration (about 1.4 GiB before compression), while still putting finite
# bounds on hostile archives. Coactivation histories can be larger; callers
# loading such models can raise all three limits explicitly.
DEFAULT_MAX_MODEL_FILE_BYTES = 2 * 1024**3
DEFAULT_MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
DEFAULT_MAX_ARRAY_BYTES = 1 * 1024**3
MAX_METADATA_BYTES = 1024**2

_LEGACY_ARRAY_DTYPES = {
    "codebook": np.dtype(np.int32),
    "receptors": np.dtype(np.int32),
    "output_map": np.dtype(np.int32),
    "cluster_offsets": np.dtype(np.int64),
    "cluster_bits": np.dtype(np.int32),
    "cluster_bit_hits": np.dtype(np.int64),
    "cluster_point_indices": np.dtype(np.int32),
    "cluster_statuses": np.dtype(np.int8),
    "cluster_created_at": np.dtype(np.int64),
    "cluster_last_seen": np.dtype(np.int64),
    "cluster_partial_hits": np.dtype(np.int64),
    "cluster_exact_hits": np.dtype(np.int64),
    "cluster_partial_errors": np.dtype(np.int64),
    "cluster_complete_errors": np.dtype(np.int64),
}
_HISTORY_ARRAY_DTYPES = {
    "cluster_history_counts": np.dtype(np.int32),
    "cluster_history_offsets": np.dtype(np.int64),
    "cluster_history_values": np.dtype(np.bool_),
}
_ARRAY_DTYPES = {**_LEGACY_ARRAY_DTYPES, **_HISTORY_ARRAY_DTYPES}
_ARCHIVE_NAMES = frozenset({"metadata", *_ARRAY_DTYPES})
_LEGACY_ARCHIVE_NAMES = frozenset({"metadata", *_LEGACY_ARRAY_DTYPES})
_TRAINING_MODES = frozenset({"untrained", "unsupervised", "supervised", "unknown"})


def _positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NPY member version {version}")
    return shape, fortran_order, np.dtype(dtype)


def _inspect_archive(
    stream: Any,
    *,
    max_file_bytes: int,
    max_uncompressed_bytes: int,
    max_array_bytes: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Validate ZIP/NPY declarations without allocating array payloads."""

    file_size = os.fstat(stream.fileno()).st_size
    if file_size <= 0:
        raise ValueError("model archive is empty")
    if file_size > max_file_bytes:
        raise ValueError(
            f"model file is {file_size} bytes; limit is {max_file_bytes} bytes"
        )

    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        expected_members = {f"{name}.npy" for name in _ARCHIVE_NAMES}
        legacy_members = {f"{name}.npy" for name in _LEGACY_ARCHIVE_NAMES}
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("model archive contains duplicate members")
        name_set = set(names)
        if name_set not in (expected_members, legacy_members):
            # Report the most useful failure against the new schema, except
            # that a complete legacy schema remains valid for v1/v2 metadata.
            missing = expected_members.difference(name_set)
            unexpected = name_set.difference(expected_members)
            if missing:
                raise ValueError(
                    "model archive is missing data: " + ", ".join(sorted(missing))
                )
            raise ValueError(
                "model archive contains unexpected data: "
                + ", ".join(sorted(unexpected))
            )
        total_size = sum(info.file_size for info in infos)
        if total_size > max_uncompressed_bytes:
            raise ValueError(
                "model uncompressed payload is "
                f"{total_size} bytes; limit is {max_uncompressed_bytes} bytes"
            )

        headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for info in infos:
            name = info.filename.removesuffix(".npy")
            member_limit = MAX_METADATA_BYTES if name == "metadata" else max_array_bytes
            if info.file_size > member_limit:
                raise ValueError(
                    f"model member {name!r} is {info.file_size} bytes; "
                    f"limit is {member_limit} bytes"
                )
            if info.flag_bits & 0x1:
                raise ValueError("encrypted model archive members are unsupported")
            with archive.open(info) as member:
                shape, fortran_order, dtype = _read_npy_header(member)
            if fortran_order:
                raise ValueError(f"model array {name!r} must use C order")
            if dtype.hasobject:
                raise ValueError(f"model array {name!r} cannot contain objects")
            element_count = prod(shape)
            payload_bytes = element_count * dtype.itemsize
            if payload_bytes > member_limit or payload_bytes > info.file_size:
                raise ValueError(f"model array {name!r} exceeds its payload limit")
            headers[name] = shape, dtype
    stream.seek(0)
    return headers


def _require_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    for key in ("format_version", "config", "alphabet", "step"):
        if key not in value:
            raise ValueError(f"metadata is missing {key!r}")
    version = value["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("metadata format_version must be an integer")
    if version not in (1, 2, MODEL_FORMAT_VERSION):
        raise ValueError(f"unsupported model format version: {version}")
    if not isinstance(value["config"], dict):
        raise ValueError("metadata config must be an object")
    if not isinstance(value["alphabet"], str):
        raise ValueError("metadata alphabet must be a string")
    step = value["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("metadata step must be a non-negative integer")
    return value


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
        # Injected memory may have been trained with targets or without them.
        # Only the validated persistence loader is allowed to restore a known
        # mode automatically.
        populated = memory is not None and any(memory.clusters)
        self._training_mode = "unknown" if populated else "untrained"
        self._context_pair: tuple[int, int] | None = None

    @property
    def alphabet(self) -> str:
        return self.encoder.alphabet

    @property
    def training_mode(self) -> str:
        """Return the mode enforced for subsequent online learning."""

        return self._training_mode

    @property
    def context_pair(self) -> tuple[int, int] | None:
        """Return the fixed supervised source/target contexts, if any."""

        return self._context_pair

    def assume_training_mode(
        self,
        mode: str,
        *,
        source_context: int | None = None,
        target_context: int | None = None,
    ) -> TextFactorModel:
        """Explicitly classify legacy or externally supplied populated memory.

        This is an opt-in escape hatch for v1 files, which did not record
        whether observations were supervised. It cannot change a mode already
        known from persisted metadata or prior learning.
        """

        if self._training_mode != "unknown":
            raise ValueError("training mode is already known and cannot be changed")
        pair = self._validate_training_choice(
            mode,
            source_context=source_context,
            target_context=target_context,
        )
        self._training_mode = mode
        self._context_pair = pair
        return self

    def _validate_training_choice(
        self,
        mode: str,
        *,
        source_context: int | None,
        target_context: int | None,
    ) -> tuple[int, int] | None:
        if mode == "unsupervised":
            if source_context is not None or target_context is not None:
                raise ValueError("unsupervised mode cannot have a context pair")
            return None
        if mode != "supervised":
            raise ValueError("mode must be 'unsupervised' or 'supervised'")
        if self.config.output_bits != self.config.input_bits:
            raise ValueError("supervised mode requires output_bits == input_bits")
        for name, value in (
            ("source_context", source_context),
            ("target_context", target_context),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be an integer")
            if not 0 <= int(value) < self.config.context_count:
                raise ValueError(f"{name} must be in [0, {self.config.context_count})")
        assert source_context is not None and target_context is not None
        return int(source_context), int(target_context)

    def _ensure_training_mode(
        self,
        mode: str,
        *,
        source_context: int | None = None,
        target_context: int | None = None,
    ) -> None:
        pair = self._validate_training_choice(
            mode,
            source_context=source_context,
            target_context=target_context,
        )
        if self._training_mode == "unknown":
            raise ValueError(
                "cannot resume learning because this populated memory's training "
                "mode is unknown; call assume_training_mode(...) explicitly"
            )
        if self._training_mode == "untrained":
            # Catch clusters inserted into a model through its public memory
            # attribute after construction.
            if any(self.memory.clusters):
                self._training_mode = "unknown"
                raise ValueError(
                    "cannot learn from externally populated memory until "
                    "assume_training_mode(...) is called"
                )
            self._training_mode = mode
            self._context_pair = pair
            return
        if self._training_mode != mode:
            raise ValueError(
                f"cannot mix {mode} learning with {self._training_mode} memory"
            )
        if mode == "supervised" and self._context_pair != pair:
            raise ValueError(
                "cannot mix supervised context operators: expected "
                f"{self._context_pair}, got {pair}"
            )

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
        self._ensure_training_mode("unsupervised")
        self.memory.observe(active)
        readout = self.memory.read(active)
        return self._result(window, offset, context, scores, readout)

    def transform_window(self, window: str, *, offset: int = 0) -> TransformResult:
        """Analyze one window without modifying the model."""

        context, scores, active = self._select_context(window, offset=offset)
        readout = self.memory.read(active)
        return self._result(window, offset, context, scores, readout)

    @property
    def recognition_encoding_id(self) -> str:
        """Local coordinate identity, preserved by an unchanged model archive."""

        digest = hashlib.sha256(memory_encoding_id(self.memory).encode())
        digest.update(self.encoder.codebook.astype("<i4", copy=False).tobytes())
        return digest.hexdigest()

    def _recognize_windows(
        self,
        windows: Iterable[tuple[str, int]],
        *,
        total_windows: int,
        observation_id: str,
        limits: RecognitionLimits | None,
        progress: Callable[[str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> RecognitionResult:
        if self.training_mode == "supervised":
            raise ValueError(
                "context recognition requires an unsupervised factor memory"
            )
        if (
            type(observation_id) is not str
            or not observation_id
            or len(observation_id) > 256
        ):
            raise ValueError(
                "observation_id must be a nonempty string of at most 256 characters"
            )

        def views() -> Iterator[ContextView]:
            for window, start in windows:
                for context in range(self.config.context_count):
                    bits = self.encoder.encode_window(
                        window, offset=start, context=context
                    )
                    sources = tuple(
                        (
                            start + local,
                            tuple(
                                int(bit)
                                for bit in self.encoder.concept_bits(
                                    character,
                                    (start + local + context) % self.config.positions,
                                )
                            ),
                        )
                        for local, character in enumerate(window)
                    )
                    yield ContextView(
                        view_id=f"w{start}:c{context}",
                        context_id=str(context),
                        bits=bits,
                        source_positions=tuple(range(start, start + len(window))),
                        observation_id=observation_id,
                        bit_sources=sources,
                    )

        return recognize_views(
            self.memory,
            views(),
            total_views=total_windows * self.config.context_count,
            limits=limits,
            encoding_id=self.recognition_encoding_id,
            progress=progress,
            cancelled=cancelled,
        )

    def recognize_window(
        self,
        window: str,
        *,
        offset: int = 0,
        observation_id: str = "input",
        limits: RecognitionLimits | None = None,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecognitionResult:
        """Return multiple located stable readouts; leave the legacy API intact."""

        self.encoder.encode_window(window, offset=offset)
        return self._recognize_windows(
            ((window, offset),),
            total_windows=1,
            observation_id=observation_id,
            limits=limits,
            progress=progress,
            cancelled=cancelled,
        )

    def recognize_text(
        self,
        text: str,
        *,
        stride: int = 1,
        observation_id: str = "input",
        limits: RecognitionLimits | None = None,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecognitionResult:
        """Scan finite fixed windows, preserving absolute source positions.

        Window boundaries are the existing explicit observation adapter, not
        learned object segmentation. Overlapping evidence may stay undetermined.
        """

        if type(text) is not str or len(text) > 16384:
            raise ValueError("text must be a string of at most 16384 characters")
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        size = self.config.frame_size
        starts = range(0, max(1, len(text) - size + 1), stride) if text else range(0)
        windows = ((text[start : start + size], start) for start in starts)
        return self._recognize_windows(
            windows,
            total_windows=len(starts),
            observation_id=observation_id,
            limits=limits,
            progress=progress,
            cancelled=cancelled,
        )

    def recognize_parts(
        self,
        parts: Sequence[str],
        *,
        observation_id: str = "input",
        limits: RecognitionLimits | None = None,
        progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> RecognitionResult:
        """Read explicitly separated sensor parts as one observation.

        Part boundaries are given by the caller. This adapter avoids treating
        every overlapping text fragment as a newly discovered object.
        """

        if (
            isinstance(parts, (str, bytes))
            or not isinstance(parts, Sequence)
            or len(parts) > 64
        ):
            raise ValueError("parts must be a sequence of at most 64 strings")
        windows: list[tuple[str, int]] = []
        start = 0
        for part in parts:
            if type(part) is not str or not part or len(part) > self.config.frame_size:
                raise ValueError("each part must be nonempty and fit frame_size")
            windows.append((part, start))
            start += len(part) + 1
        return self._recognize_windows(
            windows,
            total_windows=len(windows),
            observation_id=observation_id,
            limits=limits,
            progress=progress,
            cancelled=cancelled,
        )

    def fit_text(
        self,
        text: str,
        *,
        epochs: int = 1,
        stride: int = 1,
    ) -> TextFactorModel:
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
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
        self._ensure_training_mode(
            "supervised",
            source_context=source_context,
            target_context=target_context,
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
        if self._training_mode == "unsupervised":
            raise ValueError(
                "context-transform prediction is incompatible with unsupervised memory"
            )
        if (
            self._training_mode == "supervised"
            and self._context_pair is not None
            and source_context != self._context_pair[0]
        ):
            raise ValueError(
                "source_context does not match the learned context operator: "
                f"expected {self._context_pair[0]}, got {source_context}"
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
            "training_mode": self._training_mode,
            "context_pair": self._context_pair,
            "memory": self.memory.stats(),
            "top_factors": [
                factor.to_dict() for factor in self.top_factors(factor_limit)
            ],
        }

    def save(self, path: str | Path) -> Path:
        """Atomically save without pickle so loading cannot execute code."""

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
        history_counts: list[int] = []
        history_offsets = [0]
        flat_history: list[bool] = []

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
            history = cluster.activation_history
            if not isinstance(history, list):
                raise ValueError("cluster activation_history must be a list")
            history_count = len(history)
            if self.config.consolidation_method == "frequency":
                if history_count:
                    raise ValueError(
                        "frequency clusters cannot contain activation history"
                    )
            elif history_count != min(
                cluster.partial_hits, self.config.coactivation_history_size
            ):
                raise ValueError(
                    "coactivation cluster history count must equal "
                    "min(partial_hits, coactivation_history_size)"
                )
            history_counts.append(history_count)
            history_sums = np.zeros(len(cluster.bits), dtype=np.int64)
            for row in history:
                converted = np.asarray(row)
                if converted.dtype != np.dtype(np.bool_):
                    raise ValueError("cluster activation history rows must be boolean")
                if converted.shape != (len(cluster.bits),):
                    raise ValueError(
                        "cluster activation history rows must match cluster bits"
                    )
                history_sums += converted
                flat_history.extend(bool(value) for value in converted)
            if history_count and np.any(history_sums > cluster.bit_hits):
                raise ValueError("cluster activation history exceeds bit_hits")
            history_offsets.append(len(flat_history))

        metadata = json.dumps(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "config": self.config.to_dict(),
                "alphabet": self.alphabet,
                "step": self.memory.step,
                "training": {
                    "mode": self._training_mode,
                    "source_context": (
                        None if self._context_pair is None else self._context_pair[0]
                    ),
                    "target_context": (
                        None if self._context_pair is None else self._context_pair[1]
                    ),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        metadata_array = np.asarray(metadata)
        if metadata_array.nbytes > MAX_METADATA_BYTES:
            raise ValueError(
                f"model metadata exceeds the {MAX_METADATA_BYTES}-byte limit"
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(
                    stream,
                    metadata=metadata_array,
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
                    cluster_history_counts=np.asarray(history_counts, dtype=np.int32),
                    cluster_history_offsets=np.asarray(history_offsets, dtype=np.int64),
                    cluster_history_values=np.asarray(flat_history, dtype=np.bool_),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_file_bytes: int = DEFAULT_MAX_MODEL_FILE_BYTES,
        max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
        max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
    ) -> TextFactorModel:
        """Load a validated v1/v2/v3 archive within explicit resource bounds."""

        max_file_bytes = _positive_limit("max_file_bytes", max_file_bytes)
        max_uncompressed_bytes = _positive_limit(
            "max_uncompressed_bytes", max_uncompressed_bytes
        )
        max_array_bytes = _positive_limit("max_array_bytes", max_array_bytes)
        source = Path(path)
        try:
            return cls._load_validated(
                source,
                max_file_bytes=max_file_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_array_bytes=max_array_bytes,
            )
        except ValueError as error:
            raise ValueError(f"invalid persisted model: {error}") from error
        except (
            OSError,
            EOFError,
            KeyError,
            RuntimeError,
            TypeError,
            zipfile.BadZipFile,
        ) as error:
            raise ValueError(f"invalid persisted model: {error}") from error

    @classmethod
    def _load_validated(
        cls,
        source: Path,
        *,
        max_file_bytes: int,
        max_uncompressed_bytes: int,
        max_array_bytes: int,
    ) -> TextFactorModel:
        with source.open("rb") as stream:
            headers = _inspect_archive(
                stream,
                max_file_bytes=max_file_bytes,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_array_bytes=max_array_bytes,
            )
            metadata_shape, metadata_dtype = headers["metadata"]
            if metadata_shape != () or metadata_dtype.kind != "U":
                raise ValueError("metadata must be a scalar Unicode array")

            with np.load(stream, allow_pickle=False) as state:
                raw_metadata = state["metadata"].item()
                if not isinstance(raw_metadata, str):
                    raise ValueError("metadata must contain JSON text")
                metadata = _require_metadata(json.loads(raw_metadata))
                version = int(metadata["format_version"])
                config = ModelConfig.from_dict(dict(metadata["config"]))
                if version < 3 and config.consolidation_method == "coactivation":
                    raise ValueError(
                        "v1/v2 archives cannot restore coactivation history"
                    )

                cls._validate_array_headers(headers, config, version=version)
                array_dtypes = (
                    _ARRAY_DTYPES
                    if version == MODEL_FORMAT_VERSION
                    else _LEGACY_ARRAY_DTYPES
                )
                arrays = {name: state[name] for name in array_dtypes}
                encoder = SparseSymbolEncoder(
                    config,
                    metadata["alphabet"],
                    codebook=arrays["codebook"],
                )
                memory = CombinatorialMemory(
                    config,
                    receptors=arrays["receptors"],
                    output_map=arrays["output_map"],
                )

                offsets = arrays["cluster_offsets"]
                bits = arrays["cluster_bits"]
                bit_hits = arrays["cluster_bit_hits"]
                point_indices = arrays["cluster_point_indices"]
                statuses = arrays["cluster_statuses"]
                created_at = arrays["cluster_created_at"]
                last_seen = arrays["cluster_last_seen"]
                partial_hits = arrays["cluster_partial_hits"]
                exact_hits = arrays["cluster_exact_hits"]
                partial_errors = arrays["cluster_partial_errors"]
                complete_errors = arrays["cluster_complete_errors"]
                history_counts = (
                    arrays["cluster_history_counts"] if version == 3 else None
                )
                history_offsets = (
                    arrays["cluster_history_offsets"] if version == 3 else None
                )
                history_values = (
                    arrays["cluster_history_values"] if version == 3 else None
                )

                if offsets[0] != 0 or np.any(offsets < 0):
                    raise ValueError("cluster offsets must start at zero")
                if np.any(offsets[1:] < offsets[:-1]):
                    raise ValueError("cluster offsets must be nondecreasing")
                if int(offsets[-1]) != len(bits):
                    raise ValueError("cluster offsets do not span cluster_bits")
                if np.any(
                    (statuses < int(ClusterStatus.TEMPORARY))
                    | (statuses > int(ClusterStatus.STABLE))
                ):
                    raise ValueError("cluster_statuses contains an invalid status")
                if version == 3:
                    assert history_counts is not None
                    assert history_offsets is not None
                    assert history_values is not None
                    if history_offsets[0] != 0 or np.any(history_offsets < 0):
                        raise ValueError("cluster history offsets must start at zero")
                    if np.any(history_offsets[1:] < history_offsets[:-1]):
                        raise ValueError(
                            "cluster history offsets must be nondecreasing"
                        )
                    if int(history_offsets[-1]) != len(history_values):
                        raise ValueError(
                            "cluster history offsets do not span history values"
                        )
                    if np.any(history_counts < 0):
                        raise ValueError("cluster history counts cannot be negative")

                memory.step = int(metadata["step"])
                for index in range(len(point_indices)):
                    start = int(offsets[index])
                    end = int(offsets[index + 1])
                    activation_history: list[np.ndarray[Any, Any]] = []
                    if version == 3:
                        assert history_counts is not None
                        assert history_offsets is not None
                        assert history_values is not None
                        history_count = int(history_counts[index])
                        history_start = int(history_offsets[index])
                        history_end = int(history_offsets[index + 1])
                        expected_history_values = history_count * (end - start)
                        if history_end - history_start != expected_history_values:
                            raise ValueError(
                                "cluster history offsets do not match counts and "
                                "cluster lengths"
                            )
                        expected_history_count = min(
                            int(partial_hits[index]),
                            config.coactivation_history_size,
                        )
                        if config.consolidation_method == "frequency":
                            if history_count:
                                raise ValueError(
                                    "frequency clusters cannot contain activation "
                                    "history"
                                )
                        elif history_count != expected_history_count:
                            raise ValueError(
                                "coactivation cluster history count must equal "
                                "min(partial_hits, coactivation_history_size)"
                            )
                        if history_count:
                            rows = history_values[history_start:history_end].reshape(
                                history_count, end - start
                            )
                            activation_history = [row.copy() for row in rows]
                    cluster = Cluster(
                        bits=bits[start:end].copy(),
                        bit_hits=bit_hits[start:end].copy(),
                        created_at=int(created_at[index]),
                        last_seen=int(last_seen[index]),
                        status=ClusterStatus(int(statuses[index])),
                        partial_hits=int(partial_hits[index]),
                        exact_hits=int(exact_hits[index]),
                        partial_errors=int(partial_errors[index]),
                        complete_errors=int(complete_errors[index]),
                        activation_history=activation_history,
                    )
                    memory.add_loaded_cluster(int(point_indices[index]), cluster)

                model = cls(config, encoder=encoder, memory=memory)
                model._restore_training_metadata(metadata, len(point_indices))
                return model

    @staticmethod
    def _validate_array_headers(
        headers: Mapping[str, tuple[tuple[int, ...], np.dtype[Any]]],
        config: ModelConfig,
        *,
        version: int,
    ) -> None:
        expected_dtypes = (
            _ARRAY_DTYPES if version == MODEL_FORMAT_VERSION else _LEGACY_ARRAY_DTYPES
        )
        if set(headers) != {"metadata", *expected_dtypes}:
            raise ValueError(
                f"archive members do not match model format version {version}"
            )
        for name, expected_dtype in expected_dtypes.items():
            shape, dtype = headers[name]
            if dtype != expected_dtype:
                raise ValueError(
                    f"model array {name!r} has dtype {dtype}, expected {expected_dtype}"
                )
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            ):
                raise ValueError(f"model array {name!r} has an invalid shape")

        expected_fixed_shapes = {
            "receptors": (config.point_count, config.receptive_bits),
            "output_map": (config.point_count,),
        }
        for name, expected in expected_fixed_shapes.items():
            if headers[name][0] != expected:
                raise ValueError(
                    f"model array {name!r} has shape {headers[name][0]}, "
                    f"expected {expected}"
                )
        codebook_shape = headers["codebook"][0]
        if len(codebook_shape) != 3:
            raise ValueError("model array 'codebook' must be three-dimensional")

        point_shape = headers["cluster_point_indices"][0]
        if len(point_shape) != 1:
            raise ValueError("cluster_point_indices must be one-dimensional")
        cluster_count = point_shape[0]
        maximum_clusters = config.point_count * config.max_clusters_per_point
        if cluster_count > maximum_clusters:
            raise ValueError("persisted model exceeds its maximum cluster count")
        for name in (
            "cluster_statuses",
            "cluster_created_at",
            "cluster_last_seen",
            "cluster_partial_hits",
            "cluster_exact_hits",
            "cluster_partial_errors",
            "cluster_complete_errors",
        ):
            if headers[name][0] != (cluster_count,):
                raise ValueError("persisted cluster arrays have inconsistent shapes")
        if headers["cluster_offsets"][0] != (cluster_count + 1,):
            raise ValueError("cluster_offsets has an inconsistent shape")
        bits_shape = headers["cluster_bits"][0]
        if len(bits_shape) != 1:
            raise ValueError("cluster_bits must be one-dimensional")
        bit_count = bits_shape[0]
        if bit_count > cluster_count * config.receptive_bits:
            raise ValueError("persisted model exceeds its maximum cluster bit count")
        if headers["cluster_bit_hits"][0] != (bit_count,):
            raise ValueError("cluster_bits and cluster_bit_hits shapes differ")
        if version == MODEL_FORMAT_VERSION:
            if headers["cluster_history_counts"][0] != (cluster_count,):
                raise ValueError("cluster_history_counts has an inconsistent shape")
            if headers["cluster_history_offsets"][0] != (cluster_count + 1,):
                raise ValueError("cluster_history_offsets has an inconsistent shape")
            history_shape = headers["cluster_history_values"][0]
            if len(history_shape) != 1:
                raise ValueError("cluster_history_values must be one-dimensional")
            if history_shape[0] > bit_count * config.coactivation_history_size:
                raise ValueError(
                    "persisted model exceeds its maximum activation history size"
                )

    def _restore_training_metadata(
        self,
        metadata: Mapping[str, Any],
        cluster_count: int,
    ) -> None:
        version = int(metadata["format_version"])
        if version == 1:
            self._training_mode = "unknown" if cluster_count else "untrained"
            self._context_pair = None
            return

        training = metadata.get("training")
        if not isinstance(training, dict):
            raise ValueError(f"v{version} metadata training must be an object")
        if set(training) != {"mode", "source_context", "target_context"}:
            raise ValueError(f"v{version} metadata training fields are malformed")
        mode = training["mode"]
        if not isinstance(mode, str) or mode not in _TRAINING_MODES:
            raise ValueError(f"v{version} metadata has an invalid training mode")
        source_context = training["source_context"]
        target_context = training["target_context"]
        if mode == "supervised":
            pair = self._validate_training_choice(
                mode,
                source_context=source_context,
                target_context=target_context,
            )
        else:
            if source_context is not None or target_context is not None:
                raise ValueError(
                    f"v{version} {mode} training mode cannot have a context pair"
                )
            pair = None
        if mode == "untrained" and cluster_count:
            raise ValueError(f"populated v{version} memory cannot be marked untrained")
        if mode == "unknown" and not cluster_count:
            raise ValueError(f"empty v{version} memory cannot be marked unknown")
        self._training_mode = mode
        self._context_pair = pair

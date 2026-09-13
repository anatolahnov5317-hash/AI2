"""Source-only learned context transformations followed by one common memory.

The caller supplies trained mappings, their input coordinate name and the common
memory's coordinate name. This module discovers neither contexts nor encodings;
it has no evaluator, target, case decoder or label input. It owns no persistence:
the caller must retain both memory types and their coordinate identities.

Source positions describe the entire adapter-supplied input scope. A learned SDR
mapping does not supply exact source-bit attribution, so output ContextViews have
no bit_sources. Their located evidence must not be read as discovered boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .memory import CombinatorialMemory, MemoryReadout
from .recognition import (
    ContextView,
    RecognitionLimits,
    RecognitionResult,
    memory_encoding_id,
    recognize_views,
)
from .transforms import LearnedSDRTransform


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")


def _bits(
    value: NDArray[Any], width: int, name: str, *, allow_empty: bool
) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != (width,):
        raise ValueError(f"{name} shape must be {(width,)}")
    if not (
        np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer)
    ) or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain bool or integer 0/1 bits")
    if not allow_empty and not np.any(array):
        raise ValueError(f"{name} must have at least one active bit")
    owned = array.astype(np.bool_, copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True, slots=True)
class TransformTrace:
    context_id: str
    view_id: str
    observation_id: str
    predicted_bits: tuple[int, ...]
    source_positions: tuple[int, ...]
    transform_memory_step: int
    elapsed_seconds: float
    forwarded: bool
    origin: str = "learned_transform"
    provenance: str = "whole adapter-supplied source scope; no exact bit attribution"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    recognition: RecognitionResult
    transforms: tuple[TransformTrace, ...]
    complete: bool
    stop_reason: str | None
    input_encoding_id: str
    memory_namespace: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearnedContextPipeline:
    """Read separately trained mappings into the same factor-recognition memory.

    Mapping instances and memories remain caller-owned and must not be trained
    concurrently with recognition. The context catalogue is copied in sorted ID
    order. A global cooperative budget covers prediction and recognition; it
    cannot preempt an individual Python/NumPy call. No empty prediction fallback
    to the source or a memorized training target is allowed.
    """

    def __init__(
        self,
        memory: CombinatorialMemory,
        transforms: Mapping[str, LearnedSDRTransform],
        *,
        input_encoding_id: str,
        memory_namespace: str | None = None,
        limits: RecognitionLimits | None = None,
    ) -> None:
        if not isinstance(memory, CombinatorialMemory):
            raise ValueError("memory must be CombinatorialMemory")
        if not isinstance(transforms, Mapping) or not 1 <= len(transforms) <= 2048:
            raise ValueError("transforms must be a mapping with 1 to 2048 contexts")
        _identifier(input_encoding_id, "input_encoding_id")
        if memory_namespace is not None:
            _identifier(memory_namespace, "memory_namespace")
        if limits is not None and not isinstance(limits, RecognitionLimits):
            raise ValueError("limits must be RecognitionLimits")
        entries = tuple(transforms.items())
        widths: set[int] = set()
        for context_id, transform in entries:
            _identifier(context_id, "context_id")
            if not isinstance(transform, LearnedSDRTransform):
                raise ValueError("every transform must be LearnedSDRTransform")
            if transform.memory.config.output_bits != memory.config.input_bits:
                raise ValueError(
                    "transform output width must match common memory input"
                )
            widths.add(transform.memory.config.input_bits)
        if len(widths) != 1:
            raise ValueError("all transforms must use the same input width")
        self.memory = memory
        self.transforms = tuple(sorted(entries, key=lambda item: item[0]))
        self.input_width = widths.pop()
        self.input_encoding_id = input_encoding_id
        self.memory_namespace = memory_namespace or memory_encoding_id(memory)
        self.limits = limits or RecognitionLimits()

    def recognize(
        self,
        source: NDArray[Any],
        *,
        observation_id: str = "input",
        source_positions: tuple[int, ...] = (),
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        """Predict each context from the owned source SDR, then read common memory.

        Traces include a completed prediction even if cancellation/deadline
        prevents forwarding it. ``forwarded`` does not mean recognition finished:
        use RecognitionResult.examined_views for the completed recognition prefix.
        Every incomplete stage remains incomplete in both returned status fields.
        """
        started = perf_counter()
        _identifier(observation_id, "observation_id")
        if (
            type(source_positions) is not tuple
            or len(source_positions) > 4096
            or any(type(p) is not int or p < 0 for p in source_positions)
            or tuple(sorted(set(source_positions))) != source_positions
        ):
            raise ValueError(
                "source_positions must be sorted unique nonnegative integers"
            )
        checked = _bits(source, self.input_width, "source", allow_empty=False)
        traces: list[TransformTrace] = []
        total = len(self.transforms)

        def check_budget() -> None:
            if cancelled is not None and cancelled():
                raise InterruptedError("cancelled")
            if perf_counter() - started >= self.limits.seconds:
                raise InterruptedError("time_budget")

        def emit(message: str) -> None:
            check_budget()
            if progress is not None:
                progress(message)
            check_budget()

        def views() -> Iterator[ContextView]:
            for index, (context_id, transform) in enumerate(self.transforms):
                check_budget()
                emit(f"context-transform: {index + 1}/{total} predict {context_id}")
                owned_source = checked.copy()
                owned_source.flags.writeable = False
                prediction_started = perf_counter()
                readout = transform.predict(owned_source)
                duration = perf_counter() - prediction_started
                if not isinstance(readout, MemoryReadout):
                    raise ValueError("transform must return MemoryReadout")
                predicted = _bits(
                    readout.output,
                    self.memory.config.input_bits,
                    "predicted SDR",
                    allow_empty=True,
                )
                view_id = f"learned-{index}"
                traces.append(
                    TransformTrace(
                        context_id,
                        view_id,
                        observation_id,
                        tuple(int(bit) for bit in np.flatnonzero(predicted)),
                        source_positions,
                        transform.memory.step,
                        duration,
                        False,
                    )
                )
                check_budget()
                emit(f"context-transform: {index + 1}/{total} forward {context_id}")
                traces[-1] = replace(traces[-1], forwarded=True)
                yield ContextView(
                    view_id,
                    context_id,
                    predicted,
                    source_positions,
                    observation_id=observation_id,
                    bit_sources=(),
                )

        recognition = RecognitionResult(
            (),
            (),
            False,
            0,
            total,
            memory_step=self.memory.step,
            encoding_id=self.memory_namespace,
        )
        try:
            check_budget()
            remaining = self.limits.seconds - (perf_counter() - started)
            if remaining <= 0:
                raise InterruptedError("time_budget")
            recognition = recognize_views(
                self.memory,
                views(),
                total_views=total,
                limits=replace(self.limits, seconds=remaining),
                encoding_id=self.memory_namespace,
                progress=emit,
                cancelled=cancelled,
            )
            check_budget()
        except InterruptedError as error:
            recognition = replace(
                recognition,
                complete=False,
                stop_reason=recognition.stop_reason or str(error),
            )
        return PipelineResult(
            recognition,
            tuple(traces),
            recognition.complete,
            recognition.stop_reason,
            self.input_encoding_id,
            self.memory_namespace,
            perf_counter() - started,
        )

"""Learning a supplied SDR-to-SDR mapping through the existing factor memory.

This adapter has no encoder, rule catalogue, symbolic decoder or transformation
identifier. Selecting training pairs and deciding that they belong to one
mapping are responsibilities of the caller, not capabilities learned here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .memory import CombinatorialMemory, MemoryReadout


class LearnedSDRTransform:
    """One supervised mapping, with all evidence in CombinatorialMemory."""

    def __init__(self, config: ModelConfig) -> None:
        self.memory = CombinatorialMemory(config)

    @staticmethod
    def _checked_bits(
        active: NDArray[Any], length: int, name: str
    ) -> NDArray[np.bool_]:
        bits = np.asarray(active)
        if bits.shape != (length,):
            raise ValueError(f"{name} shape must be {(length,)}")
        if not (
            np.issubdtype(bits.dtype, np.bool_) or np.issubdtype(bits.dtype, np.integer)
        ) or np.any((bits != 0) & (bits != 1)):
            raise ValueError(f"{name} must contain bool or integer 0/1 bits")
        if not np.any(bits):
            raise ValueError(f"{name} must have at least one active bit")
        return bits.astype(np.bool_, copy=True)

    def observe(self, source: NDArray[Any], target: NDArray[Any]) -> int:
        """Observe one supplied pair; repeated epochs are not new experience."""
        checked_source = self._checked_bits(
            source, self.memory.config.input_bits, "source"
        )
        checked_target = self._checked_bits(
            target, self.memory.config.output_bits, "target"
        )
        return self.memory.observe(checked_source, target=checked_target)

    def predict(self, source: NDArray[Any]) -> MemoryReadout:
        """Read exact matches of stable local clusters without learning."""
        checked = self._checked_bits(source, self.memory.config.input_bits, "source")
        return self.memory.predict(checked)

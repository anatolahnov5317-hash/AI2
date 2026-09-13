"""Validated configuration for the text-factors model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hyperparameters for sparse encoding and associative memory.

    The defaults preserve the useful scale and thresholds of the original
    experiment while fixing its undefined unsupervised consolidation stages.
    Tests and examples intentionally use fewer points.
    """

    input_bits: int = 256
    active_bits_per_symbol: int = 8
    positions: int = 10
    frame_size: int = 5
    context_count: int = 10

    receptive_bits: int = 32
    point_count: int = 20_000
    output_bits: int = 100

    create_threshold: int = 6
    activation_threshold: int = 4
    min_active_points: int = 4
    probation_after: int = 3
    stable_after: int = 6
    prune_keep_ratio: float = 0.75
    max_clusters_per_point: int = 150

    max_complete_error_rate: float = 0.05
    max_partial_error_rate: float = 0.30
    min_error_observations: int = 5
    prediction_vote_threshold: int = 2

    seed: int = 42

    def __post_init__(self) -> None:
        positive_fields = {
            "input_bits": self.input_bits,
            "active_bits_per_symbol": self.active_bits_per_symbol,
            "positions": self.positions,
            "frame_size": self.frame_size,
            "context_count": self.context_count,
            "receptive_bits": self.receptive_bits,
            "point_count": self.point_count,
            "output_bits": self.output_bits,
            "create_threshold": self.create_threshold,
            "activation_threshold": self.activation_threshold,
            "min_active_points": self.min_active_points,
            "probation_after": self.probation_after,
            "stable_after": self.stable_after,
            "max_clusters_per_point": self.max_clusters_per_point,
            "min_error_observations": self.min_error_observations,
            "prediction_vote_threshold": self.prediction_vote_threshold,
        }
        for name, value in positive_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.input_bits > 2**31 - 1:
            raise ValueError("input_bits must fit signed 32-bit indices")
        for name, value in {
            "prune_keep_ratio": self.prune_keep_ratio,
            "max_complete_error_rate": self.max_complete_error_rate,
            "max_partial_error_rate": self.max_partial_error_rate,
        }.items():
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")

        if self.active_bits_per_symbol > self.input_bits:
            raise ValueError("active_bits_per_symbol cannot exceed input_bits")
        if self.frame_size > self.positions:
            raise ValueError("frame_size cannot exceed positions")
        if self.context_count > self.positions:
            raise ValueError("context_count cannot exceed positions")
        if self.receptive_bits > self.input_bits:
            raise ValueError("receptive_bits cannot exceed input_bits")
        if self.activation_threshold > self.create_threshold:
            raise ValueError("activation_threshold cannot exceed create_threshold")
        if self.create_threshold > self.receptive_bits:
            raise ValueError("create_threshold cannot exceed receptive_bits")
        if self.min_active_points > self.point_count:
            raise ValueError("min_active_points cannot exceed point_count")
        if self.probation_after >= self.stable_after:
            raise ValueError("probation_after must be lower than stable_after")
        if not 0.0 < self.prune_keep_ratio <= 1.0:
            raise ValueError("prune_keep_ratio must be in (0, 1]")

        for name, value in {
            "max_complete_error_rate": self.max_complete_error_rate,
            "max_partial_error_rate": self.max_partial_error_rate,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)

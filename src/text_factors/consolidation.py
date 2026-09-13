"""Bounded local Hebbian consolidation of observed binary coactivations.

This follows the floating-point interpretation of Redozubov's F_main_iter:
three default passes, per-example max normalization, decaying learning rate.
It is an uncentered single-component filter, not a general factor separator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def coactivation_weights(
    history: list[NDArray[np.bool_]], *, passes: int = 3
) -> NDArray[np.float64]:
    """Compute weights without modifying history or creating new evidence."""

    if type(passes) is not int or not 1 <= passes <= 16:
        raise ValueError("passes must be an integer in [1, 16]")
    if not isinstance(history, list) or not history or len(history) > 256:
        raise ValueError("history must contain between 1 and 256 observations")
    if any(not isinstance(row, np.ndarray) or row.ndim != 1 for row in history):
        raise ValueError("history must contain one-dimensional boolean arrays")
    width = len(history[0])
    if width == 0 or any(
        row.dtype != np.dtype(np.bool_) or row.shape != (width,) for row in history
    ):
        raise ValueError("history must contain equally sized nonempty binary rows")
    weights = np.ones(width, dtype=np.float64)
    learning_rate = 1.0 / width
    for _ in range(passes):
        for row in history:
            activity = float(np.sum(weights[row]))
            weights[row] += activity * learning_rate
            weights /= float(np.max(weights))
        learning_rate *= 0.8
    return weights

"""Small, dependency-free metrics for controlled factor experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def jaccard_similarity(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> float:
    """Jaccard similarity of two one-dimensional binary representations."""

    left = np.asarray(first, dtype=np.bool_)
    right = np.asarray(second, dtype=np.bool_)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("inputs must be one-dimensional arrays of equal shape")
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(left & right))
    return intersection / union


def bit_precision_recall(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_]
) -> tuple[float, float]:
    """Return precision and recall for active output bits."""

    predicted = np.asarray(prediction, dtype=np.bool_)
    expected = np.asarray(target, dtype=np.bool_)
    if predicted.ndim != 1 or expected.ndim != 1 or predicted.shape != expected.shape:
        raise ValueError("inputs must be one-dimensional arrays of equal shape")

    true_positive = int(np.count_nonzero(predicted & expected))
    predicted_positive = int(np.count_nonzero(predicted))
    target_positive = int(np.count_nonzero(expected))
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / target_positive if target_positive else 0.0
    return precision, recall

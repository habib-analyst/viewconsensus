"""Fit a defer threshold on held-out, labeled examples.

This is threshold fitting for a deferral policy.  It does not turn a model's
scores into formally calibrated probabilities; use dedicated calibration or
conformal methods when that stronger guarantee is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .core import ConsensusResult, consensus_from_probabilities


@dataclass(frozen=True)
class ThresholdCalibration:
    """Result of fitting an uncertainty threshold on validation examples."""

    threshold: float
    coverage: float
    accepted_accuracy: float | None
    defer_rate: float
    examples: int
    target_accuracy: float
    target_met: bool


def _result_uncertainty_and_correct(item: Any, label: Any) -> tuple[float, bool]:
    if isinstance(item, ConsensusResult):
        return item.uncertainty, item.label == label if item.label is not None else item.prediction == label
    # A threshold of 1 is above every possible winning-class standard
    # deviation for probabilities, so aggregation here never defers.
    result = consensus_from_probabilities(item, defer_threshold=1.0)
    return result.uncertainty, result.prediction == label


def fit_defer_threshold(
    validation_view_probabilities: Iterable[Any],
    labels: Sequence[Any],
    *,
    target_accuracy: float = 0.95,
) -> ThresholdCalibration:
    """Choose the broadest threshold meeting an accepted-accuracy target.

    Each validation item is either a :class:`ConsensusResult` or a
    ``[views, classes]`` probability matrix.  Labels are class indices for
    matrices, or indices/labels for results.  "Broadest" means the fitted
    policy accepts as many validation examples as possible while its accepted
    accuracy remains at or above ``target_accuracy``.  If no threshold meets
    the target, the result returns the threshold with the best accepted
    accuracy and ``target_met=False``.
    """

    items = list(validation_view_probabilities)
    if not items:
        raise ValueError("at least one validation example is required")
    if len(items) != len(labels):
        raise ValueError("validation examples and labels must have the same length")
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1")

    pairs = [
        _result_uncertainty_and_correct(item, label)
        for item, label in zip(items, labels)
    ]
    candidates = sorted({uncertainty for uncertainty, _ in pairs}) + [math.inf]

    evaluated: list[ThresholdCalibration] = []
    for threshold in candidates:
        accepted = [correct for uncertainty, correct in pairs if uncertainty <= threshold]
        coverage = len(accepted) / len(pairs)
        accuracy = float(np.mean(accepted)) if accepted else None
        evaluated.append(
            ThresholdCalibration(
                threshold=float(threshold),
                coverage=coverage,
                accepted_accuracy=accuracy,
                defer_rate=1.0 - coverage,
                examples=len(pairs),
                target_accuracy=float(target_accuracy),
                target_met=bool(accuracy is not None and accuracy >= target_accuracy),
            )
        )

    meeting = [item for item in evaluated if item.target_met]
    if meeting:
        return max(meeting, key=lambda item: (item.coverage, -item.threshold))

    def fallback_key(item: ThresholdCalibration) -> tuple[float, float, float]:
        accuracy = -1.0 if item.accepted_accuracy is None else item.accepted_accuracy
        return accuracy, item.coverage, -item.threshold

    best = max(evaluated, key=fallback_key)
    return ThresholdCalibration(**{**best.__dict__, "target_met": False})


# Short alias for callers who use calibration terminology.
calibrate_defer_threshold = fit_defer_threshold

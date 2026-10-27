"""ViewConsensus: consensus, uncertainty, and deferral for image classifiers."""

from .calibration import ThresholdCalibration, calibrate_defer_threshold, fit_defer_threshold
from .core import (
    ConsensusPanel,
    ConsensusResult,
    PredictionVerdict,
    ViewConsensus,
    ViewVote,
    consensus_from_probabilities,
    predict_with_consensus,
    validate_probabilities,
)
from .views import View, default_image_views, resolve_views

__version__ = "0.1.0"

__all__ = [
    "ConsensusPanel",
    "ConsensusResult",
    "PredictionVerdict",
    "ThresholdCalibration",
    "View",
    "ViewConsensus",
    "ViewVote",
    "calibrate_defer_threshold",
    "consensus_from_probabilities",
    "default_image_views",
    "fit_defer_threshold",
    "predict_with_consensus",
    "resolve_views",
    "validate_probabilities",
]

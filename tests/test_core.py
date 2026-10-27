import math

import numpy as np
import pytest
import torch
from PIL import Image

from viewconsensus import (
    ConsensusPanel,
    PredictionVerdict,
    consensus_from_probabilities,
    default_image_views,
    fit_defer_threshold,
)


class ConstantLogits(torch.nn.Module):
    def forward(self, value):
        logits = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
        if torch.is_tensor(value) and value.ndim == 4:
            return logits.repeat(value.shape[0], 1)
        return logits


class SequenceProbabilities:
    def __init__(self, rows):
        self.rows = list(rows)

    def __call__(self, _image):
        return self.rows.pop(0)


def test_consensus_math_mean_and_population_std():
    result = consensus_from_probabilities(
        [[0.80, 0.20], [0.60, 0.40]],
        labels=["yes", "no"],
        view_names=["front", "side"],
        defer_threshold=0.20,
    )

    assert result.prediction == 0
    assert result.label == "yes"
    assert result.consensus_probabilities == pytest.approx((0.70, 0.30))
    assert result.class_uncertainty == pytest.approx((0.10, 0.10))
    assert result.uncertainty == pytest.approx(0.10)
    assert result.agreement == pytest.approx(1.0)
    assert result.verdict is PredictionVerdict.CONFIDENT
    assert result.votes[0].confidence == pytest.approx(0.80)
    assert result.votes[1].consensus_probability == pytest.approx(0.60)


def test_defer_verdict_when_views_disagree():
    result = consensus_from_probabilities(
        [[0.95, 0.05], [0.05, 0.95]],
        labels=["cat", "dog"],
        defer_threshold=0.10,
    )

    assert result.view_probabilities[0][0] == pytest.approx(0.95)
    assert result.uncertainty == pytest.approx(0.45)
    assert result.agreement == pytest.approx(0.5)
    assert result.should_defer is True
    assert result.verdict is PredictionVerdict.UNSURE
    assert "defer to a human" in result.explain()


def test_agreeing_dummy_model_has_zero_uncertainty():
    panel = ConsensusPanel(ConstantLogits(), labels=["warm", "cool"])
    image = torch.rand(3, 16, 16)

    result = panel.predict(image)

    assert len(result.votes) == 8
    assert result.uncertainty == pytest.approx(0.0)
    assert result.agreement == 1.0
    assert result.is_confident
    assert result.label == "warm"


def test_disagreeing_dummy_model_defers_and_preserves_votes():
    model = SequenceProbabilities([[0.90, 0.10], [0.15, 0.85]])
    panel = ConsensusPanel(
        model,
        labels=["left", "right"],
        views=["original", ("changed", lambda value: value)],
        output="probabilities",
        defer_threshold=0.10,
    )

    result = panel.predict(np.zeros((4, 4), dtype=np.float32))

    assert result.uncertainty > 0.10
    assert result.should_defer
    assert [vote.name for vote in result.votes] == ["original", "changed"]
    assert [vote.prediction for vote in result.votes] == [0, 1]
    assert result.to_dict()["votes"][1]["label"] == "right"


def test_probability_validation_rejects_bad_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        consensus_from_probabilities([[0.75, 0.75]])
    with pytest.raises(ValueError, match="finite"):
        consensus_from_probabilities([[math.nan, 1.0]])


def test_default_views_cover_pil_tensor_and_numpy_inputs():
    views = default_image_views()
    assert len(views) == 8
    assert views[0].name == "original"
    assert views[-1].name == "brightness_up"

    pil_image = Image.new("RGB", (12, 10), color=(120, 40, 30))
    tensor_image = torch.rand(3, 10, 12)
    array_image = np.zeros((10, 12, 3), dtype=np.float32)

    for view in views:
        assert view(pil_image) is not None
        assert view(tensor_image) is not None
        assert view(array_image).shape in {(10, 12, 3), (12, 10, 3)}


def test_logits_temperature_changes_probability_but_not_prediction():
    panel = ConsensusPanel(
        lambda _image: [2.0, 0.0],
        views=["original"],
        temperature=2.0,
    )
    result = panel.predict(object())
    assert result.prediction == 0
    assert result.probability == pytest.approx(1 / (1 + math.exp(-1)))


def test_fit_defer_threshold_picks_broadest_target_meeting_threshold():
    validation = [
        [[0.95, 0.05], [0.95, 0.05]],  # correct, uncertainty 0
        [[0.60, 0.40], [0.40, 0.60]],  # wrong for label 1, uncertainty .10
        [[0.90, 0.10], [0.50, 0.50]],  # correct, uncertainty .20
    ]

    fitted = fit_defer_threshold(validation, [0, 1, 0], target_accuracy=0.80)

    assert fitted.threshold == pytest.approx(0.0)
    assert fitted.coverage == pytest.approx(1 / 3)
    assert fitted.accepted_accuracy == pytest.approx(1.0)
    assert fitted.target_met is True


def test_fit_defer_threshold_reports_unmet_target_honestly():
    fitted = fit_defer_threshold(
        [[[0.90, 0.10]]], [1], target_accuracy=0.95
    )
    assert fitted.target_met is False
    assert fitted.accepted_accuracy == pytest.approx(0.0)

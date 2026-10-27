"""Core consensus aggregation and the small model-wrapper API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .views import View, ViewLike, resolve_views

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class PredictionVerdict(str, Enum):
    """Human-action verdict attached to every consensus prediction."""

    CONFIDENT = "confident"
    UNSURE = "unsure"


@dataclass(frozen=True)
class ViewVote:
    """One view's vote, retained for explanation rather than discarded."""

    index: int
    name: str
    prediction: int
    label: str | None
    confidence: float
    consensus_probability: float
    probabilities: tuple[float, ...]

    @property
    def probability(self) -> float:
        """Probability this view assigned to the consensus prediction."""

        return self.consensus_probability


@dataclass(frozen=True)
class ConsensusResult:
    """The complete, explainable output of one consensus prediction."""

    prediction: int
    label: str | None
    probability: float
    consensus_probabilities: tuple[float, ...]
    uncertainty: float
    class_uncertainty: tuple[float, ...]
    verdict: PredictionVerdict
    should_defer: bool
    agreement: float
    agreement_count: int
    votes: tuple[ViewVote, ...]
    view_probabilities: tuple[tuple[float, ...], ...]
    defer_threshold: float

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Alias for :attr:`consensus_probabilities`."""

        return self.consensus_probabilities

    @property
    def is_confident(self) -> bool:
        """Whether the panel did not ask for human review."""

        return not self.should_defer

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the result."""

        return {
            "prediction": self.prediction,
            "label": self.label,
            "probability": self.probability,
            "consensus_probabilities": list(self.consensus_probabilities),
            "uncertainty": self.uncertainty,
            "class_uncertainty": list(self.class_uncertainty),
            "verdict": self.verdict.value,
            "should_defer": self.should_defer,
            "agreement": self.agreement,
            "agreement_count": self.agreement_count,
            "defer_threshold": self.defer_threshold,
            "votes": [
                {
                    "index": vote.index,
                    "name": vote.name,
                    "prediction": vote.prediction,
                    "label": vote.label,
                    "confidence": vote.confidence,
                    "consensus_probability": vote.consensus_probability,
                    "probabilities": list(vote.probabilities),
                }
                for vote in self.votes
            ],
            "view_probabilities": [list(row) for row in self.view_probabilities],
        }

    def explain(self) -> str:
        """Return a compact panel-of-views explanation."""

        label = self.label if self.label is not None else str(self.prediction)
        verdict = (
            "CONFIDENT"
            if self.verdict is PredictionVerdict.CONFIDENT
            else "UNSURE - defer to a human"
        )
        lines = [
            f"Prediction: {label} ({self.probability:.1%})",
            f"Uncertainty: +/-{self.uncertainty:.1%} across {len(self.votes)} views",
            f"Agreement: {self.agreement_count}/{len(self.votes)} views ({self.agreement:.1%})",
            f"Verdict: {verdict}",
            "Votes:",
        ]
        for vote in self.votes:
            vote_label = vote.label if vote.label is not None else str(vote.prediction)
            lines.append(f"- {vote.name}: {vote_label} ({vote.confidence:.1%})")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return the panel result as one line."""

        label = self.label if self.label is not None else str(self.prediction)
        return (
            f"{label}: {self.probability:.1%} consensus, "
            f"+/-{self.uncertainty:.1%} uncertainty, {self.verdict.value}"
        )

    __str__ = summary


def _validate_labels(
    labels: Sequence[str] | None, n_classes: int | None = None
) -> tuple[str, ...] | None:
    if labels is None:
        return None
    result = tuple(str(label) for label in labels)
    if not result:
        raise ValueError("labels cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError("labels must be unique")
    if n_classes is not None and len(result) != n_classes:
        raise ValueError(f"expected {n_classes} labels, received {len(result)}")
    return result


def validate_probabilities(view_probabilities: Any) -> np.ndarray:
    """Validate and normalize a ``[views, classes]`` probability matrix."""

    probabilities = np.asarray(view_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            "view probabilities must have shape [views, classes]; "
            f"received {probabilities.shape}"
        )
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("at least one view and one class are required")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities cannot be negative")

    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("every view must assign positive total probability")
    if np.any(np.abs(row_sums - 1.0) > 1e-3):
        raise ValueError("each view's probabilities must sum to 1 (within 0.001)")
    return probabilities / row_sums[:, None]


def consensus_from_probabilities(
    view_probabilities: Any,
    labels: Sequence[str] | None = None,
    *,
    view_names: Sequence[str] | None = None,
    defer_threshold: float = 0.10,
) -> ConsensusResult:
    """Aggregate per-view probabilities into a consensus result.

    ``uncertainty`` is the population standard deviation of the winning
    class's probability across views.  The result defers when that value is
    strictly greater than ``defer_threshold``.
    """

    if not math.isfinite(defer_threshold) or defer_threshold < 0:
        raise ValueError("defer_threshold must be a finite, non-negative number")

    probabilities = validate_probabilities(view_probabilities)
    n_views, n_classes = probabilities.shape
    checked_labels = _validate_labels(labels, n_classes)

    if view_names is None:
        names = tuple(f"view_{index + 1}" for index in range(n_views))
    else:
        names = tuple(str(name) for name in view_names)
        if len(names) != n_views:
            raise ValueError(f"expected {n_views} view names, received {len(names)}")
        if len(names) != len(set(names)):
            raise ValueError("view names must be unique")

    consensus = probabilities.mean(axis=0)
    prediction = int(np.argmax(consensus))
    class_uncertainty = probabilities.std(axis=0, ddof=0)
    uncertainty = float(class_uncertainty[prediction])
    should_defer = uncertainty > float(defer_threshold)

    votes: list[ViewVote] = []
    agreement_count = 0
    for index, row in enumerate(probabilities):
        vote_prediction = int(np.argmax(row))
        agreement_count += int(vote_prediction == prediction)
        votes.append(
            ViewVote(
                index=index,
                name=names[index],
                prediction=vote_prediction,
                label=checked_labels[vote_prediction] if checked_labels else None,
                confidence=float(row[vote_prediction]),
                consensus_probability=float(row[prediction]),
                probabilities=tuple(float(value) for value in row),
            )
        )

    return ConsensusResult(
        prediction=prediction,
        label=checked_labels[prediction] if checked_labels else None,
        probability=float(consensus[prediction]),
        consensus_probabilities=tuple(float(value) for value in consensus),
        uncertainty=uncertainty,
        class_uncertainty=tuple(float(value) for value in class_uncertainty),
        verdict=PredictionVerdict.UNSURE if should_defer else PredictionVerdict.CONFIDENT,
        should_defer=should_defer,
        agreement=agreement_count / n_views,
        agreement_count=agreement_count,
        votes=tuple(votes),
        view_probabilities=tuple(
            tuple(float(value) for value in row) for row in probabilities
        ),
        defer_threshold=float(defer_threshold),
    )


def _output_array(output: Any, *, preferred_key: str | None = None) -> np.ndarray:
    """Extract a class-score array from common model return types."""

    if isinstance(output, Mapping):
        keys = [preferred_key, "probabilities", "probs", "logits", "scores", "output"]
        for key in dict.fromkeys(key for key in keys if key is not None):
            if key in output:
                return _output_array(output[key], preferred_key=preferred_key)
        raise TypeError("mapping outputs must contain probabilities, probs, logits, scores, or output")

    if torch is not None and torch.is_tensor(output):
        array = output.detach().cpu().numpy()
    elif isinstance(output, (tuple, list)):
        if len(output) == 1:
            return _output_array(output[0], preferred_key=preferred_key)
        if output and all(
            isinstance(item, (int, float, np.integer, np.floating)) for item in output
        ):
            return np.asarray(output, dtype=np.float64)
        candidates: list[np.ndarray] = []
        for item in output:
            try:
                candidate = _output_array(item, preferred_key=preferred_key)
            except (TypeError, ValueError):
                continue
            if candidate.ndim in (1, 2):
                candidates.append(candidate)
        if not candidates:
            raise TypeError("could not find class scores in the model output")
        return candidates[0]
    else:
        array = np.asarray(output)

    array = np.asarray(array, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"model output must be [classes] or [batch, classes]; got {array.shape}")
    return array


def _softmax(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = scores / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


class ConsensusPanel:
    """Wrap a classifier and ask several transformed views to vote.

    Parameters:
        model: A callable returning class logits (by default) or
            probabilities for one input, or for a batch of tensor inputs.
        labels: Optional class labels in model-output order.
        views: ``None`` for the default eight image views, or a sequence of
            :class:`~viewconsensus.views.View`, callables, ``(name, callable)``
            pairs, and default view names.
        defer_threshold: Maximum tolerated standard deviation in the winning
            class probability across views.
        output: ``"logits"`` (default) or ``"probabilities"``.
        temperature: Positive divisor applied to logits before softmax. This
            is score scaling, not a substitute for fitting a defer threshold
            on held-out examples.
    """

    def __init__(
        self,
        model: Callable[[Any], Any],
        *,
        labels: Sequence[str] | None = None,
        views: Iterable[ViewLike] | None = None,
        defer_threshold: float = 0.10,
        output: str = "logits",
        temperature: float = 1.0,
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        if output not in {"logits", "probabilities"}:
            raise ValueError('output must be "logits" or "probabilities"')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(defer_threshold) or defer_threshold < 0:
            raise ValueError("defer_threshold must be a finite, non-negative number")

        self.model = model
        self.labels = _validate_labels(labels)
        self.views: tuple[View, ...] = resolve_views(views)
        self.defer_threshold = float(defer_threshold)
        self.output = output
        self.temperature = float(temperature)

    @property
    def view_names(self) -> tuple[str, ...]:
        """Names of the views in panel order."""

        return tuple(view.name for view in self.views)

    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        if self.output == "probabilities":
            return validate_probabilities(scores)
        return _softmax(scores, temperature=self.temperature)

    def _call_model(self, image: Any) -> np.ndarray:
        preferred = "probabilities" if self.output == "probabilities" else "logits"
        scores = _output_array(self.model(image), preferred_key=preferred)
        if scores.ndim == 2:
            if scores.shape[0] != 1:
                raise ValueError("a single-input model call returned multiple rows")
            scores = scores[0]
        return self._scores_to_probabilities(scores[None, :])[0]

    def _probabilities_for_views(self, image: Any) -> np.ndarray:
        view_images = [view(image) for view in self.views]

        # A tensor input can usually be evaluated as one batch, which keeps
        # the panel fast and gives every view the same model call path.
        if torch is not None and view_images and all(torch.is_tensor(item) for item in view_images):
            try:
                batch = torch.stack(view_images)
                context = torch.inference_mode()
                preferred = "probabilities" if self.output == "probabilities" else "logits"
                with context:
                    scores = _output_array(self.model(batch), preferred_key=preferred)
                if scores.ndim == 1:
                    scores = scores[None, :]
                if scores.shape[0] == len(view_images):
                    return self._scores_to_probabilities(scores)
            except Exception:
                # Some wrapped models only accept one image.  Fall through to
                # the per-view path instead of making that a hard failure.
                pass

        rows = [self._call_model(view_image) for view_image in view_images]
        widths = {row.shape[0] for row in rows}
        if len(widths) != 1:
            raise ValueError(f"views returned inconsistent class counts: {sorted(widths)}")
        return np.vstack(rows)

    def predict(self, image: Any) -> ConsensusResult:
        """Run every view and return the panel's explainable consensus."""

        probabilities = self._probabilities_for_views(image)
        return consensus_from_probabilities(
            probabilities,
            labels=self.labels,
            view_names=self.view_names,
            defer_threshold=self.defer_threshold,
        )

    def predict_batch(self, images: Iterable[Any]) -> tuple[ConsensusResult, ...]:
        """Predict a sequence of inputs, one panel at a time."""

        return tuple(self.predict(image) for image in images)

    __call__ = predict


# A descriptive alias matching the package name.
ViewConsensus = ConsensusPanel


def predict_with_consensus(
    model: Callable[[Any], Any], image: Any, **kwargs: Any
) -> ConsensusResult:
    """Convenience one-call wrapper around :class:`ConsensusPanel`."""

    return ConsensusPanel(model, **kwargs).predict(image)

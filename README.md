# ViewConsensus

**Do not ask an image classifier once. Ask a panel of views, show the votes, and defer when the panel disagrees.**

ViewConsensus wraps a trained image classifier and runs it on several transformed views of the same input. It returns the mean probability as the consensus, the spread across views as uncertainty, and a first-class verdict: **confident** or **unsure — defer to a human**.

I generalized this from **Consensus Diagnosis**, the panel-of-eight mechanism I built into [SonoSight](https://github.com/habib-analyst/SonoSight), a private in-browser breast-ultrasound demo. ViewConsensus makes that idea reusable for ordinary image-classification code.

> Name note: I chose **ViewConsensus** rather than ConsensusKit because ConsensusKit is already used by distributed-systems projects. The package and repository name here describe the actual mechanism: consensus across views.

## Install

ViewConsensus is not on PyPI yet. Install the GitHub version:

```bash
pip install "git+https://github.com/habib-analyst/viewconsensus.git"
```

For development:

```bash
git clone https://github.com/habib-analyst/viewconsensus.git
cd viewconsensus
python -m pip install -e ".[dev]"
python -m pytest
```

## Quickstart

```python
from PIL import Image

from viewconsensus import ConsensusPanel

panel = ConsensusPanel(
    model,  # a PyTorch model or callable returning logits
    labels=["benign", "malignant"],
    defer_threshold=0.10,
)

result = panel.predict(Image.open("scan.png"))

print(result.label)       # winning label
print(result.probability) # consensus probability for that label
print(result.uncertainty) # spread of that probability across views
print(result.verdict)     # "confident" or "unsure"

if result.should_defer:
    send_to_human_reviewer(result)
else:
    use_prediction(result)

print(result.explain())   # the full vote-by-vote panel
```

That is the whole main API. A result also exposes:

| Field | Meaning |
|---|---|
| `prediction` | Winning class index after averaging view probabilities |
| `label` | Winning label, when labels were supplied |
| `consensus_probabilities` | Mean softmax/probability for every class |
| `uncertainty` | Population standard deviation of the winning class probability across views |
| `class_uncertainty` | The same spread for every class |
| `agreement` | Fraction of individual views whose top class matches the consensus |
| `votes` | Each view's name, prediction, confidence, and full probability row |
| `verdict` | `confident` when uncertainty is at or below the threshold; otherwise `unsure` |
| `should_defer` | Boolean form of the verdict |

## What a panel looks like

The runnable example in [`examples/panel_demo.py`](examples/panel_demo.py) uses a tiny hand-configured PyTorch color model and synthetic images. It is a mechanics demo, not a benchmark. In the run recorded for this release, its warm sample produced:

```text
Prediction: warm (98.2%)
Uncertainty: +/-0.3% across 8 views
Agreement: 8/8 views (100.0%)
Verdict: CONFIDENT
Votes:
- original: warm (98.2%)
- flip_horizontal: warm (98.2%)
- flip_vertical: warm (98.2%)
- rotate_90: warm (98.2%)
- rotate_180: warm (98.2%)
- rotate_270: warm (98.2%)
- brightness_down: warm (97.4%)
- brightness_up: warm (98.7%)
```

Run it yourself:

```bash
PYTHONPATH=src python examples/panel_demo.py
```

It writes the sample images, two panel cards, and `summary.json` under `examples/output/`.

## Choose views that preserve the label

The default panel is:

1. original
2. horizontal flip
3. vertical flip
4. 90-degree rotation
5. 180-degree rotation
6. 270-degree rotation
7. brightness down
8. brightness up

That panel made sense for the original ultrasound demo. It may not make sense everywhere: rotating a digit can change its meaning, and flipping text can create a different input. Keep only label-preserving views for your domain, or supply your own:

```python
panel = ConsensusPanel(
    model,
    labels=labels,
    views=["original", "flip_horizontal", "brightness_up"],
)
```

Views can also be callables, `(name, callable)` pairs, or `View` objects. Inputs may be PyTorch tensors, NumPy arrays, or PIL images; the framework-agnostic core can aggregate any per-view probability matrix through `consensus_from_probabilities`.

## Fit the defer threshold on held-out examples

The default `defer_threshold=0.10` is a starting policy, not a universal constant. If you have labeled validation data, fit the threshold to an accepted-accuracy target:

```python
from viewconsensus import fit_defer_threshold

validation_results = [panel.predict(image) for image in validation_images]
fit = fit_defer_threshold(
    [result.view_probabilities for result in validation_results],
    validation_labels,
    target_accuracy=0.95,
)

panel = ConsensusPanel(
    model,
    labels=labels,
    defer_threshold=fit.threshold,
)

print(fit.target_met, fit.coverage, fit.accepted_accuracy)
```

This is **defer-policy fitting**, not formal probability calibration. If your application needs calibration curves, conformal coverage, or a guarantee under a stated shift model, use a dedicated framework for that job and use ViewConsensus for the inference-time panel and explanation.

## How this differs

- [`ttach`](https://github.com/qubvel/ttach) is a capable test-time augmentation wrapper, but its main output is the merged prediction tensor. ViewConsensus keeps the spread and every vote, then turns disagreement into a defer verdict.
- [TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty) is a much broader uncertainty and training framework. ViewConsensus is deliberately smaller: no retraining, no ensemble construction, and one wrapper around a model you already have.
- [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) provides rigorous conformal prediction and calibration workflows. ViewConsensus does not claim conformal guarantees; it provides augmented-view consensus and a practical, fitted deferral threshold.

Consensus can change accuracy in either direction depending on the model, data, and transformations. Validate the panel on your own held-out set instead of assuming that more views always help.

## Origin

SonoSight classified a breast-ultrasound image entirely in the browser, then ran its model on eight views of the same scan. The important product decision was not just averaging those runs. It was showing the panel, reporting disagreement, and saying when a human should review the result. ViewConsensus extracts that decision pattern into a small Python library.

## License

MIT. See [`LICENSE`](LICENSE).

"""Run ViewConsensus on synthetic images with a tiny PyTorch model.

The model's weights are set by hand to respond to red versus blue channel
means.  This is a mechanics demo, not a trained classifier or a benchmark.
It writes sample images, panel cards, and a JSON summary under
``examples/output/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from viewconsensus import ConsensusPanel, default_image_views


class TinyColorNet(torch.nn.Module):
    """A two-class global-color model implemented as one convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Conv2d(3, 2, kernel_size=96, bias=True)
        with torch.no_grad():
            self.score.weight.zero_()
            pixels = 96 * 96
            self.score.weight[0, 0] = 4.0 / pixels  # red supports class 0
            self.score.weight[0, 2] = -4.0 / pixels  # blue opposes class 0
            self.score.weight[1, 0] = -4.0 / pixels
            self.score.weight[1, 2] = 4.0 / pixels
            self.score.bias.copy_(torch.tensor([-0.10, -0.10]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.score(value).flatten(1)


class PilColorClassifier:
    """Adapt the tensor model to the single PIL images supplied by views."""

    def __init__(self) -> None:
        self.model = TinyColorNet().eval()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)[0]


def make_sample(kind: str) -> Image.Image:
    """Create a simple warm or cool synthetic sample image."""

    if kind == "warm":
        background = (185, 62, 54)
        shape = (238, 164, 92)
    elif kind == "cool":
        background = (54, 96, 184)
        shape = (105, 190, 224)
    else:  # pragma: no cover - internal guard
        raise ValueError(kind)

    image = Image.new("RGB", (96, 96), color=background)
    draw = ImageDraw.Draw(image)
    draw.ellipse((22, 18, 74, 70), fill=shape)
    draw.rectangle((10, 78, 86, 84), fill=(235, 235, 225))
    return image


def save_panel(image: Image.Image, result, path: Path) -> None:
    """Save the eight transformed thumbnails with their votes."""

    tile_width, tile_height, header_height = 160, 156, 44
    canvas = Image.new("RGB", (tile_width * 4, header_height + tile_height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        f"{result.label} | consensus {result.probability:.1%} | "
        f"uncertainty +/-{result.uncertainty:.1%} | {result.verdict.value}",
        fill=(20, 20, 20),
    )

    for index, (view, vote) in enumerate(zip(default_image_views(), result.votes)):
        column, row = index % 4, index // 4
        x, y = column * tile_width, header_height + row * tile_height
        thumbnail = view(image).convert("RGB").resize((144, 112))
        canvas.paste(thumbnail, (x + 8, y + 4))
        draw.text(
            (x + 8, y + 122),
            f"{view.name}: {vote.label} {vote.confidence:.1%}",
            fill=(20, 20, 20),
        )
    canvas.save(path)


def main() -> None:
    output_dir = Path(__file__).with_name("output")
    output_dir.mkdir(exist_ok=True)
    panel = ConsensusPanel(
        PilColorClassifier(), labels=["warm", "cool"], defer_threshold=0.10
    )

    summaries = {}
    for kind in ("warm", "cool"):
        image = make_sample(kind)
        image_path = output_dir / f"sample_{kind}.png"
        image.save(image_path)

        result = panel.predict(image)
        save_panel(image, result, output_dir / f"panel_{kind}.png")
        summaries[kind] = result.to_dict()

        print(f"\n{kind.upper()} SAMPLE")
        print(result.explain())

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote panel cards and summary to {output_dir}")


if __name__ == "__main__":
    main()

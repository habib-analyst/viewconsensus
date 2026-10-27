"""Built-in image views and view resolution helpers.

A *view* is a named transformation of one model input.  The default panel is
the eight-view panel used by Consensus Diagnosis: the original image, both
flips, three rotations, and two brightness changes.  Those transformations
are appropriate only when they preserve the label for the user's domain;
callers can pass a smaller or entirely custom set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Union

import numpy as np

try:  # Pillow is a declared dependency, but keep the error useful.
    from PIL import Image, ImageEnhance
except ImportError:  # pragma: no cover - exercised only in broken installs
    Image = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

ViewTransform = Callable[[Any], Any]


@dataclass(frozen=True)
class View:
    """A named input transformation."""

    name: str
    transform: ViewTransform

    def __call__(self, image: Any) -> Any:
        """Apply this view to *image*."""

        return self.transform(image)


def _is_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _tensor_flip(image: Any, *, horizontal: bool) -> Any:
    dim = -1 if horizontal else -2
    return torch.flip(image, dims=(dim,))


def _tensor_rotate(image: Any, quarter_turns: int) -> Any:
    return torch.rot90(image, k=quarter_turns, dims=(-2, -1))


def _tensor_brightness(image: Any, factor: float) -> Any:
    result = image * factor
    if torch.is_floating_point(image):
        # Only clamp when the source follows the usual [0, 1] image range.
        if float(image.detach().min()) >= 0.0 and float(image.detach().max()) <= 1.0:
            result = result.clamp(0.0, 1.0)
    elif torch.is_complex(image):
        raise TypeError("brightness views do not support complex tensors")
    else:
        info = torch.iinfo(image.dtype)
        result = result.clamp(info.min, info.max).to(image.dtype)
    return result


def _array_layout(image: np.ndarray) -> tuple[int, int]:
    """Return ``(height_axis, width_axis)`` for HW, HWC, and CHW arrays."""

    if image.ndim == 2:
        return 0, 1
    if image.ndim != 3:
        raise TypeError(
            "default NumPy views support HW, HWC, or CHW arrays; "
            f"received shape {image.shape}"
        )

    first, last = image.shape[0], image.shape[-1]
    first_is_channel = first in (1, 3, 4)
    last_is_channel = last in (1, 3, 4)
    if last_is_channel and (not first_is_channel or last <= first):
        return 0, 1  # HWC
    if first_is_channel:
        return 1, 2  # CHW
    raise TypeError(
        "cannot infer whether this 3D NumPy array is HWC or CHW; "
        "pass custom views for ambiguous arrays"
    )


def _array_flip(image: np.ndarray, *, horizontal: bool) -> np.ndarray:
    height_axis, width_axis = _array_layout(image)
    return np.flip(image, axis=width_axis if horizontal else height_axis).copy()


def _array_rotate(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    height_axis, width_axis = _array_layout(image)
    return np.rot90(image, k=quarter_turns, axes=(height_axis, width_axis)).copy()


def _array_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    result = image.astype(np.float64, copy=False) * factor
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        result = np.clip(result, info.min, info.max)
    elif image.size and float(np.min(image)) >= 0.0 and float(np.max(image)) <= 1.0:
        result = np.clip(result, 0.0, 1.0)
    return result.astype(image.dtype, copy=False)


def _pil_flip(image: Any, *, horizontal: bool) -> Any:
    transpose = Image.Transpose.FLIP_LEFT_RIGHT if horizontal else Image.Transpose.FLIP_TOP_BOTTOM
    return image.transpose(transpose)


def _pil_rotate(image: Any, quarter_turns: int) -> Any:
    operations = {
        1: Image.Transpose.ROTATE_90,
        2: Image.Transpose.ROTATE_180,
        3: Image.Transpose.ROTATE_270,
    }
    return image.transpose(operations[quarter_turns])


def _dispatch(
    image: Any,
    *,
    tensor: Callable[[Any], Any],
    array: Callable[[np.ndarray], np.ndarray],
    pil: Callable[[Any], Any],
) -> Any:
    if _is_tensor(image):
        return tensor(image)
    if Image is not None and isinstance(image, Image.Image):
        return pil(image)
    if isinstance(image, np.ndarray):
        return array(image)
    raise TypeError(
        "default views support PyTorch tensors, NumPy arrays, and PIL images; "
        f"received {type(image).__name__}. Pass custom views for other inputs."
    )


def _identity(image: Any) -> Any:
    return image


def _flip(image: Any, *, horizontal: bool) -> Any:
    return _dispatch(
        image,
        tensor=lambda value: _tensor_flip(value, horizontal=horizontal),
        array=lambda value: _array_flip(value, horizontal=horizontal),
        pil=lambda value: _pil_flip(value, horizontal=horizontal),
    )


def _rotate(image: Any, quarter_turns: int) -> Any:
    return _dispatch(
        image,
        tensor=lambda value: _tensor_rotate(value, quarter_turns),
        array=lambda value: _array_rotate(value, quarter_turns),
        pil=lambda value: _pil_rotate(value, quarter_turns),
    )


def _brightness(image: Any, factor: float) -> Any:
    if Image is not None and isinstance(image, Image.Image):
        return ImageEnhance.Brightness(image).enhance(factor)
    return _dispatch(
        image,
        tensor=lambda value: _tensor_brightness(value, factor),
        array=lambda value: _array_brightness(value, factor),
        pil=lambda value: ImageEnhance.Brightness(value).enhance(factor),
    )


def default_image_views() -> tuple[View, ...]:
    """Return the original eight-view Consensus Diagnosis panel."""

    return (
        View("original", _identity),
        View("flip_horizontal", lambda image: _flip(image, horizontal=True)),
        View("flip_vertical", lambda image: _flip(image, horizontal=False)),
        View("rotate_90", lambda image: _rotate(image, 1)),
        View("rotate_180", lambda image: _rotate(image, 2)),
        View("rotate_270", lambda image: _rotate(image, 3)),
        View("brightness_down", lambda image: _brightness(image, 0.90)),
        View("brightness_up", lambda image: _brightness(image, 1.10)),
    )


ViewLike = Union[View, ViewTransform, tuple[str, ViewTransform], str]


def resolve_views(views: Iterable[ViewLike] | None) -> tuple[View, ...]:
    """Normalize user-supplied views into a non-empty tuple of :class:`View`."""

    if views is None:
        return default_image_views()

    defaults: Mapping[str, View] = {view.name: view for view in default_image_views()}
    resolved: list[View] = []
    for item in views:
        if isinstance(item, View):
            resolved.append(item)
        elif isinstance(item, str):
            try:
                resolved.append(defaults[item])
            except KeyError as exc:
                choices = ", ".join(sorted(defaults))
                raise ValueError(f"unknown view {item!r}; choose from: {choices}") from exc
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            resolved.append(View(item[0], item[1]))
        elif callable(item):
            name = getattr(item, "__name__", item.__class__.__name__)
            resolved.append(View(name, item))
        else:
            raise TypeError(
                "views must contain View objects, callables, (name, callable) "
                "pairs, or default view names"
            )

    if not resolved:
        raise ValueError("at least one view is required")
    names = [view.name for view in resolved]
    if len(names) != len(set(names)):
        raise ValueError(f"view names must be unique; received {names}")
    if any(not name.strip() for name in names):
        raise ValueError("view names cannot be empty")
    return tuple(resolved)

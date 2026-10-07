from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_DISPLAY_DIMENSION = 1600
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    channels: int
    file_size_kb: float
    color_mode: str
    dtype: str
    min_intensity: float | None = None
    max_intensity: float | None = None
    mean_intensity: float | None = None
    std_intensity: float | None = None


def load_image_from_bytes(data: bytes, file_size: int = 0) -> tuple[np.ndarray, ImageInfo]:
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The selected file is not a readable image.") from exc

    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

    array = np.array(image)
    array = resize_if_large(array, MAX_DISPLAY_DIMENSION)
    return array, describe_image(array, file_size)


def resize_if_large(image: np.ndarray, max_dimension: int = MAX_DISPLAY_DIMENSION) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, new_size, interpolation=interpolation)


def describe_image(image: np.ndarray, file_size: int = 0) -> ImageInfo:
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    color_mode = "Grayscale" if channels == 1 else "RGB"
    gray = to_grayscale(image)
    return ImageInfo(
        width=width,
        height=height,
        channels=channels,
        file_size_kb=round(file_size / 1024, 2) if file_size else 0.0,
        color_mode=color_mode,
        dtype=str(image.dtype),
        min_intensity=float(np.min(gray)),
        max_intensity=float(np.max(gray)),
        mean_intensity=float(np.mean(gray)),
        std_intensity=float(np.std(gray)),
    )


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8(image)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_rgb(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def normalize_binary(mask: np.ndarray) -> np.ndarray:
    mask = ensure_uint8(mask)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(ensure_uint8(image)).save(output, format="PNG")
    return output.getvalue()


def compare_ready(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = ensure_uint8(a)
    b = ensure_uint8(b)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if a.ndim != b.ndim:
        a = to_grayscale(a)
        b = to_grayscale(b)
    return a, b


def stats_dict(image: np.ndarray) -> dict[str, Any]:
    gray = to_grayscale(image)
    return {
        "Mean": float(np.mean(gray)),
        "Median": float(np.median(gray)),
        "Std Dev": float(np.std(gray)),
        "Minimum": int(np.min(gray)),
        "Maximum": int(np.max(gray)),
        "Width": int(image.shape[1]),
        "Height": int(image.shape[0]),
        "Channels": 1 if image.ndim == 2 else int(image.shape[2]),
    }

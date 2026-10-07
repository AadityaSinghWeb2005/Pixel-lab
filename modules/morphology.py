from __future__ import annotations

import cv2
import numpy as np

from .filtering import odd_kernel
from .utils import normalize_binary, to_grayscale


KERNEL_SHAPES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS,
}


def make_kernel(shape: str = "Rectangle", size: int = 5) -> np.ndarray:
    k = odd_kernel(size)
    return cv2.getStructuringElement(KERNEL_SHAPES.get(shape, cv2.MORPH_RECT), (k, k))


def apply_morphology(
    image: np.ndarray,
    operation: str,
    kernel_shape: str = "Rectangle",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    gray = normalize_binary(to_grayscale(image))
    kernel = make_kernel(kernel_shape, kernel_size)
    iterations = max(1, int(iterations))
    op = operation.lower()
    if op == "erosion":
        return cv2.erode(gray, kernel, iterations=iterations)
    if op == "dilation":
        return cv2.dilate(gray, kernel, iterations=iterations)
    if op == "opening":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op == "closing":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if op == "morphological gradient":
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    raise ValueError(f"Unsupported morphology operation: {operation}")

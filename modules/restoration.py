from __future__ import annotations

import cv2
import numpy as np
from skimage.util import random_noise

from .filtering import bilateral_filter, gaussian_filter, mean_filter, median_filter
from .utils import ensure_uint8


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, variance: float = 0.01) -> np.ndarray:
    noisy = random_noise(ensure_uint8(image), mode="gaussian", mean=mean, var=max(variance, 0.0))
    return (noisy * 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> np.ndarray:
    noisy = random_noise(
        ensure_uint8(image),
        mode="s&p",
        amount=float(np.clip(amount, 0, 1)),
        salt_vs_pepper=float(np.clip(salt_vs_pepper, 0, 1)),
    )
    return (noisy * 255).astype(np.uint8)


def add_speckle_noise(image: np.ndarray, variance: float = 0.04) -> np.ndarray:
    noisy = random_noise(ensure_uint8(image), mode="speckle", var=max(variance, 0.0))
    return (noisy * 255).astype(np.uint8)


def restore_image(image: np.ndarray, method: str, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    method = method.lower()
    if method == "mean":
        return mean_filter(image, kernel_size)
    if method == "median":
        return median_filter(image, kernel_size)
    if method == "gaussian":
        return gaussian_filter(image, kernel_size, sigma)
    if method == "bilateral":
        return bilateral_filter(image, kernel_size, 75, 75)
    raise ValueError(f"Unsupported restoration method: {method}")


def estimate_blur(image: np.ndarray) -> float:
    gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_RGB2GRAY) if image.ndim == 3 else ensure_uint8(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

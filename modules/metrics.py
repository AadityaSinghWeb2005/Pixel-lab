from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity

from .utils import compare_ready, ensure_uint8, to_grayscale


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    reference, test = compare_ready(reference, test)
    diff = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(diff * diff))


def rmse(reference: np.ndarray, test: np.ndarray) -> float:
    return float(math.sqrt(mse(reference, test)))


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    error = mse(reference, test)
    if error == 0:
        return float("inf")
    return float(20 * math.log10(255.0 / math.sqrt(error)))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    reference, test = compare_ready(reference, test)
    reference = ensure_uint8(reference)
    test = ensure_uint8(test)
    if reference.ndim == 2:
        return float(structural_similarity(reference, test, data_range=255))
    return float(structural_similarity(reference, test, channel_axis=2, data_range=255))


def all_metrics(reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
    return {
        "MSE": mse(reference, test),
        "RMSE": rmse(reference, test),
        "PSNR": psnr(reference, test),
        "SSIM": ssim(reference, test),
    }


def histogram(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    gray = to_grayscale(image)
    values, edges = np.histogram(gray.flatten(), bins=bins, range=(0, 255))
    return values, edges

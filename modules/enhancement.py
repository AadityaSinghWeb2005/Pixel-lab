from __future__ import annotations

import cv2
import numpy as np

from .utils import ensure_uint8, to_grayscale, to_rgb


def adjust_brightness(image: np.ndarray, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(ensure_uint8(image), alpha=1.0, beta=beta)


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.convertScaleAbs(ensure_uint8(image), alpha=alpha, beta=0)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 0.01)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(ensure_uint8(image), table)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8(image)
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(to_rgb(image), cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    image = ensure_uint8(image)
    tile = max(2, int(tile_grid_size))
    engine = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile, tile))
    if image.ndim == 2:
        return engine.apply(image)
    lab = cv2.cvtColor(to_rgb(image), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = engine.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    image = ensure_uint8(image)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def grayscale(image: np.ndarray) -> np.ndarray:
    return to_grayscale(image)

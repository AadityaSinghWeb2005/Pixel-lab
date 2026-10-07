from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import prewitt

from .utils import ensure_uint8, to_grayscale


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def mean_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = odd_kernel(kernel_size)
    return cv2.blur(ensure_uint8(image), (k, k))


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    k = odd_kernel(kernel_size)
    return cv2.GaussianBlur(ensure_uint8(image), (k, k), sigmaX=float(sigma))


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = odd_kernel(kernel_size)
    return cv2.medianBlur(ensure_uint8(image), k)


def bilateral_filter(image: np.ndarray, diameter: int = 7, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    d = odd_kernel(diameter)
    return cv2.bilateralFilter(ensure_uint8(image), d, float(sigma_color), float(sigma_space))


def sobel_edges(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    gray = to_grayscale(image)
    k = odd_kernel(kernel_size)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def scharr_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def prewitt_edges(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    edges = prewitt(gray)
    return (np.clip(edges, 0, 1) * 255).astype(np.uint8)


def laplacian_edges(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    gray = to_grayscale(image)
    k = odd_kernel(kernel_size)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(lap)


def canny_edges(image: np.ndarray, low_threshold: int = 80, high_threshold: int = 160) -> np.ndarray:
    gray = to_grayscale(image)
    return cv2.Canny(gray, int(low_threshold), int(high_threshold))

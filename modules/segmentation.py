from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .filtering import canny_edges, odd_kernel
from .utils import normalize_binary, to_grayscale, to_rgb


@dataclass(frozen=True)
class RegionStat:
    label: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    perimeter: float


def global_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = to_grayscale(image)
    _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return mask


def binary_threshold(image: np.ndarray, threshold: int = 127, invert: bool = False) -> np.ndarray:
    gray = to_grayscale(image)
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, int(threshold), 255, kind)
    return mask


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c_value: int = 2) -> np.ndarray:
    gray = to_grayscale(image)
    block = max(3, odd_kernel(block_size))
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(c_value))


def canny_segmentation(image: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    return canny_edges(image, low, high)


def connected_components(image: np.ndarray) -> tuple[np.ndarray, list[RegionStat]]:
    mask = normalize_binary(image)
    total, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: list[RegionStat] = []
    for label in range(1, total):
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        regions.append(
            RegionStat(
                label=label,
                area=int(stats[label, cv2.CC_STAT_AREA]),
                centroid_x=float(centroids[label][0]),
                centroid_y=float(centroids[label][1]),
                bbox_x=int(stats[label, cv2.CC_STAT_LEFT]),
                bbox_y=int(stats[label, cv2.CC_STAT_TOP]),
                bbox_width=int(stats[label, cv2.CC_STAT_WIDTH]),
                bbox_height=int(stats[label, cv2.CC_STAT_HEIGHT]),
                perimeter=perimeter,
            )
        )
    return labels.astype(np.int32), regions


def segment_image(image: np.ndarray, method: str, **params: int | bool) -> tuple[np.ndarray, list[RegionStat]]:
    method = method.lower()
    if method == "global thresholding":
        mask = global_threshold(image, int(params.get("threshold", 127)))
    elif method == "otsu thresholding":
        mask = otsu_threshold(image)
    elif method == "adaptive thresholding":
        mask = adaptive_threshold(image, int(params.get("block_size", 11)), int(params.get("c_value", 2)))
    elif method == "binary thresholding":
        mask = binary_threshold(image, int(params.get("threshold", 127)), bool(params.get("invert", False)))
    elif method == "canny-based segmentation":
        mask = canny_segmentation(image, int(params.get("low", 80)), int(params.get("high", 160)))
    elif method == "connected components":
        mask = otsu_threshold(image)
    else:
        raise ValueError(f"Unsupported segmentation method: {method}")
    _, regions = connected_components(mask)
    return mask, regions


def overlay_regions(image: np.ndarray, mask: np.ndarray, regions: list[RegionStat]) -> np.ndarray:
    overlay = to_rgb(image).copy()
    color_mask = np.zeros_like(overlay)
    color_mask[:, :, 1] = normalize_binary(mask)
    overlay = cv2.addWeighted(overlay, 0.78, color_mask, 0.22, 0)
    for region in regions:
        if region.area < 10:
            continue
        x, y, w, h = region.bbox_x, region.bbox_y, region.bbox_width, region.bbox_height
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (37, 99, 235), 2)
        cv2.circle(overlay, (int(region.centroid_x), int(region.centroid_y)), 3, (239, 68, 68), -1)
    return overlay

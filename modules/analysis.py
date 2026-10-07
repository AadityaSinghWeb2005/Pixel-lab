from __future__ import annotations

import cv2
import numpy as np
import plotly.graph_objects as go

from .utils import stats_dict, to_grayscale, to_rgb


def grayscale_histogram_figure(image: np.ndarray) -> go.Figure:
    gray = to_grayscale(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    fig = go.Figure(data=[go.Bar(x=list(range(256)), y=hist, marker_color="#2563EB")])
    fig.update_layout(
        title="Grayscale Histogram",
        xaxis_title="Intensity",
        yaxis_title="Pixel Count",
        margin=dict(l=20, r=20, t=45, b=35),
        height=320,
    )
    return fig


def rgb_histogram_figure(image: np.ndarray) -> go.Figure:
    rgb = to_rgb(image)
    fig = go.Figure()
    colors = [("Red", "#DC2626"), ("Green", "#16A34A"), ("Blue", "#2563EB")]
    for idx, (name, color) in enumerate(colors):
        hist = cv2.calcHist([rgb], [idx], None, [256], [0, 256]).flatten()
        fig.add_trace(go.Scatter(x=list(range(256)), y=hist, mode="lines", name=name, line=dict(color=color)))
    fig.update_layout(
        title="RGB Histogram",
        xaxis_title="Intensity",
        yaxis_title="Pixel Count",
        margin=dict(l=20, r=20, t=45, b=35),
        height=320,
    )
    return fig


def intensity_distribution_figure(image: np.ndarray) -> go.Figure:
    gray = to_grayscale(image)
    fig = go.Figure(data=[go.Histogram(x=gray.flatten(), nbinsx=64, marker_color="#0F766E")])
    fig.update_layout(
        title="Intensity Distribution",
        xaxis_title="Intensity",
        yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=45, b=35),
        height=300,
    )
    return fig


def image_statistics(image: np.ndarray) -> dict[str, float | int]:
    return stats_dict(image)

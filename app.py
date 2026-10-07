from __future__ import annotations

import html
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules import analysis, enhancement, filtering, metrics, morphology, restoration, segmentation
from modules.utils import (
    describe_image,
    image_to_png_bytes,
    load_image_from_bytes,
    stats_dict,
    to_grayscale,
    to_rgb,
)


APP_TITLE = "PixelLab"
APP_SUBTITLE = "Interactive Image Processing & Analysis Laboratory"
SAMPLE_DIR = Path("assets/sample_images")

EDUCATION = {
    "Brightness": ("Change overall intensity.", "Adds or subtracts a constant from pixel values.", "Beta controls the intensity shift."),
    "Contrast": ("Separate dark and bright values.", "Multiplies each pixel by a scaling factor.", "Alpha above 1 increases contrast."),
    "Gamma": ("Correct nonlinear brightness.", "Maps intensities through a power-law curve.", "Gamma below/above 1 changes midtones."),
    "Histogram Equalization": ("Improve global contrast.", "Redistributes grayscale intensity frequencies.", "Best for low-contrast images."),
    "CLAHE": ("Improve local contrast.", "Equalizes small tiles and limits over-amplification.", "Clip limit and tile size control strength."),
    "Sharpening": ("Emphasize detail.", "Combines original image with a high-frequency component.", "Amount controls edge emphasis."),
    "Mean Filter": ("Smooth image noise.", "Averages each pixel neighborhood.", "Larger kernels blur more."),
    "Gaussian Filter": ("Smooth with distance weighting.", "Nearby pixels contribute more than distant pixels.", "Sigma controls blur spread."),
    "Median Filter": ("Remove impulse noise while preserving edges.", "Replaces pixels with neighborhood median.", "Kernel size controls neighborhood."),
    "Bilateral Filter": ("Smooth while preserving edges.", "Uses spatial and intensity similarity.", "Useful for denoising natural images."),
    "Sobel": ("Detect directional edges.", "Uses horizontal and vertical gradient filters.", "Kernel size changes gradient scale."),
    "Scharr": ("Detect strong gradients.", "A more accurate Sobel-like operator.", "Good for fine edge response."),
    "Prewitt": ("Estimate image gradients.", "Uses simple derivative masks.", "Educational baseline edge detector."),
    "Laplacian": ("Find rapid intensity changes.", "Computes second derivative response.", "Sensitive to noise."),
    "Canny": ("Detect thin, connected edges.", "Smooths, computes gradients, suppresses non-maxima, and thresholds.", "Low/high thresholds control edge linking."),
    "Segmentation": ("Separate objects or regions.", "Creates masks from intensity, edges, or components.", "Thresholds and block sizes control masks."),
    "Morphology": ("Refine binary shapes.", "Uses a structuring element to grow, shrink, open, close, or outline regions.", "Kernel shape, size, and iterations matter."),
}


def main() -> None:
    st.set_page_config(page_title="PixelLab", page_icon="PL", layout="wide")
    inject_css()
    initialize_state()

    with st.sidebar:
        st.title("PixelLab")
        st.caption(APP_SUBTITLE)
        page = st.radio(
            "Navigation",
            [
                "Home",
                "Image Upload",
                "Enhancement",
                "Filtering",
                "Noise & Restoration",
                "Edge Detection",
                "Segmentation",
                "Morphological Operations",
                "Image Analysis",
                "Quality Metrics",
                "Processing Pipeline",
                "Report",
            ],
        )
        st.divider()
        if st.session_state.image is not None:
            info = describe_image(st.session_state.image)
            st.success(f"Active image: {info.width} x {info.height}")
        else:
            st.info("No active image loaded.")
        st.caption("PixelLab is an educational and research tool. It is not a medical diagnostic system.")

    try:
        pages = {
            "Home": home_page,
            "Image Upload": upload_page,
            "Enhancement": enhancement_page,
            "Filtering": filtering_page,
            "Noise & Restoration": restoration_page,
            "Edge Detection": edge_page,
            "Segmentation": segmentation_page,
            "Morphological Operations": morphology_page,
            "Image Analysis": analysis_page,
            "Quality Metrics": metrics_page,
            "Processing Pipeline": pipeline_page,
            "Report": report_page,
        }
        pages[page]()
    except Exception as exc:
        st.error(f"PixelLab could not complete this operation: {exc}")


def initialize_state() -> None:
    defaults = {
        "image": None,
        "image_name": "No image",
        "processed": None,
        "noisy": None,
        "restored": None,
        "segmentation": None,
        "segmentation_regions": [],
        "pipeline_steps": [],
        "pipeline_result": None,
        "operations_log": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; }
        .pl-hero {
            border: 1px solid #E5E7EB;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 48%, #EFF6FF 100%);
            padding: 1.35rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .pl-hero h1 { margin: 0 0 .25rem 0; font-size: 2.3rem; letter-spacing: 0; }
        .pl-hero p { margin: 0; color: #475569; font-size: 1.05rem; }
        .pl-card {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 1rem;
            background: #FFFFFF;
            min-height: 104px;
        }
        .pl-card h3 { margin: 0 0 .35rem 0; font-size: 1rem; }
        .pl-card p { margin: 0; color: #64748B; font-size: .9rem; }
        .small-muted { color: #64748B; font-size: .9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str = APP_TITLE, subtitle: str = APP_SUBTITLE) -> None:
    st.markdown(f"<div class='pl-hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def require_image() -> np.ndarray | None:
    if st.session_state.image is None:
        st.warning("Load an image from the Image Upload page before using this section.")
        return None
    return st.session_state.image


def add_operation(name: str, params: dict, result: np.ndarray | None = None) -> None:
    st.session_state.operations_log.append({"name": name, "params": params, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    if result is not None:
        st.session_state.processed = result


def display_pair(original: np.ndarray, processed: np.ndarray, left: str = "Original Image", right: str = "Processed Image") -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.image(original, caption=left, use_container_width=True)
    with c2:
        st.image(processed, caption=right, use_container_width=True)
        key_base = "".join(ch if ch.isalnum() else "_" for ch in f"{left}_{right}").lower()
        st.download_button(
            "Download processed PNG",
            image_to_png_bytes(processed),
            file_name="pixellab_processed.png",
            mime="image/png",
            use_container_width=True,
            key=f"download_{key_base}",
        )


def education_box(key: str) -> None:
    purpose, how, params = EDUCATION.get(key, EDUCATION.get("Segmentation"))
    with st.expander("How does this work?"):
        st.markdown(f"**Algorithm:** {key}")
        st.markdown(f"**Purpose:** {purpose}")
        st.markdown(f"**How it works:** {how}")
        st.markdown(f"**Important parameters:** {params}")


def home_page() -> None:
    hero(APP_TITLE, "An interactive laboratory for understanding and experimenting with digital image processing.")
    st.subheader("Laboratory Modules")
    cards = [
        ("Image Enhancement", "Brightness, contrast, gamma, histogram equalization, CLAHE, sharpening."),
        ("Spatial Filtering", "Mean, Gaussian, median, and bilateral filters with timing."),
        ("Image Restoration", "Noise generation, denoising, and reconstruction metrics."),
        ("Edge Detection", "Sobel, Scharr, Prewitt, Laplacian, and Canny edge maps."),
        ("Segmentation", "Thresholding, adaptive methods, edge masks, and connected components."),
        ("Morphological Processing", "Erosion, dilation, opening, closing, and gradients."),
        ("Image Analysis", "Histogram dashboards and descriptive intensity statistics."),
        ("Quality Metrics", "MSE, RMSE, PSNR, and SSIM for image comparison."),
    ]
    for row_start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (title, body) in zip(cols, cards[row_start : row_start + 4]):
            col.markdown(f"<div class='pl-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)

    st.subheader("Course Concepts Covered")
    concept_cols = st.columns(2)
    with concept_cols[0]:
        st.markdown("- Histogram Processing\n- Spatial Filtering\n- Noise Reduction\n- Image Restoration")
    with concept_cols[1]:
        st.markdown("- Edge Detection\n- Image Segmentation\n- Morphological Operations\n- Image Quality Evaluation")
    st.info("Get started by opening the Image Upload page and choosing a sample image or uploading your own file.")


def upload_page() -> None:
    hero("Image Upload", "Load a file or try an included sample image.")
    mode = st.radio("Image source", ["Try Sample Image", "Upload My Image"], horizontal=True)
    image = None
    name = None
    file_size = 0
    if mode == "Upload My Image":
        upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"])
        if upload is not None:
            data = upload.getvalue()
            image, info = load_image_from_bytes(data, upload.size)
            name = upload.name
            file_size = upload.size
    else:
        samples = sorted(SAMPLE_DIR.glob("*"))
        sample_names = [path.name for path in samples if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}]
        if not sample_names:
            st.warning("No sample images were found. Run the sample generation step or upload your own image.")
            return
        selected = st.selectbox("Sample image", sample_names)
        path = SAMPLE_DIR / selected
        data = path.read_bytes()
        image, info = load_image_from_bytes(data, path.stat().st_size)
        name = selected
        file_size = path.stat().st_size

    if image is not None:
        st.session_state.image = image
        st.session_state.image_name = name or "image"
        st.session_state.processed = image
        st.session_state.operations_log.append({"name": "Loaded image", "params": {"file": st.session_state.image_name}, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        info = describe_image(image, file_size)
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.image(image, caption=st.session_state.image_name, use_container_width=True)
        with c2:
            st.subheader("Image Information")
            rows = asdict(info)
            st.dataframe(pd.DataFrame(rows.items(), columns=["Property", "Value"]), hide_index=True, use_container_width=True)


def enhancement_page() -> None:
    hero("Enhancement", "Adjust intensity, contrast, and detail.")
    image = require_image()
    if image is None:
        return
    op = st.selectbox("Enhancement method", ["Brightness", "Contrast", "Gamma", "Histogram Equalization", "CLAHE", "Sharpening"])
    result = image
    params: dict[str, float | int] = {}
    if op == "Brightness":
        beta = st.slider("Brightness shift", -120, 120, 20)
        result = enhancement.adjust_brightness(image, beta)
        params = {"beta": beta}
    elif op == "Contrast":
        alpha = st.slider("Contrast factor", 0.2, 3.0, 1.25, 0.05)
        result = enhancement.adjust_contrast(image, alpha)
        params = {"alpha": alpha}
    elif op == "Gamma":
        gamma = st.slider("Gamma", 0.2, 3.0, 1.2, 0.05)
        result = enhancement.gamma_correction(image, gamma)
        params = {"gamma": gamma}
    elif op == "Histogram Equalization":
        result = enhancement.histogram_equalization(image)
    elif op == "CLAHE":
        clip = st.slider("Clip limit", 0.5, 8.0, 2.0, 0.1)
        tile = st.slider("Tile grid size", 2, 16, 8)
        result = enhancement.clahe(image, clip, tile)
        params = {"clip_limit": clip, "tile_grid_size": tile}
    elif op == "Sharpening":
        amount = st.slider("Sharpen amount", 0.1, 3.0, 1.0, 0.1)
        result = enhancement.sharpen(image, amount)
        params = {"amount": amount}
    display_pair(image, result)
    if op in {"Histogram Equalization", "CLAHE"}:
        c1, c2 = st.columns(2)
        c1.plotly_chart(analysis.grayscale_histogram_figure(image), use_container_width=True)
        c2.plotly_chart(analysis.grayscale_histogram_figure(result), use_container_width=True)
    if st.button("Save enhancement result", use_container_width=True):
        add_operation(op, params, result)
        st.success("Enhancement result saved to the session.")
    education_box(op)


def filtering_page() -> None:
    hero("Filtering", "Smooth, denoise, and preserve structures.")
    image = require_image()
    if image is None:
        return
    method = st.selectbox("Filter type", ["Mean Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter"])
    kernel = st.slider("Kernel size", 3, 21, 5, step=2)
    sigma = st.slider("Sigma", 0.1, 8.0, 1.0, 0.1) if method == "Gaussian Filter" else 1.0
    start = time.perf_counter()
    if method == "Mean Filter":
        result = filtering.mean_filter(image, kernel)
    elif method == "Gaussian Filter":
        result = filtering.gaussian_filter(image, kernel, sigma)
    elif method == "Median Filter":
        result = filtering.median_filter(image, kernel)
    else:
        result = filtering.bilateral_filter(image, kernel, 75, 75)
    elapsed_ms = (time.perf_counter() - start) * 1000
    st.metric("Processing time", f"{elapsed_ms:.2f} ms")
    display_pair(image, result)
    if st.button("Save filtering result", use_container_width=True):
        add_operation(method, {"kernel_size": kernel, "sigma": sigma}, result)
        st.success("Filtering result saved to the session.")
    education_box(method)


def edge_page() -> None:
    hero("Edge Detection", "Explore classical gradient and edge-map algorithms.")
    image = require_image()
    if image is None:
        return
    method = st.selectbox("Edge detector", ["Sobel", "Scharr", "Prewitt", "Laplacian", "Canny"])
    params: dict[str, int] = {}
    if method in {"Sobel", "Laplacian"}:
        k = st.slider("Kernel size", 3, 7, 3, step=2)
        result = filtering.sobel_edges(image, k) if method == "Sobel" else filtering.laplacian_edges(image, k)
        params = {"kernel_size": k}
    elif method == "Canny":
        low = st.slider("Lower threshold", 0, 255, 80)
        high = st.slider("Upper threshold", 0, 255, 160)
        result = filtering.canny_edges(image, low, high)
        params = {"low_threshold": low, "high_threshold": high}
    elif method == "Scharr":
        result = filtering.scharr_edges(image)
    else:
        result = filtering.prewitt_edges(image)
    display_pair(image, result, "Original Image", "Edge Map")
    if st.button("Save edge map", use_container_width=True):
        add_operation(method, params, result)
        st.success("Edge result saved to the session.")
    education_box(method)


def restoration_page() -> None:
    hero("Noise & Restoration", "Generate noise, restore the image, and quantify reconstruction quality.")
    image = require_image()
    if image is None:
        return
    noise = st.selectbox("Noise type", ["Gaussian Noise", "Salt & Pepper Noise", "Speckle Noise"])
    if noise == "Gaussian Noise":
        variance = st.slider("Variance", 0.001, 0.1, 0.01, 0.001)
        noisy = restoration.add_gaussian_noise(image, 0.0, variance)
        noise_params = {"variance": variance}
    elif noise == "Salt & Pepper Noise":
        amount = st.slider("Amount", 0.001, 0.25, 0.05, 0.001)
        balance = st.slider("Salt vs pepper", 0.0, 1.0, 0.5, 0.05)
        noisy = restoration.add_salt_pepper_noise(image, amount, balance)
        noise_params = {"amount": amount, "salt_vs_pepper": balance}
    else:
        variance = st.slider("Variance", 0.001, 0.2, 0.04, 0.001)
        noisy = restoration.add_speckle_noise(image, variance)
        noise_params = {"variance": variance}
    st.session_state.noisy = noisy
    display_pair(image, noisy, "Original", "Noisy Image")

    st.subheader("Restoration")
    method = st.selectbox("Restoration filter", ["Mean", "Median", "Gaussian", "Bilateral"])
    kernel = st.slider("Restoration kernel size", 3, 21, 5, step=2)
    sigma = st.slider("Restoration sigma", 0.1, 8.0, 1.0, 0.1)
    restored = restoration.restore_image(noisy, method, kernel, sigma)
    st.session_state.restored = restored
    display_pair(noisy, restored, "Noisy", "Restored")
    metric_values = metrics.all_metrics(image, restored)
    metric_cards(metric_values)
    if st.button("Save restoration result", use_container_width=True):
        add_operation("Noise & Restoration", {"noise": noise, **noise_params, "method": method, "kernel_size": kernel, "sigma": sigma}, restored)
        st.success("Restored image saved to the session.")
    education_box(f"{method} Filter" if method != "Bilateral" else "Bilateral Filter")


def segmentation_page() -> None:
    hero("Segmentation", "Create masks, connected components, and region overlays.")
    image = require_image()
    if image is None:
        return
    method = st.selectbox(
        "Segmentation method",
        ["Global Thresholding", "Otsu Thresholding", "Adaptive Thresholding", "Binary Thresholding", "Canny-based Segmentation", "Connected Components"],
    )
    params: dict[str, int | bool] = {}
    if method in {"Global Thresholding", "Binary Thresholding"}:
        params["threshold"] = st.slider("Threshold", 0, 255, 127)
    if method == "Binary Thresholding":
        params["invert"] = st.checkbox("Invert binary mask")
    if method == "Adaptive Thresholding":
        params["block_size"] = st.slider("Block size", 3, 51, 11, step=2)
        params["c_value"] = st.slider("C value", -15, 15, 2)
    if method == "Canny-based Segmentation":
        params["low"] = st.slider("Lower threshold", 0, 255, 80)
        params["high"] = st.slider("Upper threshold", 0, 255, 160)
    mask, regions = segmentation.segment_image(image, method, **params)
    overlay = segmentation.overlay_regions(image, mask, regions)
    st.session_state.segmentation = mask
    st.session_state.segmentation_regions = regions
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Original Image", use_container_width=True)
    c2.image(mask, caption="Segmented Image", use_container_width=True)
    c3.image(overlay, caption="Region Overlay", use_container_width=True)
    st.download_button("Download segmentation result", image_to_png_bytes(mask), "pixellab_segmentation.png", "image/png", use_container_width=True)
    st.metric("Number of regions", len(regions))
    if regions:
        st.dataframe(pd.DataFrame([asdict(region) for region in regions]).sort_values("area", ascending=False), hide_index=True, use_container_width=True)
    if st.button("Save segmentation result", use_container_width=True):
        add_operation(method, params, overlay)
        st.success("Segmentation overlay saved to the session.")
    education_box("Segmentation")


def morphology_page() -> None:
    hero("Morphological Operations", "Refine binary shapes with structuring elements.")
    image = require_image()
    if image is None:
        return
    source = st.radio("Source image", ["Active image", "Last segmentation mask"], horizontal=True)
    base = st.session_state.segmentation if source == "Last segmentation mask" and st.session_state.segmentation is not None else image
    operation = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient"])
    shape = st.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"])
    kernel = st.slider("Kernel size", 3, 31, 5, step=2)
    iterations = st.slider("Iterations", 1, 8, 1)
    result = morphology.apply_morphology(base, operation, shape, kernel, iterations)
    display_pair(base, result, "Input", "Processed")
    if st.button("Save morphology result", use_container_width=True):
        add_operation(operation, {"kernel_shape": shape, "kernel_size": kernel, "iterations": iterations}, result)
        st.success("Morphology result saved to the session.")
    education_box("Morphology")


def analysis_page() -> None:
    hero("Image Analysis", "Histogram and intensity-statistics dashboard.")
    image = require_image()
    if image is None:
        return
    c1, c2 = st.columns([1, 1])
    c1.image(image, caption=st.session_state.image_name, use_container_width=True)
    with c2:
        st.subheader("Statistics")
        st.dataframe(pd.DataFrame(stats_dict(image).items(), columns=["Statistic", "Value"]), hide_index=True, use_container_width=True)
    st.plotly_chart(analysis.grayscale_histogram_figure(image), use_container_width=True)
    if image.ndim == 3:
        st.plotly_chart(analysis.rgb_histogram_figure(image), use_container_width=True)
    st.plotly_chart(analysis.intensity_distribution_figure(image), use_container_width=True)


def metrics_page() -> None:
    hero("Quality Metrics", "Compare a reference image against a processed image.")
    image = require_image()
    if image is None:
        return
    choices = {"Active original": image}
    for label, key in [("Last processed", "processed"), ("Noisy image", "noisy"), ("Restored image", "restored"), ("Pipeline result", "pipeline_result"), ("Segmentation mask", "segmentation")]:
        if st.session_state.get(key) is not None:
            choices[label] = st.session_state[key]
    ref_label = st.selectbox("Reference Image", list(choices.keys()), index=0)
    default_idx = min(1, len(choices) - 1)
    test_label = st.selectbox("Processed Image", list(choices.keys()), index=default_idx)
    values = metrics.all_metrics(choices[ref_label], choices[test_label])
    metric_cards(values)
    st.markdown("**MSE:** Average squared pixel error.")
    st.markdown("**PSNR:** Measures reconstruction quality using a logarithmic scale.")
    st.markdown("**SSIM:** Measures structural similarity between two images.")


def metric_cards(values: dict[str, float]) -> None:
    cols = st.columns(4)
    for col, (name, value) in zip(cols, values.items()):
        text = "inf" if value == float("inf") else f"{value:.4f}"
        col.metric(name, text)


def pipeline_page() -> None:
    hero("Processing Pipeline", "Build and run a sequence of operations.")
    image = require_image()
    if image is None:
        return
    st.subheader("Add operation")
    op = st.selectbox("Operation", ["Grayscale", "CLAHE", "Median Filter", "Gaussian Filter", "Canny", "Threshold", "Morphological Opening"])
    p1, p2, p3 = pipeline_defaults(op)
    c1, c2, c3 = st.columns(3)
    with c1:
        first = st.number_input("Parameter 1", value=float(p1), step=1.0)
    with c2:
        second = st.number_input("Parameter 2", value=float(p2), step=1.0)
    with c3:
        third = st.number_input("Parameter 3", value=float(p3), step=1.0)
    c1, c2, c3 = st.columns(3)
    if c1.button("Add operation", use_container_width=True):
        st.session_state.pipeline_steps.append({"operation": op, "params": [first, second, third]})
    if c2.button("Run pipeline", use_container_width=True):
        result = run_pipeline(image, st.session_state.pipeline_steps)
        st.session_state.pipeline_result = result
        add_operation("Processing Pipeline", {"steps": st.session_state.pipeline_steps}, result)
    if c3.button("Reset pipeline", use_container_width=True):
        st.session_state.pipeline_steps = []
        st.session_state.pipeline_result = None
    st.subheader("Pipeline")
    if st.session_state.pipeline_steps:
        labels = ["Original"] + [step["operation"] for step in st.session_state.pipeline_steps] + ["Final Result"]
        st.code("\n ↓\n".join(labels), language="text")
        st.dataframe(pd.DataFrame(st.session_state.pipeline_steps), hide_index=True, use_container_width=True)
    else:
        st.info("Add operations to build a processing sequence.")
    if st.session_state.pipeline_result is not None:
        display_pair(image, st.session_state.pipeline_result, "Original", "Pipeline Result")


def pipeline_defaults(op: str) -> tuple[int, int, int]:
    return {
        "Grayscale": (0, 0, 0),
        "CLAHE": (2, 8, 0),
        "Median Filter": (5, 0, 0),
        "Gaussian Filter": (5, 1, 0),
        "Canny": (80, 160, 0),
        "Threshold": (127, 0, 0),
        "Morphological Opening": (5, 1, 0),
    }[op]


def run_pipeline(image: np.ndarray, steps: list[dict]) -> np.ndarray:
    result = image.copy()
    for step in steps:
        op = step["operation"]
        p = step["params"]
        if op == "Grayscale":
            result = to_grayscale(result)
        elif op == "CLAHE":
            result = enhancement.clahe(result, float(p[0]), int(p[1]))
        elif op == "Median Filter":
            result = filtering.median_filter(result, int(p[0]))
        elif op == "Gaussian Filter":
            result = filtering.gaussian_filter(result, int(p[0]), float(p[1]))
        elif op == "Canny":
            result = filtering.canny_edges(result, int(p[0]), int(p[1]))
        elif op == "Threshold":
            result = segmentation.binary_threshold(result, int(p[0]))
        elif op == "Morphological Opening":
            result = morphology.apply_morphology(result, "Opening", "Rectangle", int(p[0]), int(p[1]))
    return result


def report_page() -> None:
    hero("Report", "Generate a downloadable HTML report for the current session.")
    image = require_image()
    if image is None:
        return
    if st.session_state.pipeline_result is not None:
        final = st.session_state.pipeline_result
    elif st.session_state.processed is not None:
        final = st.session_state.processed
    else:
        final = image
    metric_values = metrics.all_metrics(image, final)
    st.subheader("Report Preview")
    st.write(f"Image: {st.session_state.image_name}")
    metric_cards(metric_values)
    report = build_report_html(image, final, metric_values)
    st.download_button("Download HTML report", report.encode("utf-8"), "pixellab_report.html", "text/html", use_container_width=True)


def build_report_html(original: np.ndarray, final: np.ndarray, metric_values: dict[str, float]) -> str:
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in stats_dict(original).items())
    metrics_rows = "".join(f"<tr><td>{k}</td><td>{'inf' if v == float('inf') else f'{v:.4f}'}</td></tr>" for k, v in metric_values.items())
    operations = "".join(
        f"<li>{html.escape(item['time'])}: {html.escape(item['name'])} {html.escape(str(item['params']))}</li>"
        for item in st.session_state.operations_log
    )
    return f"""
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>PixelLab Report</title>
    <style>body{{font-family:Arial,sans-serif;margin:32px;color:#111827}}table{{border-collapse:collapse}}td,th{{border:1px solid #D1D5DB;padding:8px}}</style></head>
    <body>
    <h1>PixelLab Report</h1>
    <p>Educational image-processing session report. PixelLab is not a medical diagnostic system.</p>
    <h2>Image Information</h2><table>{rows}</table>
    <h2>Operations Performed</h2><ul>{operations or '<li>No operations recorded.</li>'}</ul>
    <h2>Quality Metrics</h2><table>{metrics_rows}</table>
    <h2>Notes</h2><p>Original and final image downloads are available inside the running app.</p>
    </body></html>
    """


if __name__ == "__main__":
    main()

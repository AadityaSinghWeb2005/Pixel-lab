import numpy as np

from modules import enhancement, filtering, metrics, morphology, restoration, segmentation
from modules.utils import describe_image, to_grayscale


def sample_rgb():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = [180, 80, 40]
    image[24:40, 24:40] = [240, 220, 200]
    return image


def test_describe_and_grayscale():
    image = sample_rgb()
    info = describe_image(image, 1024)
    assert info.width == 64
    assert info.height == 64
    assert info.channels == 3
    gray = to_grayscale(image)
    assert gray.shape == (64, 64)


def test_enhancement_functions():
    image = sample_rgb()
    assert enhancement.adjust_brightness(image, 10).shape == image.shape
    assert enhancement.adjust_contrast(image, 1.2).shape == image.shape
    assert enhancement.gamma_correction(image, 1.1).shape == image.shape
    assert enhancement.histogram_equalization(image).shape == image.shape
    assert enhancement.clahe(image, 2.0, 8).shape == image.shape
    assert enhancement.sharpen(image, 1.0).shape == image.shape


def test_filters_and_edges():
    image = sample_rgb()
    assert filtering.mean_filter(image, 3).shape == image.shape
    assert filtering.gaussian_filter(image, 3, 1.0).shape == image.shape
    assert filtering.median_filter(image, 3).shape == image.shape
    assert filtering.bilateral_filter(image, 5).shape == image.shape
    assert filtering.sobel_edges(image).shape == image.shape[:2]
    assert filtering.scharr_edges(image).shape == image.shape[:2]
    assert filtering.prewitt_edges(image).shape == image.shape[:2]
    assert filtering.laplacian_edges(image).shape == image.shape[:2]
    assert filtering.canny_edges(image).shape == image.shape[:2]


def test_noise_restoration_and_metrics():
    image = sample_rgb()
    noisy = restoration.add_salt_pepper_noise(image, 0.05, 0.5)
    restored = restoration.restore_image(noisy, "median", 3, 1.0)
    assert noisy.shape == image.shape
    assert restored.shape == image.shape
    values = metrics.all_metrics(image, restored)
    assert set(values) == {"MSE", "RMSE", "PSNR", "SSIM"}
    assert values["MSE"] >= 0
    assert values["SSIM"] <= 1


def test_segmentation_and_morphology():
    image = sample_rgb()
    mask, regions = segmentation.segment_image(image, "Otsu Thresholding")
    assert mask.shape == image.shape[:2]
    assert isinstance(regions, list)
    overlay = segmentation.overlay_regions(image, mask, regions)
    assert overlay.shape == image.shape
    opened = morphology.apply_morphology(mask, "Opening", "Rectangle", 3, 1)
    assert opened.shape == mask.shape

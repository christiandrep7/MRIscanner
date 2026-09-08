from __future__ import annotations

import numpy as np
from PIL import Image

from src.ood_detection import assess_reliability


def _gray_brain_like_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """A plausible clean single-slice scan: grayscale, roughly square, black
    background with a soft gray blob in the middle, empty corners."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    radius = min(w, h) * 0.3
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    gray = np.clip(180 - dist, 0, 180).astype(np.uint8)
    rng = np.random.default_rng(0)
    gray = np.clip(gray.astype(np.int16) + rng.integers(-5, 5, size=gray.shape), 0, 255).astype(np.uint8)
    arr = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_clean_grayscale_image_is_not_flagged():
    img = _gray_brain_like_image()
    result = assess_reliability(img)
    assert result.is_atypical is False
    assert result.reasons == []


def test_wide_aspect_ratio_image_is_flagged():
    img = _gray_brain_like_image((224, 224)).resize((450, 224))
    result = assess_reliability(img)
    assert result.is_atypical is True
    assert any("proportions" in r for r in result.reasons)


def test_heavily_tinted_image_is_flagged():
    base = _gray_brain_like_image()
    arr = np.array(base).astype(np.float32)
    arr[..., 0] = np.clip(arr[..., 0] + 80, 0, 255)  # push red channel far from green/blue
    tinted = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    result = assess_reliability(tinted)
    assert result.is_atypical is True
    assert any("colorful" in r for r in result.reasons)


def test_busy_corners_image_is_flagged():
    base = _gray_brain_like_image()
    arr = np.array(base)
    rng = np.random.default_rng(1)
    corner_size = 24
    for y0, x0 in [(0, 0), (0, arr.shape[1] - corner_size), (arr.shape[0] - corner_size, 0)]:
        noise = rng.integers(0, 255, size=(corner_size, corner_size, 3), dtype=np.uint8)
        arr[y0 : y0 + corner_size, x0 : x0 + corner_size] = noise
    result = assess_reliability(Image.fromarray(arr, mode="RGB"))
    assert result.is_atypical is True
    assert any("corners" in r for r in result.reasons)


def test_assessment_always_reports_all_three_metrics():
    result = assess_reliability(_gray_brain_like_image())
    assert set(result.metrics.keys()) == {"aspect_ratio", "color_deviation", "corner_edge_density"}


def test_assess_reliability_accepts_non_rgb_image():
    grayscale_mode_img = _gray_brain_like_image().convert("L")
    result = assess_reliability(grayscale_mode_img)  # must not raise
    assert result.is_atypical is False

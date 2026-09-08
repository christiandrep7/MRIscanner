"""Flags images that don't look like a clean, single MRI slice -- real-world
photos/screenshots found online (multi-panel PACS composites, burned-in scan
parameter text, tinted/scanned-film photos) that this project's models were
never trained to handle and answer confidently wrong on. Two separate training
attempts this session (augmentation targeting exactly this) didn't close that
gap -- this is the fallback: don't try to make the classifier invariant to
every real-world image format, just tell the user when an input doesn't look
like what the model actually learned from, so a confident wrong answer isn't
mistaken for a reliable one.

Thresholds below are calibrated, not guessed: computed from 150 real training
images (see kb explore/lesson from this session), using the 95th/99th
percentile of each metric as the "elevated"/"very elevated" cutoff. Verified
against the 3 real online images that exposed the original problem:
  - pituitary (tinted, ruler+label): color_deviation ~15.0 vs a training
    sample max of 3.74 -- ~4x over, an unambiguous outlier.
  - meningioma (4-panel PACS composite): corner_edge_density ~26.6 vs a
    training p99 of 18.0.
  - glioma (2-panel composite): aspect_ratio and corner_edge_density both
    modestly over their p95 cutoffs -- individually weak, together enough to
    flag under the "2 of 3 elevated" rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# From 150 images sampled across all 4 classes of data/Training (seed=0).
ASPECT_RATIO_P95 = 1.71
ASPECT_RATIO_P99 = 1.74
COLOR_DEVIATION_P95 = 0.5  # training p95 was ~0 (near-perfectly grayscale); small margin above noise
COLOR_DEVIATION_P99 = 1.2
CORNER_EDGE_DENSITY_P95 = 3.9
CORNER_EDGE_DENSITY_P99 = 18.0


@dataclass
class ReliabilityAssessment:
    is_atypical: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _aspect_ratio(img: Image.Image) -> float:
    w, h = img.size
    return max(w, h) / min(w, h)


def _color_deviation(arr: np.ndarray) -> float:
    """Real MRI images are essentially grayscale (R=G=B per pixel) even when
    stored as RGB. Tinted photos, colored annotation arrows, and colorized
    overlays push channels apart -- this is the single strongest signal found
    among the 3 real failing images (the tinted pituitary case)."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b))).mean())


def _corner_edge_density(arr: np.ndarray) -> float:
    """A real single-slice scan has empty (near-zero-gradient) background in
    the corners. Burned-in scan-parameter text and scale rulers -- exactly
    what broke the meningioma and pituitary test cases -- live in the corners
    and sharply raise local edge density there."""
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx**2 + gy**2)
    h, w = gray.shape
    ch, cw = max(1, int(h * 0.1)), max(1, int(w * 0.1))
    corners = [edge[:ch, :cw], edge[:ch, -cw:], edge[-ch:, :cw], edge[-ch:, -cw:]]
    return float(np.mean([c.mean() for c in corners]))


def assess_reliability(img: Image.Image) -> ReliabilityAssessment:
    """Runs all 3 heuristics and combines them: flagged if any single metric
    is very elevated (past its p99), or at least 2 of 3 are elevated (past
    their p95) together -- catches both a single strong outlier (the tinted
    pituitary case) and several individually-weak signals stacking up (the
    2-panel glioma composite, where no single metric alone was conclusive)."""
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)

    aspect = _aspect_ratio(img)
    color_dev = _color_deviation(arr)
    corner_edge = _corner_edge_density(arr)

    metrics = {"aspect_ratio": aspect, "color_deviation": color_dev, "corner_edge_density": corner_edge}
    reasons: list[str] = []
    elevated_count = 0

    if aspect > ASPECT_RATIO_P99:
        reasons.append("unusually wide/tall proportions, possibly a multi-panel image")
    elif aspect > ASPECT_RATIO_P95:
        elevated_count += 1

    if color_dev > COLOR_DEVIATION_P99:
        reasons.append("unusually colorful for an MRI scan (real scans are essentially grayscale)")
    elif color_dev > COLOR_DEVIATION_P95:
        elevated_count += 1

    if corner_edge > CORNER_EDGE_DENSITY_P99:
        reasons.append("busy detail in the image corners, possibly burned-in text or a scale ruler")
    elif corner_edge > CORNER_EDGE_DENSITY_P95:
        elevated_count += 1

    if elevated_count >= 2 and not reasons:
        reasons.append("several aspects of this image look atypical for a single clean MRI slice")

    return ReliabilityAssessment(is_atypical=bool(reasons), reasons=reasons, metrics=metrics)

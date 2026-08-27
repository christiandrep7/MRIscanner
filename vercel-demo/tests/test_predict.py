"""Lightweight tests for the Vercel serverless function's inference logic.
No torch here on purpose -- this mirrors what's actually deployed (onnxruntime
+ numpy + Pillow only), so it tests the real deployed code path, not a proxy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
import predict as p  # noqa: E402


def _synthetic_image(size: int = 96) -> Image.Image:
    arr = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_model_file_exists():
    assert Path(p._MODEL_PATH).is_file()


def test_preprocess_shape_and_dtype():
    x = p.preprocess(_synthetic_image(64))
    assert x.shape == (1, 3, p.IMAGE_SIZE, p.IMAGE_SIZE)
    assert x.dtype == np.float32


def test_preprocess_handles_grayscale_input():
    gray = Image.fromarray(np.zeros((50, 50), dtype=np.uint8), mode="L")
    x = p.preprocess(gray)
    assert x.shape == (1, 3, p.IMAGE_SIZE, p.IMAGE_SIZE)


def test_softmax_sums_to_one_and_matches_argmax():
    logits = np.array([2.0, 1.0, 0.1, -1.0], dtype=np.float32)
    probs = p.softmax(logits)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.argmax(probs) == np.argmax(logits)


def test_predict_end_to_end_returns_valid_result():
    result = p.predict(_synthetic_image())
    assert result["label"] in p.CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == set(p.CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["probabilities"][result["label"]] == result["confidence"]

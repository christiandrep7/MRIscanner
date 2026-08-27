from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.gradcam_utils import default_target_layer, generate_gradcam_overlay, preprocess_image, save_overlay
from src.model import ARCHITECTURES, build_model, build_resnet50_model, get_device


def _dummy_pil_image(size: int = 32) -> Image.Image:
    arr = np.random.default_rng(0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_preprocess_image_shape_and_normalization():
    img = _dummy_pil_image(48)
    tensor = preprocess_image(img, image_size=32)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == torch.float32
    # ImageNet-normalized tensors are typically within roughly [-3, 3]
    assert tensor.min() > -5 and tensor.max() < 5


def test_generate_gradcam_overlay_shape():
    device = get_device()
    model = build_resnet50_model(num_classes=4, pretrained=False).to(device)
    model.eval()
    img = _dummy_pil_image(32)

    overlay = generate_gradcam_overlay(model, img, device=device, image_size=32)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_generate_gradcam_overlay_works_for_every_architecture(architecture: str):
    """Regression test: gradcam_utils used to hardcode `model.layer4[-1]`, which
    only exists on ResNet -- EfficientNet-B0 and VGG16 crashed with AttributeError
    the moment Grad-CAM ran on them (caught via the Gradio UI, not by earlier
    per-module tests, since benchmark.py never calls Grad-CAM at all)."""
    device = get_device()
    model = build_model(architecture, num_classes=4, pretrained=False).to(device)
    model.eval()
    img = _dummy_pil_image(32)

    overlay = generate_gradcam_overlay(model, img, device=device, image_size=32)

    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_default_target_layer_resnet_uses_last_block():
    model = build_model("resnet50", num_classes=4, pretrained=False)
    assert default_target_layer(model) is model.layer4[-1]


@pytest.mark.parametrize("architecture", ["efficientnet_b0", "vgg16"])
def test_default_target_layer_others_use_last_conv2d(architecture: str):
    model = build_model(architecture, num_classes=4, pretrained=False)
    target = default_target_layer(model)
    assert isinstance(target, nn.Conv2d)


def test_save_overlay_writes_readable_image(tmp_path: Path):
    overlay = np.zeros((16, 16, 3), dtype=np.uint8)
    overlay[:, :, 0] = 255
    out_path = tmp_path / "nested" / "overlay.jpg"

    save_overlay(overlay, out_path)

    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.size == (16, 16)

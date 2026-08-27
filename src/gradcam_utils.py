from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD


def default_target_layer(model: nn.Module) -> nn.Module:
    """Picks the Grad-CAM target layer for whichever architecture `model` is.

    ResNet: the whole last residual block (layer4[-1]) -- the conventional
    Grad-CAM target, combining that block's skip connection.
    EfficientNet / VGG (and anything else without `layer4`): the last Conv2d
    layer in the network, found generically rather than hardcoded by index so
    this doesn't silently break if torchvision's internal layer count changes.
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError(f"No known Grad-CAM target layer for {type(model).__name__}")


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    xform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return xform(image).unsqueeze(0)


def generate_gradcam_overlay(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    image_size: int = 224,
) -> np.ndarray:
    model.eval()
    input_tensor = preprocess_image(image, image_size=image_size).to(device)
    target_layers = [default_target_layer(model)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        # targets=None -> auto-picks each sample's top predicted class (matches
        # pre-1.5 pytorch-grad-cam default; newer versions require it explicitly).
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    rgb_img = np.array(image.convert("RGB").resize((image_size, image_size))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image


def save_overlay(overlay_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

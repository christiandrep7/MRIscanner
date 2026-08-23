from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD


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
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(image.convert("RGB").resize((image_size, image_size))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image


def save_overlay(overlay_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

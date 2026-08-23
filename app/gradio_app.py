from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.checkpoint_io import load_training_checkpoint
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.gradcam_utils import generate_gradcam_overlay
from src.model import build_resnet50_model, get_device
from torchvision import transforms


CHECKPOINT_PATH = Path("checkpoints") / "best_model.pth"


def load_artifacts():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Train model first.")

    device = get_device()
    payload = load_training_checkpoint(CHECKPOINT_PATH, map_location=device)
    class_to_idx = payload["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_resnet50_model(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, idx_to_class


def preprocess(image: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


model, device, idx_to_class = load_artifacts()


def predict(image: Image.Image):
    if image is None:
        return "No image received", None

    x = preprocess(image).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx].item())

    overlay = generate_gradcam_overlay(model, image.convert("RGB"), device=device, image_size=224)
    return f"{pred_label} ({confidence:.2%})", overlay


if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload MRI image"),
        outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Grad-CAM heatmap")],
        title="Brain Tumor MRI Classifier",
        description="Upload an MRI image to get class prediction and Grad-CAM explanation.",
    )
    demo.launch()

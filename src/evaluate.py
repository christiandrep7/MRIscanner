from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import get_eval_transforms
from src.gradcam_utils import generate_gradcam_overlay, save_overlay
from src.model import build_model, get_device


def load_model_checkpoint(checkpoint_path: Path, num_classes: int, device: torch.device):
    # No ImageNet download: checkpoint contains full trained weights. The checkpoint
    # is self-describing (src.train saves an "architecture" field); old-format
    # checkpoints without it default to resnet50, the only architecture that existed
    # before checkpoints started recording it.
    payload = load_training_checkpoint(checkpoint_path, map_location=device)
    architecture = payload.get("architecture", "resnet50")
    model = build_model(architecture, num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader: DataLoader, device: torch.device):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: Path):
    # `labels` pins the matrix to every known class, even ones absent from this
    # particular y_true/y_pred (e.g. a small "secret" holdout that doesn't cover
    # all 4 classes) -- otherwise the matrix shrinks and silently misaligns
    # against `class_names` in the heatmap's tick labels.
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_report(y_true, y_pred, class_names, output_path: Path):
    # Same reasoning as plot_confusion_matrix: without `labels`, sklearn raises
    # ValueError as soon as y_true/y_pred cover fewer classes than class_names.
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    report["accuracy"] = accuracy_score(y_true, y_pred)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def generate_gradcam_samples(
    model,
    test_dataset_raw: datasets.ImageFolder,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    picked = 0
    for img_path, label_idx in test_dataset_raw.samples:
        image = Image.open(img_path).convert("RGB")
        overlay = generate_gradcam_overlay(model, image, device=device, image_size=224)
        class_name = class_names[label_idx]
        out_path = output_dir / f"gradcam_{picked+1}_{class_name}.jpg"
        save_overlay(overlay, out_path)
        picked += 1
        if picked >= num_samples:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRI classifier.")
    parser.add_argument("--checkpoint", default="checkpoints/resnet50_best_model.pth")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = DataConfig(batch_size=args.batch_size)
    test_dir = cfg.data_root / cfg.test_dir_name
    test_dataset_eval = datasets.ImageFolder(test_dir, transform=get_eval_transforms(cfg.image_size))
    test_dataset_raw = datasets.ImageFolder(test_dir)
    class_names = test_dataset_eval.classes

    loader = DataLoader(test_dataset_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    device = get_device()
    model = load_model_checkpoint(Path(args.checkpoint), num_classes=len(class_names), device=device)

    y_true, y_pred = collect_predictions(model, loader, device=device)
    output_dir = Path(args.output_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
    save_report(y_true, y_pred, class_names, output_dir / "classification_report.json")
    generate_gradcam_samples(
        model=model,
        test_dataset_raw=test_dataset_raw,
        class_names=class_names,
        device=device,
        output_dir=output_dir / "gradcam",
        num_samples=5,
    )
    print(f"Evaluation artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()

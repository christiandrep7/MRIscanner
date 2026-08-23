from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import DataConfig, TrainConfig
from src.data import build_dataloaders
from src.model import build_resnet50_model, get_device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def save_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = list(history[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def train(data_cfg: DataConfig, train_cfg: TrainConfig) -> Path:
    set_seed(data_cfg.seed)
    bundle = build_dataloaders(data_cfg)
    device = get_device()

    model = build_resnet50_model(
        num_classes=len(bundle.class_to_idx),
        pretrained=train_cfg.pretrained,
        imagenet_weights_path=train_cfg.imagenet_weights_path,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = train_cfg.checkpoint_dir / "best_model.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, bundle.train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, bundle.val_loader, criterion, None, device)

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        }
        history.append(epoch_row)
        print(epoch_row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": bundle.class_to_idx,
                    "data_config": asdict(data_cfg),
                    "train_config": asdict(train_cfg),
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= train_cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} (no val loss improvement for "
                f"{train_cfg.early_stopping_patience} epochs)."
            )
            break

    save_history(history, train_cfg.history_path)
    print(f"Saved training history to {train_cfg.history_path}")
    print(f"Best model checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRI classifier.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--imagenet-weights",
        type=str,
        default=None,
        help="Path to local resnet50-11ad3fa6.pth (if download.pytorch.org is blocked).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random ResNet50 init (no ImageNet weights; not recommended).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config = DataConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    imagenet_path = Path(args.imagenet_weights) if args.imagenet_weights else None
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        pretrained=not args.no_pretrained,
        imagenet_weights_path=imagenet_path,
    )
    train(data_config, train_config)

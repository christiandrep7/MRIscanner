from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import DataConfig, TrainConfig
from src.data import build_dataloaders
from src.model import ARCHITECTURES, build_model, get_device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json_safe(value):
    """Recursively convert Path -> str so checkpoint metadata unpickles on any
    Python version. A checkpoint saved on one Python (e.g. Colab's) that embeds a
    live pathlib.Path can fail to torch.load on another (observed: newer Pythons'
    PosixPath resolves through the internal `pathlib._local` module, which older
    Pythons don't have) -- str is portable everywhere."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    false_negative_penalty: tuple[int, float] | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """`false_negative_penalty`, if given, is (class_idx, weight): adds
    weight * mean(softmax_prob[:, class_idx]) over rows whose true label is NOT
    class_idx -- i.e. penalizes the model for leaning towards e.g. "notumor"
    on rows that are actually some other (tumor) class, on top of `criterion`.
    Applied in both train and eval so early stopping's val_loss reflects the
    same objective actually being optimized.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            logits = model(images)
            loss = criterion(logits.float(), labels)
            if false_negative_penalty is not None:
                penalty_idx, penalty_weight = false_negative_penalty
                if penalty_weight > 0:
                    is_other_class = labels != penalty_idx
                    if is_other_class.any():
                        probs = torch.softmax(logits.float(), dim=1)
                        loss = loss + penalty_weight * probs[is_other_class, penalty_idx].mean()
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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


def resume_state_path(train_cfg: TrainConfig) -> Path:
    """Saved every epoch (not just on improvement) so an interrupted run loses
    at most 1 epoch of progress -- distinct from checkpoint_path, which only
    holds the best model's weights and no optimizer/epoch state."""
    return train_cfg.checkpoint_dir / f"{train_cfg.architecture}_resume_state.pth"


def training_done_marker(train_cfg: TrainConfig) -> Path:
    """Written only once a run truly finishes (early-stopped or hit max
    epochs) -- lets a driver script (train_all, or a restart-after-crash loop)
    tell "fully done" apart from "partially done, safe to resume"."""
    return train_cfg.checkpoint_dir / f"{train_cfg.architecture}_training_done.marker"


def train(data_cfg: DataConfig, train_cfg: TrainConfig) -> Path:
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if training_done_marker(train_cfg).exists():
        # A driver script (train_all, or a restart-after-crash loop) re-invoked
        # train() for an architecture that already fully finished -- skip
        # before paying for dataset/model construction, not just before the
        # epoch loop.
        print(f"{train_cfg.architecture} already finished training -- skipping.")
        return train_cfg.checkpoint_path

    set_seed(data_cfg.seed)
    bundle = build_dataloaders(data_cfg)
    device = get_device()

    model = build_model(
        train_cfg.architecture,
        num_classes=len(bundle.class_to_idx),
        fine_tune_layers=train_cfg.fine_tune_layers,
        pretrained=train_cfg.pretrained,
        imagenet_weights_path=train_cfg.imagenet_weights_path,
    ).to(device)
    if train_cfg.init_checkpoint is not None:
        init = torch.load(train_cfg.init_checkpoint, map_location=device, weights_only=False)
        if init["class_to_idx"] != bundle.class_to_idx:
            raise ValueError(
                f"init_checkpoint classes {init['class_to_idx']} != dataset classes {bundle.class_to_idx}"
            )
        model.load_state_dict(init["model_state_dict"])
        print(f"Continuing from {train_cfg.init_checkpoint} (weights loaded, optimizer fresh).")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = (
        ReduceLROnPlateau(
            optimizer, mode="min", factor=train_cfg.lr_plateau_factor,
            patience=train_cfg.lr_plateau_patience, min_lr=train_cfg.min_learning_rate,
        )
        if train_cfg.lr_plateau_patience > 0
        else None
    )
    scaler = torch.amp.GradScaler("cuda") if train_cfg.use_amp and device.type == "cuda" else None

    # Per-class weight tensor, resolved by name against this dataset's actual
    # class_to_idx (never assume alphabetical order lines up with a hardcoded list).
    class_weights = torch.ones(len(bundle.class_to_idx))
    for class_name, weight_value in train_cfg.class_loss_weights.items():
        if class_name in bundle.class_to_idx:
            class_weights[bundle.class_to_idx[class_name]] = weight_value
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=train_cfg.label_smoothing)

    false_negative_penalty = None
    penalty_class_idx = bundle.class_to_idx.get(train_cfg.false_negative_penalty_class)
    if penalty_class_idx is not None and train_cfg.false_negative_penalty_weight > 0:
        false_negative_penalty = (penalty_class_idx, train_cfg.false_negative_penalty_weight)

    best_ckpt_path = train_cfg.checkpoint_path
    resume_path = resume_state_path(train_cfg)
    done_marker = training_done_marker(train_cfg)

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    start_epoch = 1

    if resume_path.exists():
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        if resume_state.get("architecture") == train_cfg.architecture:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if scheduler is not None and resume_state.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            best_val_loss = resume_state["best_val_loss"]
            patience_counter = resume_state["patience_counter"]
            history = resume_state["history"]
            start_epoch = resume_state["epoch"] + 1
            print(
                f"Resuming {train_cfg.architecture} from epoch {start_epoch} "
                f"(best_val_loss so far: {best_val_loss:.6f})"
            )

    if train_cfg.init_checkpoint is not None and start_epoch == 1:
        # The starting weights are the bar to beat: only epochs whose val loss is
        # lower replace them, so continued training can never silently make the
        # saved model worse on the validation data.
        baseline_loss, baseline_acc = run_epoch(model, bundle.val_loader, criterion, None, device, false_negative_penalty)
        print(f"Baseline (starting weights) on validation: loss={baseline_loss:.6f} acc={baseline_acc:.4f}")
        best_val_loss = baseline_loss
        torch.save(
            {
                "architecture": train_cfg.architecture,
                "model_state_dict": model.state_dict(),
                "class_to_idx": bundle.class_to_idx,
                "data_config": _json_safe(asdict(data_cfg)),
                "train_config": _json_safe(asdict(train_cfg)),
            },
            best_ckpt_path,
        )

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, bundle.train_loader, criterion, optimizer, device, false_negative_penalty, scaler
        )
        val_loss, val_acc = run_epoch(
            model, bundle.val_loader, criterion, None, device, false_negative_penalty
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(epoch_row)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "architecture": train_cfg.architecture,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": bundle.class_to_idx,
                    "data_config": _json_safe(asdict(data_cfg)),
                    "train_config": _json_safe(asdict(train_cfg)),
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        # Saved every `resume_save_every` epochs regardless of improvement -- if
        # training is interrupted (sleep, crash, power loss) re-running train()
        # picks up from the last saved epoch instead of starting over.
        should_stop = patience_counter >= train_cfg.early_stopping_patience
        if epoch % train_cfg.resume_save_every == 0 or should_stop or epoch == train_cfg.epochs:
            torch.save(
                {
                    "architecture": train_cfg.architecture,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "history": history,
                },
                resume_path,
            )

        if should_stop:
            print(
                f"Early stopping at epoch {epoch} (no val loss improvement for "
                f"{train_cfg.early_stopping_patience} epochs)."
            )
            break

    save_history(history, train_cfg.history_path)
    print(f"Saved training history to {train_cfg.history_path}")
    print(f"Best model checkpoint: {best_ckpt_path}")
    done_marker.touch()
    resume_path.unlink(missing_ok=True)  # run truly finished -- no longer needed
    return best_ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRI classifier.")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--imagenet-weights",
        type=str,
        default=None,
        help="Path to a local ImageNet weights .pth file (if download.pytorch.org is blocked).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random init (no ImageNet weights; not recommended).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config = DataConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    imagenet_path = Path(args.imagenet_weights) if args.imagenet_weights else None
    train_config = TrainConfig(
        architecture=args.architecture,
        epochs=args.epochs,
        learning_rate=args.lr,
        pretrained=not args.no_pretrained,
        imagenet_weights_path=imagenet_path,
    )
    train(data_config, train_config)

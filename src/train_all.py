"""Trains multiple architectures back to back and prints a comparison summary.

Thin wrapper around src.train.train() -- no training logic is duplicated here.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace

from src.config import DataConfig, TrainConfig
from src.model import ARCHITECTURES
from src.train import train


def train_all(
    data_cfg: DataConfig,
    base_train_cfg: TrainConfig,
    architectures: tuple[str, ...] = ARCHITECTURES,
) -> list[dict]:
    summary: list[dict] = []
    for architecture in architectures:
        train_cfg = replace(base_train_cfg, architecture=architecture)
        print(f"\n{'=' * 60}\nTraining {architecture}\n{'=' * 60}")
        start = time.monotonic()
        checkpoint_path = train(data_cfg, train_cfg)
        elapsed = time.monotonic() - start

        with train_cfg.history_path.open() as f:
            rows = list(csv.DictReader(f))
        best_row = min(rows, key=lambda r: float(r["val_loss"])) if rows else {}

        summary.append(
            {
                "architecture": architecture,
                "checkpoint": str(checkpoint_path),
                "epochs_run": len(rows),
                "best_val_loss": best_row.get("val_loss"),
                "best_val_accuracy": best_row.get("val_accuracy"),
                "elapsed_seconds": round(elapsed, 1),
            }
        )

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for row in summary:
        print(row)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all (or a subset of) architectures.")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_config = DataConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    base_config = TrainConfig(epochs=args.epochs, learning_rate=args.lr, pretrained=not args.no_pretrained)
    train_all(data_config, base_config, architectures=tuple(args.architectures))

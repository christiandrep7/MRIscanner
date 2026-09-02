"""Evaluates every trained architecture on data/Testing -- images no model's
training run or checkpoint-selection ever touched -- and reports a comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import get_eval_transforms
from src.evaluate import collect_predictions, load_model_checkpoint, plot_confusion_matrix, save_report
from src.model import ARCHITECTURES, get_device


@dataclass
class BenchmarkResult:
    architecture: str
    checkpoint_path: Path
    available: bool
    accuracy: float | None = None
    macro_f1: float | None = None
    num_params: int | None = None
    ms_per_image: float | None = None
    error: str | None = None


def default_checkpoint_specs(checkpoint_dir: Path = Path("checkpoints")) -> list[tuple[str, Path]]:
    return [(architecture, checkpoint_dir / f"{architecture}_best_model.pth") for architecture in ARCHITECTURES]


def benchmark_models(
    specs: list[tuple[str, Path]],
    data_cfg: DataConfig,
    output_dir: Path = Path("outputs") / "benchmark",
) -> list[BenchmarkResult]:
    device = get_device()
    test_dir = data_cfg.data_root / data_cfg.test_dir_name
    test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(data_cfg.image_size))
    # Snapshot before any remapping below -- this test folder may hold only a
    # subset of the trained classes (e.g. an external, cross-distribution
    # held-out set with just 2 of 4 classes), so its own ImageFolder-assigned
    # indices (a fresh 0..N-1 over whatever subdirectories are physically
    # present) don't line up with the model's actual (larger) output space.
    original_samples = list(test_dataset.samples)
    local_idx_to_name = {v: k for k, v in test_dataset.class_to_idx.items()}

    results: list[BenchmarkResult] = []
    for architecture, checkpoint_path in specs:
        if not checkpoint_path.exists():
            results.append(
                BenchmarkResult(
                    architecture=architecture,
                    checkpoint_path=checkpoint_path,
                    available=False,
                    error="checkpoint not found -- train this architecture first",
                )
            )
            continue

        try:
            payload = load_training_checkpoint(checkpoint_path, map_location=device)
            checkpoint_class_to_idx: dict[str, int] = payload["class_to_idx"]
            class_names = [name for name, _ in sorted(checkpoint_class_to_idx.items(), key=lambda kv: kv[1])]

            missing = [name for name in local_idx_to_name.values() if name not in checkpoint_class_to_idx]
            if missing:
                raise ValueError(
                    f"Test folder has class(es) {missing} that {architecture}'s checkpoint was never "
                    "trained on -- can't score predictions for a class the model has no output for."
                )
            # Remap into the checkpoint's own global label space (read from the
            # checkpoint, never assumed) rather than trusting this test folder's
            # local re-numbering -- rebuilt from the pristine snapshot each loop
            # iteration so one architecture's remap can't leak into the next's.
            test_dataset.samples = [
                (path, checkpoint_class_to_idx[local_idx_to_name[local_target]])
                for path, local_target in original_samples
            ]
            test_dataset.targets = [target for _, target in test_dataset.samples]
            loader = DataLoader(
                test_dataset, batch_size=data_cfg.batch_size, shuffle=False, num_workers=data_cfg.num_workers
            )

            model = load_model_checkpoint(checkpoint_path, num_classes=len(class_names), device=device)
            num_params = sum(p.numel() for p in model.parameters())

            start = time.perf_counter()
            y_true, y_pred = collect_predictions(model, loader, device=device)
            elapsed = time.perf_counter() - start
            ms_per_image = (elapsed / max(1, len(y_true))) * 1000

            model_out_dir = output_dir / architecture
            plot_confusion_matrix(y_true, y_pred, class_names, model_out_dir / "confusion_matrix.png")
            report_path = model_out_dir / "classification_report.json"
            save_report(y_true, y_pred, class_names, report_path)
            report = json.loads(report_path.read_text())

            results.append(
                BenchmarkResult(
                    architecture=architecture,
                    checkpoint_path=checkpoint_path,
                    available=True,
                    accuracy=report["accuracy"],
                    macro_f1=report["macro avg"]["f1-score"],
                    num_params=num_params,
                    ms_per_image=ms_per_image,
                )
            )
        except Exception as e:
            results.append(
                BenchmarkResult(
                    architecture=architecture,
                    checkpoint_path=checkpoint_path,
                    available=False,
                    error=str(e),
                )
            )

    return results


def save_benchmark_report(results: list[BenchmarkResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "architecture": r.architecture,
            "checkpoint": str(r.checkpoint_path),
            "available": r.available,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "num_params": r.num_params,
            "ms_per_image": r.ms_per_image,
            "error": r.error,
        }
        for r in results
    ]
    report_path = output_dir / "results.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    available = [r for r in results if r.available]
    if available:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(available))
        width = 0.35
        ax.bar(x - width / 2, [r.accuracy for r in available], width, label="Accuracy")
        ax.bar(x + width / 2, [r.macro_f1 for r in available], width, label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels([r.architecture for r in available])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model comparison on held-out test set (never trained on)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_comparison.png", dpi=150)
        plt.close(fig)

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all trained architectures on the held-out test set.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(batch_size=args.batch_size)
    specs = default_checkpoint_specs(args.checkpoint_dir)
    results = benchmark_models(specs, data_cfg, output_dir=args.output_dir)
    report_path = save_benchmark_report(results, args.output_dir)

    for r in results:
        if r.available:
            print(
                f"{r.architecture}: accuracy={r.accuracy:.4f} macro_f1={r.macro_f1:.4f} "
                f"ms/img={r.ms_per_image:.2f}"
            )
        else:
            print(f"{r.architecture}: unavailable ({r.error})")
    print(f"Wrote benchmark report to: {report_path}")


if __name__ == "__main__":
    main()

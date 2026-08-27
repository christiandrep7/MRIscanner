"""
Find images in data/secret that duplicate (or nearly duplicate) the original dataset,
build a deduplicated folder, and evaluate the trained checkpoint on it.

- Exact duplicates: MD5 of file bytes
- Near duplicates: perceptual hash (pHash) Hamming distance <= threshold
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import get_eval_transforms
from src.evaluate import (
    collect_predictions,
    load_model_checkpoint,
    plot_confusion_matrix,
    save_report,
)
from src.model import get_device

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            out.append(p)
    return out


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_of(path: Path) -> imagehash.ImageHash | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return imagehash.phash(im)
    except (UnidentifiedImageError, OSError):
        return None


def build_reference_index(
    roots: list[Path],
) -> tuple[set[str], list[imagehash.ImageHash]]:
    md5_set: set[str] = set()
    ph_list: list[imagehash.ImageHash] = []
    paths: list[Path] = []
    for r in roots:
        if r.exists():
            paths.extend(iter_images(r))
    for p in tqdm(paths, desc="Indexing original dataset (hash + phash)"):
        md5_set.add(file_md5(p))
        h = phash_of(p)
        if h is not None:
            ph_list.append(h)
    return md5_set, ph_list


def is_near_duplicate(
    ph: imagehash.ImageHash | None,
    ph_list: list[imagehash.ImageHash],
    max_hamming: int,
) -> bool:
    if ph is None:
        return False
    # For MRI, loose thresholds (e.g. 8) often flag different patients as "similar".
    # Use 0 for exact pHash match; raise max_hamming only if you accept more aggressive filtering.
    for ref in ph_list:
        if ph - ref <= max_hamming:
            return True
    return False


def dedupe_secret(
    secret_root: Path,
    out_root: Path,
    md5_set: set[str],
    ph_list: list[imagehash.ImageHash],
    max_hamming: int,
) -> dict:
    stats = {
        "total_secret": 0,
        "md5_dup": 0,
        "phash_dup": 0,
        "kept": 0,
        "bad_image": 0,
        "per_class": {},
    }
    for cls_dir in sorted([p for p in secret_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        stats["per_class"][cls] = {"total": 0, "kept": 0, "dropped": 0}
        dest_cls = out_root / cls
        dest_cls.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(iter_images(cls_dir)):
            stats["total_secret"] += 1
            stats["per_class"][cls]["total"] += 1
            m = file_md5(img_path)
            if m in md5_set:
                stats["md5_dup"] += 1
                stats["per_class"][cls]["dropped"] += 1
                continue
            ph = phash_of(img_path)
            if ph is None:
                stats["bad_image"] += 1
                stats["per_class"][cls]["dropped"] += 1
                continue
            if is_near_duplicate(ph, ph_list, max_hamming):
                stats["phash_dup"] += 1
                stats["per_class"][cls]["dropped"] += 1
                continue
            dest = dest_cls / img_path.name
            shutil.copy2(img_path, dest)
            stats["kept"] += 1
            stats["per_class"][cls]["kept"] += 1
    return stats


def remove_empty_class_dirs(test_root: Path) -> None:
    """Remove class folders with zero files (avoids torchvision ImageFolder errors)."""
    for d in list(test_root.iterdir()):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


class _LabeledImageListDataset(Dataset):
    """Paths + integer labels aligned with training ``class_to_idx``."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, label


def run_eval(
    data_root: Path,
    test_dir: str,
    checkpoint: Path,
    out_dir: Path,
    batch_size: int,
) -> dict:
    cfg = DataConfig(
        data_root=data_root,
        test_dir_name=test_dir,
        batch_size=batch_size,
        num_workers=0,
    )
    test_dir_path = cfg.data_root / cfg.test_dir_name
    remove_empty_class_dirs(test_dir_path)
    if not test_dir_path.exists() or not any(test_dir_path.iterdir()):
        raise FileNotFoundError(f"No class folders with images under {test_dir_path}.")

    payload = load_training_checkpoint(checkpoint, map_location="cpu")
    class_to_idx: dict[str, int] = payload["class_to_idx"]
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    samples: list[tuple[Path, int]] = []
    for name, idx in class_to_idx.items():
        cls_dir = test_dir_path / name
        if not cls_dir.is_dir():
            continue
        for p in iter_images(cls_dir):
            samples.append((p, idx))

    if not samples:
        raise FileNotFoundError(f"No labeled images found under {test_dir_path}.")

    tfm = get_eval_transforms(cfg.image_size)
    ds = _LabeledImageListDataset(samples, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = get_device()
    model = load_model_checkpoint(checkpoint, num_classes=num_classes, device=device)
    y_true, y_pred = collect_predictions(model, loader, device=device)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
    save_report(y_true, y_pred, class_names, out_dir / "classification_report.json")
    report = json.loads((out_dir / "classification_report.json").read_text(encoding="utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedupe secret set vs train+test, then evaluate.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--secret-testing", type=Path, default=Path("data/secret/Testing"))
    parser.add_argument("--deduped-out", type=Path, default=Path("data/secret/Testing_deduped"))
    parser.add_argument(
        "--hamming",
        type=int,
        default=0,
        help="pHash max Hamming distance to flag near-dup (0=exact pHash; MRI: avoid large values).",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet50_best_model.pth"))
    parser.add_argument("--eval-out", type=Path, default=Path("outputs/secret_eval_deduped"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_dir = args.data_root / "Training"
    test_dir = args.data_root / "Testing"
    md5_set, ph_list = build_reference_index([train_dir, test_dir])
    print(f"Reference images: md5 unique={len(md5_set)}, phash list={len(ph_list)}")

    if args.deduped_out.exists():
        shutil.rmtree(args.deduped_out)
    args.deduped_out.parent.mkdir(parents=True, exist_ok=True)

    stats = dedupe_secret(
        args.secret_testing,
        args.deduped_out,
        md5_set,
        ph_list,
        max_hamming=args.hamming,
    )
    report_path = Path("outputs") / "secret_dedupe_stats.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("Dedupe stats:", json.dumps(stats, indent=2))
    print(f"Wrote deduped images to: {args.deduped_out}")
    print(f"Wrote stats to: {report_path}")

    if stats["kept"] == 0:
        print("No images left after dedupe. Increase --hamming or check paths.")
        return

    # Evaluate on deduped folder: ImageFolder needs parent of class dirs
    eval_root = args.deduped_out.parent
    test_name = args.deduped_out.name
    report = run_eval(
        data_root=eval_root,
        test_dir=test_name,
        checkpoint=args.checkpoint,
        out_dir=args.eval_out,
        batch_size=args.batch_size,
    )
    print("Deduped secret accuracy:", report["accuracy"])
    print("Macro F1:", report["macro avg"]["f1-score"])
    print(f"Wrote eval artifacts to: {args.eval_out}")


if __name__ == "__main__":
    main()

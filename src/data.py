from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DataConfig


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PACSStyleNoise:
    """Simulates the visual noise real-world MRI images actually have when
    people find them online: burned-in scan-parameter text, scale rulers,
    borders from multi-panel layouts, and tint shifts from old scans/photos of
    film. None of this project's training data (Kaggle, BraTS, IXI) has any of
    it -- every image is a clean, unannotated single slice -- so the model
    never learned to ignore it. Confirmed with real online images this
    session: the model was confidently *wrong* ("notumor") specifically on
    images with this kind of overlay, and correct on clean ones of the same
    tumor type. Train-time only -- eval/benchmark images shouldn't be
    artificially corrupted, only made robust to real corruption.
    """

    _SAMPLE_TEXT = (
        "TR: 4500.0", "TE: 109.0", "Mag: 1.5x", "HEAD", "FOV: 24.0cm",
        "SER 1-5", "512 x 256", "W:976 L:488", "AR", "RHP", "ET: 16",
    )
    _TINTS = ((255, 230, 200), (200, 220, 255), (230, 200, 230))

    def __init__(self, p: float = 0.35):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img = img.copy()
        w, h = img.size
        noise_types = random.sample(["text", "ruler", "border", "tint"], k=random.randint(1, 3))

        if "text" in noise_types:
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            for _ in range(random.randint(1, 3)):
                text = random.choice(self._SAMPLE_TEXT)
                x = random.choice([2, max(2, w - 80)])
                y = random.choice([2, max(2, h - 15)])
                color = random.choice([(255, 255, 255), (220, 220, 220)])
                draw.text((x, y), text, fill=color, font=font)

        if "ruler" in noise_types:
            draw = ImageDraw.Draw(img)
            side_x = random.choice([2, max(2, w - 6)])
            tick_count = random.randint(8, 14)
            for i in range(tick_count):
                y = int(h * i / tick_count)
                draw.line([(side_x, y), (side_x + 4, y)], fill=(255, 255, 255), width=1)

        if "border" in noise_types:
            draw = ImageDraw.Draw(img)
            border_width = random.randint(4, 16)
            border_color = random.choice([(0, 0, 0), (255, 255, 255)])
            for side in random.sample(["left", "right", "top", "bottom"], k=random.randint(1, 2)):
                if side == "left":
                    draw.rectangle([0, 0, border_width, h], fill=border_color)
                elif side == "right":
                    draw.rectangle([w - border_width, 0, w, h], fill=border_color)
                elif side == "top":
                    draw.rectangle([0, 0, w, border_width], fill=border_color)
                else:
                    draw.rectangle([0, h - border_width, w, h], fill=border_color)

        if "tint" in noise_types:
            overlay = Image.new("RGB", img.size, random.choice(self._TINTS))
            img = Image.blend(img.convert("RGB"), overlay, alpha=random.uniform(0.08, 0.2))

        return img


class MultiPanelComposite:
    """Simulates real teaching-figure/PACS composites: 2 or 4 sub-panels of
    brain-like content side by side in one frame, instead of one image filling
    the whole canvas. Confirmed with a real failing image this session: a
    glioma case that was wrong (confidently "notumor") turned out to be a
    genuine 2-panel composite (coronal + axial side by side) -- no training
    image, even with PACSStyleNoise added, had ever looked like that, since
    text/ruler/tint noise alone doesn't change the fact that only half the
    frame is the "real" view. Built from the *same* source image (no access to
    other dataset samples from inside a single-image transform) -- each panel
    gets independently flipped so panels don't look like exact duplicates,
    which is the closest cheap approximation to genuinely different sub-views.
    Applied after geometric transforms/crop so panels land on the final canvas.
    """

    def __init__(self, p: float = 0.25):
        self.p = p

    def _random_flip(self, panel: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            panel = panel.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            panel = panel.transpose(Image.FLIP_TOP_BOTTOM)
        return panel

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        layout = random.choice(["2h", "2v", "4"])
        canvas = Image.new("RGB", (w, h), (0, 0, 0))

        if layout == "2h":
            half_w = w // 2
            canvas.paste(self._random_flip(img.resize((half_w, h))), (0, 0))
            canvas.paste(self._random_flip(img.resize((w - half_w, h))), (half_w, 0))
        elif layout == "2v":
            half_h = h // 2
            canvas.paste(self._random_flip(img.resize((w, half_h))), (0, 0))
            canvas.paste(self._random_flip(img.resize((w, h - half_h))), (0, half_h))
        else:  # "4"
            half_w, half_h = w // 2, h // 2
            quadrant = img.resize((half_w, half_h))
            for qx in (0, half_w):
                for qy in (0, half_h):
                    canvas.paste(self._random_flip(quadrant), (qx, qy))

        return canvas


@dataclass
class MRIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    train_size: int
    val_size: int
    test_size: int


def get_train_transforms(image_size: int) -> transforms.Compose:
    # RandomResizedCrop (instead of a fixed Resize) plus a bit more rotation/shift
    # than before: benchmarking showed glioma generalizing far worse than the other
    # 3 classes (test recall ~0.76-0.84 vs 0.94-1.0) despite ~98% val accuracy --
    # a classic overfit-to-Training-distribution signature, not a labeling bug (see
    # kb mistake/lesson from this session). More scale/position/rotation variation
    # forces the model to learn shape rather than the exact framing of the Training set.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # Widened from brightness/contrast=0.15 -- a real failing image this
            # session (pituitary, wrong at ~65-100% "notumor" across every attempt
            # so far) turned out to be a different acquisition/windowing style
            # entirely, not just noisy decoration; 0.15 was nowhere near enough
            # range to have ever produced anything that different at train time.
            # RandomAutocontrast/RandomEqualize directly remap the intensity
            # histogram (real torchvision ops, not custom code) -- a much closer
            # analogue of "a different scanner's windowing" than brightness scaling.
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.2),
            # After the geometric transforms (not before) so any text/ruler/border
            # added lands axis-aligned on the final canvas -- real PACS overlays are
            # never rotated, so training the model to expect rotated overlays would
            # teach the wrong invariance.
            MultiPanelComposite(p=0.25),
            PACSStyleNoise(p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _split_train_val_indices(num_items: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(num_items * val_split)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return train_indices, val_indices


def build_dataloaders(config: DataConfig) -> MRIDataBundle:
    train_dir = config.data_root / config.train_dir_name
    test_dir = config.data_root / config.test_dir_name
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at '{train_dir}' and '{test_dir}'. "
            "Run download_mri_dataset.py first."
        )

    full_train_for_aug = datasets.ImageFolder(train_dir, transform=get_train_transforms(config.image_size))
    full_train_for_eval = datasets.ImageFolder(train_dir, transform=get_eval_transforms(config.image_size))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(config.image_size))

    train_indices, val_indices = _split_train_val_indices(
        num_items=len(full_train_for_aug),
        val_split=config.val_split,
        seed=config.seed,
    )

    train_subset = Subset(full_train_for_aug, train_indices)
    val_subset = Subset(full_train_for_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_to_idx = full_train_for_aug.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return MRIDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        train_size=len(train_subset),
        val_size=len(val_subset),
        test_size=len(test_dataset),
    )


def show_random_batch(data_bundle: MRIDataBundle, output_path: Path | None = None, max_images: int = 12) -> None:
    images, labels = next(iter(data_bundle.train_loader))
    images = images[:max_images]
    labels = labels[:max_images]

    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis("off")
            continue
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * IMAGENET_STD) + IMAGENET_MEAN
        img = img.clip(0, 1)
        label_name = data_bundle.idx_to_class[int(labels[i].item())]
        ax.imshow(img)
        ax.set_title(label_name)
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    cfg = DataConfig()
    bundle = build_dataloaders(cfg)
    print("Class mapping:", bundle.class_to_idx)
    print(
        f"Sizes -> train: {bundle.train_size}, val: {bundle.val_size}, test: {bundle.test_size}"
    )
    show_random_batch(bundle, output_path=Path("outputs") / "sample_train_batch.png")
    print("Saved sample batch image to outputs/sample_train_batch.png")

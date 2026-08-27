from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import DataConfig, TrainConfig
from src.model import ARCHITECTURES, build_model

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
TEST_IMAGE_SIZE = 32


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(TEST_IMAGE_SIZE, TEST_IMAGE_SIZE, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _populate_split(root: Path, split_name: str, per_class: int, seed_offset: int) -> None:
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = root / split_name / class_name
        for i in range(per_class):
            seed = seed_offset + class_idx * 1000 + i
            _make_image(class_dir / f"img_{i}.jpg", seed=seed)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Builds a fake data/Training + data/Testing tree: 10 train / 6 test images per class."""
    data_root = tmp_path / "data"
    _populate_split(data_root, "Training", per_class=10, seed_offset=0)
    _populate_split(data_root, "Testing", per_class=6, seed_offset=10_000)
    return data_root


@pytest.fixture
def tiny_data_config(tiny_dataset: Path) -> DataConfig:
    return DataConfig(
        data_root=tiny_dataset,
        image_size=TEST_IMAGE_SIZE,
        batch_size=2,
        num_workers=0,
        val_split=0.2,
        seed=42,
    )


@pytest.fixture
def tiny_train_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
        history_dir=tmp_path / "outputs",
        pretrained=False,
    )


def make_fake_checkpoint(
    checkpoint_dir: Path,
    architecture: str,
    data_config: DataConfig,
    train_config: TrainConfig | None = None,
) -> Path:
    """A {architecture}_best_model.pth with the exact shape src/train.py writes,
    without running real training. Reused by model/train/benchmark/UI tests."""
    train_config = train_config or TrainConfig(
        architecture=architecture, epochs=1, checkpoint_dir=checkpoint_dir, pretrained=False
    )
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    model = build_model(architecture, num_classes=len(class_to_idx), pretrained=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{architecture}_best_model.pth"
    torch.save(
        {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "data_config": asdict(data_config),
            "train_config": asdict(train_config),
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def fake_checkpoint(tmp_path: Path, tiny_data_config: DataConfig, tiny_train_config: TrainConfig) -> Path:
    return make_fake_checkpoint(tmp_path / "checkpoints", "resnet50", tiny_data_config, tiny_train_config)


@pytest.fixture
def fake_checkpoint_factory(tmp_path: Path, tiny_data_config: DataConfig):
    """Callable(architecture) -> Path, for tests needing checkpoints for multiple architectures."""

    def _factory(architecture: str) -> Path:
        return make_fake_checkpoint(tmp_path / "checkpoints", architecture, tiny_data_config)

    return _factory


@pytest.fixture
def all_fake_checkpoints(fake_checkpoint_factory) -> dict[str, Path]:
    return {architecture: fake_checkpoint_factory(architecture) for architecture in ARCHITECTURES}

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import DataConfig
from src.data import (
    _split_train_val_indices,
    build_dataloaders,
    get_eval_transforms,
    get_train_transforms,
)


def test_build_dataloaders_class_mapping(tiny_data_config: DataConfig):
    bundle = build_dataloaders(tiny_data_config)
    assert bundle.class_to_idx == {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3,
    }
    assert bundle.idx_to_class == {v: k for k, v in bundle.class_to_idx.items()}


def test_build_dataloaders_split_sizes(tiny_data_config: DataConfig):
    bundle = build_dataloaders(tiny_data_config)
    total_train_pool = 4 * 10  # 4 classes x 10 train images from tiny_dataset
    assert bundle.train_size + bundle.val_size == total_train_pool
    expected_val = int(total_train_pool * tiny_data_config.val_split)
    assert bundle.val_size == expected_val
    assert bundle.test_size == 4 * 6  # 4 classes x 6 test images


def test_dataloader_batch_shapes(tiny_data_config: DataConfig):
    bundle = build_dataloaders(tiny_data_config)
    images, labels = next(iter(bundle.train_loader))
    assert images.shape[1:] == (3, tiny_data_config.image_size, tiny_data_config.image_size)
    assert images.dtype == torch.float32
    assert labels.shape[0] == images.shape[0]


def test_missing_dataset_dirs_raise_with_helpful_message(tmp_path: Path):
    cfg = DataConfig(data_root=tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="download_mri_dataset.py"):
        build_dataloaders(cfg)


def test_split_train_val_indices_is_deterministic_for_seed():
    train_a, val_a = _split_train_val_indices(num_items=100, val_split=0.2, seed=42)
    train_b, val_b = _split_train_val_indices(num_items=100, val_split=0.2, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 20
    assert len(train_a) == 80
    # current implementation shuffles a flat index range with no class stratification
    assert set(train_a) | set(val_a) == set(range(100))
    assert set(train_a).isdisjoint(set(val_a))


def test_split_train_val_indices_differs_across_seeds():
    _, val_a = _split_train_val_indices(num_items=100, val_split=0.2, seed=1)
    _, val_b = _split_train_val_indices(num_items=100, val_split=0.2, seed=2)
    assert val_a != val_b


def test_train_transforms_include_augmentation_and_eval_does_not():
    train_tfm = get_train_transforms(image_size=32)
    eval_tfm = get_eval_transforms(image_size=32)
    train_repr = repr(train_tfm)
    eval_repr = repr(eval_tfm)
    assert "RandomHorizontalFlip" in train_repr
    assert "RandomRotation" in train_repr
    assert "RandomHorizontalFlip" not in eval_repr
    assert "RandomRotation" not in eval_repr

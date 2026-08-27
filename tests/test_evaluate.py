from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import DataConfig
from src.data import get_eval_transforms
from src.evaluate import (
    collect_predictions,
    generate_gradcam_samples,
    load_model_checkpoint,
    plot_confusion_matrix,
    save_report,
)
from src.model import get_device


def test_load_model_checkpoint_and_predict(tiny_data_config: DataConfig, fake_checkpoint: Path):
    device = get_device()
    model = load_model_checkpoint(fake_checkpoint, num_classes=4, device=device)
    assert not model.training  # .eval() was called

    test_dir = tiny_data_config.data_root / tiny_data_config.test_dir_name
    dataset = datasets.ImageFolder(test_dir, transform=get_eval_transforms(tiny_data_config.image_size))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    y_true, y_pred = collect_predictions(model, loader, device=device)

    assert y_true.shape == y_pred.shape
    assert len(y_true) == len(dataset)
    assert set(np.unique(y_pred)).issubset({0, 1, 2, 3})


def test_plot_confusion_matrix_and_save_report(tmp_path: Path):
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 2, 0, 0])
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    cm_path = tmp_path / "cm.png"
    report_path = tmp_path / "report.json"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    save_report(y_true, y_pred, class_names, report_path)

    assert cm_path.exists() and cm_path.stat().st_size > 0
    report = json.loads(report_path.read_text())
    assert "accuracy" in report
    for name in class_names:
        assert name in report
        assert "precision" in report[name]
        assert "recall" in report[name]
        assert "f1-score" in report[name]


def test_save_report_and_confusion_matrix_survive_missing_classes(tmp_path: Path):
    """Regression test: a small holdout set (e.g. the 'secret' dataset) can easily
    contain only 1-2 of the 4 known classes. Without `labels=` pinned to the full
    class list, sklearn raises ValueError instead of just reporting zeros for the
    classes that never showed up."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 1, 0])
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    cm_path = tmp_path / "cm.png"
    report_path = tmp_path / "report.json"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    save_report(y_true, y_pred, class_names, report_path)

    report = json.loads(report_path.read_text())
    assert report["notumor"]["support"] == 0
    assert report["pituitary"]["support"] == 0
    assert report["glioma"]["support"] == 3


def test_generate_gradcam_samples_writes_requested_count(
    tiny_data_config: DataConfig, fake_checkpoint: Path, tmp_path: Path
):
    device = get_device()
    model = load_model_checkpoint(fake_checkpoint, num_classes=4, device=device)
    test_dir = tiny_data_config.data_root / tiny_data_config.test_dir_name
    raw_dataset = datasets.ImageFolder(test_dir)
    out_dir = tmp_path / "gradcam"

    generate_gradcam_samples(
        model=model,
        test_dataset_raw=raw_dataset,
        class_names=raw_dataset.classes,
        device=device,
        output_dir=out_dir,
        num_samples=3,
    )

    written = list(out_dir.glob("gradcam_*.jpg"))
    assert len(written) == 3

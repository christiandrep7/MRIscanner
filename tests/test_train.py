from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from src.config import DataConfig, TrainConfig
from src.model import ARCHITECTURES
from src.train import run_epoch, save_history, set_seed, train


def test_train_writes_checkpoint_and_history(tiny_data_config: DataConfig, tiny_train_config: TrainConfig):
    tiny_train_config.epochs = 2
    ckpt_path = train(tiny_data_config, tiny_train_config)

    assert ckpt_path == tiny_train_config.checkpoint_dir / "resnet50_best_model.pth"
    assert ckpt_path.exists()

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert set(payload.keys()) >= {"architecture", "model_state_dict", "class_to_idx", "data_config", "train_config"}
    assert payload["architecture"] == "resnet50"
    assert payload["class_to_idx"] == {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3,
    }

    assert tiny_train_config.history_path.exists()
    with tiny_train_config.history_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == tiny_train_config.epochs
    for row in rows:
        assert set(row.keys()) == {"epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_train_works_for_every_architecture(
    architecture: str, tiny_data_config: DataConfig, tiny_train_config: TrainConfig
):
    tiny_train_config.architecture = architecture
    tiny_train_config.epochs = 1

    ckpt_path = train(tiny_data_config, tiny_train_config)

    assert ckpt_path == tiny_train_config.checkpoint_dir / f"{architecture}_best_model.pth"
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["architecture"] == architecture


def test_early_stopping_halts_before_max_epochs(tiny_data_config: DataConfig, tiny_train_config: TrainConfig):
    tiny_train_config.epochs = 10
    tiny_train_config.early_stopping_patience = 1
    tiny_train_config.learning_rate = 0.0  # weights never update

    train(tiny_data_config, tiny_train_config)

    with tiny_train_config.history_path.open() as f:
        rows = list(csv.DictReader(f))
    val_losses = [float(row["val_loss"]) for row in rows]
    # With lr=0.0 the weights never update, but BatchNorm running stats still
    # drift each epoch from the forward pass alone (more so now with heavier
    # train-time augmentation), so "best epoch" isn't pinned to epoch 1 -- the
    # real contract is: it stops well short of the 10-epoch budget, and within
    # patience(=1) epochs of whichever epoch actually had the lowest val_loss.
    assert len(rows) < tiny_train_config.epochs
    best_epoch_idx = val_losses.index(min(val_losses))
    assert len(rows) <= best_epoch_idx + 1 + tiny_train_config.early_stopping_patience


def test_save_history_noop_on_empty_list(tmp_path: Path):
    out = tmp_path / "history.csv"
    save_history([], out)
    assert not out.exists()


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.rand(3)
    set_seed(123)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_run_epoch_eval_mode_does_not_update_weights(tiny_data_config: DataConfig):
    from src.data import build_dataloaders
    from src.model import build_resnet50_model, get_device

    bundle = build_dataloaders(tiny_data_config)
    device = get_device()
    model = build_resnet50_model(num_classes=4, pretrained=False).to(device)
    before = model.fc[1].weight.clone()

    run_epoch(model, bundle.val_loader, torch.nn.CrossEntropyLoss(), optimizer=None, device=device)

    after = model.fc[1].weight
    assert torch.equal(before, after)

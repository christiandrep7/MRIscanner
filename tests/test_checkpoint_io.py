from __future__ import annotations

from pathlib import Path

import torch

from src.checkpoint_io import load_training_checkpoint


def test_round_trip_simple_dict(tmp_path: Path):
    payload = {
        "model_state_dict": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "class_to_idx": {"a": 0, "b": 1},
    }
    path = tmp_path / "ckpt.pth"
    torch.save(payload, path)

    loaded = load_training_checkpoint(path)

    assert loaded["class_to_idx"] == payload["class_to_idx"]
    assert torch.equal(loaded["model_state_dict"]["w"], payload["model_state_dict"]["w"])


def test_falls_back_when_weights_only_true_cannot_unpickle(tmp_path: Path, monkeypatch):
    """Simulates older checkpoints containing structures weights_only=True rejects."""
    payload = {"class_to_idx": {"a": 0}}
    path = tmp_path / "ckpt.pth"
    torch.save(payload, path)

    calls: list[bool] = []
    real_load = torch.load

    def fake_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        if kwargs.get("weights_only") is True:
            raise RuntimeError("simulated weights_only failure")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", fake_load)

    loaded = load_training_checkpoint(path)

    assert loaded["class_to_idx"] == {"a": 0}
    assert calls == [True, False]


def test_map_location_is_respected(tmp_path: Path):
    payload = {"model_state_dict": {"w": torch.tensor([1.0])}}
    path = tmp_path / "ckpt.pth"
    torch.save(payload, path)

    loaded = load_training_checkpoint(path, map_location="cpu")

    assert loaded["model_state_dict"]["w"].device.type == "cpu"

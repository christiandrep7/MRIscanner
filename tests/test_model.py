from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model import ARCHITECTURES, build_model, build_resnet50_model, get_device


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_pass_shape(architecture: str):
    model = build_model(architecture, num_classes=4, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 4)


def test_resnet50_freeze_unfreeze_respects_fine_tune_layers():
    model = build_model("resnet50", num_classes=4, pretrained=False, fine_tune_layers=2)

    for param in model.fc.parameters():
        assert param.requires_grad is True
    # last 2 blocks (layer3, layer4) trainable, earlier ones + stem frozen
    for param in model.layer3.parameters():
        assert param.requires_grad is True
    for param in model.layer4.parameters():
        assert param.requires_grad is True
    for param in model.layer1.parameters():
        assert param.requires_grad is False
    for param in model.layer2.parameters():
        assert param.requires_grad is False
    for param in model.conv1.parameters():
        assert param.requires_grad is False


def test_efficientnet_b0_freeze_unfreeze_respects_fine_tune_layers():
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False, fine_tune_layers=2)

    for param in model.classifier.parameters():
        assert param.requires_grad is True
    # `features` has 9 stages; last 2 (index 7, 8) trainable, the rest (incl. the
    # index-0 stem) frozen.
    for param in model.features[8].parameters():
        assert param.requires_grad is True
    for param in model.features[7].parameters():
        assert param.requires_grad is True
    for param in model.features[6].parameters():
        assert param.requires_grad is False
    for param in model.features[0].parameters():
        assert param.requires_grad is False


def test_vgg16_freeze_unfreeze_respects_fine_tune_layers():
    model = build_model("vgg16", num_classes=4, pretrained=False, fine_tune_layers=2)

    for param in model.classifier.parameters():
        assert param.requires_grad is True
    # features has 5 conv blocks (split at each MaxPool2d); last 2 trainable.
    # Blocks, by index into the flat `features` Sequential: [0:5), [5:10), [10:17),
    # [17:24), [24:31) -- so the last 2 blocks start at index 17.
    for param in model.features[17:].parameters():
        assert param.requires_grad is True
    for param in model.features[:17].parameters():
        assert param.requires_grad is False


def test_fine_tune_layers_defaults_to_at_least_one_block():
    model = build_resnet50_model(num_classes=4, pretrained=False, fine_tune_layers=0)
    for param in model.layer4.parameters():
        assert param.requires_grad is True
    for param in model.layer3.parameters():
        assert param.requires_grad is False


def test_missing_local_imagenet_weights_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pth"
    with pytest.raises(FileNotFoundError):
        build_resnet50_model(num_classes=4, pretrained=True, imagenet_weights_path=missing)


def test_build_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("resnet9000", num_classes=4, pretrained=False)


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cuda")


def test_get_device_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")

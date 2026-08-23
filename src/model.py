from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

# Default filename torchvision uses when downloading ResNet50 ImageNet weights.
_RESNET50_HUB_NAME = "resnet50-11ad3fa6.pth"


def default_resnet50_hub_path() -> Path:
    """Where torch hub stores downloaded ResNet50 weights (after first successful download)."""
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / _RESNET50_HUB_NAME


def _load_imagenet_weights_from_file(model: nn.Module, path: Path) -> None:
    """Load official ImageNet ResNet50 state_dict (1000-class head) from a local .pth file."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)


def build_resnet50_model(
    num_classes: int = 4,
    fine_tune_layers: int = 2,
    *,
    pretrained: bool = True,
    imagenet_weights_path: Path | str | None = None,
) -> nn.Module:
    """
    Build ResNet50 with a custom classification head.

    - If ``pretrained=False``, starts from random weights (use when you will load a full
      checkpoint immediately, e.g. evaluate / Gradio).
    - If ``pretrained=True``, loads ImageNet weights from (in order):
      1) ``imagenet_weights_path`` if provided and the file exists
      2) ``~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`` if it exists
      3) Otherwise downloads from pytorch.org (requires working DNS/network)
    """
    model = models.resnet50(weights=None)

    if pretrained:
        path: Path | None = None
        if imagenet_weights_path is not None:
            path = Path(imagenet_weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"ImageNet weights not found: {path}")
        elif default_resnet50_hub_path().is_file():
            path = default_resnet50_hub_path()

        if path is not None:
            _load_imagenet_weights_from_file(model, path)
        else:
            try:
                w = models.ResNet50_Weights.DEFAULT
                pretrained_model = models.resnet50(weights=w)
                model.load_state_dict(pretrained_model.state_dict())
            except Exception as e:
                hub = default_resnet50_hub_path()
                raise RuntimeError(
                    "Could not download ResNet50 ImageNet weights (network/DNS/firewall?).\n\n"
                    "Fix options:\n"
                    "  1) Use a browser/VPN on another machine, download:\n"
                    f"     https://download.pytorch.org/models/{_RESNET50_HUB_NAME}\n"
                    f"     Save as: {hub}\n"
                    "  2) Or pass the file path when training:\n"
                    f"     python -m src.train --imagenet-weights \"C:\\path\\to\\{_RESNET50_HUB_NAME}\"\n"
                ) from e

    # Freeze everything first, then unfreeze last block(s) and head.
    for param in model.parameters():
        param.requires_grad = False

    blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    for block in blocks[-max(1, fine_tune_layers) :]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    device = get_device()
    model = build_resnet50_model(num_classes=4).to(device)
    dummy = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("Forward pass OK. Output shape:", tuple(out.shape))

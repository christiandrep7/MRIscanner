from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ("resnet50", "efficientnet_b0", "vgg16")


def default_hub_path(weights: models.WeightsEnum) -> Path:
    """Where torch hub stores a downloaded ImageNet weights file (after first success)."""
    filename = Path(weights.url).name
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename


def _load_imagenet_weights_from_file(model: nn.Module, path: Path) -> None:
    """Load an official ImageNet state_dict (1000-class head) from a local .pth file."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)


def _load_pretrained_backbone(
    model: nn.Module,
    weights_enum: models.WeightsEnum,
    build_with_weights,
    *,
    imagenet_weights_path: Path | str | None,
) -> None:
    """
    Loads ImageNet weights into `model` in-place, from (in order):
      1) `imagenet_weights_path` if given
      2) the torch hub cache, if that file already exists
      3) a fresh download from pytorch.org (requires working DNS/network)
    """
    path: Path | None = None
    if imagenet_weights_path is not None:
        path = Path(imagenet_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"ImageNet weights not found: {path}")
    else:
        hub_path = default_hub_path(weights_enum.DEFAULT)
        if hub_path.is_file():
            path = hub_path

    if path is not None:
        _load_imagenet_weights_from_file(model, path)
        return

    try:
        pretrained_model = build_with_weights(weights_enum.DEFAULT)
        model.load_state_dict(pretrained_model.state_dict())
    except Exception as e:
        hub = default_hub_path(weights_enum.DEFAULT)
        raise RuntimeError(
            "Could not download ImageNet weights (network/DNS/firewall?).\n\n"
            "Fix options:\n"
            "  1) Use a browser/VPN on another machine, download:\n"
            f"     {weights_enum.DEFAULT.url}\n"
            f"     Save as: {hub}\n"
            "  2) Or pass the file path explicitly (--imagenet-weights on the training CLI).\n"
        ) from e


def _freeze_all_then_unfreeze(model: nn.Module, blocks: list[nn.Module], head: nn.Module, fine_tune_layers: int) -> None:
    # Freeze the *whole* model first -- not just `blocks` -- so any stem/stray
    # layers outside the named block list (e.g. ResNet's conv1/bn1, which aren't
    # part of layer1..4) are frozen too, matching "everything but the last N
    # blocks + head is frozen".
    for param in model.parameters():
        param.requires_grad = False
    for block in blocks[-max(1, fine_tune_layers) :]:
        for param in block.parameters():
            param.requires_grad = True
    for param in head.parameters():
        param.requires_grad = True


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
    - If ``pretrained=True``, loads ImageNet weights (local file, hub cache, or download —
      see ``_load_pretrained_backbone``).
    """
    model = models.resnet50(weights=None)
    if pretrained:
        _load_pretrained_backbone(
            model,
            models.ResNet50_Weights,
            models.resnet50,
            imagenet_weights_path=imagenet_weights_path,
        )

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))

    blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
    _freeze_all_then_unfreeze(model, blocks, model.fc, fine_tune_layers)
    return model


def build_efficientnet_b0_model(
    num_classes: int = 4,
    fine_tune_layers: int = 2,
    *,
    pretrained: bool = True,
    imagenet_weights_path: Path | str | None = None,
) -> nn.Module:
    """Build EfficientNet-B0 with a custom classification head.

    `model.features` is already 9 top-level stages (stem, 7 MBConv stages, final
    conv); the last `fine_tune_layers` of those stay trainable, everything else in
    `features` is frozen. `model.classifier` (Dropout + Linear) is always trainable.
    """
    model = models.efficientnet_b0(weights=None)
    if pretrained:
        _load_pretrained_backbone(
            model,
            models.EfficientNet_B0_Weights,
            models.efficientnet_b0,
            imagenet_weights_path=imagenet_weights_path,
        )

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))

    blocks = list(model.features)
    _freeze_all_then_unfreeze(model, blocks, model.classifier, fine_tune_layers)
    return model


def _vgg_conv_blocks(features: nn.Sequential) -> list[nn.Module]:
    """Groups VGG's flat `features` Sequential into its 5 standard conv blocks,
    splitting after each MaxPool2d (VGG16: blocks of 2,2,3,3,3 conv+ReLU layers)."""
    blocks: list[list[nn.Module]] = [[]]
    for layer in features:
        blocks[-1].append(layer)
        if isinstance(layer, nn.MaxPool2d):
            blocks.append([])
    if not blocks[-1]:
        blocks.pop()
    return [nn.Sequential(*block) for block in blocks]


def build_vgg16_model(
    num_classes: int = 4,
    fine_tune_layers: int = 2,
    *,
    pretrained: bool = True,
    imagenet_weights_path: Path | str | None = None,
) -> nn.Module:
    """Build VGG16 with a custom classification head.

    VGG's `features` has no natural block attributes like ResNet's `layer1..4`, so
    it's grouped into its 5 standard conv blocks (split at each MaxPool2d). The last
    `fine_tune_layers` blocks stay trainable; the rest of `features` is frozen. All of
    `classifier` (3 Linear layers) is always trainable, matching ResNet50/EfficientNet's
    "head fully trainable" convention here.
    """
    model = models.vgg16(weights=None)
    if pretrained:
        _load_pretrained_backbone(
            model,
            models.VGG16_Weights,
            models.vgg16,
            imagenet_weights_path=imagenet_weights_path,
        )

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    blocks = _vgg_conv_blocks(model.features)
    _freeze_all_then_unfreeze(model, blocks, model.classifier, fine_tune_layers)
    return model


_BUILDERS = {
    "resnet50": build_resnet50_model,
    "efficientnet_b0": build_efficientnet_b0_model,
    "vgg16": build_vgg16_model,
}


def build_model(architecture: str, num_classes: int = 4, fine_tune_layers: int = 2, **kwargs) -> nn.Module:
    try:
        builder = _BUILDERS[architecture]
    except KeyError:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from: {ARCHITECTURES}") from None
    return builder(num_classes=num_classes, fine_tune_layers=fine_tune_layers, **kwargs)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_WEIGHTS_ENUMS = {
    "resnet50": models.ResNet50_Weights,
    "efficientnet_b0": models.EfficientNet_B0_Weights,
    "vgg16": models.VGG16_Weights,
}


if __name__ == "__main__":
    device = get_device()
    for architecture in ARCHITECTURES:
        weights = _WEIGHTS_ENUMS[architecture].DEFAULT
        print(f"{architecture}: ImageNet weights URL: {weights.url}")
        print(f"{architecture}: expected hub cache path: {default_hub_path(weights)}")

        model = build_model(architecture, num_classes=4, pretrained=False).to(device)
        dummy = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(dummy)
        print(f"{architecture}: forward pass OK. Output shape:", tuple(out.shape))

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import DataConfig, TrainConfig
from src.model import build_resnet50_model


def _checkpoint_that_always_predicts_meningioma(tmp_path: Path, tiny_data_config: DataConfig) -> Path:
    """Zeroing the head's weight matrix makes its output equal the bias for every
    input (Dropout is disabled in eval mode), guaranteeing a deterministic,
    input-independent prediction -- every glioma test image is then a reliable
    'wrong glioma' misclassification without depending on random init luck."""
    class_to_idx = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    model = build_resnet50_model(num_classes=len(class_to_idx), pretrained=False)
    head = model.fc[1]
    head.weight.data.zero_()
    head.bias.data = torch.tensor([0.0, 10.0, 0.0, 0.0])

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "data_config": asdict(tiny_data_config),
            "train_config": asdict(TrainConfig(pretrained=False)),
        },
        ckpt_path,
    )
    return ckpt_path


def test_import_has_no_side_effects():
    sys.modules.pop("make_fig4", None)
    module = __import__("make_fig4")
    assert hasattr(module, "main")


def test_main_saves_misclassified_glioma_overlays(tiny_data_config: DataConfig, tmp_path: Path):
    checkpoint = _checkpoint_that_always_predicts_meningioma(tmp_path, tiny_data_config)
    out_dir = tmp_path / "fig4_out"

    import make_fig4

    saved = make_fig4.main(
        checkpoint=checkpoint,
        data_config=tiny_data_config,
        out_dir=out_dir,
        max_saved=3,
    )

    assert saved == 3  # 6 glioma test images available, all mispredicted, capped at max_saved
    written = list(out_dir.glob("wrong_glioma_pred_meningioma_*.jpg"))
    assert len(written) == 3


def test_main_respects_max_saved_cap(tiny_data_config: DataConfig, tmp_path: Path):
    checkpoint = _checkpoint_that_always_predicts_meningioma(tmp_path, tiny_data_config)
    out_dir = tmp_path / "fig4_out_capped"

    import make_fig4

    saved = make_fig4.main(
        checkpoint=checkpoint,
        data_config=tiny_data_config,
        out_dir=out_dir,
        max_saved=1,
    )

    assert saved == 1
    assert len(list(out_dir.glob("*.jpg"))) == 1

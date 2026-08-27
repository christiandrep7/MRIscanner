from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.gradio_app as gradio_app
from src.config import DataConfig, TrainConfig
from src.model import ARCHITECTURES, build_model


def test_available_architectures_lists_only_existing_checkpoints(all_fake_checkpoints: dict[str, Path]):
    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    found = gradio_app.available_architectures(checkpoint_dir)
    assert set(found) == set(all_fake_checkpoints.keys())


def test_available_architectures_empty_when_nothing_trained(tmp_path: Path):
    assert gradio_app.available_architectures(tmp_path) == []


def test_preprocess_shape():
    img = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8))
    tensor = gradio_app.preprocess(img)
    assert tensor.shape == (1, 3, 224, 224)


def test_predict_with_missing_checkpoint_reports_not_trained(tmp_path: Path):
    from src.model import get_device

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    label, overlay = gradio_app.predict_with("resnet50", img, get_device(), checkpoint_dir=tmp_path)
    assert label == "Not trained yet"
    assert overlay is None


def test_predict_with_trained_checkpoint_returns_label_and_overlay(fake_checkpoint: Path):
    from src.model import get_device

    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    )
    label, overlay = gradio_app.predict_with(
        "resnet50", img, get_device(), checkpoint_dir=fake_checkpoint.parent
    )
    assert "(" in label and "%" in label  # "<class> (xx.xx%)"
    assert overlay.shape[-1] == 3


def test_predict_all_handles_none_image():
    outputs = gradio_app.predict_all(None, list(ARCHITECTURES))
    assert outputs[0] == "No image uploaded."
    per_model = outputs[1:]
    assert len(per_model) == len(ARCHITECTURES) * 2
    labels = per_model[0::2]
    overlays = per_model[1::2]
    assert all(label == "No image received" for label in labels)
    assert all(overlay is None for overlay in overlays)


def test_predict_all_skips_unselected_architectures(all_fake_checkpoints: dict[str, Path], monkeypatch):
    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", checkpoint_dir)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    outputs = gradio_app.predict_all(img, ["resnet50"])
    labels = outputs[1:][0::2]

    for architecture, label in zip(ARCHITECTURES, labels):
        if architecture == "resnet50":
            assert label != "(not selected)"
        else:
            assert label == "(not selected)"


def test_predict_all_runs_selected_models_concurrently(all_fake_checkpoints: dict[str, Path], monkeypatch):
    """Regression test for the sequential-loop version: 3 models each pausing
    briefly should overlap in wall-clock time, not add up serially."""
    import threading
    import time

    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", checkpoint_dir)

    starts: dict[str, float] = {}
    lock = threading.Lock()

    def fake_predict_with(architecture, image, device, checkpoint_dir=None):
        with lock:
            starts[architecture] = time.monotonic()
        time.sleep(0.3)
        return f"{architecture}-label", None

    monkeypatch.setattr(gradio_app, "predict_with", fake_predict_with)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    t0 = time.monotonic()
    gradio_app.predict_all(img, list(ARCHITECTURES))
    elapsed = time.monotonic() - t0

    # Serial would be >= 0.9s (3 x 0.3s); concurrent should land well under that.
    assert elapsed < 0.7, f"expected concurrent execution, took {elapsed:.2f}s"
    assert len(starts) == 3
    spread = max(starts.values()) - min(starts.values())
    assert spread < 0.2, f"model start times were {spread:.2f}s apart -- looks sequential, not concurrent"


def test_run_benchmark_returns_table_and_chart(
    tiny_data_config, all_fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch
):
    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(gradio_app, "BENCHMARK_OUTPUT_DIR", tmp_path / "benchmark_out")
    monkeypatch.setattr(gradio_app, "DataConfig", lambda: tiny_data_config)

    rows, chart = gradio_app.run_benchmark(list(ARCHITECTURES))

    assert len(rows) == 3
    assert chart is not None
    assert Path(chart).exists()


def test_run_benchmark_respects_selected_subset(
    tiny_data_config, all_fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch
):
    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(gradio_app, "BENCHMARK_OUTPUT_DIR", tmp_path / "benchmark_out")
    monkeypatch.setattr(gradio_app, "DataConfig", lambda: tiny_data_config)

    rows, _chart = gradio_app.run_benchmark(["resnet50"])

    assert len(rows) == 1
    assert rows[0][0] == "resnet50"


def test_run_benchmark_empty_selection_returns_nothing(tiny_data_config, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(gradio_app, "DataConfig", lambda: tiny_data_config)

    rows, chart = gradio_app.run_benchmark([])

    assert rows == []
    assert chart is None


def test_pick_random_test_image_returns_pil_image_and_true_label(tiny_dataset: Path):
    from tests.conftest import CLASS_NAMES

    image, true_label = gradio_app.pick_random_test_image(tiny_dataset)
    assert isinstance(image, Image.Image)
    assert true_label in CLASS_NAMES


def test_pick_random_test_image_none_when_no_testing_dir(tmp_path: Path):
    assert gradio_app.pick_random_test_image(tmp_path) == (None, None)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("glioma (98.21%)", "glioma"),
        ("notumor (50.00%)", "notumor"),
        ("(not selected)", None),
        ("Not trained yet", None),
        ("No image received", None),
    ],
)
def test_label_only(text: str, expected: str | None):
    assert gradio_app._label_only(text) == expected


def test_consensus_message_empty_when_fewer_than_two_predictions():
    assert gradio_app._consensus_message({"resnet50": "glioma"}) == ""
    assert gradio_app._consensus_message({"resnet50": None, "vgg16": None}) == ""


def test_consensus_message_agreement():
    msg = gradio_app._consensus_message({"resnet50": "glioma", "vgg16": "glioma"})
    assert "agree" in msg
    assert "glioma" in msg


def test_consensus_message_disagreement():
    msg = gradio_app._consensus_message({"resnet50": "glioma", "vgg16": "meningioma"})
    assert "disagree" in msg
    assert "resnet50" in msg and "vgg16" in msg


def test_ground_truth_message_empty_without_true_label():
    assert gradio_app._ground_truth_message(None, {"resnet50": "glioma"}) == ""


def test_ground_truth_message_empty_when_nothing_predicted():
    assert gradio_app._ground_truth_message("glioma", {"resnet50": None}) == ""


def test_ground_truth_message_all_correct():
    msg = gradio_app._ground_truth_message("glioma", {"resnet50": "glioma", "vgg16": "glioma"})
    assert "True label" in msg and "glioma" in msg
    assert "2/2 correct" in msg
    assert "❌" not in msg


def test_ground_truth_message_some_wrong():
    msg = gradio_app._ground_truth_message(
        "glioma", {"resnet50": "glioma", "efficientnet_b0": "meningioma", "vgg16": "glioma"}
    )
    assert "2/3 correct" in msg
    assert "resnet50 ✅" in msg
    assert "efficientnet_b0 ❌ (said meningioma)" in msg
    assert "vgg16 ✅" in msg
    assert "glioma" in msg and "meningioma" in msg


def _force_constant_prediction_checkpoint(
    tmp_path: Path, architecture: str, class_idx: int, tiny_data_config: DataConfig
) -> Path:
    """Zeroing the final classifier layer's weight makes its output equal the bias
    alone (Dropout is disabled in eval mode) -- a deterministic, input-independent
    prediction, so multi-model agreement/disagreement can be tested reliably."""
    class_to_idx = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    model = build_model(architecture, num_classes=4, pretrained=False)

    if architecture == "vgg16":
        head = model.classifier[6]
    else:
        head = model.classifier[1] if architecture == "efficientnet_b0" else model.fc[1]
    head.weight.data.zero_()
    bias = torch.zeros(4)
    bias[class_idx] = 10.0
    head.bias.data = bias

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{architecture}_best_model.pth"
    torch.save(
        {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "data_config": asdict(tiny_data_config),
            "train_config": asdict(TrainConfig(architecture=architecture, pretrained=False)),
        },
        ckpt_path,
    )
    return ckpt_path


def test_predict_all_reports_agreement_when_models_match(tiny_data_config: DataConfig, tmp_path: Path, monkeypatch):
    _force_constant_prediction_checkpoint(tmp_path, "resnet50", class_idx=0, tiny_data_config=tiny_data_config)
    _force_constant_prediction_checkpoint(
        tmp_path, "efficientnet_b0", class_idx=0, tiny_data_config=tiny_data_config
    )
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    outputs = gradio_app.predict_all(img, ["resnet50", "efficientnet_b0"])

    assert "agree" in outputs[0]
    assert "glioma" in outputs[0]


def test_predict_all_reports_disagreement_when_models_differ(
    tiny_data_config: DataConfig, tmp_path: Path, monkeypatch
):
    _force_constant_prediction_checkpoint(tmp_path, "resnet50", class_idx=0, tiny_data_config=tiny_data_config)
    _force_constant_prediction_checkpoint(
        tmp_path, "efficientnet_b0", class_idx=1, tiny_data_config=tiny_data_config
    )
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    outputs = gradio_app.predict_all(img, ["resnet50", "efficientnet_b0"])

    assert "disagree" in outputs[0]
    assert "glioma" in outputs[0] and "meningioma" in outputs[0]


def test_predict_all_marks_correctness_against_true_label(
    tiny_data_config: DataConfig, tmp_path: Path, monkeypatch
):
    _force_constant_prediction_checkpoint(tmp_path, "resnet50", class_idx=0, tiny_data_config=tiny_data_config)
    _force_constant_prediction_checkpoint(
        tmp_path, "efficientnet_b0", class_idx=1, tiny_data_config=tiny_data_config
    )
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    outputs = gradio_app.predict_all(img, ["resnet50", "efficientnet_b0"], true_label="glioma")

    assert "True label" in outputs[0] and "glioma" in outputs[0]
    assert "1/2 correct" in outputs[0]
    assert "resnet50 ✅" in outputs[0]
    assert "efficientnet_b0 ❌ (said meningioma)" in outputs[0]
    assert "disagree" in outputs[0]  # ground truth line and consensus line coexist


def test_predict_all_no_ground_truth_line_for_manual_upload(fake_checkpoint: Path, monkeypatch):
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", fake_checkpoint.parent)
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

    outputs = gradio_app.predict_all(img, ["resnet50"])  # true_label defaults to None

    assert "True label" not in outputs[0]

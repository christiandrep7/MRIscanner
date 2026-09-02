from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.benchmark import benchmark_models, default_checkpoint_specs, save_benchmark_report
from src.config import DataConfig
from src.model import ARCHITECTURES


def _make_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_default_checkpoint_specs_covers_all_architectures(tmp_path: Path):
    specs = default_checkpoint_specs(tmp_path)
    assert [architecture for architecture, _ in specs] == list(ARCHITECTURES)
    for architecture, path in specs:
        assert path == tmp_path / f"{architecture}_best_model.pth"


def test_benchmark_models_skips_missing_checkpoint_gracefully(
    tiny_data_config: DataConfig, fake_checkpoint_factory, tmp_path: Path
):
    """Only 2 of 3 architectures trained -- the third should be reported as
    unavailable, not crash the whole comparison."""
    resnet_ckpt = fake_checkpoint_factory("resnet50")
    efficientnet_ckpt = fake_checkpoint_factory("efficientnet_b0")
    missing_vgg_ckpt = resnet_ckpt.parent / "vgg16_best_model.pth"

    specs = [
        ("resnet50", resnet_ckpt),
        ("efficientnet_b0", efficientnet_ckpt),
        ("vgg16", missing_vgg_ckpt),
    ]

    results = benchmark_models(specs, tiny_data_config, output_dir=tmp_path / "benchmark_out")

    by_arch = {r.architecture: r for r in results}
    assert by_arch["resnet50"].available is True
    assert by_arch["resnet50"].accuracy is not None
    assert 0.0 <= by_arch["resnet50"].accuracy <= 1.0
    assert by_arch["resnet50"].num_params is not None
    assert by_arch["resnet50"].ms_per_image is not None

    assert by_arch["efficientnet_b0"].available is True

    assert by_arch["vgg16"].available is False
    assert by_arch["vgg16"].error is not None
    assert by_arch["vgg16"].accuracy is None


def test_benchmark_models_writes_per_model_artifacts(
    tiny_data_config: DataConfig, all_fake_checkpoints: dict[str, Path], tmp_path: Path
):
    specs = [(architecture, path) for architecture, path in all_fake_checkpoints.items()]
    output_dir = tmp_path / "benchmark_out"

    results = benchmark_models(specs, tiny_data_config, output_dir=output_dir)

    assert all(r.available for r in results)
    for architecture in all_fake_checkpoints:
        assert (output_dir / architecture / "confusion_matrix.png").exists()
        assert (output_dir / architecture / "classification_report.json").exists()


def test_benchmark_models_scores_partial_class_test_set_in_checkpoint_label_space(
    tiny_data_config: DataConfig, all_fake_checkpoints: dict[str, Path], tmp_path: Path
):
    """A cross-distribution held-out set with only 2 of the 4 trained classes
    physically present (exactly the shape of BraTS/IXI's ExternalTesting) --
    real bug caught running this for real: building the eval model with
    num_classes taken from this folder's own (2-class) ImageFolder discovery,
    instead of the checkpoint's actual (4-class) trained output size, crashed
    every architecture with a state_dict shape mismatch."""
    external_root = tiny_data_config.data_root / "ExternalOnly"
    for class_name in ["glioma", "notumor"]:
        for i in range(4):
            _make_image(external_root / class_name / f"img_{i}.jpg", seed=i)

    external_cfg = replace(tiny_data_config, test_dir_name="ExternalOnly")
    specs = [(architecture, path) for architecture, path in all_fake_checkpoints.items()]

    results = benchmark_models(specs, external_cfg, output_dir=tmp_path / "benchmark_external")

    assert all(r.available for r in results), [r.error for r in results if not r.available]
    for architecture in all_fake_checkpoints:
        report = json.loads(
            (tmp_path / "benchmark_external" / architecture / "classification_report.json").read_text()
        )
        # All 4 trained classes appear (even meningioma/pituitary, zero support
        # here) -- proves this scored in the checkpoint's full label space, not
        # a fresh 0/1 renumbering of just the 2 classes physically present.
        assert "meningioma" in report
        assert "pituitary" in report


def test_save_benchmark_report_writes_json_and_chart(
    tiny_data_config: DataConfig, all_fake_checkpoints: dict[str, Path], tmp_path: Path
):
    specs = [(architecture, path) for architecture, path in all_fake_checkpoints.items()]
    output_dir = tmp_path / "benchmark_out"
    results = benchmark_models(specs, tiny_data_config, output_dir=output_dir)

    report_path = save_benchmark_report(results, output_dir)

    assert report_path == output_dir / "results.json"
    summary = json.loads(report_path.read_text())
    assert len(summary) == 3
    assert {row["architecture"] for row in summary} == set(ARCHITECTURES)
    assert all(row["available"] for row in summary)
    assert (output_dir / "accuracy_comparison.png").exists()


def test_save_benchmark_report_skips_chart_when_nothing_available(tmp_path: Path):
    from src.benchmark import BenchmarkResult

    results = [
        BenchmarkResult(
            architecture="resnet50",
            checkpoint_path=tmp_path / "missing.pth",
            available=False,
            error="checkpoint not found -- train this architecture first",
        )
    ]
    output_dir = tmp_path / "benchmark_out"

    report_path = save_benchmark_report(results, output_dir)

    assert report_path.exists()
    assert not (output_dir / "accuracy_comparison.png").exists()

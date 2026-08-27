"""End-to-end smoke tests: drive the real CLI entry points via subprocess in a
temp working directory, exactly as a user would invoke them from a terminal.
Uses the tiny synthetic dataset instead of the real ~7000-image Kaggle dataset
so the full pipeline can be proven wired-correctly without a GPU or a download.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tests.conftest import CLASS_NAMES, _populate_split

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture
def e2e_workspace(tmp_path: Path) -> Path:
    """A temp CWD with data/Training + data/Testing populated, mirroring what a
    real user's project directory looks like right after download_mri_dataset.py."""
    data_root = tmp_path / "data"
    _populate_split(data_root, "Training", per_class=10, seed_offset=0)
    _populate_split(data_root, "Testing", per_class=6, seed_offset=10_000)
    return tmp_path


def test_train_cli_end_to_end(e2e_workspace: Path):
    result = _run(
        [
            "-m",
            "src.train",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--no-pretrained",
        ],
        cwd=e2e_workspace,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (e2e_workspace / "checkpoints" / "resnet50_best_model.pth").exists()
    assert (e2e_workspace / "outputs" / "resnet50_metrics_history.csv").exists()


def test_evaluate_cli_end_to_end(e2e_workspace: Path):
    train_result = _run(
        ["-m", "src.train", "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--no-pretrained"],
        cwd=e2e_workspace,
    )
    assert train_result.returncode == 0, train_result.stdout + train_result.stderr

    eval_result = _run(
        [
            "-m",
            "src.evaluate",
            "--checkpoint",
            "checkpoints/resnet50_best_model.pth",
            "--output-dir",
            "outputs/eval",
            "--batch-size",
            "2",
        ],
        cwd=e2e_workspace,
    )

    assert eval_result.returncode == 0, eval_result.stdout + eval_result.stderr
    eval_dir = e2e_workspace / "outputs" / "eval"
    assert (eval_dir / "confusion_matrix.png").exists()
    report = json.loads((eval_dir / "classification_report.json").read_text())
    assert "accuracy" in report
    gradcam_files = list((eval_dir / "gradcam").glob("*.jpg"))
    assert len(gradcam_files) == 5


def test_secret_dedupe_eval_cli_end_to_end(e2e_workspace: Path):
    train_result = _run(
        ["-m", "src.train", "--epochs", "1", "--batch-size", "2", "--num-workers", "0", "--no-pretrained"],
        cwd=e2e_workspace,
    )
    assert train_result.returncode == 0, train_result.stdout + train_result.stderr

    secret_dir = e2e_workspace / "data" / "secret" / "Testing" / "glioma"
    secret_dir.mkdir(parents=True)
    original = next((e2e_workspace / "data" / "Training" / "glioma").glob("*.jpg"))
    shutil.copy2(original, secret_dir / "duplicate.jpg")
    Image.new("RGB", (32, 32), color=(5, 5, 5)).save(secret_dir / "novel.jpg")

    result = _run(["-m", "src.secret_dedupe_eval", "--hamming", "0"], cwd=e2e_workspace)

    assert result.returncode == 0, result.stdout + result.stderr
    stats = json.loads((e2e_workspace / "outputs" / "secret_dedupe_stats.json").read_text())
    assert stats["md5_dup"] == 1
    assert stats["kept"] == 1
    assert (e2e_workspace / "data" / "secret" / "Testing_deduped" / "glioma" / "novel.jpg").exists()
    assert (e2e_workspace / "outputs" / "secret_eval_deduped" / "classification_report.json").exists()


def test_secret_overlap_report_cli_end_to_end(e2e_workspace: Path):
    secret_dir = e2e_workspace / "data" / "secret" / "Testing" / "glioma"
    secret_dir.mkdir(parents=True)
    original = next((e2e_workspace / "data" / "Training" / "glioma").glob("*.jpg"))
    shutil.copy2(original, secret_dir / "identical.jpg")

    with Image.open(original) as im:
        pixels = im.convert("RGB")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", "reposted")
    pixels.save(secret_dir / "reposted.png", pnginfo=meta)

    result = _run(["-m", "src.secret_overlap_report", "--no-eval"], cwd=e2e_workspace)

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(
        (e2e_workspace / "outputs" / "secret_overlap_sha256_summary.json").read_text()
    )
    assert summary["sha256_overlap"] == 1
    assert summary["phash_exact_diff_sha_candidates"] == 1
    assert (e2e_workspace / "data" / "secret" / "Testing_sha256_only" / "glioma" / "reposted.png").exists()
    assert (e2e_workspace / "outputs" / "phash_overlap_pairs_sample.csv").exists()


def test_full_pipeline_train_then_evaluate_then_dedupe_then_overlap(e2e_workspace: Path):
    """The complete real-world workflow from the README, back to back."""
    train_result = _run(
        ["-m", "src.train", "--epochs", "2", "--batch-size", "2", "--num-workers", "0", "--no-pretrained"],
        cwd=e2e_workspace,
    )
    assert train_result.returncode == 0, train_result.stdout + train_result.stderr

    eval_result = _run(
        ["-m", "src.evaluate", "--checkpoint", "checkpoints/resnet50_best_model.pth", "--batch-size", "2"],
        cwd=e2e_workspace,
    )
    assert eval_result.returncode == 0, eval_result.stdout + eval_result.stderr

    secret_dir = e2e_workspace / "data" / "secret" / "Testing"
    for class_name in CLASS_NAMES:
        (secret_dir / class_name).mkdir(parents=True)
        Image.new("RGB", (32, 32), color=(3, 3, 3)).save(secret_dir / class_name / "novel.jpg")

    dedupe_result = _run(["-m", "src.secret_dedupe_eval", "--hamming", "0"], cwd=e2e_workspace)
    assert dedupe_result.returncode == 0, dedupe_result.stdout + dedupe_result.stderr

    overlap_result = _run(["-m", "src.secret_overlap_report", "--no-eval"], cwd=e2e_workspace)
    assert overlap_result.returncode == 0, overlap_result.stdout + overlap_result.stderr

    assert (e2e_workspace / "checkpoints" / "resnet50_best_model.pth").exists()
    assert (e2e_workspace / "outputs" / "eval" / "classification_report.json").exists()
    assert (e2e_workspace / "outputs" / "secret_dedupe_stats.json").exists()
    assert (e2e_workspace / "outputs" / "secret_overlap_sha256_summary.json").exists()


def test_train_all_then_benchmark_cli_end_to_end(e2e_workspace: Path):
    """The 3-model comparison workflow: train every architecture, then benchmark
    all of them on data/Testing -- scans none of them ever trained on."""
    train_all_result = _run(
        [
            "-m",
            "src.train_all",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--no-pretrained",
        ],
        cwd=e2e_workspace,
    )
    assert train_all_result.returncode == 0, train_all_result.stdout + train_all_result.stderr

    for architecture in ("resnet50", "efficientnet_b0", "vgg16"):
        assert (e2e_workspace / "checkpoints" / f"{architecture}_best_model.pth").exists()
        assert (e2e_workspace / "outputs" / f"{architecture}_metrics_history.csv").exists()

    benchmark_result = _run(["-m", "src.benchmark"], cwd=e2e_workspace)
    assert benchmark_result.returncode == 0, benchmark_result.stdout + benchmark_result.stderr

    report = json.loads((e2e_workspace / "outputs" / "benchmark" / "results.json").read_text())
    assert {row["architecture"] for row in report} == {"resnet50", "efficientnet_b0", "vgg16"}
    assert all(row["available"] for row in report)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in report)
    assert (e2e_workspace / "outputs" / "benchmark" / "accuracy_comparison.png").exists()

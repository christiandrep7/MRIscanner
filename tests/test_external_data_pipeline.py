"""Tests for run_external_data_pipeline.py's own logic: stage-marker gating
and the dedupe/merge/split step. Deliberately does NOT test the download
stages (real network calls to Kaggle) -- those are thin API-glue, same as
this repo's other download_*.py scripts, which aren't unit-tested either."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import run_external_data_pipeline as pipeline
from src.external_data import find_patient_overlap


def _make_png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)).save(path)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_reclaim_disk_after_brats_download_keeps_only_t1ce_and_seg(tmp_path: Path):
    patient_dir = tmp_path / "BraTS20_Training_001"
    _touch(patient_dir / "BraTS20_Training_001_flair.nii")
    _touch(patient_dir / "BraTS20_Training_001_t1.nii")
    _touch(patient_dir / "BraTS20_Training_001_t1ce.nii")  # must survive -- filename contains "_t1."
    _touch(patient_dir / "BraTS20_Training_001_t2.nii")
    _touch(patient_dir / "BraTS20_Training_001_seg.nii")
    _touch(tmp_path / "brats20-dataset-training-validation.zip")

    pipeline._reclaim_disk_after_brats_download(tmp_path)

    remaining = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert remaining == {"BraTS20_Training_001_t1ce.nii", "BraTS20_Training_001_seg.nii"}


def test_reclaim_disk_after_ixi_download_removes_only_the_zip(tmp_path: Path):
    _touch(tmp_path / "IXI002-Guys-0828-T1.nii")
    _touch(tmp_path / "ixi-healthy-brain-mri-t1-t2.zip")

    pipeline._reclaim_disk_after_ixi_download(tmp_path)

    remaining = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert remaining == {"IXI002-Guys-0828-T1.nii"}


def test_stage_marker_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "pipeline_state")

    assert not pipeline._stage_done("some_stage")
    pipeline._mark_stage_done("some_stage")
    assert pipeline._stage_done("some_stage")


def test_stage_dedupe_and_merge_splits_and_skips_when_done(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "pipeline_state")
    monkeypatch.setattr(pipeline, "EXTERNAL_SLICES_DIR", tmp_path / "external_raw_slices")
    monkeypatch.setattr(pipeline, "EXTERNAL_DEDUPED_DIR", tmp_path / "external_deduped")
    monkeypatch.setattr(pipeline, "EXTERNAL_TEST_FRACTION", 0.2)

    # 10 patients x 3 slices, none overlapping the (empty) existing dataset.
    for patient in range(10):
        for sl in range(3):
            _make_png(
                tmp_path / "external_raw_slices" / "glioma" / f"BraTS20_Training_{patient:03d}_slice{sl}.png",
                seed=patient * 10 + sl,
            )

    pipeline.stage_dedupe_and_merge()

    train_images = list((tmp_path / "data" / "Training" / "glioma").glob("*.png"))
    test_images = list((tmp_path / "data" / "ExternalTesting" / "glioma").glob("*.png"))
    assert len(train_images) + len(test_images) == 30
    assert len(test_images) == 6  # 20% of 10 patients, all 3 slices of each
    assert find_patient_overlap(tmp_path / "data" / "Training", tmp_path / "data" / "ExternalTesting") == set()
    assert pipeline._stage_done("dedupe_and_merge")

    # Re-running should skip (no-op) rather than re-splitting already-moved files.
    pipeline.stage_dedupe_and_merge()
    assert len(list((tmp_path / "data" / "Training" / "glioma").glob("*.png"))) == len(train_images)


def test_stage_dedupe_and_merge_drops_exact_duplicate_of_existing_image(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "pipeline_state")
    monkeypatch.setattr(pipeline, "EXTERNAL_SLICES_DIR", tmp_path / "external_raw_slices")
    monkeypatch.setattr(pipeline, "EXTERNAL_DEDUPED_DIR", tmp_path / "external_deduped")

    _make_png(tmp_path / "data" / "Training" / "glioma" / "BraTS20_Training_900_slice1.png", seed=42)
    _make_png(tmp_path / "external_raw_slices" / "glioma" / "BraTS20_Training_901_slice1.png", seed=42)  # byte-identical
    _make_png(tmp_path / "external_raw_slices" / "glioma" / "BraTS20_Training_902_slice1.png", seed=99)

    pipeline.stage_dedupe_and_merge()

    merged_names = {
        p.name
        for p in (tmp_path / "data" / "Training" / "glioma").glob("*.png")
    } | {
        p.name
        for p in (tmp_path / "data" / "ExternalTesting" / "glioma").glob("*.png")
    }
    assert "BraTS20_Training_901_slice1.png" not in merged_names
    assert "BraTS20_Training_902_slice1.png" in merged_names


def test_stage_remove_train_test_duplicates_quarantines_train_side_only(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "pipeline_state")

    rng = np.random.default_rng(5)
    base = Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8)).resize((128, 128), Image.BICUBIC).convert("RGB")
    other = Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8)).resize((128, 128), Image.BICUBIC).convert("RGB")
    (tmp_path / "data" / "Training" / "glioma").mkdir(parents=True)
    (tmp_path / "data" / "Testing" / "glioma").mkdir(parents=True)
    base.save(tmp_path / "data" / "Training" / "glioma" / "twin.png")
    other.save(tmp_path / "data" / "Training" / "glioma" / "keep.png")
    base.resize((90, 90)).save(tmp_path / "data" / "Testing" / "glioma" / "twin_of_train.jpg")

    pipeline.stage_remove_train_test_duplicates()

    assert not (tmp_path / "data" / "Training" / "glioma" / "twin.png").exists()
    assert (tmp_path / "data" / "Training" / "glioma" / "keep.png").exists()
    assert (tmp_path / "data" / "Testing" / "glioma" / "twin_of_train.jpg").exists()
    assert (tmp_path / "data" / "_quarantined_duplicates_of_test" / "Testing" / "glioma" / "twin.png").exists()
    assert pipeline._stage_done("remove_train_test_duplicates")

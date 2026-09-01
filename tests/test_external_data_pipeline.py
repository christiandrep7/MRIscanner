"""Tests for run_external_data_pipeline.py's own logic: stage-marker gating
and the dedupe/merge/split step. Deliberately does NOT test the download
stages (real network calls to Kaggle) -- those are thin API-glue, same as
this repo's other download_*.py scripts, which aren't unit-tested either."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import run_external_data_pipeline as pipeline


def _make_png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)).save(path)


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

    # 10 brand-new glioma slices, none overlapping the (empty) existing dataset.
    for i in range(10):
        _make_png(tmp_path / "external_raw_slices" / "glioma" / f"img{i}.png", seed=i)

    pipeline.stage_dedupe_and_merge()

    train_images = list((tmp_path / "data" / "Training" / "glioma").glob("*.png"))
    test_images = list((tmp_path / "data" / "ExternalTesting" / "glioma").glob("*.png"))
    assert len(train_images) + len(test_images) == 10
    assert len(test_images) == 2  # 20% of 10
    assert pipeline._stage_done("dedupe_and_merge")

    # Re-running should skip (no-op) rather than re-splitting already-moved files.
    pipeline.stage_dedupe_and_merge()
    assert len(list((tmp_path / "data" / "Training" / "glioma").glob("*.png"))) == len(train_images)


def test_stage_dedupe_and_merge_drops_exact_duplicate_of_existing_image(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "pipeline_state")
    monkeypatch.setattr(pipeline, "EXTERNAL_SLICES_DIR", tmp_path / "external_raw_slices")
    monkeypatch.setattr(pipeline, "EXTERNAL_DEDUPED_DIR", tmp_path / "external_deduped")

    _make_png(tmp_path / "data" / "Training" / "glioma" / "existing.png", seed=42)
    _make_png(tmp_path / "external_raw_slices" / "glioma" / "duplicate.png", seed=42)  # byte-identical
    _make_png(tmp_path / "external_raw_slices" / "glioma" / "genuinely_new.png", seed=99)

    pipeline.stage_dedupe_and_merge()

    merged_names = {
        p.name
        for p in (tmp_path / "data" / "Training" / "glioma").glob("*.png")
    } | {
        p.name
        for p in (tmp_path / "data" / "ExternalTesting" / "glioma").glob("*.png")
    }
    assert "duplicate.png" not in merged_names
    assert "genuinely_new.png" in merged_names

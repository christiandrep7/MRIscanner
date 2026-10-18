"""End-to-end check of the data assembly + leakage gate on a synthetic tree (no network)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import run_continue_training as rct
from src.frozen_benchmark import LeakageError

CLASSES = rct.CLASSES


def _smooth(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8)).resize((96, 96), Image.BICUBIC).convert("RGB")


def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.fixture
def world(tmp_path: Path, monkeypatch) -> rct.Paths:
    monkeypatch.chdir(tmp_path)
    seed = 0

    def nxt() -> int:
        nonlocal seed
        seed += 1
        return seed

    for cls in CLASSES:  # Kaggle-style Training / Testing (no patient ids)
        for i in range(6):
            _save(_smooth(nxt()), tmp_path / "data" / "Training" / cls / f"tr_{i}.jpg")
        for i in range(8):
            _save(_smooth(nxt()), tmp_path / "data" / "Testing" / cls / f"te_{i}.jpg")
    # one Testing image is a re-saved copy of a Training image (a leaky twin)
    twin_src = _smooth(9999)
    _save(twin_src, tmp_path / "data" / "Training" / "glioma" / "tr_twin.jpg")
    _save(twin_src.resize((70, 70)), tmp_path / "data" / "Testing" / "glioma" / "te_twin.jpg")
    # old ExternalTesting patients (seen by the starting weights) must be pooled into Training
    for s in range(2):
        _save(_smooth(nxt()), tmp_path / "data" / "ExternalTesting" / "glioma" / f"BraTS20_Training_900_slice{s}.png")

    st = tmp_path / "external_new" / "slices"
    for p in range(10):
        for s in range(3):
            _save(_smooth(nxt()), st / "brats_val" / "glioma" / f"BraTS20_Validation_{p:03d}_slice{s}.png")
            _save(_smooth(nxt()), st / "oasis" / "notumor" / f"OAS1_{p:04d}_scan1_slice{s}.png")
    for cls in CLASSES:
        for i in range(12):
            _save(_smooth(nxt()), st / "pmram" / cls / f"PMRAM_{cls}_{i:04d}.png")
    return rct.Paths(tmp_path / "work")


def _all_files(root: Path) -> set[str]:
    return {p.name for p in root.rglob("*") if p.is_file()}


def test_assembly_pools_old_external_test_excludes_twin_and_passes_gate(world: rct.Paths):
    rct.stage_assemble_data(world)
    d = world.data

    assert "BraTS20_Training_900_slice0.png" in _all_files(d / "Training")  # seen before -> training
    assert "te_twin.jpg" not in _all_files(d / "Testing") | _all_files(d / "Validation")
    assert "te_twin.jpg" in _all_files(d / "_excluded_leaky_eval")
    assert _all_files(d / "PMRAMTesting") and _all_files(d / "ExternalTesting") and _all_files(d / "Validation")

    rct.stage_gate(world)  # must not raise
    assert world.manifest.exists()


def test_every_new_patient_lands_in_exactly_one_split(world: rct.Paths):
    from src.external_data import patient_id_of

    rct.stage_assemble_data(world)
    owners: dict[str, set[str]] = {}
    for split in ("Training", "Validation", "ExternalTesting"):
        for f in (world.data / split).rglob("*.png"):
            pid = patient_id_of(f)
            if pid and not pid.startswith("BraTS20_Training"):
                owners.setdefault(pid, set()).add(split)
    assert len(owners) == 20 and all(len(v) == 1 for v in owners.values())


def test_gate_aborts_if_an_image_is_planted_in_both_train_and_test(world: rct.Paths):
    rct.stage_assemble_data(world)
    victim = next((world.data / "Testing" / "notumor").glob("*.jpg"))
    _save(Image.open(victim).copy(), world.data / "Training" / "notumor" / "planted.jpg")
    with pytest.raises(LeakageError):
        rct.stage_gate(world)


def test_gate_aborts_if_frozen_test_set_is_edited_after_freezing(world: rct.Paths):
    rct.stage_assemble_data(world)
    rct.stage_gate(world)
    next((world.data / "Testing" / "glioma").glob("*.jpg")).unlink()
    with pytest.raises(LeakageError, match="Frozen benchmark changed"):
        rct.stage_gate(world)


def test_done_marker_is_ignored_when_its_output_is_gone(world: rct.Paths):
    rct._mark(world, "assemble_data")
    assert rct._done(world, "assemble_data") is True
    assert rct._done(world, "assemble_data", world.data / "Validation") is False
    assert not (world.state / "assemble_data.done").exists()


def test_pmram_images_duplicating_kaggle_scans_are_dropped_not_tested_on(world: rct.Paths):
    kaggle_scan = next((world.data / "Training" / "pituitary").glob("tr_0.jpg"))
    _save(Image.open(kaggle_scan).resize((64, 64)), world.staging / "slices" / "pmram" / "pituitary" / "PMRAM_pituitary_dup.png")
    rct.stage_assemble_data(world)
    assert "PMRAM_pituitary_dup.png" in _all_files(world.data / "_excluded_pmram_duplicates")
    for split in ("Training", "Validation", "PMRAMTesting"):
        assert "PMRAM_pituitary_dup.png" not in _all_files(world.data / split)


def test_phash_lookalikes_with_different_patient_ids_do_not_trip_the_gate(tmp_path: Path):
    from src.frozen_benchmark import assert_no_leakage

    img = _smooth(5)
    _save(img, tmp_path / "Training" / "glioma" / "BraTS20_Training_001_slice5.png")
    _save(img.resize((90, 90)), tmp_path / "Testing" / "glioma" / "BraTS20_Validation_002_slice5.png")  # same look, different patient
    assert assert_no_leakage(tmp_path / "Training", [tmp_path / "Testing"]).clean

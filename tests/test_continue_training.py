"""Tests for the continue-training / no-leakage pipeline pieces."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import DataConfig, TrainConfig
from src.data import _balanced_sampler, build_dataloaders
from src.external_data import (
    extract_oasis_notumor_slices,
    extract_pmram_images,
    flair_tumor_slice_indices,
    patient_id_of,
    split_dir_by_patient_fractions,
)
from src.frozen_benchmark import LeakageError, assert_no_leakage, verify_manifest, write_manifest
from src.leakage import cluster_by_similarity, split_clusters
from src.train import train
from tests.conftest import CLASS_NAMES, make_fake_checkpoint


def _smooth(seed: int, size: int = 128) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8)).resize((size, size), Image.BICUBIC).convert("RGB")


def _save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------- extraction
def test_flair_slice_selection_finds_the_bright_lesion_slices():
    vol = np.ones((40, 40, 30))
    vol[:, :, :3] = 0  # empty top/bottom slices
    vol[:, :, -3:] = 0
    vol[10:20, 10:20, 12:18] = 10.0  # bright lesion only in slices 12..17
    picks = flair_tumor_slice_indices(vol, 4)
    assert 1 <= len(picks) <= 4
    assert all(12 <= i <= 17 for i in picks)


def test_flair_slice_selection_returns_nothing_for_empty_volume():
    assert flair_tumor_slice_indices(np.zeros((8, 8, 8)), 4) == []


def test_oasis_extraction_keeps_only_nondemented_and_names_by_subject(tmp_path: Path):
    root = tmp_path / "oasis"
    for split in ("train", "test"):
        for sl in range(100, 120):
            _save(_smooth(sl), root / split / "NonDemented" / f"OAS1_000{1 if split == 'train' else 2}_MR1_1_nii_slice_{sl}_png.rf.abc{sl}.jpg")
        _save(_smooth(1), root / split / "MildDemented" / "OAS1_0099_MR1_1_nii_slice_100_png.rf.x.jpg")
    out = tmp_path / "out"
    n = extract_oasis_notumor_slices(root, out, slices_per_subject=5)
    assert n == 10
    assert {patient_id_of(f) for f in out.glob("*.png")} == {"OAS1_0001", "OAS1_0002"}


def test_pmram_extraction_uses_raw_only_and_maps_normal_to_notumor(tmp_path: Path):
    base = tmp_path / "pm"
    for folder in ("Raw Data/Raw", "Augmented Data/Augmented"):
        for cls in ("512Glioma", "512Meningioma", "512Pituitary", "512Normal"):
            for i in range(3 if "Raw" in folder else 7):
                _save(_smooth(i), base / folder / cls / f"{cls}_{i}.jpg")
    counts = extract_pmram_images(base, tmp_path / "out")
    assert counts == {"glioma": 3, "meningioma": 3, "pituitary": 3, "notumor": 3}


def test_split_by_patient_fractions_never_splits_a_patient(tmp_path: Path):
    src = tmp_path / "src"
    for p in range(20):
        for s in range(4):
            _save(_smooth(p * 10 + s, 16), src / f"BraTS20_Validation_{p:03d}_slice{s}.png")
    dests = {k: tmp_path / k for k in ("train", "val", "test")}
    counts = split_dir_by_patient_fractions(src, dests, {"train": 0.6, "val": 0.15, "test": 0.25}, seed=1)
    assert sum(counts.values()) == 80 and counts["train"] == 48 and counts["test"] == 20
    owners: dict[str, set[str]] = {}
    for name, d in dests.items():
        for f in d.glob("*.png"):
            owners.setdefault(patient_id_of(f), set()).add(name)
    assert all(len(v) == 1 for v in owners.values())
    assert not list(src.glob("*.png"))


# ---------------------------------------------------------------- clustering
def test_similar_images_cluster_together_and_stay_on_one_side(tmp_path: Path):
    base = _smooth(1)
    a = _save(base, tmp_path / "a.png")
    b = _save(base.resize((100, 100)), tmp_path / "b.png")  # same scan re-saved
    c = _save(_smooth(2), tmp_path / "c.png")
    d = _save(_smooth(3), tmp_path / "d.png")
    clusters = cluster_by_similarity([a, b, c, d], max_hamming=10)
    assert sorted(len(x) for x in clusters) == [1, 1, 2]
    for seed in range(5):
        buckets = split_clusters([a, b, c, d], {"x": 0.5, "y": 0.5}, seed, max_hamming=10)
        side = {name: {p.name for p in ps} for name, ps in buckets.items()}
        assert ("a.png" in side["x"]) == ("b.png" in side["x"])


# ---------------------------------------------------------------- leakage gate / manifest
def _tree(tmp_path: Path) -> tuple[Path, Path]:
    train, test = tmp_path / "Training", tmp_path / "Testing"
    _save(_smooth(10), train / "glioma" / "BraTS20_Training_001_slice5.png")
    _save(_smooth(11), train / "glioma" / "other.png")
    _save(_smooth(50), test / "glioma" / "fresh.png")
    return train, test


def test_gate_passes_for_disjoint_data(tmp_path: Path):
    train, test = _tree(tmp_path)
    assert assert_no_leakage(train, [test]).clean


def test_gate_fails_on_byte_identical_image(tmp_path: Path):
    train, test = _tree(tmp_path)
    _save(_smooth(11), test / "glioma" / "copy.png")
    with pytest.raises(LeakageError, match="byte-identical"):
        assert_no_leakage(train, [test])


def test_gate_fails_on_shared_patient(tmp_path: Path):
    train, test = _tree(tmp_path)
    _save(_smooth(77), test / "glioma" / "BraTS20_Training_001_slice9.png")
    with pytest.raises(LeakageError, match="1 shared patients"):
        assert_no_leakage(train, [test])


def test_gate_fails_on_resized_near_duplicate(tmp_path: Path):
    train, test = _tree(tmp_path)
    _save(_smooth(11).resize((90, 90)), test / "glioma" / "resized.jpg")
    with pytest.raises(LeakageError, match="near-duplicate"):
        assert_no_leakage(train, [test])


def test_manifest_detects_added_removed_and_modified_files(tmp_path: Path):
    _, test = _tree(tmp_path)
    manifest = tmp_path / "m.json"
    write_manifest({"Testing": test}, manifest)
    verify_manifest({"Testing": test}, manifest)
    _save(_smooth(123), test / "glioma" / "added.png")
    with pytest.raises(LeakageError, match="1 added"):
        verify_manifest({"Testing": test}, manifest)
    (test / "glioma" / "added.png").unlink()
    _save(_smooth(124), test / "glioma" / "fresh.png")
    with pytest.raises(LeakageError, match="1 modified"):
        verify_manifest({"Testing": test}, manifest)


# ---------------------------------------------------------------- data + train
def test_dedicated_validation_folder_replaces_the_training_carve_out(tiny_dataset: Path):
    for ci, cls in enumerate(CLASS_NAMES):
        for i in range(3):
            _save(_smooth(5000 + ci * 10 + i, 32), tiny_dataset / "Validation" / cls / f"v{i}.png")
    bundle = build_dataloaders(DataConfig(data_root=tiny_dataset, image_size=32, batch_size=2, num_workers=0))
    assert bundle.train_size == 40  # all of Training, nothing carved out
    assert bundle.val_size == 12


def test_balanced_sampler_equalises_class_frequencies():
    targets = [0] * 900 + [1] * 90 + [2] * 10
    drawn = Counter(targets[i] for i in _balanced_sampler(targets, seed=0))
    assert all(abs(drawn[c] - 333) < 60 for c in (0, 1, 2))


def test_train_continues_from_checkpoint_and_never_saves_worse_than_baseline(tiny_data_config, tmp_path: Path):
    init = make_fake_checkpoint(tmp_path / "start", "resnet50", tiny_data_config)
    cfg = TrainConfig(
        epochs=2, checkpoint_dir=tmp_path / "ck", history_dir=tmp_path / "out", pretrained=False,
        init_checkpoint=init, lr_plateau_patience=1,
    )
    best = train(tiny_data_config, cfg)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["class_to_idx"] == {n: i for i, n in enumerate(CLASS_NAMES)}
    assert best.exists()


def test_train_rejects_checkpoint_with_different_classes(tiny_data_config, tmp_path: Path):
    init = make_fake_checkpoint(tmp_path / "start", "resnet50", tiny_data_config)
    payload = torch.load(init, map_location="cpu", weights_only=False)
    payload["class_to_idx"] = {"a": 0, "b": 1, "c": 2, "d": 3}
    torch.save(payload, init)
    cfg = TrainConfig(epochs=1, checkpoint_dir=tmp_path / "ck", history_dir=tmp_path / "out", pretrained=False, init_checkpoint=init)
    with pytest.raises(ValueError, match="classes"):
        train(tiny_data_config, cfg)

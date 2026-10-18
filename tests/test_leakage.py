from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.leakage import clean_test_files, find_test_images_with_train_duplicates, quarantine_train_duplicates_of_test


def _smooth_image(seed: int, size: int = 128) -> Image.Image:
    """Low-frequency random pattern: stable perceptual hash, distinct per seed."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(8, 8), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BICUBIC).convert("RGB")


def _save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _setup(tmp_path: Path):
    train, test = tmp_path / "Training", tmp_path / "Testing"
    a = _smooth_image(1)
    _save(a, train / "glioma" / "a.png")                                   # original in train
    _save(_smooth_image(2), train / "glioma" / "b.png")                    # unrelated train image
    _save(a.resize((100, 100)), test / "glioma" / "a_resized_copy.jpg")    # same scan, re-encoded + resized
    _save(_smooth_image(3), test / "glioma" / "c_unique.png")              # genuinely unseen
    return train, test


def test_finds_resized_reencoded_copy_but_not_unique_image(tmp_path: Path):
    train, test = _setup(tmp_path)
    dups = find_test_images_with_train_duplicates(train, test, max_hamming=5)
    assert {p.name: [q.name for q in qs] for p, qs in dups.items()} == {"a_resized_copy.jpg": ["a.png"]}


def test_clean_test_files_excludes_duplicated_images(tmp_path: Path):
    train, test = _setup(tmp_path)
    assert {p.name for p in clean_test_files(train, test, max_hamming=5)} == {"c_unique.png"}


def test_quarantine_moves_train_side_and_leaves_test_set_intact(tmp_path: Path):
    train, test = _setup(tmp_path)
    q = tmp_path / "quarantine"
    moved = quarantine_train_duplicates_of_test(train, test, q, max_hamming=5)
    assert moved == 1
    assert not (train / "glioma" / "a.png").exists()
    assert (train / "glioma" / "b.png").exists()
    assert (q / "glioma" / "a.png").exists()
    assert len(list(test.rglob("*.*"))) == 2  # test set untouched
    assert find_test_images_with_train_duplicates(train, test, max_hamming=5) == {}


def test_quarantine_is_a_noop_when_nothing_overlaps(tmp_path: Path):
    train, test = tmp_path / "Training", tmp_path / "Testing"
    _save(_smooth_image(10), train / "notumor" / "x.png")
    _save(_smooth_image(11), test / "notumor" / "y.png")
    assert quarantine_train_duplicates_of_test(train, test, tmp_path / "q", max_hamming=5) == 0

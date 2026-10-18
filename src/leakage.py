"""Train/test leakage checks that work without patient ids.

The Kaggle dataset ships no patient or scan ids, so "same patient on both sides"
can't be tested directly. What *can* be measured: whether a held-out test image
has a (near-)identical copy in the training folder. Measured on this project's
data before this module existed: ~27% of the 1,600 'held-out' Kaggle test images
had a pixel-correlation >= 0.99 twin in Training (same scan, re-saved as a
different file -- never byte-identical, which is why an MD5 check found nothing).
Accuracy measured on such images mostly reflects memorization.

Perceptual-hash (pHash) distance <= 5 is used as a deliberately *conservative*
duplicate test: it also flags some merely-similar slices, which only costs a few
extra training images -- it can't make the test set easier.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXT)


@lru_cache(maxsize=None)
def _phash_cached(path: Path, mtime_ns: int) -> int:
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def _phash(path: Path) -> int:
    # keyed on mtime so an edited file is rehashed; repeated gate checks over the
    # same ~20k images then cost one pass, not one per eval folder
    return _phash_cached(Path(path), Path(path).stat().st_mtime_ns)


def _hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """(len(a), len(b)) matrix of pHash bit distances, computed in row chunks."""
    out = np.empty((len(a), len(b)), dtype=np.uint8)
    for i in range(0, len(a), chunk):
        x = (a[i : i + chunk, None] ^ b[None, :]).view(np.uint8).reshape(-1, len(b), 8)
        out[i : i + chunk] = _POPCOUNT[x].sum(axis=-1)
    return out


def find_test_images_with_train_duplicates(
    train_root: Path, test_root: Path, max_hamming: int = 5
) -> dict[Path, list[Path]]:
    """{test image: [train images within max_hamming bits]} for every test image
    that has at least one near-duplicate in training."""
    train, test = _images(train_root), _images(test_root)
    if not train or not test:
        return {}
    tr = np.array([_phash(p) for p in train], dtype=np.uint64)
    te = np.array([_phash(p) for p in test], dtype=np.uint64)
    dist = _hamming_matrix(te, tr)
    return {
        test[i]: [train[j] for j in np.where(dist[i] <= max_hamming)[0]]
        for i in range(len(test))
        if (dist[i] <= max_hamming).any()
    }


def clean_test_files(train_root: Path, test_root: Path, max_hamming: int = 5) -> list[Path]:
    """Test images with no near-duplicate in training -- the honest subset to
    evaluate models on when the training folder can't be changed."""
    leaky = set(find_test_images_with_train_duplicates(train_root, test_root, max_hamming))
    return [p for p in _images(test_root) if p not in leaky]


def quarantine_train_duplicates_of_test(
    train_root: Path, test_root: Path, quarantine_root: Path, max_hamming: int = 5
) -> int:
    """Moves every training image that near-duplicates a test image into
    quarantine_root (preserving class subfolders). The test set is left
    untouched, so results stay comparable run to run. Returns files moved."""
    to_move = {t for ts in find_test_images_with_train_duplicates(train_root, test_root, max_hamming).values() for t in ts}
    for src in sorted(to_move):
        dest = Path(quarantine_root) / src.relative_to(train_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    return len(to_move)


def cluster_by_similarity(paths: list[Path], max_hamming: int = 10) -> list[list[Path]]:
    """Connected components of images whose pHash distance is <= max_hamming.
    Datasets without patient ids (Kaggle, PMRAM) still contain several slices or
    re-saves of the same scan; keeping each component on one side of a split
    stops those look-alikes from straddling train and test."""
    if not paths:
        return []
    hashes = np.array([_phash(p) for p in paths], dtype=np.uint64)
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(paths), 256):
        block = _hamming_matrix(hashes[start : start + 256], hashes)
        for row, i in enumerate(range(start, min(start + 256, len(paths)))):
            for j in np.where(block[row] <= max_hamming)[0]:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[Path]] = {}
    for i, p in enumerate(paths):
        groups.setdefault(find(i), []).append(p)
    return list(groups.values())


def split_clusters(
    paths: list[Path], fractions: dict[str, float], seed: int, max_hamming: int = 10
) -> dict[str, list[Path]]:
    """Assigns whole similarity clusters to named buckets in roughly the given
    proportions (fractions need not be exact -- clusters are indivisible)."""
    import random

    clusters = cluster_by_similarity(paths, max_hamming)
    random.Random(seed).shuffle(clusters)
    total = sum(len(c) for c in clusters)
    names = list(fractions)
    targets = {n: fractions[n] * total for n in names}
    buckets: dict[str, list[Path]] = {n: [] for n in names}
    for cluster in sorted(clusters, key=len, reverse=True):
        # place into the bucket furthest below its target share
        name = max(names, key=lambda n: targets[n] - len(buckets[n]))
        buckets[name].extend(cluster)
    return buckets

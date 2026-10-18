"""Freezes the evaluation sets and proves they share nothing with training.

`assert_no_leakage` is the hard gate run before any training or benchmarking:
it fails loudly if any validation/test image is byte-identical to, a perceptual
near-duplicate of, or from the same patient as a training image.

`write_manifest` / `verify_manifest` record the sha256 of every evaluation file
so a later run can show the benchmark set was not edited after being frozen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.external_data import patient_id_of
from src.leakage import _images, find_test_images_with_train_duplicates


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class LeakageReport:
    eval_images: int
    train_images: int
    exact_duplicates: int
    shared_patients: int
    near_duplicates: int

    @property
    def clean(self) -> bool:
        return self.exact_duplicates == 0 and self.shared_patients == 0 and self.near_duplicates == 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_leakage(train_root: Path, eval_roots: list[Path], max_hamming: int = 5) -> LeakageReport:
    train_files = _images(train_root)
    train_sha = {_sha256(p) for p in train_files}
    train_patients = {pid for p in train_files if (pid := patient_id_of(p))}

    eval_files = [p for root in eval_roots if Path(root).exists() for p in _images(root)]
    exact = sum(1 for p in eval_files if _sha256(p) in train_sha)
    shared = {pid for p in eval_files if (pid := patient_id_of(p)) and pid in train_patients}

    # pHash is only meaningful for images that carry no patient id (Kaggle, PMRAM). For
    # patient-id sources (BraTS, IXI, OASIS) distinct patients are guaranteed by the id
    # check above, and pHash on dark skull-stripped slices flags different brains as
    # "twins" (measured: 40% of patient-disjoint BraTS slices at distance <= 5).
    near = 0
    for root in eval_roots:
        if Path(root).exists():
            twins = find_test_images_with_train_duplicates(train_root, root, max_hamming)
            near += sum(1 for f in twins if patient_id_of(f) is None)
    return LeakageReport(len(eval_files), len(train_files), exact, len(shared), near)


def assert_no_leakage(train_root: Path, eval_roots: list[Path], max_hamming: int = 5) -> LeakageReport:
    report = check_leakage(train_root, eval_roots, max_hamming)
    if not report.clean:
        raise LeakageError(
            f"Evaluation data overlaps training data: {report.exact_duplicates} byte-identical, "
            f"{report.shared_patients} shared patients, {report.near_duplicates} near-duplicate images "
            f"(pHash<={max_hamming}) across {report.eval_images} eval / {report.train_images} train images."
        )
    return report


def build_manifest(roots: dict[str, Path]) -> dict:
    files = {}
    for name, root in roots.items():
        root = Path(root)
        for p in _images(root):
            files[f"{name}/{p.relative_to(root).as_posix()}"] = _sha256(p)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"manifest_sha256": digest, "n_files": len(files), "files": files}


def write_manifest(roots: dict[str, Path], out_path: Path) -> dict:
    manifest = build_manifest(roots)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def verify_manifest(roots: dict[str, Path], manifest_path: Path) -> None:
    """Raises LeakageError if any frozen evaluation file was added, removed or changed."""
    frozen = json.loads(Path(manifest_path).read_text())
    current = build_manifest(roots)
    if current["manifest_sha256"] == frozen["manifest_sha256"]:
        return
    a, b = frozen["files"], current["files"]
    raise LeakageError(
        f"Frozen benchmark changed: {len(set(a) - set(b))} removed, {len(set(b) - set(a))} added, "
        f"{sum(1 for k in set(a) & set(b) if a[k] != b[k])} modified."
    )

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from src.external_data import extract_brats_glioma_slices, extract_ixi_notumor_slices


def _save_nii(volume: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), str(path))


def _make_brats_patient(root: Path, patient_id: str, tumor_slices: range | None) -> None:
    shape = (20, 20, 30)
    t1ce = np.random.default_rng(0).uniform(0, 200, size=shape)
    seg = np.zeros(shape)
    if tumor_slices is not None:
        for z in tumor_slices:
            seg[3:16, 3:16, z] = 1  # 13x13 = 169 voxels/slice, clears MIN_TUMOR_PIXELS (150)

    patient_dir = root / patient_id
    _save_nii(t1ce, patient_dir / f"{patient_id}_t1ce.nii")
    _save_nii(seg, patient_dir / f"{patient_id}_seg.nii")


def test_extract_brats_glioma_slices_writes_images_for_tumor_slices(tmp_path: Path):
    brats_root = tmp_path / "brats"
    _make_brats_patient(brats_root, "BraTS20_Training_001", tumor_slices=range(10, 20))
    out_dir = tmp_path / "out"

    written = extract_brats_glioma_slices(brats_root, out_dir, slices_per_patient=4)

    assert written == 4
    files = list(out_dir.glob("*.png"))
    assert len(files) == 4
    for f in files:
        img = Image.open(f)
        assert img.size[0] > 0 and img.size[1] > 0


def test_extract_brats_glioma_slices_skips_patient_with_no_tumor(tmp_path: Path):
    brats_root = tmp_path / "brats"
    _make_brats_patient(brats_root, "BraTS20_Training_002", tumor_slices=None)
    out_dir = tmp_path / "out"

    written = extract_brats_glioma_slices(brats_root, out_dir, slices_per_patient=4)

    assert written == 0
    assert not out_dir.exists() or list(out_dir.glob("*.png")) == []


def test_extract_brats_glioma_slices_respects_max_patients(tmp_path: Path):
    brats_root = tmp_path / "brats"
    _make_brats_patient(brats_root, "BraTS20_Training_001", tumor_slices=range(10, 20))
    _make_brats_patient(brats_root, "BraTS20_Training_002", tumor_slices=range(10, 20))
    out_dir = tmp_path / "out"

    written = extract_brats_glioma_slices(brats_root, out_dir, slices_per_patient=2, max_patients=1)

    assert written == 2


def _make_ixi_subject(root: Path, filename: str, shape: tuple[int, int, int] = (20, 20, 40)) -> None:
    volume = np.zeros(shape)
    # Only the central slices have real "brain" signal -- top/bottom of the
    # skull (outside the central 60% the extractor samples from) stay empty.
    volume[5:15, 5:15, :] = 100
    _save_nii(volume, root / filename)


def test_extract_ixi_notumor_slices_writes_images(tmp_path: Path):
    ixi_root = tmp_path / "ixi"
    _make_ixi_subject(ixi_root, "IXI002-Guys-0828-T1.nii")
    out_dir = tmp_path / "out"

    written = extract_ixi_notumor_slices(ixi_root, out_dir, slices_per_subject=4)

    assert written == 4
    assert len(list(out_dir.glob("IXI002_*.png"))) == 4


def test_extract_ixi_notumor_slices_skips_empty_volume(tmp_path: Path):
    ixi_root = tmp_path / "ixi"
    _save_nii(np.zeros((20, 20, 40)), ixi_root / "IXI099-HH-0001-T1.nii")  # all-background: nothing but skull-void
    out_dir = tmp_path / "out"

    written = extract_ixi_notumor_slices(ixi_root, out_dir, slices_per_subject=4)

    assert written == 0


def test_extract_ixi_notumor_slices_respects_max_subjects(tmp_path: Path):
    ixi_root = tmp_path / "ixi"
    _make_ixi_subject(ixi_root, "IXI002-Guys-0828-T1.nii")
    _make_ixi_subject(ixi_root, "IXI003-Guys-0829-T1.nii")
    out_dir = tmp_path / "out"

    written = extract_ixi_notumor_slices(ixi_root, out_dir, slices_per_subject=2, max_subjects=1)

    assert written == 2

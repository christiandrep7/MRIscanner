"""Extracts 2D slices from external NIfTI-format MRI datasets (BraTS, IXI) and
saves them as PNGs matching this project's ImageFolder-per-class layout.

Why: the Kaggle "Brain Tumor MRI Dataset" this project trains on is (per its own
documented provenance, and confirmed by external research) a combination of just
3 source collections from a narrow set of scanners/institutions -- models trained
on it generalize poorly to real-world scans from elsewhere. BraTS (glioma, 19
real institutions) and IXI (healthy controls, 3 real hospitals) are genuinely
independent sources, not folded into the Kaggle set, used here to add real
distributional diversity to exactly the 2 classes where it's obtainable without
requiring a personal data-use agreement (meningioma/pituitary alternatives all
either are the Kaggle set's own source data, or require TCIA's restricted
license agreement -- a real per-person legal step, not something automatable).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

# A slice with fewer than this fraction of non-background pixels is mostly
# empty (top/bottom of the skull) -- not a useful training image either way.
MIN_BRAIN_FRACTION = 0.05
# BraTS seg mask: any nonzero label (1=necrotic/non-enhancing, 2=edema,
# 4=enhancing tumor) means "glioma is visible in this slice". A slice needs at
# least this many tumor pixels to count as a clear, unambiguous example (a
# handful of stray voxels isn't a usable "this looks like glioma" image).
MIN_TUMOR_PIXELS = 150


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Percentile-clip then min-max scale to 0-255, matching the contrast range
    of a typical windowed clinical MRI slice (raw NIfTI intensities have no
    fixed range, unlike an already-windowed JPEG)."""
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _brain_fraction(slice_2d: np.ndarray) -> float:
    threshold = slice_2d.max() * 0.05 if slice_2d.max() > 0 else 0
    return float((slice_2d > threshold).mean())


def _load_axial_volume(nii_path: Path) -> np.ndarray:
    """Reorients to the closest canonical (RAS+) orientation before returning
    the array, so axis 2 is always the axial (inferior-superior) plane
    regardless of how the source file stored its axes -- NIfTI orientation
    conventions aren't guaranteed consistent across datasets/scanners, and
    slicing a raw un-reoriented volume can silently give a sagittal or coronal
    slice instead (caught by visually spot-checking a real extracted slice)."""
    img = nib.as_closest_canonical(nib.load(str(nii_path)))
    return img.get_fdata()


def _save_axial_slice(volume: np.ndarray, axial_index: int, out_path: Path) -> None:
    slice_2d = np.rot90(volume[:, :, axial_index])
    img = Image.fromarray(_normalize_to_uint8(slice_2d), mode="L").convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def extract_brats_glioma_slices(
    brats_root: Path,
    out_dir: Path,
    slices_per_patient: int = 6,
    max_patients: int | None = None,
) -> int:
    """BraTS patient folders each hold *_t1ce.nii (T1 contrast-enhanced, the
    modality closest to this project's existing images) and *_seg.nii (tumor
    segmentation mask). Picks slices_per_patient axial slices per patient where
    the mask shows a clear tumor, evenly spread through the tumor's extent
    (not just the single biggest slice) for some within-patient variety.
    Returns the number of images written."""
    patient_dirs = sorted(p for p in brats_root.rglob("BraTS20_Training_*") if p.is_dir())
    if max_patients is not None:
        patient_dirs = patient_dirs[:max_patients]

    written = 0
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        t1ce_path = patient_dir / f"{patient_id}_t1ce.nii"
        seg_path = patient_dir / f"{patient_id}_seg.nii"
        if not t1ce_path.exists() or not seg_path.exists():
            continue

        t1ce = _load_axial_volume(t1ce_path)
        seg = _load_axial_volume(seg_path)

        tumor_pixel_counts = (seg > 0).sum(axis=(0, 1))
        qualifying = np.where(tumor_pixel_counts >= MIN_TUMOR_PIXELS)[0]
        if len(qualifying) == 0:
            continue

        # Evenly spaced picks across the qualifying range, not just the top-N
        # by tumor size -- avoids every saved slice looking near-identical.
        pick_indices = np.linspace(0, len(qualifying) - 1, min(slices_per_patient, len(qualifying)))
        chosen = sorted({qualifying[int(round(i))] for i in pick_indices})

        for slice_idx in chosen:
            out_path = out_dir / f"{patient_id}_slice{slice_idx}.png"
            _save_axial_slice(t1ce, slice_idx, out_path)
            written += 1

    return written


def extract_ixi_notumor_slices(
    ixi_root: Path,
    out_dir: Path,
    slices_per_subject: int = 6,
    max_subjects: int | None = None,
) -> int:
    """IXI subjects are all healthy controls -- every slice is a legitimate
    "notumor" example. Picks slices_per_subject evenly spaced axial slices from
    the central 60% of the volume (skips the mostly-empty top/bottom of the
    skull) per subject. Returns the number of images written."""
    nii_files = sorted(ixi_root.rglob("*.nii")) + sorted(ixi_root.rglob("*.nii.gz"))
    if max_subjects is not None:
        nii_files = nii_files[:max_subjects]

    written = 0
    for nii_path in nii_files:
        subject_id = nii_path.stem.replace(".nii", "").split("-")[0]
        try:
            volume = _load_axial_volume(nii_path)
        except Exception:  # noqa: BLE001 -- a corrupt/unreadable download shouldn't kill the whole run
            continue
        if volume.ndim != 3:
            continue

        depth = volume.shape[2]
        lo, hi = int(depth * 0.2), int(depth * 0.8)
        candidate_indices = list(range(lo, hi))
        if not candidate_indices:
            continue
        pick_indices = np.linspace(0, len(candidate_indices) - 1, min(slices_per_subject, len(candidate_indices)))
        chosen = sorted({candidate_indices[int(round(i))] for i in pick_indices})

        for slice_idx in chosen:
            slice_2d = np.rot90(volume[:, :, slice_idx])
            if _brain_fraction(slice_2d) < MIN_BRAIN_FRACTION:
                continue
            out_path = out_dir / f"{subject_id}_slice{slice_idx}.png"
            _save_axial_slice(volume, slice_idx, out_path)
            written += 1

    return written

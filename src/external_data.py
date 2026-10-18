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

import random
import re
from pathlib import Path

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
    import nibabel as nib  # lazy: only extraction needs it; src.data imports this module on servers without nibabel

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


def flair_tumor_slice_indices(flair: np.ndarray, k: int) -> list[int]:
    """Axial slices that very likely contain tumor, chosen from a FLAIR volume
    alone (BraTS *validation* patients ship no segmentation mask). A slice scores
    by how many brain voxels are > mean + 2 SD of the volume's brain intensities --
    FLAIR makes edema/tumor bright. Candidates scoring >= 70% of the best slice are
    spread evenly to k picks. Measured on 12 BraTS training patients with real
    masks: 96/96 picked slices (k=8) contained >= 150 tumor pixels."""
    brain = flair > 0
    if not brain.any():
        return []
    mu, sd = flair[brain].mean(), flair[brain].std()
    score = ((flair > mu + 2.0 * sd) & brain).sum(axis=(0, 1)).astype(float)
    brain_area = brain.sum(axis=(0, 1))
    score[brain_area < 0.3 * brain_area.max()] = 0  # skip mostly-empty top/bottom slices
    if score.max() <= 0:
        return []
    candidates = np.where(score >= 0.7 * score.max())[0]
    picks = np.linspace(0, len(candidates) - 1, min(k, len(candidates)))
    return sorted({int(candidates[int(round(i))]) for i in picks})


def extract_brats_validation_glioma_slices(brats_root: Path, out_dir: Path, slices_per_patient: int = 8) -> int:
    """BraTS 2020 validation patients (never used by the original training run):
    t1ce for the image, FLAIR to pick tumor slices (no seg mask is released)."""
    written = 0
    for patient_dir in sorted(p for p in brats_root.rglob("BraTS20_Validation_*") if p.is_dir()):
        pid = patient_dir.name
        t1ce_path, flair_path = patient_dir / f"{pid}_t1ce.nii", patient_dir / f"{pid}_flair.nii"
        if not t1ce_path.exists() or not flair_path.exists():
            continue
        t1ce, flair = _load_axial_volume(t1ce_path), _load_axial_volume(flair_path)
        for idx in flair_tumor_slice_indices(flair, slices_per_patient):
            _save_axial_slice(t1ce, idx, out_dir / f"{pid}_slice{idx}.png")
            written += 1
    return written


_OASIS_RE = re.compile(r"^(OAS1_\d+)_MR\d_(\d+)_nii_slice_(\d+)_png")


def extract_oasis_notumor_slices(oasis_root: Path, out_dir: Path, slices_per_subject: int = 8) -> int:
    """OASIS-1 healthy controls (CDR 0, folder NonDemented) from the Roboflow export
    of OASIS. These are skull-stripped T1 slices -- exactly what BraTS glioma slices
    look like, which is the point: healthy skull-stripped examples stop the model
    from learning 'no skull = glioma'. Picks evenly spaced slices per subject."""
    by_subject: dict[str, list[Path]] = {}
    for split in ("train", "test"):
        for f in sorted((oasis_root / split / "NonDemented").glob("*.jpg")):
            m = _OASIS_RE.match(f.name)
            if m:
                by_subject.setdefault(m.group(1), []).append(f)
    written = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for subject, files in sorted(by_subject.items()):
        files.sort(key=lambda f: (int(_OASIS_RE.match(f.name).group(2)), int(_OASIS_RE.match(f.name).group(3))))
        picks = np.linspace(0, len(files) - 1, min(slices_per_subject, len(files)))
        for i in sorted({int(round(x)) for x in picks}):
            scan, sl = _OASIS_RE.match(files[i].name).group(2, 3)
            with Image.open(files[i]) as im:
                im.convert("RGB").save(out_dir / f"{subject}_scan{scan}_slice{sl}.png")
            written += 1
    return written


PMRAM_CLASS_DIRS = {"512Glioma": "glioma", "512Meningioma": "meningioma", "512Pituitary": "pituitary", "512Normal": "notumor"}


def extract_pmram_images(pmram_root: Path, out_root: Path) -> dict[str, int]:
    """PMRAM (Bangladeshi hospital MRI, 4 classes). Only the *Raw* images are used:
    the 'Augmented' folder is made from the raw ones, so mixing both would put
    copies of the same scan on both sides of any split. No patient ids exist, so
    callers must split with src.leakage.split_clusters. Returns counts per class."""
    counts: dict[str, int] = {}
    for src_name, class_name in PMRAM_CLASS_DIRS.items():
        files = sorted(d for d in pmram_root.rglob(src_name) if d.is_dir() and d.parent.name == "Raw")
        n = 0
        for d in files:
            for f in sorted(d.glob("*.jpg")):
                dest = out_root / class_name / f"PMRAM_{class_name}_{n:04d}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(f) as im:
                    im.convert("RGB").save(dest)
                n += 1
        counts[class_name] = n
    return counts


# Every extracted slice is named <patient>_slice<N>.png, so the patient a slice
# came from can always be recovered from the filename alone.
_PATIENT_RE = re.compile(r"^(BraTS20_(?:Training|Validation)_\d+|IXI\d+|OAS1_\d+)(?:_scan\d+)?_slice\d+\.png$")


def patient_id_of(path: Path | str) -> str | None:
    """Patient/subject id embedded in an extracted slice's filename, or None for
    files that don't carry one (e.g. the Kaggle dataset, which ships no ids)."""
    m = _PATIENT_RE.match(Path(path).name)
    return m.group(1) if m else None


def split_by_patient(
    src_dir: Path, train_dest: Path, test_dest: Path, test_fraction: float, seed: int
) -> tuple[int, int]:
    """Moves slices from src_dir into train_dest/test_dest, assigning *whole
    patients* to one side. A slice-level shuffle (what this pipeline did first)
    lets the same brain's neighbouring slices land on both sides, so the test
    set partly measures recognizing patients already seen in training -- every
    one of the original external test slices had a same-patient slice in
    training. Returns (n_train_slices, n_test_slices)."""
    by_patient: dict[str, list[Path]] = {}
    for img in sorted(src_dir.glob("*.png")):
        pid = patient_id_of(img)
        if pid is None:
            raise ValueError(f"Can't tell which patient {img.name} belongs to; refusing to split by slice.")
        by_patient.setdefault(pid, []).append(img)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n_test_patients = max(1, round(len(patients) * test_fraction)) if len(patients) > 1 else 0
    test_patients = set(patients[:n_test_patients])

    train_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)
    n_train = n_test = 0
    for pid, imgs in by_patient.items():
        dest, is_test = (test_dest, True) if pid in test_patients else (train_dest, False)
        for img in imgs:
            img.rename(dest / img.name)
        if is_test:
            n_test += len(imgs)
        else:
            n_train += len(imgs)
    return n_train, n_test


def find_patient_overlap(dir_a: Path, dir_b: Path) -> set[str]:
    """Patient ids that have slices under both directories (searched recursively).
    Should be empty for any train/test pair."""
    ids = lambda root: {pid for f in Path(root).rglob("*.png") if (pid := patient_id_of(f))}
    return ids(dir_a) & ids(dir_b)


def split_dir_by_patient_fractions(
    src_dir: Path, dests: dict[str, Path], fractions: dict[str, float], seed: int
) -> dict[str, int]:
    """Moves slices from src_dir into the named destination folders, giving each
    *whole patient* to one bucket (generalises split_by_patient to N buckets,
    e.g. train / val / test). Returns the slice count per bucket."""
    by_patient: dict[str, list[Path]] = {}
    for img in sorted(src_dir.glob("*.png")):
        pid = patient_id_of(img)
        if pid is None:
            raise ValueError(f"Can't tell which patient {img.name} belongs to; refusing to split by slice.")
        by_patient.setdefault(pid, []).append(img)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    names = list(fractions)
    assignment: dict[str, str] = {}
    start = 0
    for i, name in enumerate(names):
        count = len(patients) - start if i == len(names) - 1 else round(len(patients) * fractions[name])
        for pid in patients[start : start + count]:
            assignment[pid] = name
        start += count

    counts = {name: 0 for name in names}
    for pid, imgs in by_patient.items():
        dest = dests[assignment[pid]]
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            img.rename(dest / img.name)
        counts[assignment[pid]] += len(imgs)
    return counts

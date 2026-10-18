"""Continue training the published checkpoints on more, genuinely unseen data,
with an evaluation set that provably shares nothing with training.

What it does (every stage is resumable -- rerun the same command after any
interruption, and keep --work-dir on Google Drive when running on Colab):

 1. fetch the starting checkpoints (GitHub release) -- never retrained from scratch
 2. add new data the starting weights never saw:
      BraTS 2020 *validation* patients (glioma, FLAIR-selected slices)
      OASIS-1 healthy subjects (notumor; skull-stripped, so BraTS-vs-IXI skull shortcut is broken)
      PMRAM Bangladeshi hospital MRI (all 4 classes, raw images only)
 3. re-split everything so nothing evaluated is also trained on:
      old ExternalTesting patients were seen by the starting weights -> pooled back into Training
      Kaggle Testing images with a near-duplicate in Training -> excluded
      remaining Kaggle test -> Validation / Testing by similarity cluster
      new patients (BraTS val, OASIS) -> Training / Validation / ExternalTesting by patient
      PMRAM -> Training / Validation / PMRAMTesting by similarity cluster
 4. hard leakage gate + frozen sha256 manifest of the benchmark (aborts on any overlap)
 5. fine-tune each architecture from its checkpoint (balanced sampling, LR plateau schedule,
    early stopping, up to --epochs epochs) and keep a model only if it beats the starting weights
 6. benchmark on Testing and ExternalTesting, re-verify the manifest, write outputs/benchmark_v2
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from dataclasses import replace
from pathlib import Path

from src.config import DataConfig, TrainConfig
from src.external_data import (
    extract_brats_validation_glioma_slices,
    extract_oasis_notumor_slices,
    extract_pmram_images,
    patient_id_of,
    split_dir_by_patient_fractions,
)
from src.frozen_benchmark import assert_no_leakage, verify_manifest, write_manifest
from src.leakage import find_test_images_with_train_duplicates, split_clusters
from src.model import ARCHITECTURES

BRATS_DATASET = "awsaf49/brats20-dataset-training-validation"
OASIS_DATASET = "shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset"
PMRAM_DATASET = "orvile/pmram-bangladeshi-brain-cancer-mri-dataset"
BRATS_VAL_PREFIX = "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
# Reported benchmark sets (frozen). Validation is separate and only drives early stopping.
TEST_DIRS = ("Testing", "ExternalTesting", "PMRAMTesting")

PATIENT_FRACTIONS = {"train": 0.60, "val": 0.15, "test": 0.25}
PMRAM_FRACTIONS = {"train": 0.50, "val": 0.15, "test": 0.35}
KAGGLE_CLEAN_FRACTIONS = {"val": 0.30, "test": 0.70}
SPLIT_SEED = 42
GATE_MAX_HAMMING = 5
CLUSTER_MAX_HAMMING = 10  # wider than the gate, so look-alikes always land on the same side


class Paths:
    def __init__(self, work_dir: Path):
        self.work = work_dir
        self.state = work_dir / "pipeline_state_v2"
        self.data = Path("data")
        self.staging = Path("external_new")
        self.start_ckpts = work_dir / "checkpoints_start"
        self.ckpts = work_dir / "checkpoints_v2"
        self.outputs = work_dir / "outputs_v2"
        self.manifest = work_dir / "benchmark_v2" / "frozen_manifest.json"


def _done(paths: Paths, name: str, artifact: Path | None = None) -> bool:
    """A stage is done only if its marker exists AND its output still exists. The
    state dir may live on Google Drive while data/ sits on Colab's ephemeral disk,
    so after a runtime reset the marker survives but the files don't."""
    marker = paths.state / f"{name}.done"
    if marker.exists() and artifact is not None and not artifact.exists():
        marker.unlink()
        return False
    return marker.exists()


def _mark(paths: Paths, name: str) -> None:
    paths.state.mkdir(parents=True, exist_ok=True)
    (paths.state / f"{name}.done").touch()


def _kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _move_all(src_root: Path, dest_root: Path) -> int:
    n = 0
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        (dest_root / cls_dir.name).mkdir(parents=True, exist_ok=True)
        for f in sorted(cls_dir.iterdir()):
            f.rename(dest_root / cls_dir.name / f.name)
            n += 1
    return n


# ---------------------------------------------------------------- stage 1
def stage_fetch_checkpoints(paths: Paths) -> None:
    import download_checkpoints

    paths.start_ckpts.mkdir(parents=True, exist_ok=True)
    for arch in ARCHITECTURES:
        if not (paths.start_ckpts / f"{arch}_best_model.pth").exists():
            local = Path("checkpoints") / f"{arch}_best_model.pth"
            if local.exists():
                shutil.copy2(local, paths.start_ckpts / local.name)
            else:
                download_checkpoints.download_checkpoint(arch, paths.start_ckpts)
        print(f"  starting checkpoint ready: {arch}")


# ---------------------------------------------------------------- stage 2
def _download_brats_validation(staging_raw: Path) -> None:
    """Per-file download (t1ce + flair only, ~8 MB per patient) instead of the 7 GB archive."""
    api = _kaggle_api()
    for n in range(1, 126):
        pid = f"BraTS20_Validation_{n:03d}"
        out = staging_raw / pid
        for modality in ("t1ce", "flair"):
            target = out / f"{pid}_{modality}.nii"
            if target.exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            for attempt in range(6):
                try:
                    api.dataset_download_file(
                        BRATS_DATASET, f"{BRATS_VAL_PREFIX}/{pid}/{pid}_{modality}.nii", path=str(out), force=True, quiet=True
                    )
                    break
                except Exception:  # noqa: BLE001 -- Kaggle rate-limits (429); back off and retry
                    time.sleep(6 * (attempt + 1))
            for z in out.glob("*.zip"):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
                z.unlink()
            time.sleep(1.2)


def stage_new_data(paths: Paths) -> None:
    if _done(paths, "new_data", paths.staging / "slices"):
        print("[skip] new data already staged")
        return
    st = paths.staging
    api = _kaggle_api()

    brats_raw = st / "raw" / "brats_val"
    print("[run] BraTS validation patients (glioma)...")
    _download_brats_validation(brats_raw)
    n = extract_brats_validation_glioma_slices(brats_raw, st / "slices" / "brats_val" / "glioma", slices_per_patient=8)
    print(f"  -> {n} glioma slices")
    assert n > 500, f"only {n} BraTS validation slices extracted"
    shutil.rmtree(brats_raw, ignore_errors=True)

    print("[run] OASIS-1 healthy subjects (notumor, ~1 GB download)...")
    oasis_raw = st / "raw" / "oasis"
    api.dataset_download_files(OASIS_DATASET, path=str(oasis_raw), unzip=True, quiet=True)
    n = extract_oasis_notumor_slices(oasis_raw, st / "slices" / "oasis" / "notumor", slices_per_subject=8)
    print(f"  -> {n} notumor slices")
    assert n > 500, f"only {n} OASIS slices extracted"
    shutil.rmtree(oasis_raw, ignore_errors=True)

    print("[run] PMRAM hospital MRI (4 classes)...")
    pm_raw = st / "raw" / "pmram"
    api.dataset_download_files(PMRAM_DATASET, path=str(pm_raw), unzip=True, quiet=True)
    counts = extract_pmram_images(pm_raw, st / "slices" / "pmram")
    print(f"  -> {counts}")
    assert all(counts.get(c, 0) > 100 for c in CLASSES), counts
    shutil.rmtree(pm_raw, ignore_errors=True)
    _mark(paths, "new_data")


# ---------------------------------------------------------------- stage 3
def _drop_eval_images_with_train_twins(paths: Paths, eval_name: str) -> int:
    """Moves eval images that have a near-duplicate in Training out of the eval set."""
    eval_root = paths.data / eval_name
    twins = find_test_images_with_train_duplicates(paths.data / "Training", eval_root, GATE_MAX_HAMMING)
    for f in twins:
        if patient_id_of(f) is not None:
            continue  # patient-id sources are kept apart by patient, see frozen_benchmark.check_leakage
        dest = paths.data / "_excluded_leaky_eval" / eval_name / f.relative_to(eval_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.rename(dest)
    return sum(1 for f in twins if patient_id_of(f) is None)


def _ensure_old_external_data_in_training(data_root: Path) -> None:
    """The starting weights were trained on BraTS (369 patients) + IXI slices. On a fresh
    machine (Colab) those aren't on disk, so rebuild them and put ALL of them in Training --
    the starting weights saw them, so none may be used for validation or testing."""
    train = data_root / "Training"
    has_brats = any((train / "glioma").glob("BraTS20_Training_*")) if (train / "glioma").exists() else False
    has_ixi = any((train / "notumor").glob("IXI*")) if (train / "notumor").exists() else False
    if has_brats and has_ixi:
        return
    import run_external_data_pipeline as old

    print("[run] Rebuilding the BraTS/IXI training slices the starting checkpoints were trained on...")
    old.stage_process_brats()
    old.stage_process_ixi()
    for cls in ("glioma", "notumor"):
        src = old.EXTERNAL_SLICES_DIR / cls
        (train / cls).mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.png")):
            f.rename(train / cls / f.name)
    shutil.rmtree(old.EXTERNAL_SLICES_DIR, ignore_errors=True)


def stage_assemble_data(paths: Paths) -> None:
    if _done(paths, "assemble_data", paths.data / "Validation"):
        print("[skip] data already assembled")
        return
    d, st = paths.data, paths.staging / "slices"
    if not (d / "Training").exists():
        import sys

        import download_mri_dataset

        saved_argv = sys.argv
        sys.argv = ["download_mri_dataset"]  # its main() runs argparse over sys.argv; ours has other flags
        try:
            download_mri_dataset.main()
        finally:
            sys.argv = saved_argv

    _ensure_old_external_data_in_training(d)

    # The starting weights were trained with these patients' slices -> they can't be test data.
    old_ext = d / "ExternalTesting"
    if old_ext.exists():
        moved = _move_all(old_ext, d / "Training")
        print(f"[run] pooled {moved} old ExternalTesting images back into Training (seen by the starting weights)")
        shutil.rmtree(old_ext, ignore_errors=True)

    # Patient-level sources: BraTS validation, OASIS.
    for source, cls in (("brats_val", "glioma"), ("oasis", "notumor")):
        counts = split_dir_by_patient_fractions(
            st / source / cls,
            {"train": d / "Training" / cls, "val": d / "Validation" / cls, "test": d / "ExternalTesting" / cls},
            PATIENT_FRACTIONS,
            SPLIT_SEED,
        )
        print(f"[run] {source}: {counts} slices -> train/val/test (whole patients)")

    # PMRAM is mostly a re-publication of the Kaggle/Br35H scans (measured: 72% of its raw images
    # have a pHash-identical twin in Kaggle). Drop every PMRAM image within CLUSTER_MAX_HAMMING of
    # any Kaggle image so what remains is genuinely new; then split similarity clusters per class.
    for cls in CLASSES:
        pm_dir = st / "pmram" / cls
        dupes: set[Path] = set()
        for kaggle_root in (d / "Training", d / "Testing"):
            dupes |= set(find_test_images_with_train_duplicates(kaggle_root, pm_dir, CLUSTER_MAX_HAMMING))
        for f in dupes:
            (d / "_excluded_pmram_duplicates" / cls).mkdir(parents=True, exist_ok=True)
            f.rename(d / "_excluded_pmram_duplicates" / cls / f.name)
        print(f"[run] pmram/{cls}: dropped {len(dupes)} images that duplicate Kaggle scans")
        files = sorted(pm_dir.glob("*.png"))
        buckets = split_clusters(files, PMRAM_FRACTIONS, SPLIT_SEED, CLUSTER_MAX_HAMMING)
        for bucket, dest_root in (("train", "Training"), ("val", "Validation"), ("test", "PMRAMTesting")):
            (d / dest_root / cls).mkdir(parents=True, exist_ok=True)
            for f in buckets[bucket]:
                f.rename(d / dest_root / cls / f.name)
        print(f"[run] pmram/{cls}: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    # Kaggle Testing: only images without a Training twin are a fair test; split the rest into val/test by cluster.
    kaggle_test = d / "Testing"
    leaky = find_test_images_with_train_duplicates(d / "Training", kaggle_test, GATE_MAX_HAMMING)
    for f in leaky:
        dest = d / "_excluded_leaky_eval" / "Testing" / f.relative_to(kaggle_test)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.rename(dest)
    print(f"[run] Kaggle Testing: excluded {len(leaky)} images that have a near-duplicate in Training")
    for cls in CLASSES:
        files = sorted((kaggle_test / cls).glob("*.*"))
        buckets = split_clusters(files, KAGGLE_CLEAN_FRACTIONS, SPLIT_SEED, CLUSTER_MAX_HAMMING)
        (d / "Validation" / cls).mkdir(parents=True, exist_ok=True)
        for f in buckets["val"]:
            f.rename(d / "Validation" / cls / f.name)
        print(f"[run] Kaggle clean {cls}: val={len(buckets['val'])} test={len(buckets['test'])}")

    # New-data eval images that happen to look like a Training image are dropped too.
    for name in ("Validation", "ExternalTesting", "PMRAMTesting"):
        print(f"[run] {name}: dropped {_drop_eval_images_with_train_twins(paths, name)} images with a Training twin")
    _mark(paths, "assemble_data")


# ---------------------------------------------------------------- stage 4
def stage_gate(paths: Paths) -> None:
    eval_dirs = {name: paths.data / name for name in ("Validation",) + TEST_DIRS}
    test_dirs = {k: eval_dirs[k] for k in TEST_DIRS}
    report = assert_no_leakage(paths.data / "Training", list(eval_dirs.values()), GATE_MAX_HAMMING)
    # validation drives early stopping, so it must not overlap the reported test sets either
    assert_no_leakage(eval_dirs["Validation"], list(test_dirs.values()), GATE_MAX_HAMMING)
    print(
        f"[gate] PASS: {report.eval_images} eval images vs {report.train_images} training images -- "
        "0 byte-identical, 0 shared patients, 0 near-duplicates"
    )
    if paths.manifest.exists():
        verify_manifest(test_dirs, paths.manifest)
        print("[gate] frozen manifest verified unchanged")
    else:
        m = write_manifest(test_dirs, paths.manifest)
        print(f"[gate] froze benchmark manifest: {m['n_files']} files, sha256 {m['manifest_sha256'][:16]}...")


# ---------------------------------------------------------------- stage 5/6
def stage_train(paths: Paths, args: argparse.Namespace) -> None:
    from src.train import train

    data_cfg = DataConfig(batch_size=args.batch_size, num_workers=args.num_workers, balanced_sampling=True)
    for arch in args.architectures:
        cfg = TrainConfig(
            architecture=arch,
            epochs=args.epochs,
            learning_rate=args.lr,
            early_stopping_patience=args.patience,
            lr_plateau_patience=max(2, args.patience // 4),
            init_checkpoint=paths.start_ckpts / f"{arch}_best_model.pth",
            class_loss_weights={},  # balanced sampling already equalises classes; don't double-count
            fine_tune_layers=args.fine_tune_layers,
            use_amp=True,
            resume_save_every=args.resume_every,
            checkpoint_dir=paths.ckpts,
            history_dir=paths.outputs,
        )
        print(f"\n{'=' * 60}\nContinuing {arch} from the published checkpoint\n{'=' * 60}")
        train(data_cfg, cfg)


def stage_benchmark(paths: Paths) -> None:
    from src.benchmark import benchmark_models, default_checkpoint_specs, save_benchmark_report

    eval_dirs = {k: paths.data / k for k in TEST_DIRS}
    verify_manifest(eval_dirs, paths.manifest)
    summary = {}
    for test_name in TEST_DIRS:
        for label, ckpt_dir in (("before", paths.start_ckpts), ("after", paths.ckpts)):
            specs = default_checkpoint_specs(ckpt_dir)
            out = paths.outputs / "benchmark_v2" / f"{test_name}_{label}"
            results = benchmark_models(specs, DataConfig(test_dir_name=test_name), output_dir=out)
            save_benchmark_report(results, out)
            for r in results:
                if r.available:
                    summary[f"{test_name}/{label}/{r.architecture}"] = {"accuracy": r.accuracy, "macro_f1": r.macro_f1}
                    print(f"  {test_name:16s} {label:6s} {r.architecture:16s} acc={r.accuracy:.4f} macro_f1={r.macro_f1:.4f}")
    verify_manifest(eval_dirs, paths.manifest)  # nothing moved while benchmarking
    (paths.outputs / "benchmark_v2" / "summary.json").write_text(json.dumps(summary, indent=1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, default=Path("."), help="checkpoints/state live here (use a Google Drive path on Colab)")
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument("--epochs", type=int, default=300, help="ceiling; early stopping normally ends runs sooner")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--fine-tune-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-every", type=int, default=1, help="save resume state every N epochs (use ~5 on Drive for VGG16)")
    p.add_argument("--prepare-only", action="store_true", help="build data + run the leakage gate, then stop")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.work_dir)
    stage_fetch_checkpoints(paths)
    stage_new_data(paths)
    stage_assemble_data(paths)
    stage_gate(paths)
    if args.prepare_only:
        return
    stage_train(paths, args)
    stage_gate(paths)
    stage_benchmark(paths)
    print("\nDone.")


if __name__ == "__main__":
    main()

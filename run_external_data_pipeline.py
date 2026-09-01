"""End-to-end driver: download BraTS (glioma) + IXI (notumor), extract slices,
dedupe against the existing dataset, merge, train all 3 architectures, and
benchmark against both the original and a fresh external-only held-out set.

Designed to survive being interrupted (laptop sleep, crash, power loss) and
re-run from the same command: every stage is gated by a marker/existence
check, so a re-invocation skips whatever already finished and resumes
whatever didn't -- training itself resumes at the epoch level (see
src/train.py's resume_state_path/training_done_marker), not just per-stage.

Usage:
    python run_external_data_pipeline.py
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from src.benchmark import benchmark_models, default_checkpoint_specs, save_benchmark_report
from src.config import DataConfig, TrainConfig
from src.external_data import extract_brats_glioma_slices, extract_ixi_notumor_slices
from src.secret_dedupe_eval import build_reference_index, dedupe_secret
from src.train_all import train_all

STATE_DIR = Path("pipeline_state")
EXTERNAL_RAW_DIR = Path("external_raw")
EXTERNAL_SLICES_DIR = Path("external_raw_slices")
EXTERNAL_DEDUPED_DIR = Path("external_deduped")
BRATS_DATASET = "awsaf49/brats20-dataset-training-validation"
IXI_DATASET = "wailrami/ixi-healthy-brain-mri-t1-t2"
EXTERNAL_TEST_FRACTION = 0.2
SPLIT_SEED = 42


def _marker(name: str) -> Path:
    return STATE_DIR / f"{name}.done"


def _stage_done(name: str) -> bool:
    return _marker(name).exists()


def _mark_stage_done(name: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _marker(name).touch()


def stage_download_kaggle_dataset() -> None:
    if _stage_done("download_kaggle"):
        print("[skip] Original Kaggle dataset already present.")
        return
    train_dir = Path("data/Training")
    if not (train_dir.exists() and any(train_dir.iterdir())):
        print("[run] Downloading original Kaggle brain-tumor-mri-dataset...")
        import download_mri_dataset

        download_mri_dataset.main()
    _mark_stage_done("download_kaggle")


def stage_download_external_raw() -> None:
    if _stage_done("download_external_raw"):
        print("[skip] BraTS/IXI raw downloads already present.")
        return
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    brats_dir = EXTERNAL_RAW_DIR / "brats"
    ixi_dir = EXTERNAL_RAW_DIR / "ixi"
    brats_dir.mkdir(parents=True, exist_ok=True)
    ixi_dir.mkdir(parents=True, exist_ok=True)

    print("[run] Downloading BraTS2020 (~7GB)...")
    api.dataset_download_files(BRATS_DATASET, path=str(brats_dir), unzip=True, quiet=False)
    print("[run] Downloading IXI (~4GB)...")
    api.dataset_download_files(IXI_DATASET, path=str(ixi_dir), unzip=True, quiet=False)

    _mark_stage_done("download_external_raw")


def stage_extract_slices() -> None:
    if _stage_done("extract_slices"):
        print("[skip] External slices already extracted.")
        return
    print("[run] Extracting glioma slices from BraTS...")
    n_glioma = extract_brats_glioma_slices(
        EXTERNAL_RAW_DIR / "brats", EXTERNAL_SLICES_DIR / "glioma", slices_per_patient=6
    )
    print(f"  -> {n_glioma} glioma slices")
    print("[run] Extracting notumor slices from IXI...")
    n_notumor = extract_ixi_notumor_slices(
        EXTERNAL_RAW_DIR / "ixi", EXTERNAL_SLICES_DIR / "notumor", slices_per_subject=4
    )
    print(f"  -> {n_notumor} notumor slices")

    assert n_glioma > 100, f"Only {n_glioma} glioma slices extracted -- check the BraTS download/layout."
    assert n_notumor > 100, f"Only {n_notumor} notumor slices extracted -- check the IXI download/layout."

    _mark_stage_done("extract_slices")


def stage_dedupe_and_merge() -> None:
    if _stage_done("dedupe_and_merge"):
        print("[skip] External data already deduped and merged.")
        return

    print("[run] Deduping extracted slices against the existing dataset...")
    reference_md5, reference_phash = build_reference_index([Path("data/Training"), Path("data/Testing")])
    stats = dedupe_secret(
        secret_root=EXTERNAL_SLICES_DIR,
        out_root=EXTERNAL_DEDUPED_DIR,
        md5_set=reference_md5,
        ph_list=reference_phash,
        max_hamming=5,
    )
    print(f"  -> dedupe stats: {stats}")

    print("[run] Splitting into data/Training (merge) and data/ExternalTesting (held out)...")
    rng = random.Random(SPLIT_SEED)
    for class_name in ["glioma", "notumor"]:
        src_dir = EXTERNAL_DEDUPED_DIR / class_name
        if not src_dir.exists():
            print(f"  no deduped images for {class_name}, skipping")
            continue
        images = sorted(src_dir.glob("*.png"))
        rng.shuffle(images)
        n_test = int(len(images) * EXTERNAL_TEST_FRACTION)
        test_images, train_images = images[:n_test], images[n_test:]

        train_dest = Path("data/Training") / class_name
        test_dest = Path("data/ExternalTesting") / class_name
        train_dest.mkdir(parents=True, exist_ok=True)
        test_dest.mkdir(parents=True, exist_ok=True)

        for p in train_images:
            p.rename(train_dest / p.name)
        for p in test_images:
            p.rename(test_dest / p.name)

        print(f"  {class_name}: {len(train_images)} -> Training, {len(test_images)} -> ExternalTesting")

    _mark_stage_done("dedupe_and_merge")


def stage_cleanup_raw_downloads() -> None:
    if _stage_done("cleanup_raw"):
        return
    print("[run] Freeing disk space (raw BraTS/IXI downloads no longer needed)...")
    shutil.rmtree(EXTERNAL_RAW_DIR, ignore_errors=True)
    shutil.rmtree(EXTERNAL_SLICES_DIR, ignore_errors=True)
    shutil.rmtree(EXTERNAL_DEDUPED_DIR, ignore_errors=True)
    _mark_stage_done("cleanup_raw")


def stage_train_all() -> None:
    # No stage-level marker here on purpose -- src/train.py itself tracks
    # per-architecture completion (training_done_marker) and per-epoch resume
    # state (resume_state_path), so re-running train_all after an interruption
    # already does the right thing: skip finished architectures, resume the
    # interrupted one from its last completed epoch.
    print("[run] Training all 3 architectures (resumable per-architecture/per-epoch)...")
    data_config = DataConfig()
    base_config = TrainConfig()
    train_all(data_config, base_config)


def stage_benchmark() -> None:
    print("[run] Benchmarking against the original data/Testing set...")
    import src.benchmark as benchmark_module

    benchmark_module.main()

    print("[run] Benchmarking against data/ExternalTesting (the real generalization test)...")
    external_cfg = DataConfig(test_dir_name="ExternalTesting")
    specs = default_checkpoint_specs(Path("checkpoints"))
    external_results = benchmark_models(specs, external_cfg, output_dir=Path("outputs/benchmark_external"))
    save_benchmark_report(external_results, Path("outputs/benchmark_external"))
    for r in external_results:
        if r.available:
            print(f"  {r.architecture}: accuracy={r.accuracy:.4f} macro_f1={r.macro_f1:.4f}")
        else:
            print(f"  {r.architecture}: unavailable ({r.error})")


def main() -> None:
    stage_download_kaggle_dataset()
    stage_download_external_raw()
    stage_extract_slices()
    stage_dedupe_and_merge()
    stage_cleanup_raw_downloads()
    stage_train_all()
    stage_benchmark()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

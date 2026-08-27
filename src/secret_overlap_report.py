"""
Cryptographic overlap check (SHA256 byte identity) vs original data/Training + data/Testing.

Also exports a *sample* of cases where pHash matches a reference image but file bytes differ
(recompression, cropping, or rare pHash collision — worth eyeballing a few rows).

Does not use loose pHash thresholds (those are misleading on MRI).
"""
from __future__ import annotations

import argparse
import sys
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from src.secret_dedupe_eval import iter_images, phash_of, run_eval


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_reference_sha_and_phash_map(
    roots: list[Path],
) -> tuple[set[str], dict[str, tuple[Path, str]]]:
    """SHA256 of every reference image + first ref path per exact pHash string."""
    paths: list[Path] = []
    for r in roots:
        if r.exists():
            paths.extend(iter_images(r))

    sha_set: set[str] = set()
    phash_to_ref: dict[str, tuple[Path, str]] = {}

    for p in tqdm(paths, desc="Reference: SHA256 + pHash map"):
        sha = file_sha256(p)
        sha_set.add(sha)
        ph = phash_of(p)
        if ph is None:
            continue
        key = str(ph)
        if key not in phash_to_ref:
            phash_to_ref[key] = (p, sha)

    return sha_set, phash_to_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="SHA256 overlap report + optional eval.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--secret-testing", type=Path, default=Path("data/secret/Testing"))
    parser.add_argument(
        "--out-filtered",
        type=Path,
        default=Path("data/secret/Testing_sha256_only"),
        help="Folder of secret images after removing byte-identical overlaps.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=Path("outputs/phash_overlap_pairs_sample.csv"),
        help="Sample CSV: pHash match but different SHA256 vs a reference image.",
    )
    parser.add_argument("--sample-pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet50_best_model.pth"))
    parser.add_argument("--eval-out", type=Path, default=Path("outputs/secret_eval_sha256_only"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    train_dir = args.data_root / "Training"
    test_dir = args.data_root / "Testing"
    ref_sha_set, phash_to_ref = build_reference_sha_and_phash_map([train_dir, test_dir])

    stats: dict = {
        "reference_unique_sha256": len(ref_sha_set),
        "reference_phash_keys": len(phash_to_ref),
        "total_secret": 0,
        "sha256_overlap": 0,
        "phash_exact_diff_sha_candidates": 0,
        "kept_sha256_only": 0,
        "per_class": {},
    }

    class_dirs = sorted([p for p in args.secret_testing.iterdir() if p.is_dir()])
    for d in class_dirs:
        stats["per_class"][d.name] = {
            "total": 0,
            "dropped_sha256": 0,
            "kept": 0,
        }

    pair_rows: list[dict] = []

    if args.out_filtered.exists():
        shutil.rmtree(args.out_filtered)
    args.out_filtered.mkdir(parents=True, exist_ok=True)

    for cls_dir in class_dirs:
        cls = cls_dir.name
        dest_cls = args.out_filtered / cls
        dest_cls.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(iter_images(cls_dir)):
            stats["total_secret"] += 1
            stats["per_class"][cls]["total"] += 1

            s_sha = file_sha256(img_path)
            if s_sha in ref_sha_set:
                stats["sha256_overlap"] += 1
                stats["per_class"][cls]["dropped_sha256"] += 1
                continue

            stats["kept_sha256_only"] += 1
            stats["per_class"][cls]["kept"] += 1
            shutil.copy2(img_path, dest_cls / img_path.name)

            ph = phash_of(img_path)
            if ph is None:
                continue
            key = str(ph)
            if key in phash_to_ref:
                ref_path, ref_sha = phash_to_ref[key]
                if s_sha != ref_sha:
                    stats["phash_exact_diff_sha_candidates"] += 1
                    pair_rows.append(
                        {
                            "secret_class": cls,
                            "secret_path": str(img_path.resolve()),
                            "secret_sha256": s_sha,
                            "ref_path": str(ref_path.resolve()),
                            "ref_sha256": ref_sha,
                            "phash": key,
                        }
                    )

    random.seed(args.seed)
    random.shuffle(pair_rows)
    sample_rows = pair_rows[: args.sample_pairs]

    args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    if sample_rows:
        with args.pairs_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            w.writeheader()
            w.writerows(sample_rows)

    summary_path = Path("outputs") / "secret_overlap_sha256_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote pHash sample CSV ({len(sample_rows)} rows): {args.pairs_csv}")
    print(f"Wrote SHA256-filtered secret set: {args.out_filtered}")
    sys.stdout.flush()

    if args.no_eval or stats["kept_sha256_only"] == 0:
        print("Skipping eval (--no-eval or no images kept).")
        return

    report = run_eval(
        data_root=args.out_filtered.parent,
        test_dir=args.out_filtered.name,
        checkpoint=args.checkpoint,
        out_dir=args.eval_out,
        batch_size=args.batch_size,
    )
    print("SHA256-only secret accuracy:", report["accuracy"])
    print("Macro F1:", report["macro avg"]["f1-score"])
    print(f"Wrote eval: {args.eval_out}")


if __name__ == "__main__":
    main()

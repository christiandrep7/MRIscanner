from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, PngImagePlugin

from src.secret_dedupe_eval import iter_images
from src.secret_overlap_report import build_reference_sha_and_phash_map, file_sha256


def test_file_sha256_stable_and_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(a)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(b)
    Image.new("RGB", (8, 8), color=(9, 9, 9)).save(tmp_path / "c.png")

    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) != file_sha256(tmp_path / "c.png")


def test_build_reference_map_indexes_training_and_testing(tiny_dataset: Path):
    train_dir = tiny_dataset / "Training"
    test_dir = tiny_dataset / "Testing"
    sha_set, phash_to_ref = build_reference_sha_and_phash_map([train_dir, test_dir])

    total_images = sum(1 for _ in iter_images(train_dir)) + sum(1 for _ in iter_images(test_dir))
    assert len(sha_set) == total_images
    assert len(phash_to_ref) > 0
    for path, sha in phash_to_ref.values():
        assert path.exists()
        assert len(sha) == 64  # sha256 hex digest length


def test_sha256_overlap_and_phash_diff_sha_candidate_via_cli(tiny_dataset: Path, tmp_path: Path, monkeypatch):
    """Exercises the module's real detection logic end-to-end (not just its helpers):
    one byte-identical secret image (SHA256 overlap, dropped) and one PNG-metadata
    "reposted" copy (same pHash, different SHA256 -> flagged as a review candidate)."""
    train_dir = tiny_dataset / "Training"
    original = iter_images(train_dir / "glioma")[0]

    secret_root = tmp_path / "secret" / "Testing"
    (secret_root / "glioma").mkdir(parents=True)

    # 1) Byte-identical copy -> should be dropped as a SHA256 overlap.
    shutil.copy2(original, secret_root / "glioma" / "identical.png")

    # 2) Same pixels, different file bytes (metadata) -> SHA256 differs but
    #    pHash matches the reference exactly -> flagged as a review candidate.
    with Image.open(original) as im:
        pixels = im.convert("RGB")
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", "reposted copy")
    pixels.save(secret_root / "glioma" / "reposted.png", pnginfo=meta)

    # 3) A genuinely novel image -> kept, no candidate pair.
    Image.new("RGB", (32, 32), color=(7, 7, 7)).save(secret_root / "glioma" / "novel.png")

    import sys

    # secret_overlap_report.main() writes outputs/secret_overlap_sha256_summary.json
    # relative to the CWD (hardcoded, not a CLI arg) -- sandbox into tmp_path so the
    # test doesn't pollute the real project's outputs/ directory.
    monkeypatch.chdir(tmp_path)

    out_filtered = tmp_path / "filtered"
    pairs_csv = tmp_path / "pairs.csv"
    argv = [
        "secret_overlap_report",
        "--data-root",
        str(tiny_dataset),
        "--secret-testing",
        str(secret_root),
        "--out-filtered",
        str(out_filtered),
        "--pairs-csv",
        str(pairs_csv),
        "--no-eval",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from src import secret_overlap_report

    secret_overlap_report.main()

    summary_path = Path("outputs") / "secret_overlap_sha256_summary.json"
    assert summary_path.exists()
    import json

    stats = json.loads(summary_path.read_text())
    assert stats["sha256_overlap"] == 1
    assert stats["kept_sha256_only"] == 2  # reposted.png + novel.png
    assert stats["phash_exact_diff_sha_candidates"] == 1

    kept_files = {p.name for p in out_filtered.glob("glioma/*")}
    assert kept_files == {"reposted.png", "novel.png"}

    assert pairs_csv.exists()
    csv_text = pairs_csv.read_text()
    assert "secret_path" in csv_text and "reposted.png" in csv_text

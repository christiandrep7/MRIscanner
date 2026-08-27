from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from src.config import DataConfig
from src.secret_dedupe_eval import (
    build_reference_index,
    dedupe_secret,
    file_md5,
    is_near_duplicate,
    iter_images,
    phash_of,
    remove_empty_class_dirs,
    run_eval,
)


def test_file_md5_stable_and_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(a)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(b)
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(tmp_path / "c.jpg")

    assert file_md5(a) == file_md5(b)
    assert file_md5(a) != file_md5(tmp_path / "c.jpg")


def test_iter_images_finds_only_valid_extensions(tmp_path: Path):
    Image.new("RGB", (4, 4)).save(tmp_path / "img.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "img.png")
    (tmp_path / "notes.txt").write_text("hi")

    found = {p.name for p in iter_images(tmp_path)}
    assert found == {"img.jpg", "img.png"}


def test_build_reference_index_counts_all_training_and_testing(tiny_dataset: Path):
    train_dir = tiny_dataset / "Training"
    test_dir = tiny_dataset / "Testing"
    md5_set, ph_list = build_reference_index([train_dir, test_dir])

    total_images = sum(1 for _ in iter_images(train_dir)) + sum(1 for _ in iter_images(test_dir))
    assert len(md5_set) == total_images  # all fixture images are distinct
    assert len(ph_list) == total_images


def test_dedupe_secret_drops_exact_md5_duplicate_keeps_novel(tiny_dataset: Path, tmp_path: Path):
    train_dir = tiny_dataset / "Training"
    test_dir = tiny_dataset / "Testing"
    md5_set, ph_list = build_reference_index([train_dir, test_dir])

    secret_root = tmp_path / "secret"
    (secret_root / "glioma").mkdir(parents=True)
    # Exact byte-for-byte duplicate of an existing training image.
    original = iter_images(train_dir / "glioma")[0]
    shutil.copy2(original, secret_root / "glioma" / "duplicate.jpg")
    # A genuinely new image.
    Image.new("RGB", (32, 32), color=(9, 9, 9)).save(secret_root / "glioma" / "novel.jpg")

    out_root = tmp_path / "deduped"
    stats = dedupe_secret(secret_root, out_root, md5_set, ph_list, max_hamming=0)

    assert stats["total_secret"] == 2
    assert stats["md5_dup"] == 1
    assert stats["kept"] == 1
    assert stats["per_class"]["glioma"]["kept"] == 1
    kept_files = list((out_root / "glioma").iterdir())
    assert len(kept_files) == 1
    assert kept_files[0].name == "novel.jpg"


def test_dedupe_secret_drops_exact_phash_duplicate_with_different_bytes(tmp_path: Path):
    """pHash is computed on *decoded pixels*; MD5 is computed on *file bytes*.
    Saving identical pixels through PNG with different embedded metadata gives
    byte-identical decode (phash matches exactly) but different file bytes
    (MD5 differs) -- the exact "recompressed/reposted" case this script targets,
    without relying on JPEG re-encoding to coincidentally preserve every DCT bit.
    """
    from PIL import PngImagePlugin

    reference_img = Image.new("RGB", (32, 32), color=(120, 60, 200))

    train_dir = tmp_path / "reference" / "Training"
    ref_class_dir = train_dir / "meningioma"
    ref_class_dir.mkdir(parents=True)
    reference_img.save(ref_class_dir / "original.png")

    md5_set, ph_list = build_reference_index([train_dir])

    secret_root = tmp_path / "secret"
    (secret_root / "meningioma").mkdir(parents=True)
    duplicate = secret_root / "meningioma" / "duplicate_with_metadata.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", "re-saved copy with different metadata")
    reference_img.save(duplicate, pnginfo=meta)

    assert file_md5(duplicate) not in md5_set  # confirm it's not an MD5 hit
    assert phash_of(duplicate) in ph_list  # confirm the fixture is a real phash match

    out_root = tmp_path / "deduped"
    stats = dedupe_secret(secret_root, out_root, md5_set, ph_list, max_hamming=0)

    assert stats["phash_dup"] == 1
    assert stats["kept"] == 0


def test_is_near_duplicate_none_phash_is_false():
    assert is_near_duplicate(None, [], max_hamming=0) is False


def test_phash_of_corrupted_file_returns_none(tmp_path: Path):
    bad_file = tmp_path / "corrupt.jpg"
    bad_file.write_bytes(b"not a real image")
    assert phash_of(bad_file) is None


def test_dedupe_secret_treats_corrupted_image_as_bad_image(tiny_dataset: Path, tmp_path: Path):
    train_dir = tiny_dataset / "Training"
    test_dir = tiny_dataset / "Testing"
    md5_set, ph_list = build_reference_index([train_dir, test_dir])

    secret_root = tmp_path / "secret"
    (secret_root / "glioma").mkdir(parents=True)
    (secret_root / "glioma" / "corrupt.jpg").write_bytes(b"not a real image")

    out_root = tmp_path / "deduped"
    stats = dedupe_secret(secret_root, out_root, md5_set, ph_list, max_hamming=0)

    assert stats["bad_image"] == 1
    assert stats["kept"] == 0
    assert stats["per_class"]["glioma"]["dropped"] == 1


def test_remove_empty_class_dirs_removes_only_empty(tmp_path: Path):
    root = tmp_path / "root"
    empty_dir = root / "empty"
    full_dir = root / "full"
    empty_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(full_dir / "a.jpg")

    remove_empty_class_dirs(root)

    assert not empty_dir.exists()
    assert full_dir.exists()


def test_run_eval_produces_report(tiny_data_config: DataConfig, fake_checkpoint: Path, tmp_path: Path):
    # Reuse the tiny Testing split itself as the "deduped secret" set for this eval check.
    out_dir = tmp_path / "eval_out"
    report = run_eval(
        data_root=tiny_data_config.data_root,
        test_dir=tiny_data_config.test_dir_name,
        checkpoint=fake_checkpoint,
        out_dir=out_dir,
        batch_size=4,
    )
    assert "accuracy" in report
    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "classification_report.json").exists()

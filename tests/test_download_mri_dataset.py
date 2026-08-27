from __future__ import annotations

from pathlib import Path

from download_mri_dataset import (
    count_images,
    find_kaggle_token,
    has_kaggle_credentials,
    print_auth_help,
)


def test_find_kaggle_token_checks_kaggle_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    token = kaggle_dir / "kaggle.json"
    token.write_text("{}")

    found = find_kaggle_token()
    assert found == token


def test_find_kaggle_token_checks_config_kaggle_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_dir = tmp_path / ".config" / "kaggle"
    config_dir.mkdir(parents=True)
    token = config_dir / "kaggle.json"
    token.write_text("{}")

    found = find_kaggle_token()
    assert found == token


def test_find_kaggle_token_returns_none_when_absent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert find_kaggle_token() is None


def test_print_auth_help_mentions_both_candidate_paths(capsys):
    print_auth_help()
    out = capsys.readouterr().out
    assert ".kaggle" in out
    assert ".config" in out
    assert "kaggle.com" in out


def test_has_kaggle_credentials_false_when_nothing_present(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    assert has_kaggle_credentials() is False


def test_has_kaggle_credentials_true_via_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "some-token-value")
    assert has_kaggle_credentials() is True


def test_has_kaggle_credentials_true_via_access_token_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "access_token").write_text("some-token-value")
    assert has_kaggle_credentials() is True


def test_has_kaggle_credentials_true_via_legacy_kaggle_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text("{}")
    assert has_kaggle_credentials() is True


def test_count_images_prints_per_class_counts(tiny_dataset: Path, capsys):
    count_images(tiny_dataset)
    out = capsys.readouterr().out
    assert "Training:" in out
    assert "Testing:" in out
    assert "glioma: 10 images" in out
    assert out.count(": 10 images") == 4  # one per class, Training
    assert out.count(": 6 images") == 4  # one per class, Testing


def test_count_images_skips_missing_split(tmp_path: Path, capsys):
    root = tmp_path / "partial"
    (root / "Training" / "glioma").mkdir(parents=True)
    (root / "Training" / "glioma" / "a.png").write_bytes(b"\x89PNG\r\n")

    count_images(root)

    out = capsys.readouterr().out
    assert "Training:" in out
    assert "Testing:" not in out

from __future__ import annotations

from pathlib import Path

import pytest

import download_checkpoints as dc


def test_release_url_format():
    url = dc.release_url("resnet50")
    assert url == (
        "https://github.com/christiandrep7/MRIscanner/releases/download/"
        "pretrained-models-v1/resnet50_best_model.pth"
    )


def test_download_checkpoint_calls_urlretrieve_with_correct_args(tmp_path: Path, monkeypatch):
    calls = []

    def fake_urlretrieve(url, dest, reporthook=None):
        calls.append((url, dest))
        Path(dest).write_bytes(b"fake checkpoint bytes")

    monkeypatch.setattr(dc.urllib.request, "urlretrieve", fake_urlretrieve)

    dest = dc.download_checkpoint("vgg16", tmp_path)

    assert dest == tmp_path / "vgg16_best_model.pth"
    assert dest.exists()
    assert len(calls) == 1
    assert calls[0][0] == dc.release_url("vgg16")


def test_main_skips_existing_checkpoint(tmp_path: Path, monkeypatch, capsys):
    existing = tmp_path / "resnet50_best_model.pth"
    existing.write_bytes(b"already here")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt download for an existing checkpoint")

    monkeypatch.setattr(dc.urllib.request, "urlretrieve", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        ["download_checkpoints.py", "--architectures", "resnet50", "--checkpoint-dir", str(tmp_path)],
    )

    dc.main()

    out = capsys.readouterr().out
    assert "already have" in out
    assert existing.read_bytes() == b"already here"  # untouched


def test_main_downloads_missing_checkpoint(tmp_path: Path, monkeypatch):
    def fake_urlretrieve(url, dest, reporthook=None):
        Path(dest).write_bytes(b"downloaded")

    monkeypatch.setattr(dc.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(
        "sys.argv",
        ["download_checkpoints.py", "--architectures", "efficientnet_b0", "--checkpoint-dir", str(tmp_path)],
    )

    dc.main()

    assert (tmp_path / "efficientnet_b0_best_model.pth").read_bytes() == b"downloaded"


def test_main_exits_on_http_error(tmp_path: Path, monkeypatch):
    import urllib.error

    def fake_urlretrieve(url, dest, reporthook=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(dc.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(
        "sys.argv",
        ["download_checkpoints.py", "--architectures", "vgg16", "--checkpoint-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit):
        dc.main()

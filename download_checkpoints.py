"""Downloads pretrained model checkpoints from this repo's GitHub Release, so you
can run predictions/benchmarks immediately without training anything yourself.

Usage:
    python download_checkpoints.py
    python download_checkpoints.py --architectures resnet50 vgg16
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

RELEASE_TAG = "pretrained-models-v1"
REPO = "christiandrep7/MRIscanner"
ARCHITECTURES = ("resnet50", "efficientnet_b0", "vgg16")


def release_url(architecture: str) -> str:
    filename = f"{architecture}_best_model.pth"
    return f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{filename}"


def download_checkpoint(architecture: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / f"{architecture}_best_model.pth"
    url = release_url(architecture)

    last_reported = -1

    def report(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_reported
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = int(downloaded / total_size * 100)
        if pct == last_reported:
            return
        last_reported = pct
        print(f"\r  {architecture}: {pct:3d}% ({downloaded // 1_000_000}MB/{total_size // 1_000_000}MB)", end="", flush=True)

    print(f"Downloading {architecture} checkpoint from {url}")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints from the GitHub Release.")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    for architecture in args.architectures:
        dest = args.checkpoint_dir / f"{architecture}_best_model.pth"
        if dest.exists():
            print(f"{architecture}: already have {dest}, skipping (delete it first to re-download)")
            continue
        try:
            download_checkpoint(architecture, args.checkpoint_dir)
        except urllib.error.HTTPError as e:
            print(f"\n{architecture}: download failed ({e}). Check {release_url(architecture)} still exists.")
            sys.exit(1)

    print(f"\nDone. Checkpoints are in: {args.checkpoint_dir.resolve()}")
    print("Run `python -m app.gradio_app` to try them out.")


if __name__ == "__main__":
    main()

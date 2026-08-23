import argparse
import os
import sys
import zipfile
from pathlib import Path


DEFAULT_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


def find_kaggle_token() -> Path | None:
    home = Path.home()
    candidates = [
        home / ".kaggle" / "kaggle.json",
        home / ".config" / "kaggle" / "kaggle.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def print_auth_help() -> None:
    print("Kaggle API token not found.")
    print("Create one at: https://www.kaggle.com/settings")
    print("Then place kaggle.json in one of these locations:")
    print(f"  - {Path.home() / '.kaggle' / 'kaggle.json'}")
    print(f"  - {Path.home() / '.config' / 'kaggle' / 'kaggle.json'}")


def run_download(dataset: str, out_dir: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Missing dependency: kaggle")
        print("Install it with: pip install kaggle")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading dataset '{dataset}' to: {out_dir}")
    api.dataset_download_files(dataset, path=str(out_dir), unzip=False, quiet=False)

    zip_name = f"{dataset.split('/')[-1]}.zip"
    zip_path = out_dir / zip_name
    if not zip_path.exists():
        zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zips:
            raise FileNotFoundError("Download finished, but no zip file was found.")
        zip_path = zips[0]
    return zip_path


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting '{zip_path.name}' into: {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def count_images(root: Path) -> None:
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    print("\nDataset summary:")
    for split in ["Training", "Testing"]:
        split_path = root / split
        if not split_path.exists():
            continue
        print(f"\n{split}:")
        class_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        for class_dir in class_dirs:
            image_count = sum(1 for p in class_dir.rglob("*") if p.suffix.lower() in valid_ext)
            print(f"  - {class_dir.name}: {image_count} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract a brain MRI dataset from Kaggle."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where zip is downloaded and extracted (default: data)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    token = find_kaggle_token()
    if token is None:
        print_auth_help()
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    zip_path = run_download(args.dataset, output_dir)
    extract_zip(zip_path, output_dir)
    count_images(output_dir)

    if not args.keep_zip:
        try:
            os.remove(zip_path)
            print(f"\nDeleted zip: {zip_path}")
        except OSError:
            print(f"\nCould not delete zip: {zip_path}")

    print("\nDone. You can now inspect images under:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()

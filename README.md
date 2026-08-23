# MRIscanner

End-to-end brain MRI tumor classification project using PyTorch.

## Quick start

1. Create and activate a virtual environment
2. Install dependencies
3. Download data
4. Train baseline model
5. Evaluate + run Gradio app

## Commands

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python .\download_mri_dataset.py
python -m src.data
python -m src.train --epochs 12
python -m src.evaluate --checkpoint checkpoints/best_model.pth
python .\app\gradio_app.py
```

## If training fails: cannot download ResNet50 (`getaddrinfo` / no internet)

Torchvision tries to download ImageNet weights from `download.pytorch.org`. If DNS/network blocks it:

1. Download **`resnet50-11ad3fa6.pth`** in a browser (or another PC):  
   https://download.pytorch.org/models/resnet50-11ad3fa6.pth
2. Save it to the hub cache folder (create folders if needed):  
   `%USERPROFILE%\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth`
3. Run training again: `python -m src.train --epochs 12`

Or pass the file path explicitly:

```powershell
python -m src.train --epochs 12 --imagenet-weights "C:\path\to\resnet50-11ad3fa6.pth"
```

**Note:** `src.evaluate` and `app/gradio_app.py` no longer download ImageNet weights; they only load your trained `checkpoints/best_model.pth`.

## External / “secret” holdout + duplicate check

After you place a second dataset under `data/secret/Testing/<class>/...`, you can remove images that are **byte-identical** (MD5) or **exact pHash matches** to anything in `data/Training` or `data/Testing`, then re-evaluate:

```powershell
python -m src.secret_dedupe_eval --hamming 0
```

- `--hamming 0` = exact perceptual hash only (recommended for MRI; large values flag many non-duplicates).
- Writes `data/secret/Testing_deduped/`, `outputs/secret_dedupe_stats.json`, and `outputs/secret_eval_deduped/`.

Run Gradio as a module:

```powershell
python -m app.gradio_app
```

## Strict overlap report (SHA256) + pHash “review sample”

Byte-identical images across datasets share the same **SHA256**. This script removes those from `data/secret/Testing`, saves `data/secret/Testing_sha256_only/`, writes summary JSON, and exports a **sample CSV** of “same pHash, different SHA256” pairs to spot recompressed/reposted images:

```powershell
python -m src.secret_overlap_report
```

Outputs: `outputs/secret_overlap_sha256_summary.json`, `outputs/phash_overlap_pairs_sample.csv`, `outputs/secret_eval_sha256_only/`.

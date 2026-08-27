# MRIscanner

Brain MRI tumor classification (glioma / meningioma / pituitary / no tumor) using PyTorch.
Three architectures — ResNet50, EfficientNet-B0, VGG16 — trained and compared side by
side, with a Gradio UI for predictions, Grad-CAM explainability, model agreement/consensus
checking, and benchmarking against a held-out test set.

## Quick start

```bash
git clone https://github.com/christiandrep7/MRIscanner.git
cd MRIscanner
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Just want to use the trained models? (no training, no Kaggle account, ~1 minute)

Pretrained checkpoints for all three architectures are published on this repo's
[Releases page](https://github.com/christiandrep7/MRIscanner/releases/tag/pretrained-models-v1) —
download them and go straight to predicting:

```bash
python download_checkpoints.py
python -m app.gradio_app
```

Open **http://127.0.0.1:7860** and upload any MRI scan to get a prediction, confidence
score, and Grad-CAM heatmap from each model — no dataset, no training, no GPU required.

Held-out test accuracy for the published checkpoints: **ResNet50 95.0%**,
**VGG16 94.2%**, **EfficientNet-B0 92.7%** (see [Comparing multiple architectures](#comparing-multiple-architectures)
below for the full breakdown).

**Note:** the **"🎲 Try a random never-seen scan"** button and the **Benchmark** tab both
need the actual dataset on disk (to pull a real, labeled test image) — those two features
need the "Download the real dataset yourself" step below too. Everything else (uploading
your own scan, single or multi-model prediction, the agreement/consensus check) works with
just the downloaded checkpoints.

### Just want to check the code works? (~2 minutes, no data or GPU needed)

```bash
pip install pytest pytest-cov
pytest -q
```

This builds a tiny synthetic image set on the fly and runs the real training, evaluation,
Grad-CAM, benchmark, and Gradio code paths against it — no Kaggle account, no download, no
GPU required. See [Running tests](#running-tests) below for coverage and slower e2e runs.

### Want to download the real dataset yourself, or retrain from scratch?

1. Get a free Kaggle API token: kaggle.com → your profile → Settings → API →
   **Create New Token**. Either save the downloaded `kaggle.json` to
   `~/.kaggle/kaggle.json`, or (newer Kaggle accounts) use the **API Tokens** section
   and save the token to `~/.kaggle/access_token`.
2. Download the dataset (~160MB, 7000 images):
   ```bash
   python download_mri_dataset.py
   ```
   This alone is enough to unlock the random-scan button and Benchmark tab if you've
   already downloaded the pretrained checkpoints above.
3. (Optional) Retrain from scratch instead of using the pretrained checkpoints:
   ```bash
   python -m src.train_all --epochs 15   # trains all 3 architectures
   ```
   Training all three takes roughly 1–3 hours depending on your machine (faster with a
   CUDA GPU or Apple Silicon's MPS backend, which this project uses automatically when
   available — see `src/model.py::get_device`).
4. Launch the app:
   ```bash
   python -m app.gradio_app
   ```
   Open **http://127.0.0.1:7860**. Upload a scan (or click **"🎲 Try a random
   never-seen scan"** to test against images no model trained on — it'll also show you
   the true label and mark each model ✅/❌), pick which model(s) to run, and use the
   **Benchmark** tab to compare all three (or just one) on the held-out test set.

Only have some checkpoints (pretrained or your own)? The app and CLI tools all handle
missing checkpoints gracefully (reported as "not trained yet" / skipped, not a crash) — you
don't need all three to start using it.

## Comparing multiple architectures

Three backbones are supported: `resnet50`, `efficientnet_b0`, `vgg16`. Each trains to
its own checkpoint (`checkpoints/{architecture}_best_model.pth`) and history file
(`outputs/{architecture}_metrics_history.csv`), so training one never overwrites another.

**Published checkpoint results** (held-out test set, 1600 images none of them trained on
or used for checkpoint selection):

| Architecture | Accuracy | Macro F1 | Params | Speed (ms/image, Apple M3) |
|---|---|---|---|---|
| ResNet50 | 95.0% | 0.949 | 23.5M | 33.7 |
| EfficientNet-B0 | 92.7% | 0.925 | 4.0M | 17.6 (fastest) |
| VGG16 | 94.2% | 0.941 | 134.3M | 47.1 |

ResNet50 wins on accuracy; EfficientNet-B0 is ~2.7x faster than VGG16 for a small accuracy
trade-off. Reproduce this yourself with `python -m src.benchmark` once you have
checkpoints (pretrained or your own) and the dataset downloaded.

Train one:

```bash
python -m src.train --architecture efficientnet_b0 --epochs 15
```

Train all three back to back (prints a comparison summary at the end):

```bash
python -m src.train_all --epochs 15
```

Then benchmark every trained architecture on `data/Testing` -- scans none of them
were trained on or used for checkpoint selection:

```bash
python -m src.benchmark
```

Writes `outputs/benchmark/results.json` and `outputs/benchmark/accuracy_comparison.png`.
Architectures without a checkpoint yet are reported as unavailable, not a crash.

Checkpoints are self-describing (they record which architecture they were trained
with), so `src.evaluate`, `src.secret_dedupe_eval`, `src.benchmark`, and the Gradio app
all load the right model automatically from the checkpoint file alone.

## If training fails: cannot download ImageNet weights (`getaddrinfo` / no internet)

Torchvision tries to download ImageNet weights from `download.pytorch.org`. If DNS/network blocks it:

1. Download the weights file in a browser (or another PC). The exact filename depends
   on the architecture -- run `python -m src.model` to print each architecture's
   expected weights URL, or check the error message when training fails.
2. Save it to the hub cache folder (create folders if needed):  
   `%USERPROFILE%\.cache\torch\hub\checkpoints\<filename>.pth`
3. Run training again.

Or pass the file path explicitly:

```bash
python -m src.train --architecture resnet50 --epochs 15 --imagenet-weights "/path/to/resnet50-11ad3fa6.pth"
```

**Note:** `src.evaluate`, `src.benchmark`, and `app/gradio_app.py` never download ImageNet
weights; they only load your trained checkpoints.

## External / “secret” holdout + duplicate check

After you place a second dataset under `data/secret/Testing/<class>/...`, you can remove images that are **byte-identical** (MD5) or **exact pHash matches** to anything in `data/Training` or `data/Testing`, then re-evaluate:

```bash
python -m src.secret_dedupe_eval --hamming 0
```

- `--hamming 0` = exact perceptual hash only (recommended for MRI; large values flag many non-duplicates).
- Writes `data/secret/Testing_deduped/`, `outputs/secret_dedupe_stats.json`, and `outputs/secret_eval_deduped/`.

## Running tests

The test suite doesn't need the real Kaggle dataset or a GPU — it builds a tiny
synthetic 4-class image set on the fly and runs the real code paths (training,
evaluation, Grad-CAM, dedupe/overlap, Gradio inference) against it, plus true
end-to-end CLI smoke tests via subprocess.

```bash
pip install -r requirements.txt pytest pytest-cov
pytest -q                                          # full suite
pytest -q --cov=src --cov=app --cov-report=term-missing   # with coverage
pytest -q tests/test_e2e_pipeline.py               # just the CLI smoke tests (slower)
```

CI (`.github/workflows/tests.yml`) runs the same suite on every push/PR.

## Strict overlap report (SHA256) + pHash “review sample”

Byte-identical images across datasets share the same **SHA256**. This script removes those from `data/secret/Testing`, saves `data/secret/Testing_sha256_only/`, writes summary JSON, and exports a **sample CSV** of “same pHash, different SHA256” pairs to spot recompressed/reposted images:

```bash
python -m src.secret_overlap_report
```

Outputs: `outputs/secret_overlap_sha256_summary.json`, `outputs/phash_overlap_pairs_sample.csv`, `outputs/secret_eval_sha256_only/`.

## Deploying to Hugging Face Spaces

Hugging Face Spaces is the natural host for this app: it's built for exactly this case
(a persistent Gradio server, room for PyTorch model weights, a free CPU tier) — unlike
serverless platforms (Vercel, Netlify, AWS Lambda), which time out long before a
multi-model benchmark run finishes and don't keep a process alive between requests.

1. Create a free account at [huggingface.co](https://huggingface.co), then
   [create a new Space](https://huggingface.co/new-space): SDK = **Gradio**,
   Hardware = **CPU basic** (free).
2. Clone your new Space (a separate git repo from this one):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   ```
3. Copy this project's `app/`, `src/`, and `requirements.txt` into it, and add a
   top-level `app.py` (Spaces looks for this by default):
   ```python
   from app.gradio_app import build_ui

   demo = build_ui()
   demo.launch()
   ```
4. For a CPU-only Space, add this to the top of `requirements.txt` so pip doesn't try
   to pull an irrelevant multi-gigabyte CUDA build of torch:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```
5. Track your trained checkpoints with git-lfs (Spaces supports this natively — no
   100MB-per-file ceiling like a plain GitHub repo):
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes app.py app/ src/ requirements.txt checkpoints/*.pth
   git commit -m "Deploy MRIscanner"
   git push
   ```
6. The Space builds automatically and gives you a public URL within a few minutes.

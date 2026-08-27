# MRIscanner — Vercel demo

A lightweight, single-model public demo of the brain tumor MRI classifier, built to
actually run on Vercel's free tier. This is **not** the full app — see the
[main README](../README.md) for the complete Gradio UI (3-model comparison, Grad-CAM,
consensus check, benchmark tab), which needs a persistent server and doesn't fit
Vercel's serverless model.

**Live:** https://vercel-demo-vert-eta.vercel.app

## Why a separate, smaller app

Vercel serverless functions are stateless, size-capped, and time-limited — a poor fit
for a multi-hundred-MB PyTorch app with a persistent Gradio server. This demo instead:

- Runs **only EfficientNet-B0** (92.7% held-out accuracy, the fastest and smallest of
  the three architectures) — VGG16 (528MB) and ResNet50 (96MB) don't fit comfortably in
  a size-capped serverless function alongside a full ML runtime.
- Uses **ONNX Runtime instead of PyTorch** at inference time — `onnxruntime` + `numpy`
  + `Pillow` is a small fraction of PyTorch/torchvision's footprint, keeping the
  deployed function well within Vercel's limits.
- Drops **Grad-CAM, multi-model consensus, and the benchmark tab** — Grad-CAM needs
  gradient hooks that ONNX Runtime's inference-only graph doesn't expose the same way,
  and the other two need either multiple models loaded at once or the real dataset on
  disk, neither of which fits this deployment's constraints.
- Custom static HTML/JS frontend instead of Gradio (Gradio needs a persistent process;
  Vercel doesn't run one).

## How the model was exported

From the main project root, with a trained `checkpoints/efficientnet_b0_best_model.pth`
present:

```python
import torch
from src.model import build_model
from src.checkpoint_io import load_training_checkpoint

payload = load_training_checkpoint("checkpoints/efficientnet_b0_best_model.pth", map_location="cpu")
model = build_model("efficientnet_b0", num_classes=len(payload["class_to_idx"]), pretrained=False)
model.load_state_dict(payload["model_state_dict"])
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "vercel-demo/model/efficientnet_b0.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13,
)
```

Verified to match the original PyTorch model's predictions within floating-point
tolerance (~5e-5 max logit difference) before deploying.

## Structure

```
vercel-demo/
  api/predict.py       # serverless function: preprocess + ONNX inference
  public/index.html     # static frontend (upload, preview, results)
  model/efficientnet_b0.onnx
  pyproject.toml        # PEP 621 metadata + [tool.vercel] entrypoint (uv-based build)
  requirements.txt      # onnxruntime, numpy, Pillow only -- no torch
```

## Redeploying

```bash
cd vercel-demo
npm install --no-save vercel   # or use a global install
./node_modules/.bin/vercel login
./node_modules/.bin/vercel --prod --yes
```

To regenerate the ONNX model after retraining, rerun the export snippet above, then
redeploy.

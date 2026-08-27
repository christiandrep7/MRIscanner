from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.benchmark import benchmark_models, default_checkpoint_specs, save_benchmark_report
from src.checkpoint_io import load_training_checkpoint
from src.config import DataConfig
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.gradcam_utils import generate_gradcam_overlay
from src.model import ARCHITECTURES, build_model, get_device

CHECKPOINT_DIR = Path("checkpoints")
BENCHMARK_OUTPUT_DIR = Path("outputs") / "benchmark"
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def checkpoint_path_for(architecture: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    return checkpoint_dir / f"{architecture}_best_model.pth"


def available_architectures(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[str]:
    return [a for a in ARCHITECTURES if checkpoint_path_for(a, checkpoint_dir).exists()]


def load_model_for(architecture: str, device: torch.device, checkpoint_dir: Path = CHECKPOINT_DIR):
    payload = load_training_checkpoint(checkpoint_path_for(architecture, checkpoint_dir), map_location=device)
    class_to_idx = payload["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = build_model(architecture, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_class


def preprocess(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return tfm(image.convert("RGB")).unsqueeze(0)


def predict_with(architecture: str, image: Image.Image, device: torch.device, checkpoint_dir: Path = CHECKPOINT_DIR):
    """Returns (label_text, overlay_or_None). Missing checkpoints are reported, not crashed on."""
    if not checkpoint_path_for(architecture, checkpoint_dir).exists():
        return "Not trained yet", None

    model, idx_to_class = load_model_for(architecture, device, checkpoint_dir)
    x = preprocess(image).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx].item())

    overlay = generate_gradcam_overlay(model, image.convert("RGB"), device=device, image_size=224)
    return f"{pred_label} ({confidence:.2%})", overlay


def pick_random_test_image(data_root: Path = Path("data")) -> tuple[Image.Image | None, str | None]:
    """Loads a random image from data/Testing -- a scan no model has ever trained on
    or used for checkpoint selection -- plus its true class (the folder it's filed
    under), so the UI can say whether each model actually got it right.
    Returns (None, None) if no test data is present."""
    test_dir = data_root / "Testing"
    if not test_dir.exists():
        return None, None

    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    random.shuffle(class_dirs)
    for class_dir in class_dirs:
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXT]
        if images:
            image = Image.open(random.choice(images)).convert("RGB")
            return image, class_dir.name
    return None, None


def _label_only(prediction_text: str) -> str | None:
    """"glioma (98.21%)" -> "glioma"; None for placeholders like "(not selected)"."""
    if " (" not in prediction_text:
        return None
    return prediction_text.split(" (")[0]


def _consensus_message(labels_by_architecture: dict[str, str | None]) -> str:
    predicted = {arch: label for arch, label in labels_by_architecture.items() if label is not None}
    if len(predicted) < 2:
        return ""
    distinct = set(predicted.values())
    if len(distinct) == 1:
        return f"✅ All {len(predicted)} selected models agree: **{next(iter(distinct))}**"
    parts = ", ".join(f"{arch} → {label}" for arch, label in predicted.items())
    return f"⚠️ Models disagree: {parts}"


def _ground_truth_message(true_label: str | None, labels_by_architecture: dict[str, str | None]) -> str:
    """Only meaningful for scans pulled via the random-test-scan button, where the
    true class is known from which data/Testing folder the image came from."""
    if true_label is None:
        return ""
    predicted = {arch: label for arch, label in labels_by_architecture.items() if label is not None}
    if not predicted:
        return ""
    num_correct = sum(1 for label in predicted.values() if label == true_label)
    per_model = ", ".join(
        f"{arch} ✅" if label == true_label else f"{arch} ❌ (said {label})"
        for arch, label in predicted.items()
    )
    return f"🎯 True label: **{true_label}** -- {num_correct}/{len(predicted)} correct ({per_model})"


def _predict_one(architecture: str, image: Image.Image, device: torch.device, num_threads: int):
    # Each worker thread pins its own torch intra-op thread count so N models
    # running concurrently split the machine's cores N ways instead of each
    # one trying to claim all of them (oversubscription -- slower than serial,
    # not faster). Only meaningful for CPU; a real CUDA/MPS device ignores it.
    torch.set_num_threads(max(1, num_threads))
    # CHECKPOINT_DIR read here (not as predict_with's default parameter) so a
    # reconfigured CHECKPOINT_DIR is always honored -- default parameter
    # values are bound once at function-definition time, not re-read per call.
    return predict_with(architecture, image, device, checkpoint_dir=CHECKPOINT_DIR)


def predict_all(image: Image.Image | None, selected_architectures: list[str], true_label: str | None = None) -> list:
    if image is None:
        outputs: list = []
        for _ in ARCHITECTURES:
            outputs.extend(["No image received", None])
        return ["No image uploaded.", *outputs]

    device = get_device()
    selected = [a for a in ARCHITECTURES if a in selected_architectures]
    # Split available cores across however many models are actually running
    # concurrently this call -- selecting just 1 model still gets full-core
    # torch performance, exactly like before this change.
    cpu_count = os.cpu_count() or 1
    threads_per_model = max(1, cpu_count // max(1, len(selected))) if device.type == "cpu" else cpu_count

    results: dict[str, tuple[str, object]] = {}
    if selected:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(_predict_one, architecture, image, device, threads_per_model): architecture
                for architecture in selected
            }
            for future in as_completed(futures):
                architecture = futures[future]
                results[architecture] = future.result()

    # Torch's global thread count is process-wide state -- restore it for
    # whatever runs next (a later single-model call, a benchmark run, etc.)
    # now that this call's workers have all finished.
    torch.set_num_threads(cpu_count)

    outputs = []
    labels_by_architecture: dict[str, str | None] = {}
    for architecture in ARCHITECTURES:
        if architecture in results:
            label, overlay = results[architecture]
            labels_by_architecture[architecture] = _label_only(label)
        else:
            label, overlay = "(not selected)", None
        outputs.extend([label, overlay])

    messages = [
        _ground_truth_message(true_label, labels_by_architecture),
        _consensus_message(labels_by_architecture),
    ]
    summary = "\n\n".join(m for m in messages if m)
    return [summary, *outputs]


def run_benchmark(
    selected_architectures: list[str], progress: gr.Progress = gr.Progress()
) -> tuple[list[list], str | None]:
    progress(0.0, desc="Loading held-out test set...")
    data_cfg = DataConfig()
    specs = [s for s in default_checkpoint_specs(CHECKPOINT_DIR) if s[0] in selected_architectures]
    if not specs:
        return [], None

    progress(0.2, desc=f"Evaluating {len(specs)} model(s) on scans none of them trained on...")
    results = benchmark_models(specs, data_cfg, output_dir=BENCHMARK_OUTPUT_DIR)
    save_benchmark_report(results, BENCHMARK_OUTPUT_DIR)
    progress(1.0, desc="Done")

    rows = []
    for r in results:
        if r.available:
            rows.append(
                [r.architecture, f"{r.accuracy:.4f}", f"{r.macro_f1:.4f}", f"{r.num_params:,}", f"{r.ms_per_image:.1f}"]
            )
        else:
            rows.append([r.architecture, "-", "-", "-", r.error])

    chart_path = BENCHMARK_OUTPUT_DIR / "accuracy_comparison.png"
    chart = str(chart_path) if chart_path.exists() else None
    return rows, chart


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
        gr.Markdown("# Brain Tumor MRI Classifier")

        with gr.Tab("Predict"):
            gr.Markdown(
                "Upload an MRI scan (or try a random one the models have never seen), "
                "pick which trained model(s) to run, and -- if you pick more than one -- "
                "see whether they agree."
            )
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload MRI image")
                model_select = gr.CheckboxGroup(
                    choices=list(ARCHITECTURES),
                    value=available_architectures(),
                    label="Models to run",
                )
            with gr.Row():
                predict_button = gr.Button("Predict", variant="primary")
                random_scan_button = gr.Button("🎲 Try a random never-seen scan")

            consensus_output = gr.Markdown()
            true_label_state = gr.State(None)

            result_components: list = []
            with gr.Row():
                for architecture in ARCHITECTURES:
                    with gr.Column():
                        gr.Markdown(f"**{architecture}**")
                        label_out = gr.Textbox(label="Prediction", interactive=False)
                        image_out = gr.Image(label="Grad-CAM heatmap")
                        result_components.extend([label_out, image_out])

            predict_outputs = [consensus_output, *result_components]
            # A manually uploaded image has no known ground truth -- reset
            # true_label_state before predicting so a stale label from a
            # previous random-scan click can't leak into this run's results.
            predict_button.click(lambda: None, inputs=[], outputs=[true_label_state]).then(
                predict_all, inputs=[image_input, model_select, true_label_state], outputs=predict_outputs
            )
            random_scan_button.click(pick_random_test_image, inputs=[], outputs=[image_input, true_label_state]).then(
                predict_all, inputs=[image_input, model_select, true_label_state], outputs=predict_outputs
            )

        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Runs the selected trained model(s) against `data/Testing` -- scans none "
                "of them were trained on or used for checkpoint selection. Pick one model "
                "to benchmark it individually, or all three to compare them at once."
            )
            benchmark_model_select = gr.CheckboxGroup(
                choices=list(ARCHITECTURES),
                value=list(ARCHITECTURES),
                label="Models to benchmark",
            )
            benchmark_button = gr.Button("Run Benchmark", variant="primary")
            results_table = gr.Dataframe(
                headers=["Architecture", "Accuracy", "Macro F1", "Params", "ms/image (or error)"],
                label="Comparison",
            )
            chart_output = gr.Image(label="Accuracy comparison chart")
            benchmark_button.click(
                run_benchmark, inputs=[benchmark_model_select], outputs=[results_table, chart_output]
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()

    # Custom routes (the async job API a stateless proxy like the Vercel demo
    # polls) have to be added to a FastAPI app *before* Gradio is mounted onto
    # it -- demo.app isn't built until launch(), which is too late.
    import uvicorn
    from fastapi import FastAPI

    from app.async_api import attach_async_routes

    fastapi_app = FastAPI()
    attach_async_routes(fastapi_app)
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

    # 0.0.0.0 + $PORT: works both locally (defaults to 127.0.0.1-equivalent on 7860)
    # and on cloud hosts (Render, etc.) that assign a port via the PORT env var.
    uvicorn.run(fastapi_app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

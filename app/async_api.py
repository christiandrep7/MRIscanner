"""A small async job API mounted onto the Gradio app's own FastAPI instance
(same port, no extra infra) -- lets an external stateless proxy (the Vercel
demo) start a prediction/benchmark job and poll for its result without
holding a connection open, since a full 3-model run can take well over a
minute on constrained hardware.

Design: submit -> background thread -> in-memory job store -> poll.
Each Vercel invocation is stateless, so this can't rely on any client-side
Job object; the job_id + polling GET is the only thing that has to survive
across separate HTTP requests, which an in-memory dict on the long-running
EC2 process handles fine.
"""
from __future__ import annotations

import base64
import io
import threading
import time
import uuid
from typing import Any

import numpy as np
from PIL import Image

_JOB_TTL_SECONDS = 30 * 60  # stale jobs are dropped so this dict can't grow forever
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _overlay_to_data_url(overlay: np.ndarray | None) -> str | None:
    if overlay is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _prune_stale_jobs() -> None:
    cutoff = time.monotonic() - _JOB_TTL_SECONDS
    with _lock:
        stale = [job_id for job_id, job in _jobs.items() if job["created"] < cutoff]
        for job_id in stale:
            del _jobs[job_id]


def _set_job(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs[job_id].update(fields)


def _run_predict_job(
    job_id: str, image: Image.Image, selected_architectures: list[str], true_label: str | None = None
) -> None:
    from app.gradio_app import predict_all

    try:
        outputs = predict_all(image, selected_architectures, true_label)
        summary = outputs[0]
        per_model = []
        pairs = outputs[1:]
        for i in range(0, len(pairs), 2):
            label, overlay = pairs[i], pairs[i + 1]
            per_model.append({"label": label, "overlay": _overlay_to_data_url(overlay)})
        _set_job(job_id, status="done", result={"summary": summary, "models": per_model})
    except Exception as e:  # noqa: BLE001 -- surfaced to the poller, not swallowed
        _set_job(job_id, status="error", error=str(e))


def _run_benchmark_job(job_id: str, selected_architectures: list[str]) -> None:
    from app.gradio_app import run_benchmark

    try:
        rows, chart_path = run_benchmark(selected_architectures)
        chart_data_url = None
        if chart_path:
            with open(chart_path, "rb") as f:
                chart_data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        _set_job(job_id, status="done", result={"rows": rows, "chart": chart_data_url})
    except Exception as e:  # noqa: BLE001
        _set_job(job_id, status="error", error=str(e))


def attach_async_routes(app) -> None:
    """Adds /api/jobs/* routes to a FastAPI app (before Gradio is mounted onto it
    -- see gr.mount_gradio_app in gradio_app.py's __main__ block)."""
    from fastapi import Body
    from fastapi.responses import JSONResponse

    @app.post("/api/jobs/predict")
    def start_predict(payload: dict = Body(...)):
        _prune_stale_jobs()
        try:
            image_b64 = payload["image"]
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"invalid image data: {e}"})
        selected = payload.get("selected_architectures") or ["resnet50", "efficientnet_b0", "vgg16"]
        true_label = payload.get("true_label")

        job_id = str(uuid.uuid4())
        with _lock:
            _jobs[job_id] = {"status": "running", "created": time.monotonic()}
        threading.Thread(
            target=_run_predict_job, args=(job_id, image, selected, true_label), daemon=True
        ).start()
        return {"job_id": job_id}

    @app.post("/api/jobs/benchmark")
    def start_benchmark(payload: dict = Body(...)):
        _prune_stale_jobs()
        selected = payload.get("selected_architectures") or ["resnet50", "efficientnet_b0", "vgg16"]

        job_id = str(uuid.uuid4())
        with _lock:
            _jobs[job_id] = {"status": "running", "created": time.monotonic()}
        threading.Thread(target=_run_benchmark_job, args=(job_id, selected), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            return {"status": "not_found"}
        return {k: v for k, v in job.items() if k != "created"}

    @app.get("/api/random-scan")
    def random_scan():
        # Fast (pick a file + base64-encode it) -- no background job needed,
        # unlike predict/benchmark which can take minutes.
        from app.gradio_app import pick_random_test_image

        image, true_label = pick_random_test_image()
        if image is None:
            return JSONResponse(
                status_code=404,
                content={"error": "no test dataset on this server (data/Testing missing)"},
            )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"image": data_url, "true_label": true_label}

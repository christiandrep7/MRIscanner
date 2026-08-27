from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.async_api import _overlay_to_data_url, attach_async_routes


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_client() -> TestClient:
    app = FastAPI()
    attach_async_routes(app)
    return TestClient(app)


def _poll_until_done(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")


def test_overlay_to_data_url_roundtrip():
    overlay = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay[:, :, 0] = 255
    url = _overlay_to_data_url(overlay)
    assert url.startswith("data:image/png;base64,")

    decoded = base64.b64decode(url.split(",", 1)[1])
    image = Image.open(io.BytesIO(decoded))
    assert image.size == (8, 8)


def test_overlay_to_data_url_none_passthrough():
    assert _overlay_to_data_url(None) is None


def test_job_status_not_found_for_unknown_id():
    client = _make_client()
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.json() == {"status": "not_found"}


def test_predict_job_end_to_end(monkeypatch, tmp_path: Path, fake_checkpoint: Path):
    import app.gradio_app as gradio_app

    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", fake_checkpoint.parent)

    client = _make_client()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    payload = {"image": _image_to_b64(image), "selected_architectures": ["resnet50"]}

    start_resp = client.post("/api/jobs/predict", json=payload)
    assert start_resp.status_code == 200
    job_id = start_resp.json()["job_id"]

    result = _poll_until_done(client, job_id)
    assert result["status"] == "done"
    assert "resnet50" in result["result"]["summary"] or result["result"]["summary"] == ""
    models = result["result"]["models"]
    assert len(models) == 3
    assert models[0]["overlay"] is not None  # resnet50 was selected
    assert models[1]["overlay"] is None  # efficientnet_b0 was not


def test_predict_job_reports_error_for_bad_image_data():
    client = _make_client()
    resp = client.post("/api/jobs/predict", json={"image": "not-valid-base64!!!"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_predict_job_with_true_label_marks_ground_truth(monkeypatch, fake_checkpoint: Path):
    import app.gradio_app as gradio_app

    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", fake_checkpoint.parent)

    client = _make_client()
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    payload = {
        "image": _image_to_b64(image),
        "selected_architectures": ["resnet50"],
        "true_label": "glioma",
    }

    start_resp = client.post("/api/jobs/predict", json=payload)
    job_id = start_resp.json()["job_id"]

    result = _poll_until_done(client, job_id)
    assert result["status"] == "done"
    # Only 1 model selected -> _ground_truth_message requires 2+ predictions to
    # say anything, so this just confirms the parameter reaches predict_all
    # without erroring; the multi-model case is covered by test_gradio_app.py.
    assert result["result"]["summary"] == "" or "True label" in result["result"]["summary"]


def test_random_scan_returns_image_and_true_label(monkeypatch):
    import app.gradio_app as gradio_app

    fake_image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    monkeypatch.setattr(gradio_app, "pick_random_test_image", lambda: (fake_image, "meningioma"))

    client = _make_client()
    resp = client.get("/api/random-scan")

    assert resp.status_code == 200
    body = resp.json()
    assert body["true_label"] == "meningioma"
    assert body["image"].startswith("data:image/png;base64,")


def test_random_scan_404_when_no_dataset(monkeypatch):
    import app.gradio_app as gradio_app

    monkeypatch.setattr(gradio_app, "pick_random_test_image", lambda: (None, None))

    client = _make_client()
    resp = client.get("/api/random-scan")

    assert resp.status_code == 404
    assert "error" in resp.json()


def test_benchmark_job_end_to_end(monkeypatch, tiny_data_config, all_fake_checkpoints):
    import app.gradio_app as gradio_app

    checkpoint_dir = next(iter(all_fake_checkpoints.values())).parent
    monkeypatch.setattr(gradio_app, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(gradio_app, "DataConfig", lambda: tiny_data_config)

    client = _make_client()
    start_resp = client.post("/api/jobs/benchmark", json={"selected_architectures": ["resnet50"]})
    job_id = start_resp.json()["job_id"]

    result = _poll_until_done(client, job_id, timeout_s=60.0)
    assert result["status"] == "done"
    assert len(result["result"]["rows"]) == 1
    assert result["result"]["rows"][0][0] == "resnet50"

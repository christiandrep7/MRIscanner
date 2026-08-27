"""Vercel Python serverless function: brain MRI tumor classification via ONNX
(EfficientNet-B0). No torch/torchvision here on purpose -- keeps the deployed
function small enough to fit Vercel's serverless size limits. Exported from
the full PyTorch checkpoint in the main MRIscanner project; see
vercel-demo/README.md for how it was produced and how to regenerate it.
"""
from __future__ import annotations

import base64
import json
import os
from http.server import BaseHTTPRequestHandler
from io import BytesIO

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "efficientnet_b0.onnx")
_session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def predict(image: Image.Image) -> dict:
    x = preprocess(image)
    logits = _session.run(["logits"], {"input": x})[0][0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            image_data = body.get("image", "")
            if "," in image_data:  # strip a data: URL prefix if present
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            result = predict(image)
            self._send_json(200, result)
        except Exception as e:  # noqa: BLE001 -- surfaced to the caller, not swallowed
            self._send_json(400, {"error": str(e)})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

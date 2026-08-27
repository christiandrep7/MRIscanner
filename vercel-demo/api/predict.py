"""Vercel serverless function: thin relay to the EC2-hosted async job API.

All actual compute (all 3 models, Grad-CAM, benchmark) runs on EC2 -- a full
run can take well over a minute on that instance's constrained hardware,
longer than Vercel's serverless function timeout allows for a single request.
So POSTs start a background job on EC2 and return instantly with a job_id;
GETs (polled by the frontend every few seconds) check that job's status.
No torch/onnx/numpy here at all -- this file only ever forwards JSON.

Routing (single entrypoint handles everything under /api/predict):
  POST {job_type: "predict", ...}   -> EC2 /api/jobs/predict
  POST {job_type: "benchmark", ...} -> EC2 /api/jobs/benchmark
  GET  ?job_id=...                  -> EC2 /api/jobs/{job_id}  (poll, either job type)
  GET  ?random=1                    -> EC2 /api/random-scan    (synchronous, not a job)
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

EC2_BASE_URL = "http://3.145.114.146:7860"
_TIMEOUT_SECONDS = 15  # relay calls are instant (submit/poll/random) -- never the long compute itself


def _relay(url: str, *, data: bytes | None = None) -> tuple[int, dict]:
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        return resp.status, json.loads(resp.read())


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) or b"{}"
            job_type = json.loads(body).get("job_type", "predict")
            endpoint = "benchmark" if job_type == "benchmark" else "predict"
            status, payload = _relay(f"{EC2_BASE_URL}/api/jobs/{endpoint}", data=body)
            self._send_json(status, payload)
        except Exception as e:  # noqa: BLE001 -- surfaced to the caller, not swallowed
            self._send_json(502, {"error": f"could not reach compute backend: {e}"})

    def do_GET(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            job_id = query.get("job_id", [None])[0]
            is_random = query.get("random", [None])[0]

            if job_id:
                status, payload = _relay(f"{EC2_BASE_URL}/api/jobs/{job_id}")
            elif is_random:
                status, payload = _relay(f"{EC2_BASE_URL}/api/random-scan")
            else:
                self._send_json(400, {"error": "expected a job_id or random query parameter"})
                return
            self._send_json(status, payload)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": f"could not reach compute backend: {e}"})

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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

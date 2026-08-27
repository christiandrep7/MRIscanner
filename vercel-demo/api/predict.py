"""Vercel serverless function: thin relay to the EC2-hosted async job API.

All actual compute (all 3 models, Grad-CAM) runs on EC2 -- a full 3-model run
can take well over a minute on that instance's constrained hardware, which is
longer than Vercel's serverless function timeout allows for a single request.
So this relays to /api/jobs/predict (returns instantly with a job_id) and
/api/jobs/{job_id} (polled by the frontend every couple seconds) rather than
holding one request open for the whole run. No torch/onnx/numpy here at all --
this file only ever forwards JSON.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

EC2_BASE_URL = "http://3.145.114.146:7860"
_TIMEOUT_SECONDS = 15  # relay calls are instant (submit) or instant (poll) -- never the long compute itself


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) or b"{}"
            req = Request(
                f"{EC2_BASE_URL}/api/jobs/predict",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                self._send_json(resp.status, json.loads(resp.read()))
        except Exception as e:  # noqa: BLE001 -- surfaced to the caller, not swallowed
            self._send_json(502, {"error": f"could not reach compute backend: {e}"})

    def do_GET(self) -> None:
        try:
            query = parse_qs(urlparse(self.path).query)
            job_id = query.get("job_id", [None])[0]
            if not job_id:
                self._send_json(400, {"error": "missing job_id query parameter"})
                return
            with urlopen(f"{EC2_BASE_URL}/api/jobs/{job_id}", timeout=_TIMEOUT_SECONDS) as resp:
                self._send_json(resp.status, json.loads(resp.read()))
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

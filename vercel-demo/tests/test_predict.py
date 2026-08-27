"""Tests for the Vercel relay function. It has no ML logic of its own --
everything here is about correctly forwarding to (and relaying failures from)
the EC2 compute backend, since that's the entire job of this file now."""
from __future__ import annotations

import io
import json
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
import predict as p  # noqa: E402


class _FakeUpstreamHandler:
    """A tiny real HTTP server standing in for the EC2 backend, so tests
    exercise predict.py's actual urllib calls instead of mocking them away."""

    def __init__(self, response_body: dict, status: int = 200):
        self.response_body = response_body
        self.status = status
        self.received_path = None
        self.received_body = None


def _start_fake_upstream(fake: _FakeUpstreamHandler) -> tuple[HTTPServer, int]:
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            fake.received_body = json.loads(self.rfile.read(length) or b"{}")
            fake.received_path = self.path
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(fake.response_body).encode())

        def do_GET(self):
            fake.received_path = self.path
            self.send_response(fake.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(fake.response_body).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


class _FakeRequestHandler(p.handler):
    """Bypasses BaseHTTPRequestHandler's socket-based __init__ so do_POST/do_GET
    can be invoked directly in a test with a controlled request body/path."""

    def __init__(self, body: bytes = b"", path: str = "/"):
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(body))}
        self.path = path
        self._responses: list[tuple[int, dict]] = []

    def send_response(self, status):
        self._status = status

    def send_header(self, *_args):
        pass

    def end_headers(self):
        pass

    def log_message(self, *args):
        pass


def test_predict_post_relays_to_upstream_and_returns_job_id(monkeypatch):
    fake = _FakeUpstreamHandler({"job_id": "abc-123"})
    server, port = _start_fake_upstream(fake)
    monkeypatch.setattr(p, "EC2_BASE_URL", f"http://127.0.0.1:{port}")

    body = json.dumps({"image": "xyz", "selected_architectures": ["resnet50"]}).encode()
    h = _FakeRequestHandler(body=body)
    h.do_POST()

    assert h._status == 200
    assert json.loads(h.wfile.getvalue()) == {"job_id": "abc-123"}
    assert fake.received_path == "/api/jobs/predict"
    assert fake.received_body == {"image": "xyz", "selected_architectures": ["resnet50"]}
    server.shutdown()


def test_predict_get_relays_job_id_from_query_string(monkeypatch):
    fake = _FakeUpstreamHandler({"status": "done", "result": {"summary": "ok", "models": []}})
    server, port = _start_fake_upstream(fake)
    monkeypatch.setattr(p, "EC2_BASE_URL", f"http://127.0.0.1:{port}")

    h = _FakeRequestHandler(path="/api/predict?job_id=abc-123")
    h.do_GET()

    assert h._status == 200
    assert json.loads(h.wfile.getvalue())["status"] == "done"
    assert fake.received_path == "/api/jobs/abc-123"
    server.shutdown()


def test_predict_get_missing_job_id_returns_400():
    h = _FakeRequestHandler(path="/api/predict")
    h.do_GET()

    assert h._status == 400
    assert "error" in json.loads(h.wfile.getvalue())


def test_predict_post_returns_502_when_upstream_unreachable(monkeypatch):
    monkeypatch.setattr(p, "EC2_BASE_URL", "http://127.0.0.1:1")  # nothing listens here

    h = _FakeRequestHandler(body=json.dumps({"image": "xyz"}).encode())
    h.do_POST()

    assert h._status == 502
    assert "error" in json.loads(h.wfile.getvalue())


def test_predict_get_returns_502_when_upstream_unreachable(monkeypatch):
    monkeypatch.setattr(p, "EC2_BASE_URL", "http://127.0.0.1:1")

    h = _FakeRequestHandler(path="/api/predict?job_id=abc-123")
    h.do_GET()

    assert h._status == 502
    assert "error" in json.loads(h.wfile.getvalue())

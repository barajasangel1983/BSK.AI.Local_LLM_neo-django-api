"""Mock-server tests for the Docling client (no real BSK calls)."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from ingestion.docling_client import DoclingClient, DoclingError


def make_server(status_states):
    """Start an in-process Docling stub.

    ``status_states`` is the sequence of states the status endpoint returns
    (the last one is held if the list runs out).
    """

    class Handler(BaseHTTPRequestHandler):
        states = list(status_states)

        def log_message(self, *args):  # silence
            pass

        def _send_json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.startswith("/v1/convert/source/async"):
                self._send_json({"task_id": "task-123"})
            else:
                self._send_json({"detail": "not found"}, 404)

        def do_GET(self):
            if self.path.startswith("/v1/status/poll/task-123"):
                state = (
                    self.states.pop(0)
                    if len(self.states) > 1
                    else self.states[0]
                )
                self._send_json({"state": state, "task_id": "task-123"})
            elif self.path.startswith("/v1/result/task-123"):
                self._send_json(
                    {"document": {"name": "test.pdf", "markdown": "# Hi"}}
                )
            else:
                self._send_json({"detail": "not found"}, 404)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def docling_server():
    server, thread = make_server(["in_progress", "in_progress", "finished"])
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


class FakeClock:
    """Monotonic-style clock that only advances when we tell it to."""

    def __init__(self):
        self._v = 1000.0

    def __call__(self):
        return self._v

    def advance(self, seconds):
        self._v += seconds


def test_submit_poll_fetch(docling_server):
    client = DoclingClient(base_url=docling_server)

    task_id = client.submit_file(b"%PDF-1.4 fake", "test.pdf")
    assert task_id == "task-123"

    status = client.poll_status(task_id, sleep=lambda s: None)
    assert status["state"] == "finished"

    result = client.fetch_result(task_id)
    assert result["document"]["markdown"] == "# Hi"


def test_failed_task_raises():
    server, thread = make_server(["failed"])
    try:
        client = DoclingClient(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        with pytest.raises(DoclingError, match="failed"):
            client.poll_status("task-123", sleep=lambda s: None)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_poll_timeout():
    server, thread = make_server(["in_progress"])
    try:
        client = DoclingClient(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        clock = FakeClock()

        def fake_sleep(delay):
            # Simulate wall time passing; also advance past the deadline.
            clock.advance(max(delay, 120.0))

        with pytest.raises(DoclingError, match="timed out"):
            client.poll_status(
                "task-123",
                timeout_s=600.0,
                sleep=fake_sleep,
                now=clock,
            )
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_progress_callback_receives_updates():
    server, thread = make_server(["in_progress", "finished"])
    try:
        client = DoclingClient(base_url=f"http://127.0.0.1:{server.server_address[1]}")
        events = []
        client.submit_file(b"x", "x.pdf", progress=events.append)
        client.poll_status("task-123", sleep=lambda s: None, progress=events.append)
        assert any("submitted" in e for e in events)
        assert any("polling" in e for e in events)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_submit_http_error():
    """Submit against a down server raises a requests connection error."""
    client = DoclingClient(base_url="http://127.0.0.1:1")
    with pytest.raises((DoclingError, requests.exceptions.ConnectionError)):
        client.submit_file(b"x", "x.pdf")

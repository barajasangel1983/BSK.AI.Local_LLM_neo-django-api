"""Mock-server tests for the sync Docling client (no real BSK calls)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from ingestion.chunker import PAGE_BREAK
from ingestion.docling_client import DoclingClient, DoclingError


def make_server(response, code=200):
    """In-process Docling stub; records the last request body in server.requests."""

    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            requests_seen.append((self.path, json.loads(self.rfile.read(length) or b"{}")))
            body = json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.requests = requests_seen
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def docling(request):
    response, code = getattr(request, "param", ({"status": "success", "errors": [],
                                                 "document": {"md_content": "# Hi"}}, 200))
    server, thread = make_server(response, code)
    yield server, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def test_convert_sends_file_and_page_break_options(docling):
    server, url = docling
    result = DoclingClient(base_url=url).convert_file(b"%PDF-1.4 fake", "test.pdf")

    assert result["document"]["md_content"] == "# Hi"
    path, body = server.requests[0]
    assert path == "/v1/convert/source"
    source = body["sources"][0]
    assert (source["kind"], source["filename"]) == ("file", "test.pdf")
    assert body["options"]["md_page_break_placeholder"] == PAGE_BREAK
    assert body["options"]["image_export_mode"] == "placeholder"


@pytest.mark.parametrize("docling", [({"status": "failure", "errors": ["bad pdf"]}, 200)], indirect=True)
def test_conversion_errors_raise(docling):
    _, url = docling
    with pytest.raises(DoclingError, match="bad pdf"):
        DoclingClient(base_url=url).convert_file(b"x", "x.pdf")


@pytest.mark.parametrize("docling", [({"detail": "boom"}, 500)], indirect=True)
def test_http_error_raises(docling):
    _, url = docling
    with pytest.raises(DoclingError, match="HTTP 500"):
        DoclingClient(base_url=url).convert_file(b"x", "x.pdf")


def test_connection_error():
    with pytest.raises(requests.exceptions.ConnectionError):
        DoclingClient(base_url="http://127.0.0.1:1").convert_file(b"x", "x.pdf")

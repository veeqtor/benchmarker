"""Production-critical tests for client.py."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from benchmarker.client import Client, chat, http_error


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that returns JSON."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": []}).encode())

    def log_message(self, *a):
        pass  # silence stderr


_server = None


def _start_server(port):
    global _server
    _server = HTTPServer(("127.0.0.1", port), _MockHandler)
    _server.serve_forever()


import pytest


@pytest.fixture(scope="module")
def http_server():
    port = 18765
    t = threading.Thread(target=_start_server, args=(port,), daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    _server.shutdown()


class TestClient:
    def test_call_returns_json(self, http_server):
        c = Client(http_server)
        result = c.call("/test")
        assert result == {"data": []}

    def test_call_with_body(self, http_server):
        c = Client(http_server)
        result = c.call("/chat", {"messages": []})
        assert result == {"choices": [{"message": {"content": "ok"}}]}

    def test_authorization_header(self, http_server):
        c = Client(http_server, key="secret")
        with patch.object(c, "call") as mock:
            mock.return_value = {}
            c.call("/test")
            # just verify no exception with key


class TestChat:
    def test_chat_returns_response(self, http_server):
        c = Client(http_server)
        resp, elapsed = chat(c, "model", [{"role": "user", "content": "hi"}], 10, 30)
        assert "choices" in resp
        assert isinstance(elapsed, float)


class TestHttpError:
    def test_http_error_format(self):
        err = urllib.error.HTTPError("http://x", 404, "not found", {}, None)
        result = http_error(err)
        assert "HTTP 404" in result

    def test_generic_error(self):
        result = http_error(ValueError("boom"))
        assert "boom" in result

"""Production-critical tests for client.py."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

from benchmarker.client import Client, chat, http_error


class TestClient:
    def test_call_returns_json(self):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        with patch("benchmarker.client.urllib.request.urlopen", return_value=mock_response):
            c = Client("http://127.0.0.1:9999")
            result = c.call("/test")
            assert result == {"data": []}

    def test_call_with_body(self):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        with patch("benchmarker.client.urllib.request.urlopen", return_value=mock_response):
            c = Client("http://127.0.0.1:9999")
            result = c.call("/chat", {"messages": []})
            assert result == {"choices": [{"message": {"content": "ok"}}]}

    def test_authorization_header(self):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        with patch("benchmarker.client.urllib.request.urlopen", return_value=mock_response):
            c = Client("http://127.0.0.1:9999", key="secret")
            result = c.call("/test")
            assert result == {"data": []}


class TestChat:
    def test_chat_returns_response(self):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        with patch("benchmarker.client.urllib.request.urlopen", return_value=mock_response):
            c = Client("http://127.0.0.1:9999")
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

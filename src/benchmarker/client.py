"""HTTP client for llama-swap and llama-server APIs.

Provides a simple POST/GET client and helpers for chat completions.
"""

import json
import time
import urllib.error
import urllib.request


class Client:
    """HTTP client for llama-swap / llama-server REST APIs."""

    def __init__(self, url, key=""):
        self.url = url.rstrip("/")
        self.key = key

    def call(self, path, body=None, timeout=900):
        """Send a JSON request and return the parsed response."""
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path, data, h, method="POST" if body is not None else "GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def base(self, model):
        """Return (base_path, props) by trying /upstream/<model> first, then root."""
        for b in (f"/upstream/{model}", ""):
            try:
                return b, self.call(b + "/props")
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    continue
                raise
        raise RuntimeError("cannot read /props")


def http_error(e):
    """Format an HTTP error into a readable string."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = ""
        return f"HTTP {e.code}: {body[:200]}"
    return str(e)[:200]


def chat(client, model, messages, max_tokens, timeout, think=False):
    """Send a chat completion request and return (response, elapsed_seconds)."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "cache_prompt": False,
        "messages": messages,
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    data = client.call("/v1/chat/completions", body, timeout=timeout)
    return data, time.time() - t0


__all__ = [
    "Client",
    "chat",
    "http_error",
]

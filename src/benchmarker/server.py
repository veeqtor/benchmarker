"""Server lifecycle management.

Tracks running llama-server instances, provides start/stop/wait_vram_released,
and builds server/bench flags from settings.
"""

import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request

from benchmarker.client import Client
from benchmarker.config import RESULTS
from benchmarker.utils import free_port, port_free

# ── Track running servers ────────────────────────────────────────────────────
ACTIVE_SERVERS = []


def stop_all():
    """Stop all tracked servers."""
    for s in ACTIVE_SERVERS:
        s.stop()


# ── Server class ─────────────────────────────────────────────────────────────
class Server:
    """Wraps a llama-server process for benchmarking one model/setting."""

    def __init__(self, cfg, m, ctx, s, tag):
        self.cfg, self.m, self.ctx, self.s = cfg, m, ctx, s
        os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
        self.log_path = os.path.join(RESULTS, "logs", f"{m['name']}-{tag}.log")
        self.proc = None
        self.port = cfg["port"] or free_port()

    def cmd(self):
        fit = ["--fit", "off"] if self.cfg.get("_fit_flag") else []
        return (
            [
                self.cfg["server"],
                "-m",
                self.m["path"],
                "-c",
                str(self.ctx),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ]
            + self.cfg["common_flags"]
            + fit
            + self.m.get("mm_flags", [])
            + server_flags(self.m, self.s)
        )

    def start(self):
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} is in use")
        t0 = time.time()
        self.logf = open(self.log_path, "w")
        self.logf.write("$ " + " ".join(shlex.quote(c) for c in self.cmd()) + "\n")
        self.logf.flush()
        self.proc = subprocess.Popen(
            self.cmd(),
            stdout=self.logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=self.cfg.get("_env"),
        )
        ACTIVE_SERVERS.append(self)
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() - t0 < self.cfg["load_timeout_s"]:
            if self.proc.poll() is not None:
                return False, time.time() - t0
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return True, time.time() - t0
            except Exception:
                pass

            time.sleep(0.5)
        return False, time.time() - t0

    def client(self):
        return Client(f"http://127.0.0.1:{self.port}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=30)
                except Exception:
                    pass

        if self.proc:
            self.logf.close()
        if self in ACTIVE_SERVERS:
            ACTIVE_SERVERS.remove(self)
        self.proc = None

    def error_line(self):
        try:
            lines = open(self.log_path, errors="replace").read().splitlines()[1:]
        except OSError:
            return ""
        for pat in (
            r"out of memory|failed to allocate|unable to allocate|cudaMalloc|hipMalloc",
            r"error|failed",
        ):
            for l in lines:
                if re.search(pat, l, re.IGNORECASE):
                    return l.strip()[:200]
        return lines[-1].strip()[:200] if lines else ""

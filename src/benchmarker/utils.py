"""General utilities used across the benchmarker package."""

import datetime
import os
import re
import socket
import subprocess
import sys


def fmt_dur(sec):
    """Format seconds into a human-readable duration string."""
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}:{s:02d}"


def xp(p):
    """Expand user vars (~, $VAR) in a path string."""
    return os.path.expanduser(os.path.expandvars(p)) if isinstance(p, str) else p


def tilde(p):
    """Replace the home directory prefix with ~."""
    home = os.path.expanduser("~")
    return "~" + p[len(home) :] if p.startswith(home + os.sep) else p


def now():
    """Return the current time as an ISO-8601 string."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def run_text(cmd, env=None, timeout=60):
    """Run a command and return stdout+stderr as a single string."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout + r.stderr
    except Exception as e:
        return f"ERROR: {e}"


def free_port():
    """Find a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_free(port):
    """Check if a TCP port is available on localhost."""
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def build_id(cfg):
    """Parse the llama.cpp build version from the server --version output."""
    m = re.search(r"version: (\S+)", run_text([cfg["server"], "--version"], cfg.get("_env")))
    return m.group(1) if m else "?"


def service_active(name):
    """Check if a systemd service is currently active."""
    try:
        r = subprocess.run(["systemctl", "is-active", "--quiet", name], capture_output=True)
    except FileNotFoundError:
        return False
    return r.returncode == 0


def other_servers():
    """Find running llama-server processes (other than the one managed by this tool)."""
    out = run_text(["pgrep", "-a", "-x", "llama-server"])
    return [l for l in out.splitlines() if l.strip() and "pgrep" not in l]


def api_key(cfg, required=True):
    """Read the API key from environment variable or key file."""
    v = os.environ.get(cfg["key_env"], "").strip()
    if v:
        return v
    path = cfg["key_file"]
    try:
        text = open(path).read()
    except FileNotFoundError:
        text = None
    except PermissionError:
        sys.exit(
            f"Cannot read {path}. Run: sudo chown $USER: {path}, or export {cfg['key_env']}=<key>"
        )
    if text is not None:
        lines = [
            l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")
        ]
        for l in lines:
            k, sep, val = l.partition("=")
            if sep and k.strip().removeprefix("export ").strip() == cfg["key_env"]:
                return val.strip().strip('"').strip("'")
        if len(lines) == 1 and "=" not in lines[0]:
            return lines[0]
    if required:
        sys.exit(f"No API key: set {cfg['key_env']} or key_file in models.toml, or use --no-key")
    return ""


__all__ = [
    "api_key",
    "build_id",
    "fmt_dur",
    "free_port",
    "now",
    "other_servers",
    "port_free",
    "run_text",
    "service_active",
    "tilde",
    "xp",
]

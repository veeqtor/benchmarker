#!/usr/bin/env python3
"""llmbench - benchmark and tuning tool for local GGUF models (llama.cpp + llama-swap).

Commands
  doctor   Check paths, GPU devices, llama-swap, API key and permissions
  init     Scan the models folder and add new models to models.toml (finds mmproj files)
  info     Show each model: architecture, layers, experts, native context, size, vision
  sweep    Fit (text + image), speed, tuning and optional long-prompt accuracy, then report
  fill     Long-prompt accuracy test through the running llama-swap
  vision   Image test through the running llama-swap (vision models)
  status   Show the progress of the running (or last) sweep, fill or vision command
  report   Write one Markdown report with all results (send this file for review)
  swap     Write a llama-swap config from the latest results (--install to apply it)

Typical use
  ./llmbench.py doctor
  ./llmbench.py init
  ./llmbench.py sweep all --stop-swap            # fit + speed + tune + image check
  ./llmbench.py sweep qwen3.8-27b --accuracy --stop-swap
  ./llmbench.py swap --install

Paths are found automatically (llama.cpp in /opt/llama.cpp or ~/llama.cpp, models in
/srv/models or ~/models). Change them in models.toml [defaults], with environment
variables (LLMBENCH_LLAMA_DIR, LLMBENCH_MODELS_DIR, LLMBENCH_BACKEND) or with options.

Decision rules (target context, speed limits, recall test) are in PROFILES: strict (the rules
of 2.2), balanced (default) and long. Choose with --profile, LLMBENCH_PROFILE or
profile = "..." in models.toml. 'report' and 'swap' judge saved results with the chosen profile.
The rules are tested by test_llmbench.py (python3 -m unittest test_llmbench).

Only the Python standard library is used (Python 3.11 or later).
"""

import argparse
import atexit
import base64
import collections
import contextlib
import csv
import datetime
import fcntl
import glob
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import zlib

VERSION = "2.3"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CONFIG = os.path.join(HERE, "models.toml")
RESULTS = os.path.join(HERE, "results")

DEFAULTS = {
    # paths ("" = find automatically)
    "llama_dir": "",  # folder of the llama.cpp checkout
    "backend": "rocm",  # build folder: <llama_dir>/build-<backend>/bin
    "server": "",  # full path to llama-server (overrides llama_dir/backend)
    "bench": "",  # full path to llama-bench
    "models_dir": "",  # model paths in models.toml are relative to this folder
    # GPU
    "visible_devices": "auto",  # auto = only the GPU with the most VRAM; "" = off; "0" = this index
    "gpu_sysfs": "",  # "" = the AMD GPU with the most VRAM in /sys/class/drm
    "min_free_vram_mb": 512,  # use 1024 if the screen is on this GPU
    "min_free_ram_mb": 4096,
    # llama-swap
    "swap_url": "http://127.0.0.1:9931",
    "swap_service": "llama-swap",
    "swap_config": "/etc/llama-swap.yaml",
    "key_env": "LLAMA_SWAP_API_KEY",  # environment variable, also read from key_file
    "key_file": "/etc/llama-swap.env",  # KEY=value file or a file with only the key
    "swap_api_keys": True,  # write apiKeys: ["${env.<key_env>}"] into the generated config
    "swap_health_timeout": 600,
    "swap_ttl": 1800,
    "swap_start_port": 10001,
    # test server
    "port": 0,  # 0 = a free port
    "common_flags": ["--jinja", "-fa", "on", "--parallel", "1"],  # same as production
    "load_timeout_s": 900,
    "request_timeout_s": 7200,
    # fit
    "moe_contexts": [32768, 65536, 131072, 196608, 262144],
    "dense_contexts": [16384, 32768, 49152, 65536, 81920, 98304, 131072, 163840, 196608, 262144],
    "moe_cache_types": ["q8_0"],
    "dense_cache_types": ["q8_0", "q4_0"],
    "fit_warmup": True,  # send a short text request before the memory is measured
    "image_fit": True,  # vision models: send an image before the memory is measured
    "image_px": 896,  # size of the test image (square)
    # speed and tuning
    "bench_depths": [0, 8192, 32768],
    "bench_deep_fill_pct": 50,  # contexts >= target_context: also measure speed at this fill (0 = off)
    "bench_reps": 2,
    "bench_fa": "1",
    "tune_ubatch": [256, 512, 1024, 2048],
    # accuracy
    "fill_pct": [25, 50, 75, 90],  # fill levels of the 'fill' command (llama-swap)
    "accuracy_max_steps": 4,  # sweep --accuracy: test at most this many contexts per cache type
    "vision_runs": 3,  # image questions per vision test
    # The decision rules (speed, accuracy, target) come from the profile below.
    "profile": "balanced",
}

# Decision rules. Order of priority: DEFAULTS < PROFILES[profile] < models.toml [defaults].
# All profiles must define the same keys (checked by test_llmbench.py).
PROFILES = {
    # Rules of version 2.2: exact recall of 3 values at 50% and 90% fill, 10% speed loss.
    "strict": {
        "target_context": 65536,
        "speed_tolerance_pct": 10,  # a context is slow if tg128 is more than this much below the best
        "min_tg_tps": 0,  # minimum generation speed (t/s) with a filled context (0 = off)
        "needle_style": "line",  # long-prompt test: "line" (Line 123: ...) or "keyed" (Project name: ...)
        "needles": 3,  # values to find in one test
        "accuracy_pct": [50, 90],  # fill levels of the long-prompt test
        "accuracy_min_pct": 100,  # share of the values that must be correct
    },
    # Default. A context is good if it keeps 75% of the speed, stays above 15 t/s with a filled
    # context, and recalls 4 of 5 values at 50% and 75% fill.
    "balanced": {
        "target_context": 131072,
        "speed_tolerance_pct": 25,
        "min_tg_tps": 15,
        "needle_style": "keyed",
        "needles": 5,
        "accuracy_pct": [50, 75],
        "accuracy_min_pct": 80,
    },
    # Largest context first: slower generation is accepted and recall is checked at 50% fill only.
    "long": {
        "target_context": 131072,
        "speed_tolerance_pct": 40,
        "min_tg_tps": 10,
        "needle_style": "keyed",
        "needles": 5,
        "accuracy_pct": [50],
        "accuracy_min_pct": 80,
    },
}
NEEDLE_STYLES = ("keyed", "line")
CACHE_QUALITY = [
    "f32",
    "bf16",
    "f16",
    "q8_0",
    "q5_1",
    "q5_0",
    "q4_1",
    "q4_0",
    "iq4_nl",
]  # best first

PATH_KEYS = ("llama_dir", "server", "bench", "models_dir", "key_file", "gpu_sysfs", "swap_config")
LLAMA_DIR_CANDIDATES = ["/opt/llama.cpp", "~/llama.cpp", "/usr/local/llama.cpp"]
MODELS_DIR_CANDIDATES = ["/srv/models", "~/models", "~/.cache/llama.cpp"]
MMPROJ_PREFERENCE = ["f16", "bf16", "q8_0", "f32"]
DEVICE_ENV = {
    "rocm": "HIP_VISIBLE_DEVICES",
    "hip": "HIP_VISIBLE_DEVICES",
    "vulkan": "GGML_VK_VISIBLE_DEVICES",
    "cuda": "CUDA_VISIBLE_DEVICES",
}
ACTIVE_SERVERS = []


# ---------------------------------------------------------------- utilities
def fmt_dur(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}:{s:02d}"


class Progress:
    """Where the run is: model x of y, phase, step, elapsed time.
    Every log line gets a short tag, a banner starts each model and phase, a heartbeat line appears
    when nothing was printed for a minute, and results/progress.json is updated for the 'status'
    command. All methods do nothing until begin() is called."""

    def __init__(self, clock=time.time):
        self.clock, self.active = clock, False
        self.last_output = 0.0
        self._stop = threading.Event()

    def begin(self, kind, models):
        self.active, self.kind, self.models = True, kind, list(models)
        self.t0 = self.model_t0 = self.last_output = self.clock()
        self.done_secs, self.mi, self.model, self._open = [], 0, "", False
        self.phase_name, self.step, self.total, self.detail, self.state = "", 0, 0, "", "running"
        self._write()

    def start_model(self, name):
        if not self.active:
            return
        self.finish_model()
        self.mi, self.model, self.model_t0, self._open = self.mi + 1, name, self.clock(), True
        self.phase_name, self.step, self.total, self.detail = "", 0, 0, ""
        self._write()

    def finish_model(self):
        if self._open:
            self.done_secs.append(self.clock() - self.model_t0)
            self._open = False

    def phase(self, name, total=0, detail=""):
        if not self.active:
            return
        self.phase_name, self.step, self.total, self.detail = name, 0, total, detail
        self.say(f"---- {self.label()} ----")
        self._write()

    def tick(self, detail=""):
        if not self.active:
            return
        self.step += 1
        self.detail = detail or self.detail
        self._write()

    def note(self, detail):
        if not self.active:
            return
        self.detail = detail
        self._write()

    def step_text(self):
        if self.total:
            return f"{self.step}/{self.total}"
        return f"#{self.step}" if self.step else ""

    def eta_rest_s(self):
        """Seconds for the models after this one, from the models finished so far (None = unknown)."""
        left = len(self.models) - self.mi
        if not self.done_secs or left <= 0:
            return None
        return sum(self.done_secs) / len(self.done_secs) * left

    def tag(self):
        parts = [
            f"{self.mi}/{len(self.models)}",
            self.phase_name,
            self.step_text(),
            f"+{fmt_dur(self.clock() - self.t0)}",
        ]
        return "[" + " ".join(p for p in parts if p) + "]"

    def label(self):
        eta = self.eta_rest_s()
        parts = [
            f"{self.kind} model {self.mi}/{len(self.models)}: {self.model}",
            f"{self.phase_name} {self.step_text()}".strip(),
            self.detail,
            f"run {fmt_dur(self.clock() - self.t0)}",
            f"model {fmt_dur(self.clock() - self.model_t0)}",
        ]
        if eta is not None:
            parts.append(f"later models ~{fmt_dur(eta)}")
        return " | ".join(p for p in parts if p)

    def say(self, text):
        self.last_output = self.clock()
        print(text, flush=True)

    def _write(self):
        """results/progress.json, for 'llmbench status' in a second terminal. Never stops the run."""
        if not self.active:
            return
        try:
            os.makedirs(RESULTS, exist_ok=True)
            data = {
                "run_id": RUN_ID,
                "kind": self.kind,
                "state": self.state,
                "pid": os.getpid(),
                "model": self.model,
                "model_index": self.mi,
                "model_total": len(self.models),
                "models": self.models,
                "phase": self.phase_name,
                "step": self.step,
                "total": self.total,
                "detail": self.detail,
                "elapsed_s": round(self.clock() - self.t0),
                "model_elapsed_s": round(self.clock() - self.model_t0),
                "eta_rest_s": None if self.eta_rest_s() is None else round(self.eta_rest_s()),
                "updated": time.time(),
                "label": self.label(),
            }
            path = os.path.join(RESULTS, "progress.json")
            with open(path + ".tmp", "w") as f:
                json.dump(data, f, indent=2)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def start_heartbeat(self, every=60):
        """Print one line when nothing was printed for 'every' seconds (model load, long prompts)."""

        def loop():
            while not self._stop.wait(min(10, every)):
                if self.active and self.clock() - self.last_output >= every:
                    self.say(f"  ... still working: {self.label()}")
                    self._write()

        self._stop.clear()
        threading.Thread(target=loop, daemon=True).start()

    def end(self, state="finished"):
        if self.active:
            self.finish_model()
            self.state, self.phase_name, self.step, self.total = state, state, 0, 0
            self.detail = f"{self.mi} of {len(self.models)} models"
            self._write()
            self.say(f"---- {self.kind} {state} after {fmt_dur(self.clock() - self.t0)} ----")
        self._stop.set()
        self.active = False


PROGRESS = Progress()


@contextlib.contextmanager
def tracked(kind, names):
    """Show and record the progress of one command; the final state is finished, stopped or failed."""
    PROGRESS.begin(kind, names)
    PROGRESS.start_heartbeat()
    state = "failed"
    try:
        yield
        state = "finished"
    except KeyboardInterrupt:
        state = "stopped"
        raise
    finally:
        PROGRESS.end(state)


def log(msg=""):
    if PROGRESS.active and msg.strip() and not msg.lstrip().startswith("==="):
        msg = f"{PROGRESS.tag()} {msg}"
    PROGRESS.last_output = PROGRESS.clock()
    print(msg, flush=True)


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def xp(p):
    return os.path.expanduser(os.path.expandvars(p)) if isinstance(p, str) else p


def tilde(p):
    home = os.path.expanduser("~")
    return "~" + p[len(home) :] if p.startswith(home + os.sep) else p


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def append_csv(name, row):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fields = list(row.keys())
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:  # older format: keep it, start a new file
            os.rename(path, path[:-4] + f".old-{RUN_ID}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def state_path(model, backend):
    return os.path.join(RESULTS, f"state-{model}@{backend}.json")


def save_state(state):
    os.makedirs(RESULTS, exist_ok=True)
    tmp = state_path(state["model"], state["backend"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path(state["model"], state["backend"]))


def load_states(backend=None):
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "state-*.json"))):
        with open(p) as f:
            s = json.load(f)
        s.setdefault("backend", "rocm")  # files from version 1
        if backend and s["backend"] != backend:
            continue
        key = s["model"] if backend else f"{s['model']}@{s['backend']}"
        if key not in out or s["run_id"] > out[key]["run_id"]:
            out[key] = s
    return out


def run_text(cmd, env=None, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.stdout + r.stderr
    except Exception as e:
        return f"ERROR: {e}"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------- GGUF metadata
_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def gguf_meta(path):
    """Read the key/value metadata of a GGUF file (no tensors)."""
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        f.read(4)  # version
        _, nkv = struct.unpack("<QQ", f.read(16))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    if n > 256:
                        for _ in range(n):
                            (ln,) = struct.unpack("<Q", f.read(8))
                            f.seek(ln, 1)
                        return f"<{n} strings>"
                    return [rstr() for _ in range(n)]
                if at in _FMT:
                    size = struct.calcsize(_FMT[at])
                    if n > 256:
                        f.seek(size * n, 1)
                        return f"<{n} items>"
                    return [struct.unpack(_FMT[at], f.read(size))[0] for _ in range(n)]
                return [rval(at) for _ in range(n)]
            return struct.unpack(_FMT[t], f.read(struct.calcsize(_FMT[t])))[0]

        for _ in range(nkv):
            key = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            meta[key] = rval(t)
    return meta


def model_files(path):
    if not re.search(r"-\d{5}-of-\d{5}\.gguf$", path):
        return [path]
    return sorted(glob.glob(re.sub(r"-\d{5}-of-", "-*-of-", path)))


def is_mmproj(path):
    return "mmproj" in os.path.basename(path).lower()


def is_extra_part(path):
    return bool(re.search(r"-(\d{5})-of-\d{5}\.gguf$", path)) and not re.search(
        r"-00001-of-\d{5}\.gguf$", path
    )


def model_info(path):
    meta = gguf_meta(path)
    arch = meta.get("general.architecture", "?")
    g = lambda k, d=0: meta.get(f"{arch}.{k}", d)
    experts = g("expert_count", 0) or 0
    kv = g("attention.head_count_kv", 0)
    if isinstance(kv, list):  # hybrid models: 0 = no KV cache in that layer
        attn_layers = sum(1 for x in kv if x)
        kv_heads = max(kv) if kv else 0
    else:
        attn_layers, kv_heads = None, kv
    size = sum(os.path.getsize(p) for p in model_files(path))
    return {
        "name": meta.get("general.name", os.path.basename(path)),
        "arch": arch,
        "layers": int(g("block_count", 0) or 0),
        "native_ctx": int(g("context_length", 0) or 0),
        "experts": int(experts),
        "experts_used": int(g("expert_used_count", 0) or 0),
        "kv_heads": kv_heads,
        "attn_layers": attn_layers,
        "file_type": meta.get("general.file_type", ""),
        "size_mib": round(size / 1048576),
        "type": "moe" if experts else "dense",
    }


def find_mmproj(model_path):
    """Best mmproj file in the folder of the model (F16 first), or ''."""
    files = [
        p for p in glob.glob(os.path.join(os.path.dirname(model_path), "*.gguf")) if is_mmproj(p)
    ]
    if not files:
        return ""

    def rank(p):
        b = os.path.basename(p).lower()
        for i, q in enumerate(MMPROJ_PREFERENCE):
            if re.search(rf"[-_.]{q}[-_.]", b + "."):
                return i
        return len(MMPROJ_PREFERENCE)

    return sorted(files, key=lambda p: (rank(p), p))[0]


# ---------------------------------------------------------------- paths
def first_existing(cands, test=os.path.isdir):
    for c in cands:
        c = xp(c)
        if c and test(c):
            return c
    return ""


def resolve_paths(cfg):
    for k in PATH_KEYS:
        cfg[k] = xp(cfg[k])
    if not cfg["llama_dir"]:
        cfg["llama_dir"] = first_existing(
            [os.environ.get("LLAMA_CPP_DIR", "")] + LLAMA_DIR_CANDIDATES
        )
    backend = cfg["backend"]
    for tool, key in (("llama-server", "server"), ("llama-bench", "bench")):
        if cfg[key]:
            continue
        cands = []
        if cfg["llama_dir"]:
            cands += [
                os.path.join(cfg["llama_dir"], f"build-{backend}", "bin", tool),
                os.path.join(cfg["llama_dir"], "build", "bin", tool),
            ]
        cands.append(shutil.which(tool) or "")
        cfg[key] = first_existing(cands, lambda p: os.access(p, os.X_OK))
    if not cfg["models_dir"]:
        cfg["models_dir"] = first_existing(MODELS_DIR_CANDIDATES) or xp("~/models")
    cfg["models_dir"] = (
        os.path.realpath(cfg["models_dir"])
        if os.path.isdir(cfg["models_dir"])
        else cfg["models_dir"]
    )
    return cfg


def resolve_model_path(cfg, p):
    p = xp(p)
    return p if os.path.isabs(p) else os.path.join(cfg["models_dir"], p)


def rel_model_path(cfg, p):
    real = os.path.realpath(p)
    base = cfg["models_dir"].rstrip(os.sep) + os.sep
    return real[len(base) :] if real.startswith(base) else tilde(p)


# ---------------------------------------------------------------- config
def read_toml():
    if not os.path.exists(CONFIG):
        return {}
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def merge_config(user_defaults, profile_name=None):
    """DEFAULTS < profile < the [defaults] table of models.toml. Pure function (no file access).
    Also records which values of models.toml replace a value of the profile ('_overrides')."""
    user = dict(user_defaults)
    name = profile_name or user.get("profile") or DEFAULTS["profile"]
    if name not in PROFILES:
        sys.exit(f"Unknown profile '{name}'. Known: {', '.join(PROFILES)}")
    cfg = dict(DEFAULTS)
    cfg.update(PROFILES[name])
    cfg.update(user)
    cfg["profile"] = name
    cfg["_overrides"] = {
        k: (user[k], PROFILES[name][k])
        for k in PROFILES[name]
        if k in user and user[k] != PROFILES[name][k]
    }
    return cfg


def check_config(cfg):
    """Stop with a clear message on values that would give wrong results."""
    problems = []
    if cfg["needle_style"] not in NEEDLE_STYLES:
        problems.append(f"needle_style must be one of {NEEDLE_STYLES}")
    if not 1 <= int(cfg["needles"]) <= 10:
        problems.append("needles must be 1 to 10")
    if not 1 <= int(cfg["accuracy_min_pct"]) <= 100:
        problems.append("accuracy_min_pct must be 1 to 100")
    if not cfg["accuracy_pct"] or not all(1 <= int(p) <= 95 for p in cfg["accuracy_pct"]):
        problems.append("accuracy_pct must be a list of fill levels from 1 to 95")
    if not 0 <= int(cfg["speed_tolerance_pct"]) < 100:
        problems.append("speed_tolerance_pct must be 0 to 99")
    if problems:
        sys.exit("Config error: " + "; ".join(problems))


def profile_summary(cfg):
    return (
        f"Profile {cfg['profile']}: target context {cfg['target_context']}, speed loss up to "
        f"{cfg['speed_tolerance_pct']}%, minimum {cfg['min_tg_tps']} t/s with a filled context, recall "
        f"{required_hits(cfg, int(cfg['needles']))}/{cfg['needles']} values ({cfg['needle_style']} test) "
        f"at {'/'.join(str(p) for p in cfg['accuracy_pct'])}% fill"
    )


def override_lines(cfg):
    """Warnings for values in models.toml [defaults] that replace a value of the profile."""
    return [
        f"models.toml sets {k} = {mine}; profile '{cfg['profile']}' uses {theirs}. "
        f"Remove the line from [defaults] to use the profile."
        for k, (mine, theirs) in cfg.get("_overrides", {}).items()
    ]


def base_config(args=None, raw=None):
    raw = read_toml() if raw is None else raw
    cfg = merge_config(
        raw.get("defaults", {}),
        (getattr(args, "profile", None) if args else None) or os.environ.get("LLMBENCH_PROFILE"),
    )
    check_config(cfg)
    for env, key in (
        ("LLMBENCH_LLAMA_DIR", "llama_dir"),
        ("LLMBENCH_MODELS_DIR", "models_dir"),
        ("LLMBENCH_BACKEND", "backend"),
        ("LLMBENCH_SWAP_URL", "swap_url"),
    ):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    for key in ("llama_dir", "models_dir", "backend", "server", "bench"):
        v = getattr(args, key, None) if args else None
        if v:
            cfg[key] = v
    if args is not None and getattr(args, "backend", None) and not getattr(args, "server", None):
        cfg["server"] = cfg["bench"] = ""  # a backend option means: use that build folder
    return resolve_paths(cfg)


def load_config(args=None, need_models=True):
    raw = read_toml()
    if need_models and not raw:
        sys.exit(f"No config: {CONFIG}. Run: {sys.argv[0]} init")
    cfg = base_config(args, raw)
    models = {}
    for name, m in raw.get("models", {}).items():
        if m.get("skip"):
            continue
        path = resolve_model_path(cfg, m["path"])
        if not os.path.exists(path):
            warn(f"{name}: file not found: {path}")
            continue
        try:
            info = model_info(path)
        except Exception as e:
            warn(f"{name}: cannot read {path}: {e}")
            continue
        mtype = m.get("type", info["type"])
        mm = m.get("mmproj", "auto")
        mmproj = find_mmproj(path) if mm == "auto" else (resolve_model_path(cfg, mm) if mm else "")
        if mmproj and not os.path.exists(mmproj):
            warn(f"{name}: mmproj not found: {mmproj} (vision tests are off)")
            mmproj = ""
        extra = [xp(str(x)) for x in m.get("extra_flags", [])]
        if "--mmproj" in extra:  # version 1 style: mmproj in extra_flags
            i = extra.index("--mmproj")
            mmproj = mmproj or (extra[i + 1] if i + 1 < len(extra) else "")
            del extra[i : i + 2]
        offload = m.get("mmproj_offload", True)
        if "--no-mmproj-offload" in extra:
            extra.remove("--no-mmproj-offload")
            offload = False
        mm_flags = []
        if mmproj:
            mm_flags = ["--mmproj", mmproj] + ([] if offload else ["--no-mmproj-offload"])
        models[name] = {
            "name": name,
            "display_name": m.get("display_name", ""),
            "path": path,
            "info": info,
            "type": mtype,
            "contexts": sorted(
                int(c)
                for c in m.get("contexts", cfg[f"{mtype}_contexts"])
                if not info["native_ctx"] or int(c) <= info["native_ctx"]
            ),
            "cache_types": m.get("cache_types", cfg[f"{mtype}_cache_types"]),
            "mmproj": mmproj,
            "mmproj_offload": offload,
            "mm_flags": mm_flags,
            "extra_flags": extra,
            "run_flags": [str(x) for x in m.get("run_flags", [])],
            "tools": m.get("tools", True),
        }
    return cfg, models


def pick_models(models, names):
    if not names or names == ["all"]:
        return list(models.values())
    out = []
    for n in names:
        if n not in models:
            sys.exit(f"Unknown model '{n}'. Known: {', '.join(models) or '(none)'}")
        out.append(models[n])
    return out


def need_tools(cfg):
    for k, tool in (("server", "llama-server"), ("bench", "llama-bench")):
        if not cfg[k] or not os.access(cfg[k], os.X_OK):
            sys.exit(
                f"{tool} not found (backend '{cfg['backend']}'). Set llama_dir or {k} in "
                f"models.toml [defaults], or use --llama-dir / --{k}. Run: {sys.argv[0]} doctor"
            )


# ---------------------------------------------------------------- GPU devices
def parse_devices(text):
    devs = []
    for m in re.finditer(r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+) MiB", text, re.MULTILINE):
        devs.append(
            {
                "kind": m.group(1),
                "index": int(m.group(2)),
                "name": m.group(3).strip(),
                "mib": int(m.group(4)),
            }
        )
    return devs


IGPU_NAME = re.compile(
    r"Radeon( \d{3,4}M| Graphics| \d{4}S)?$|Radeon\(TM\) Graphics|Vega \d+$", re.IGNORECASE
)


def pick_main_gpu(cfg, gpus):
    """The dedicated GPU. An integrated GPU reports shared RAM, so its size is not useful:
    use the dedicated VRAM size from sysfs, and skip typical iGPU names."""
    try:
        total, _, _ = vram_mb(cfg)
    except SystemExit:
        total = 0
    if total:
        near = [d for d in gpus if abs(d["mib"] - total) <= max(1024, total * 0.05)]
        if len(near) == 1:
            return near[0]
    dgpu = [d for d in gpus if not IGPU_NAME.search(d["name"])]
    return max(dgpu or gpus, key=lambda d: d["mib"])


def setup_devices(cfg):
    """Make only the wanted GPU visible to llama.cpp. Sets cfg['_env'] and cfg['_devenv']."""
    env = dict(os.environ)
    var = DEVICE_ENV.get(cfg["backend"].lower(), "")
    devenv = {}
    for v in set(DEVICE_ENV.values()) | {"ROCR_VISIBLE_DEVICES"}:
        if os.environ.get(v):
            devenv[v] = os.environ[v]
    vis = str(cfg["visible_devices"])
    if var and vis not in ("auto", "") and var not in devenv:
        devenv[var] = vis
        env[var] = vis
    devs = parse_devices(run_text([cfg["server"], "--list-devices"], env)) if cfg["server"] else []
    gpus = [d for d in devs if d["kind"].lower() != "cpu"]
    if var and vis == "auto" and var not in devenv and len(gpus) > 1:
        best = pick_main_gpu(cfg, gpus)
        env[var] = devenv[var] = str(best["index"])
        if not cfg.get("_quiet_devices"):
            log(f"{len(gpus)} GPUs found. Using only {best['name']} ({var}={best['index']}).")
        devs = parse_devices(run_text([cfg["server"], "--list-devices"], env))
        gpus = [d for d in devs if d["kind"].lower() != "cpu"]
    if len(gpus) > 1:
        warn(
            f"{len(gpus)} GPUs are visible: {', '.join(d['name'] for d in gpus)}. "
            f"Set visible_devices in models.toml or {var or 'the device variable'}."
        )
    cfg["_env"], cfg["_devenv"], cfg["_devices"] = env, devenv, gpus
    return gpus


def gpu_dir(cfg):
    if cfg["gpu_sysfs"]:
        return cfg["gpu_sysfs"]
    best = None
    for d in glob.glob("/sys/class/drm/card*/device"):
        p = os.path.join(d, "mem_info_vram_total")
        if os.path.exists(p):
            total = int(open(p).read())
            if not best or total > best[0]:
                best = (total, d)
    if not best:
        sys.exit("No AMD GPU found in /sys/class/drm (set gpu_sysfs in models.toml)")
    cfg["gpu_sysfs"] = best[1]
    return best[1]


def vram_mb(cfg):
    d = gpu_dir(cfg)
    total = int(open(os.path.join(d, "mem_info_vram_total")).read()) // 1048576
    used = int(open(os.path.join(d, "mem_info_vram_used")).read()) // 1048576
    return total, used, total - used


def ram_free_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def physical_cores():
    cores, phys = set(), None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                cores.add((phys, line.split(":")[1].strip()))
    except OSError:
        pass
    return len(cores) or max(1, (os.cpu_count() or 2) // 2)


def build_id(cfg):
    m = re.search(r"version: (\S+)", run_text([cfg["server"], "--version"], cfg.get("_env")))
    return m.group(1) if m else "?"


def service_active(name):
    try:
        r = subprocess.run(["systemctl", "is-active", "--quiet", name], capture_output=True)
    except FileNotFoundError:
        return False
    return r.returncode == 0


def other_servers():
    out = run_text(["pgrep", "-a", "-x", "llama-server"])
    return [l for l in out.splitlines() if l.strip() and "pgrep" not in l]


# ---------------------------------------------------------------- API key
def api_key(cfg, required=True):
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


# ---------------------------------------------------------------- settings
def server_flags(m, s):
    f = ["-ngl", str(s["ngl"])]
    if s.get("ncmoe"):
        f += ["--n-cpu-moe", str(s["ncmoe"])]
    f += ["-ctk", s["cache"], "-ctv", s["cache"]]
    if s.get("ub"):
        f += ["-ub", str(s["ub"])]
        if s["ub"] > 2048:
            f += ["-b", str(s["ub"])]
    if s.get("threads"):
        f += ["-t", str(s["threads"])]
    return f + m.get("extra_flags", [])


def bench_flags(cfg, s):
    f = ["-ngl", str(s["ngl"]), "-fa", cfg["bench_fa"], "-ctk", s["cache"], "-ctv", s["cache"]]
    if s.get("ncmoe"):
        f += ["--n-cpu-moe", str(s["ncmoe"])]
    if s.get("ub"):
        f += ["-ub", str(s["ub"])]
    if s.get("threads"):
        f += ["-t", str(s["threads"])]
    return f


def sdesc(s):
    parts = [f"ngl={s['ngl']}", f"kv={s['cache']}"]
    if s.get("ncmoe"):
        parts.append(f"n-cpu-moe={s['ncmoe']}")
    if s.get("ub"):
        parts.append(f"ub={s['ub']}")
    if s.get("threads"):
        parts.append(f"t={s['threads']}")
    return " ".join(parts)


# ---------------------------------------------------------------- test image
_FONT = {  # 5x7 digits
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
_COLORS = {"red": (220, 30, 30), "green": (20, 160, 40), "blue": (30, 60, 220)}


def test_image(px, seed=None):
    """PNG with a 4-digit number in a coloured frame. Returns (png_bytes, number, colour)."""
    rnd = random.Random(seed)
    number = "".join(rnd.choice("123456789") for _ in range(4))
    colour = rnd.choice(sorted(_COLORS))
    col = _COLORS[colour]
    w = h = px
    img = [bytearray(b"\xff" * (w * 3)) for _ in range(h)]
    border = max(6, px // 25)
    for y in range(h):
        for x in list(range(border)) + list(range(w - border, w)):
            img[y][x * 3 : x * 3 + 3] = bytes(col)
    for y in list(range(border)) + list(range(h - border, h)):
        img[y][:] = bytes(col) * w
    scale = max(4, int(w * 0.6) // (4 * 6))
    tw, th = 4 * 6 * scale - scale, 7 * scale
    ox, oy = (w - tw) // 2, (h - th) // 2
    for i, d in enumerate(number):
        for ry, row in enumerate(_FONT[d]):
            for rx, bit in enumerate(row):
                if bit == "1":
                    for y in range(oy + ry * scale, oy + (ry + 1) * scale):
                        x0 = ox + (i * 6 + rx) * scale
                        img[y][x0 * 3 : (x0 + scale) * 3] = b"\x00" * (3 * scale)
    raw = b"".join(b"\x00" + bytes(r) for r in img)
    chunk = lambda t, d: (
        struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return png, number, colour


def image_message(png, text):
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


VISION_Q = (
    "Read the number in the image and name the colour of the frame. "
    "Answer only in this form: NUMBER=<digits>, COLOUR=<red|green|blue>."
)


# ---------------------------------------------------------------- HTTP client
class Client:
    def __init__(self, url, key=""):
        self.url, self.key = url.rstrip("/"), key

    def call(self, path, body=None, timeout=900):
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
        for b in (f"/upstream/{model}", ""):
            try:
                return b, self.call(b + "/props")
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    continue
                raise
        raise RuntimeError("cannot read /props")


def http_error(e):
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = ""
        return f"HTTP {e.code}: {body[:200]}"
    return str(e)[:200]


def chat(client, model, messages, max_tokens, timeout, think=False):
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


# ---------------------------------------------------------------- server control
class Server:
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


def stop_all():
    for s in ACTIVE_SERVERS:
        s.stop()


def wait_vram_released(cfg, baseline_used, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, used, _ = vram_mb(cfg)
        if used <= baseline_used + 256:
            return
        time.sleep(0.5)
    warn(f"VRAM not released after {timeout}s (used {used} MiB, before {baseline_used} MiB)")


# ---------------------------------------------------------------- phase 1: fit
class Fitter:
    def __init__(self, cfg, m):
        self.cfg, self.m, self.cache = cfg, m, {}
        _, self.baseline_used, _ = vram_mb(cfg)

    def warmup(self, srv):
        """Run the work a real client does, so all buffers are allocated before measuring."""
        c, m, t = srv.client(), self.m, self.cfg["request_timeout_s"]
        if self.cfg["fit_warmup"]:
            chat(c, m["name"], [{"role": "user", "content": "Say OK."}], 4, t)
        if m["mmproj"] and self.cfg["image_fit"]:
            png, _, _ = test_image(self.cfg["image_px"], seed=1)
            chat(c, m["name"], image_message(png, "Describe the image in three words."), 8, t)

    def fits(self, ctx, s):
        key = (ctx, json.dumps({k: v for k, v in s.items() if k != "threads"}, sort_keys=True))
        if key in self.cache:
            return self.cache[key]
        PROGRESS.tick(f"fit test: ctx={ctx} {sdesc(s)}")
        wait_vram_released(self.cfg, self.baseline_used)
        _, _, free0 = vram_mb(self.cfg)
        ram0 = ram_free_mb()
        srv = Server(self.cfg, self.m, ctx, s, "fit")
        vfree = rfree = err = ""
        ok, secs, status = False, 0.0, "FAIL"
        try:
            try:
                ok, secs = srv.start()
            except RuntimeError as e:
                err = str(e)
            if ok:
                try:
                    self.warmup(srv)
                except Exception as e:
                    ok, err = False, "request failed: " + http_error(e)
                    status = "FAIL_REQUEST"
            if ok:
                time.sleep(1.0)
                _, _, vfree = vram_mb(self.cfg)
                rfree = ram_free_mb()
                if vfree < self.cfg["min_free_vram_mb"] or rfree < self.cfg["min_free_ram_mb"]:
                    status, ok = "LOW_MARGIN", False
                else:
                    status = "OK"
            elif not err:
                status, err = "FAIL", srv.error_line()
        finally:
            srv.stop()
            wait_vram_released(self.cfg, self.baseline_used)
        append_csv(
            "fit.csv",
            {
                "run_id": RUN_ID,
                "time": now(),
                "backend": self.cfg["backend"],
                "model": self.m["name"],
                "context": ctx,
                "setting": sdesc(s),
                "vision": "yes" if self.m["mmproj"] else "",
                "status": status,
                "load_s": round(secs, 1),
                "vram_free_mib": vfree,
                "vram_used_by_model_mib": (free0 - vfree) if vfree != "" else "",
                "ram_free_mib": rfree,
                "ram_used_by_model_mib": (ram0 - rfree) if rfree != "" else "",
                "error": err,
            },
        )
        log(
            f"  fit   ctx={ctx:<7} {sdesc(s):<44} {status:<12} load={secs:5.1f}s  "
            f"VRAM free={vfree} MiB  RAM free={rfree} MiB  {err[:70]}"
        )
        self.cache[key] = ok
        return ok


def sweep_moe(cfg, m, fitter):
    L = m["info"]["layers"] or 64
    settings = []
    for cache in m["cache_types"]:
        lo = 0
        mk = lambda n, c=cache: {"ngl": L + 1, "ncmoe": n, "cache": c}
        for ctx in m["contexts"]:
            a, b = lo, L  # fewest expert layers in RAM; never fewer than for a smaller context
            while a < b:
                mid = (a + b) // 2
                if fitter.fits(ctx, mk(mid)):
                    b = mid
                else:
                    a = mid + 1
            if a == L and not fitter.fits(ctx, mk(L)):
                log(f"  {m['name']}: ctx={ctx} does not fit even with all experts in RAM")
                break
            lo = a
            settings.append({"context": ctx, "setting": mk(a)})
    return settings


def sweep_dense(cfg, m, fitter):
    L = m["info"]["layers"] or 64
    cands = m["contexts"]
    settings = []
    for cache in m["cache_types"]:
        mk = lambda n, c=cache: {"ngl": n, "cache": c}
        ngl = L + 1
        if not fitter.fits(cands[0], mk(ngl)):
            if not fitter.fits(cands[0], mk(0)):
                log(f"  {m['name']}: {cache}: smallest context does not fit")
                continue
            a, b = 0, L
            while a < b:  # largest ngl that fits
                mid = (a + b + 1) // 2
                if fitter.fits(cands[0], mk(mid)):
                    a = mid
                else:
                    b = mid - 1
            ngl = a
            log(f"  {m['name']}: {cache}: {L - ngl} of {L} layers must stay in RAM")
            if m["mmproj"] and m["mmproj_offload"]:
                log(
                    f"  hint: set mmproj_offload = false for {m['name']} in models.toml. The image "
                    "encoder then stays in RAM and more layers fit on the GPU"
                )
        a, b = 0, len(cands) - 1
        while a < b:  # largest context that fits
            mid = (a + b + 1) // 2
            if fitter.fits(cands[mid], mk(ngl)):
                a = mid
            else:
                b = mid - 1
        settings.append({"context": cands[a], "setting": mk(ngl)})
    return settings


# ---------------------------------------------------------------- phase 2: speed
def run_bench(cfg, m, s, depths, extra=None, reps=None, prompt=512):
    cmd = (
        [
            cfg["bench"],
            "-m",
            m["path"],
            "-p",
            str(prompt),
            "-n",
            "128",
            "-d",
            ",".join(str(d) for d in depths),
            "-r",
            str(reps or cfg["bench_reps"]),
            "-o",
            "jsonl",
        ]
        + bench_flags(cfg, s)
        + (extra or [])
    )
    r = subprocess.run(cmd, capture_output=True, text=True, env=cfg.get("_env"))
    rows = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "bench-raw.jsonl"), "a") as f:
        f.writelines(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "backend": cfg["backend"],
                    "model": m["name"],
                    "setting": sdesc(s),
                    **d,
                }
            )
            + "\n"
            for d in rows
        )
    if not rows:
        log(f"  bench failed: {' '.join(shlex.quote(c) for c in cmd)}")
        log("  " + (r.stderr.strip().splitlines() or ["(no output)"])[-1][:200])
    return rows


def speed_depths(cfg, ctx):
    """(depths, deep). depths = the usual filled-context depths that fit in ctx. deep = one more depth
    for contexts from target_context up: bench_deep_fill_pct percent of ctx, because that is the speed
    a long session has. None if not needed."""
    depths = sorted(d for d in cfg["bench_depths"] if d + 1024 <= ctx) or [0]
    pct = cfg["bench_deep_fill_pct"]
    if pct and ctx >= cfg["target_context"]:
        deep = int(ctx * pct / 100) // 1024 * 1024
        if deep > max(depths):
            return depths, deep
    return depths, None


def bench_cached(done, cfg, m, s, depths, **kw):
    """run_bench, but the same flags and depths run only once (many contexts share one setting)."""
    key = json.dumps([bench_flags(cfg, s), depths, kw], sort_keys=True)
    if key not in done:
        done[key] = run_bench(cfg, m, s, depths, **kw)
    return done[key]


def speed_phase(cfg, m, settings):
    done = {}
    PROGRESS.phase("speed", len(settings))
    for st in settings:
        s, ctx = st["setting"], st["context"]
        PROGRESS.tick(f"llama-bench: ctx={ctx} {sdesc(s)}")
        depths, deep = speed_depths(cfg, ctx)
        rows = list(bench_cached(done, cfg, m, s, depths))
        if deep:  # one repetition, generation only: filling the context is the slow part
            rows += bench_cached(done, cfg, m, s, [deep], reps=1, prompt=0)
        res = {}
        for d in rows:
            kind = "pp512" if d.get("n_prompt", 0) > 0 else "tg128"
            dep = int(d.get("n_depth", 0))
            res[f"{kind}@{dep}"] = round(d.get("avg_ts", 0), 1)
            append_csv(
                "speed.csv",
                {
                    "run_id": RUN_ID,
                    "time": now(),
                    "backend": cfg["backend"],
                    "model": m["name"],
                    "context": ctx,
                    "setting": sdesc(s),
                    "test": kind,
                    "depth": dep,
                    "tokens_per_s": round(d.get("avg_ts", 0), 2),
                    "stddev": round(d.get("stddev_ts", 0), 2),
                    "build": cfg["_build"],
                },
            )
        st["speed"] = res
        log(
            f"  speed ctx={ctx:<7} {sdesc(s):<44} "
            + "  ".join(f"{k}={v}" for k, v in sorted(res.items()))
        )


# ---------------------------------------------------------------- phase 3: tune
def tune_phase(cfg, m, st, fitter):
    s = dict(st["setting"])
    ubs = cfg["tune_ubatch"]
    threads = [None]
    if m["type"] == "moe" or s["ngl"] <= m["info"]["layers"]:  # the CPU does part of the work
        p = physical_cores()
        threads = sorted({max(1, p // 2), max(1, p - 2), p})
    extra = ["-ub", ",".join(map(str, ubs)), "-b", str(max(ubs + [2048]))]
    if threads != [None]:
        extra += ["-t", ",".join(map(str, threads))]
    PROGRESS.phase("tune", detail=f"llama-bench: ctx={st['context']} {sdesc(s)}")
    log(f"  tune  ctx={st['context']} {sdesc(s)}  ubatch={ubs} threads={threads}")
    rows = run_bench(
        cfg, m, {k: v for k, v in s.items() if k not in ("ub", "threads")}, [0], extra
    )
    combos = {}
    for d in rows:
        key = (int(d.get("n_ubatch", 0)), int(d.get("n_threads", 0)))
        kind = "pp" if d.get("n_prompt", 0) > 0 else "tg"
        combos.setdefault(key, {})[kind] = d.get("avg_ts", 0)
    if not combos:
        return None
    for (ub, t), v in sorted(combos.items()):
        append_csv(
            "tune.csv",
            {
                "run_id": RUN_ID,
                "time": now(),
                "backend": cfg["backend"],
                "model": m["name"],
                "context": st["context"],
                "setting": sdesc(s),
                "ubatch": ub,
                "threads": t,
                "pp512": round(v.get("pp", 0), 1),
                "tg128": round(v.get("tg", 0), 1),
                "build": cfg["_build"],
            },
        )
        log(
            f"        ub={ub:<5} t={t:<3} pp512={v.get('pp', 0):8.1f}  tg128={v.get('tg', 0):6.1f}"
        )
    best_tg = max(v.get("tg", 0) for v in combos.values())
    ranked = sorted(
        combos.items(),
        key=lambda kv: (kv[1].get("tg", 0) >= best_tg * 0.97, kv[1].get("pp", 0)),
        reverse=True,
    )
    for (ub, t), _ in ranked:  # a larger ubatch needs more VRAM: check that it still loads
        cand = dict(s, ub=ub, threads=t if threads != [None] else None)
        if fitter.fits(st["context"], cand):
            log(f"  tune  best: {sdesc(cand)}")
            return cand
    return None


# ---------------------------------------------------------------- phase 4: accuracy
_SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
LETTERS = "ABCDEFGHIJ"


def pseudo_word(rnd):
    return "".join(rnd.choice(_SYLLABLES) for _ in range(3))


def haystack(style, n, seed):
    """n distinct (key, value, text) entries for the long-prompt test.
    "line":  Line 123: the value is 45678.      (find a value by line number)
    "keyed": Project kelmora: access code 45678. (find a value by a unique name, like a symbol in code)"""
    rnd = random.Random(seed)
    items, seen = [], set()
    for i in range(n):
        if style == "line":
            key, value = str(i), (i * 7919 + 13) % 100000
            text = f"Line {i}: the value is {value}."
        else:
            key = pseudo_word(rnd)
            while key in seen:
                key = pseudo_word(rnd)
            seen.add(key)
            value = rnd.randrange(10000, 100000)
            text = f"Project {key}: access code {value}."
        items.append((key, value, text))
    return items


def needle_positions(n_lines, count):
    """Line numbers of the values to ask for, spread from 10% to 90% of the text."""
    if count == 1:
        return [n_lines // 2]
    return [int(n_lines * (0.1 + 0.8 * k / (count - 1))) for k in range(count)]


def needle_question(style, keys):
    noun, what = ("line", "values on") if style == "line" else ("project", "access codes of")
    names = ", ".join(f"{noun} {k}" for k in keys[:-1])
    names = f"{names} and {noun} {keys[-1]}" if len(keys) > 1 else f"{noun} {keys[0]}"
    form = ", ".join(f"{LETTERS[i]}=<number>" for i in range(len(keys)))
    return f"What are the {what} {names}? Answer only in this form: {form}."


def score_answer(answer, expected):
    """Count the correct values. With "A=123, B=456" the letter must match; without letters the
    number anywhere in the answer counts. Returns (hits, indices of the wrong values)."""
    got = {l.upper(): v for l, v in re.findall(r"\b([A-Ja-j])\s*[=:]\s*(\d+)", answer)}
    hits, missed = 0, []
    for i, value in enumerate(expected):
        ok = got[LETTERS[i]] == str(value) if LETTERS[i] in got else str(value) in answer
        if ok:
            hits += 1
        else:
            missed.append(i)
    return hits, missed


def required_hits(cfg, count):
    return max(1, math.ceil(count * cfg["accuracy_min_pct"] / 100))


def fill_one(cfg, client, model, pct, think=False, max_tokens=0):
    """Fill the context to pct percent and ask for several hidden values."""
    max_tokens = max_tokens or (8192 if think else 256)
    margin = 300
    style, count = cfg["needle_style"], int(cfg["needles"])
    base, props = client.base(model)
    n_ctx = int(
        props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx") or 0
    )
    budget = min(int(n_ctx * pct / 100), n_ctx - max_tokens - margin)
    if budget < 1000:
        raise RuntimeError(f"context {n_ctx} too small for {pct}%")
    tokens = lambda t: len(client.call(base + "/tokenize", {"content": t})["tokens"])
    seed = random.randrange(10**9)
    per_line = tokens("\n".join(x[2] for x in haystack(style, 2000, seed))) / 2000
    lines = int(budget / per_line)
    header = f"Test run {random.randint(100000, 999999)}. Read all lines.\n"
    for _ in range(6):
        items = haystack(style, lines, seed)
        text = header + "\n".join(x[2] for x in items)
        used = tokens(text)
        if used <= budget:
            break
        lines -= int((used - budget) / per_line) + 50
    pos = needle_positions(len(items), count)
    expected = [items[p][1] for p in pos]
    question = needle_question(style, [items[p][0] for p in pos])
    data, secs = chat(
        client,
        model,
        [{"role": "user", "content": text + "\n\n" + question}],
        max_tokens,
        cfg["request_timeout_s"],
        think,
    )
    answer = (data["choices"][0]["message"].get("content") or "").strip()
    hits, missed = score_answer(answer, expected)
    need = required_hits(cfg, count)
    note = ""
    if missed:
        where = ",".join(f"{round(100 * pos[i] / len(items))}%" for i in missed)
        note = f"wrong at depth {where}; expected {expected}, got: {answer[:120]}"
    tm = data.get("timings", {})
    return {
        "n_ctx": n_ctx,
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", used),
        "prompt_tps": round(tm.get("prompt_per_second", 0), 1),
        "gen_tps": round(tm.get("predicted_per_second", 0), 1),
        "ttft_s": round(tm.get("prompt_ms", 0) / 1000, 1),
        "seconds": round(secs, 1),
        "score": f"{hits}/{count}",
        "style": style,
        "status": "PASS" if hits >= need else ("PARTIAL" if hits else "MISS"),
        "error": note,
    }


FILL_FIELDS = (
    "n_ctx",
    "prompt_tokens",
    "prompt_tps",
    "gen_tps",
    "ttft_s",
    "seconds",
    "score",
    "status",
    "style",
    "error",
)


def fill_series(
    cfg, client, model, pcts, label, think=False, meta=None, stop_on_fail=False, tick=False
):
    """Long-prompt tests at the given fill levels, smallest first. meta = {"context", "cache"} of the
    test server. With stop_on_fail the series ends at the first test that is not PASS."""
    meta, results = meta or {}, []
    for pct in sorted(pcts):
        (PROGRESS.tick if tick else PROGRESS.note)(f"long-prompt test at {pct}% fill, {label}")
        try:
            r = fill_one(cfg, client, model, pct, think)
        except Exception as e:
            r = {
                "status": "HTTP" if isinstance(e, urllib.error.HTTPError) else "FAIL",
                "error": http_error(e),
            }
        row = {
            "run_id": RUN_ID,
            "time": now(),
            "backend": cfg["backend"],
            "label": label,
            "model": model,
            "context": meta.get("context", ""),
            "cache": meta.get("cache", ""),
            "pct": pct,
        }
        for k in FILL_FIELDS:
            row[k] = r.get(k, "")
        row["error"] = str(row["error"]).replace("\n", " ")
        append_csv("accuracy.csv", row)
        log(
            f"  fill  {pct:>3}%  ctx={row['n_ctx']} prompt={row['prompt_tokens']} pp={row['prompt_tps']} t/s "
            f"tg={row['gen_tps']} t/s ttft={row['ttft_s']}s  {row['score']} {row['status']} {row['error'][:80]}"
        )
        results.append(row)
        if stop_on_fail and row["status"] != "PASS":
            break
    return results


def vision_series(cfg, client, model, label, runs=None, tick=False):
    """Read a generated number and frame colour. Measures image prompt speed and correctness."""
    rows = []
    for i in range(runs or cfg["vision_runs"]):
        (PROGRESS.tick if tick else PROGRESS.note)(f"image test, {label}")
        png, number, colour = test_image(cfg["image_px"], seed=f"{RUN_ID}-{model}-{i}")
        row = {
            "run_id": RUN_ID,
            "time": now(),
            "backend": cfg["backend"],
            "label": label,
            "model": model,
            "image_px": cfg["image_px"],
            "expected": f"{number} {colour}",
        }
        try:
            data, secs = chat(
                client, model, image_message(png, VISION_Q), 64, cfg["request_timeout_s"]
            )
            ans = (data["choices"][0]["message"].get("content") or "").strip()
            mt = re.search(r"NUMBER\s*[=:]\s*([\d\s,.]+)", ans, re.IGNORECASE)
            got_n = re.sub(r"\D", "", mt.group(1)) if mt else ""
            ok_n = got_n == number if got_n else number in re.sub(r"\D", "", ans)
            ok_c = bool(re.search(rf"\b{colour}\b", ans, re.IGNORECASE))
            tm = data.get("timings", {})
            row.update(
                {
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", ""),
                    "prompt_tps": round(tm.get("prompt_per_second", 0), 1),
                    "ttft_s": round(tm.get("prompt_ms", 0) / 1000, 2),
                    "seconds": round(secs, 1),
                    "score": f"{int(ok_n) + int(ok_c)}/2",
                    "status": "PASS" if ok_n and ok_c else ("PARTIAL" if ok_n or ok_c else "MISS"),
                    "answer": ans[:120].replace("\n", " "),
                }
            )
        except Exception as e:
            row.update(
                {
                    "prompt_tokens": "",
                    "prompt_tps": "",
                    "ttft_s": "",
                    "seconds": "",
                    "score": "0/2",
                    "status": "FAIL",
                    "answer": http_error(e).replace("\n", " "),
                }
            )
        append_csv("vision.csv", row)
        log(
            f"  image {i + 1}/{runs or cfg['vision_runs']}  tokens={row['prompt_tokens']} "
            f"ttft={row['ttft_s']}s  expected={row['expected']}  {row['score']} {row['status']}  "
            f"{row['answer'][:60]}"
        )
        rows.append(row)
    return rows


def accuracy_search(cfg, m, settings, rec):
    """Per KV cache type: test the largest context that is fast enough. If a fill test fails, test the
    next smaller one (at most accuracy_max_steps). Stop at the first context that passes."""
    acc = []
    PROGRESS.phase(
        "accuracy",
        sum(
            min(cfg["accuracy_max_steps"], sum(1 for g in p["gates"] if g["speed_ok"]))
            for p in rec["per_cache"].values()
        ),
    )
    for cache, p in rec["per_cache"].items():
        fast = {g["context"] for g in p["gates"] if g["speed_ok"]}
        cands = sorted(
            (st for st in settings if st["setting"]["cache"] == cache and st["context"] in fast),
            key=lambda x: -x["context"],
        )
        if not cands:
            log(f"  accuracy: kv={cache}: no context passes the speed rules, nothing to test")
            continue
        passed = False
        for st in cands[: cfg["accuracy_max_steps"]]:
            PROGRESS.tick(f"long-prompt tests: ctx={st['context']} {sdesc(st['setting'])}")
            log(f"  accuracy: ctx={st['context']} {sdesc(st['setting'])}")
            a, _ = served_phase(cfg, m, st, "accuracy", True, False, stop_on_fail=True)
            acc += a
            if a and all(x["status"] == "PASS" for x in a) and len(a) == len(cfg["accuracy_pct"]):
                passed = True
                log(f"  accuracy: ctx={st['context']} PASS")
                break
            if any(x["status"] in ("FAIL", "HTTP") for x in a):
                log(
                    "  accuracy: request error, see the error column (not counted as a model failure)"
                )
        if not passed:
            warn(
                f"{m['name']} kv={cache}: no tested context passed the long-prompt test. "
                "Add smaller values to contexts in models.toml, use another cache type, "
                'or use profile "long".'
            )
    return acc


def served_phase(cfg, m, st, tag, accuracy, vision, stop_on_fail=False):
    """Start one server for this setting and run the accuracy and/or image tests on it."""
    srv = Server(cfg, m, st["context"], st["setting"], tag)
    acc, vis = [], []
    try:
        PROGRESS.note(f"loading the model: ctx={st['context']} {sdesc(st['setting'])}")
        ok, _ = srv.start()
        if not ok:
            log(f"  {tag}: server did not start ({srv.error_line()})")
            return acc, vis
        label = f"{m['name']} {tag} ctx={st['context']} {sdesc(st['setting'])}"
        if accuracy:
            acc = fill_series(
                cfg,
                srv.client(),
                m["name"],
                cfg["accuracy_pct"],
                label,
                meta={"context": st["context"], "cache": st["setting"]["cache"]},
                stop_on_fail=stop_on_fail,
            )
        if vision:
            vis = vision_series(cfg, srv.client(), m["name"], label, tick=True)
    finally:
        srv.stop()
    return acc, vis


# ---------------------------------------------------------------- decision rules
# Everything below is free of file, network and process access, so test_llmbench.py can test it.
def speed_at_depth(st):
    """(depth, t/s) of the deepest tg128 measurement of a setting. This is what a long session feels like."""
    tg = [
        (int(k.split("@")[1]), v) for k, v in st.get("speed", {}).items() if k.startswith("tg128@")
    ]
    return max(tg) if tg else (0, 0)


def deep_tg(st):
    return speed_at_depth(st)[1]


def speed_gate(cfg, st, best_tg):
    """(ok, note). Slow = more than speed_tolerance_pct below the best tg128 of this cache type,
    or below min_tg_tps with a filled context."""
    if not st.get("speed"):
        return True, "speed not measured"
    tg0 = st["speed"].get("tg128@0", 0)
    depth, deep = speed_at_depth(st)
    tol = cfg["speed_tolerance_pct"]
    if best_tg and tg0 < best_tg * (1 - tol / 100):
        return (
            False,
            f"too slow: {tg0} t/s is {100 * (1 - tg0 / best_tg):.0f}% below the best ({best_tg}), limit {tol}%",
        )
    if cfg["min_tg_tps"] and deep < cfg["min_tg_tps"]:
        return (
            False,
            f"too slow: {deep} t/s with {depth} tokens in the context, minimum {cfg['min_tg_tps']}",
        )
    return True, ""


def acc_key(a):
    """(cache, context) of a saved long-prompt result. Results of version 2.2 and older have only the label."""
    ctx, cache = a.get("context"), a.get("cache")
    if not ctx:
        c = re.search(r"ctx=(\d+)", a.get("label", ""))
        ctx = c.group(1) if c else 0
    if not cache:
        k = re.search(r"kv=(\S+)", a.get("label", ""))
        cache = k.group(1) if k else "*"
    return cache, int(ctx)


def acc_verdict(cfg, a):
    """'pass', 'fail' or None (request error, no valid result). Uses the score and the current rules,
    so saved results can be judged again after a rule change."""
    mt = re.fullmatch(r"(\d+)/(\d+)", str(a.get("score", "")))
    if a.get("status") in ("FAIL", "HTTP") or not mt:
        return None
    hits, n = int(mt.group(1)), int(mt.group(2))
    return "pass" if hits >= required_hits(cfg, n) else "fail"


def accuracy_state(cfg, acc, cache, context):
    """('PASS' | 'FAIL' | 'UNTESTED', 'score@pct% ...') for one cache type and context.
    Only fill levels in accuracy_pct count; other results stay in the report as information."""
    rows = []
    for a in acc:
        c, x = acc_key(a)
        if x == context and c in (cache, "*") and int(a.get("pct") or 0) in cfg["accuracy_pct"]:
            v = acc_verdict(cfg, a)
            if v:
                rows.append((a, v))
    if not rows:
        return "UNTESTED", ""
    detail = " ".join(
        f"{a['score']}@{a['pct']}%" for a, _ in sorted(rows, key=lambda r: int(r[0]["pct"]))
    )
    return ("FAIL" if any(v == "fail" for _, v in rows) else "PASS"), detail


def gate_rows(cfg, sts, acc):
    """One decision row per context for one cache type. sts = settings of that cache type, with speed."""
    best_tg = max((s.get("speed", {}).get("tg128@0", 0) for s in sts), default=0)
    rows = []
    for st in sts:
        speed_ok, note = speed_gate(cfg, st, best_tg)
        astate, recall = accuracy_state(cfg, acc, st["setting"]["cache"], st["context"])
        depth, deep = speed_at_depth(st)
        rows.append(
            {
                "cache": st["setting"]["cache"],
                "context": st["context"],
                "setting": sdesc(st["setting"]),
                "tg128": st.get("speed", {}).get("tg128@0", ""),
                "tg_deep": deep,
                "depth": depth,
                "speed_ok": speed_ok,
                "speed_note": note,
                "accuracy": astate,
                "recall": recall,
            }
        )
    accuracy_ran = any(r["accuracy"] != "UNTESTED" for r in rows)
    for r in rows:
        r["ok"] = r["speed_ok"] and (
            r["accuracy"] == "PASS" or (r["accuracy"] == "UNTESTED" and not accuracy_ran)
        )
        why = []
        if not r["speed_ok"]:
            why.append(r["speed_note"])
        if r["accuracy"] == "FAIL":
            why.append(
                f"long-prompt test failed ({r['recall']}, needs {cfg['accuracy_min_pct']}%)"
            )
        elif r["accuracy"] == "UNTESTED" and accuracy_ran:
            why.append("long-prompt test not run for this context (or no valid result)")
        r["reason"] = "; ".join(why)
    return rows


def cache_rank(cache):
    return CACHE_QUALITY.index(cache) if cache in CACHE_QUALITY else len(CACHE_QUALITY)


def choose_cache(cfg, picks):
    """Among the cache types that reach target_context: the best quality (f16 > q8_0 > q4_0) that keeps
    at least 85% of the fastest one's speed with a filled context. If none reaches the target: the one
    with the largest context, then more layers on the GPU, then the faster one."""
    target = cfg["target_context"]
    reach = [c for c, p in picks.items() if p["reliable"] and p["optimum"]["context"] >= target]
    if reach:
        fastest = max(deep_tg(picks[c]["optimum"]) for c in reach)
        return min(
            (c for c in reach if deep_tg(picks[c]["optimum"]) >= 0.85 * fastest), key=cache_rank
        )
    return max(
        picks,
        key=lambda c: (
            picks[c]["reliable"],
            picks[c]["optimum"]["context"],
            picks[c]["optimum"]["setting"].get("ngl", 0),
            deep_tg(picks[c]["optimum"]),
            -cache_rank(c),
        ),
    )


def recommend(cfg, settings, acc):
    """Optimum = the largest context that passes the speed rule and the long-prompt rule.
    Without long-prompt results the speed rule decides alone (rec['verified'] is False)."""
    by_cache = {}
    for st in settings:
        by_cache.setdefault(st["setting"]["cache"], []).append(st)
    picks = {}
    for cache, sts in by_cache.items():
        sts.sort(key=lambda x: x["context"])
        gates = gate_rows(cfg, sts, acc)
        good = [g["context"] for g in gates if g["ok"]]
        if good:
            best = max(good)
        else:  # nothing passes: the largest fast context not known to fail, flagged unreliable
            maybe = [g["context"] for g in gates if g["speed_ok"] and g["accuracy"] != "FAIL"]
            best = max(maybe) if maybe else sts[0]["context"]
        picks[cache] = {
            "maximum": sts[-1],
            "optimum": next(s for s in sts if s["context"] == best),
            "gates": gates,
            "reliable": bool(good),
        }
    if not picks:
        return None
    choice = choose_cache(cfg, picks)
    chosen = picks[choice]
    gate = next(g for g in chosen["gates"] if g["context"] == chosen["optimum"]["context"])
    return {
        "per_cache": picks,
        "choice": choice,
        "recommended": chosen["optimum"],
        "profile": cfg["profile"],
        "target_context": cfg["target_context"],
        "verified": gate["accuracy"] == "PASS",
        "target_met": chosen["reliable"] and chosen["optimum"]["context"] >= cfg["target_context"],
    }


def why_below_target(rec):
    """One line: which rule keeps the chosen cache type under target_context ('' if the target is met)."""
    if rec["target_met"]:
        return ""
    chosen, target = rec["per_cache"][rec["choice"]], rec["target_context"]
    above = [g for g in chosen["gates"] if g["context"] >= target]
    if not above:
        tried = set(rec["per_cache"])
        hint = (
            'add "q4_0" to cache_types of this model'
            if "q4_0" not in tried
            else "set mmproj_offload = false (vision models), use a smaller quantization"
        )
        return (
            f"no tested context >= {target} fits in memory (largest fit: {chosen['maximum']['context']}). "
            f"Try: {hint}, or lower min_free_vram_mb"
        )
    return f"ctx {above[0]['context']}: {above[0]['reason'] or 'no result'}"


Final = collections.namedtuple("Final", "context setting speed rec")


def final_setting(cfg, s):
    """Recommendation of a saved result, computed with the current rules.
    Returns Final(context, setting, speed, rec) or None."""
    if not s.get("settings"):
        return None
    rec = recommend(cfg, s["settings"], s.get("accuracy", []))
    if not rec:
        return None
    st = rec["recommended"]
    setting, tuned = dict(st["setting"]), s.get("tuned")
    if tuned:
        base = {k: v for k, v in tuned.items() if k not in ("ub", "threads")}
        if base == st["setting"]:
            setting = tuned
        else:  # the choice changed after tuning: keep threads, keep ubatch only if it is the default
            if tuned.get("threads"):
                setting["threads"] = tuned["threads"]
            if tuned.get("ub") and tuned["ub"] <= 512:
                setting["ub"] = tuned["ub"]
    return Final(st["context"], setting, st.get("speed", {}), rec)


# ---------------------------------------------------------------- commands
def cmd_doctor(args):
    raw = read_toml()
    cfg = base_config(args, raw)
    ok = True

    def line(status, what, detail=""):
        nonlocal ok
        ok = ok and status != "FAIL"
        log(f"[{status:<4}] {what:<22} {detail}")

    log(f"llmbench {VERSION}   config {CONFIG}{'' if raw else ' (missing: run init)'}")
    line("OK" if cfg["llama_dir"] else "WARN", "llama.cpp folder", cfg["llama_dir"] or "not found")
    for k in ("server", "bench"):
        line(
            "OK" if cfg[k] and os.access(cfg[k], os.X_OK) else "FAIL",
            f"{k} ({cfg['backend']})",
            cfg[k] or "not found",
        )
    others = (
        sorted(glob.glob(os.path.join(cfg["llama_dir"], "build-*"))) if cfg["llama_dir"] else []
    )
    if others:
        line("INFO", "other backends", ", ".join(os.path.basename(p)[6:] for p in others))
    line("OK" if os.path.isdir(cfg["models_dir"]) else "FAIL", "models folder", cfg["models_dir"])
    line("INFO", "profile", profile_summary(cfg)[len("Profile ") :])
    for text in override_lines(cfg):
        line("WARN", "profile override", text)
    if cfg["server"]:
        gpus = setup_devices(cfg)
        line(
            "OK" if len(gpus) == 1 else ("FAIL" if not gpus else "WARN"),
            "GPU devices",
            "; ".join(f"{d['kind']}{d['index']} {d['name']} {d['mib']} MiB" for d in gpus)
            or "none",
        )
        if cfg["_devenv"]:
            line(
                "INFO", "device variables", " ".join(f"{k}={v}" for k, v in cfg["_devenv"].items())
            )
        h = run_text([cfg["server"], "--help"], cfg["_env"])
        for flag in ("--n-cpu-moe", "--mmproj", "--no-mmproj-offload", "--fit"):
            line(
                "OK" if flag in h else "WARN",
                f"server flag {flag}",
                "" if flag in h else "missing (update llama.cpp)",
            )
        line("INFO", "llama.cpp build", build_id(cfg))
    try:
        total, used, free = vram_mb(cfg)
        line(
            "OK" if used < 1024 else "WARN",
            "VRAM (sysfs)",
            f"{gpu_dir(cfg)} total={total} used={used} MiB",
        )
    except SystemExit as e:
        line("FAIL", "VRAM (sysfs)", str(e))
    line("INFO", "RAM available", f"{ram_free_mb()} MiB, physical cores {physical_cores()}")
    groups = run_text(["id", "-nG"]).split()
    line(
        "OK" if {"render", "video"} <= set(groups) or os.geteuid() == 0 else "FAIL",
        "groups render+video",
        " ".join(groups),
    )
    procs = other_servers()
    line(
        "OK" if not procs else "WARN",
        "other llama-server",
        f"{len(procs)} running" if procs else "none",
    )
    line(
        "OK" if service_active(cfg["swap_service"]) else "WARN",
        "llama-swap service",
        "active" if service_active(cfg["swap_service"]) else "not active",
    )
    try:
        key = api_key(cfg, required=False)
    except SystemExit as e:
        key = ""
        line("FAIL", "API key file", str(e))
    line(
        "OK" if key else "WARN",
        "API key",
        f"from ${cfg['key_env']} or {cfg['key_file']}" if key else "not found",
    )
    try:
        models = Client(cfg["swap_url"], key).call("/v1/models", timeout=10).get("data", [])
        line(
            "OK",
            "llama-swap API",
            f"{cfg['swap_url']}  models: {', '.join(x['id'] for x in models)}",
        )
    except Exception as e:
        line("WARN", "llama-swap API", f"{cfg['swap_url']}: {http_error(e)}")
    line(
        "OK" if os.path.exists(cfg["swap_config"]) else "WARN",
        "llama-swap config",
        cfg["swap_config"],
    )
    if raw:
        _, models = load_config(args)
        for m in models.values():
            line(
                "INFO",
                f"model {m['name']}"[:22],
                f"{m['type']} {m['info']['size_mib']} MiB"
                + (f", vision: {os.path.basename(m['mmproj'])}" if m["mmproj"] else ""),
            )
    log("\nResult: " + ("ready" if ok else "fix the FAIL lines"))


def cmd_init(args):
    raw = read_toml()
    cfg = base_config(args, raw)
    models_dir = cfg["models_dir"]
    if not os.path.isdir(models_dir):
        sys.exit(f"Models folder not found: {models_dir}. Use --models-dir.")
    existing = raw.get("models", {})
    known = {os.path.realpath(resolve_model_path(cfg, v["path"])) for v in existing.values()}
    mains = [
        p
        for p in sorted(glob.glob(os.path.join(models_dir, "**", "*.gguf"), recursive=True))
        if not is_mmproj(p) and not is_extra_part(p)
    ]
    per_dir = {}
    for p in mains:
        per_dir.setdefault(os.path.dirname(p), []).append(p)
    found = []
    for p in mains:
        if os.path.realpath(p) in known:
            continue
        d = os.path.dirname(p)
        stem = re.sub(r"-00001-of-\d{5}$", "", os.path.basename(p)[:-5])
        name = os.path.basename(d) if d != models_dir and len(per_dir[d]) == 1 else stem
        name = re.sub(r"[^a-z0-9._-]", "-", name.lower())
        while name in existing or name in [n for n, _ in found]:
            name += "-2"
        found.append((name, p))
    new = not os.path.exists(CONFIG)
    if not new and found:
        with open(CONFIG, "rb") as f:
            text = f.read()
        if text and not text.endswith(b"\n\n"):
            with open(CONFIG, "a") as f:
                f.write("\n" if text.endswith(b"\n") else "\n\n")
    with open(CONFIG, "a") as f:
        if new:
            f.write(
                "# llmbench model list. Change values here; delete a block to remove a model.\n"
                "# Any value of DEFAULTS or of the profile in llmbench.py can be set in [defaults].\n"
                "# Model paths are relative to models_dir (absolute paths also work).\n\n"
                "[defaults]\n"
                f'# models_dir = "{tilde(models_dir)}"        # found: {tilde(models_dir)}\n'
                f'# llama_dir = "{tilde(cfg["llama_dir"])}"   # found: {tilde(cfg["llama_dir"]) or "-"}\n'
                f'backend = "{cfg["backend"]}"                   # build-<backend>/bin, e.g. rocm or vulkan\n'
                f"min_free_vram_mb = {DEFAULTS['min_free_vram_mb']}   # 1024 if the screen is on this GPU\n"
                f"min_free_ram_mb = {DEFAULTS['min_free_ram_mb']}\n"
                f'profile = "{DEFAULTS["profile"]}"                   # strict, balanced or long (see README)\n'
                "# target_context = 131072             # any value of the profile can be replaced here\n\n"
            )
        for name, p in found:
            try:
                info = model_info(p)
            except Exception as e:
                log(f"skip {p}: {e}")
                continue
            mm = find_mmproj(p)
            f.write(
                f'[models."{name}"]\n'
                f'path = "{rel_model_path(cfg, p)}"\n'
                f"# detected: {info['arch']}, {info['type']}, {info['layers']} layers, "
                f"{info['experts']} experts, native context {info['native_ctx']}, {info['size_mib']} MiB\n"
                + (
                    f'# vision: {os.path.basename(mm)} is used (mmproj = "auto"). mmproj = "" turns it off\n'
                    f"# mmproj_offload = false              # image encoder in RAM: less VRAM, slower images\n"
                    if mm
                    else ""
                )
                + f'# display_name = "{info["name"]}"\n'
                f"# contexts = [32768, 65536, 131072]   # optional: own list\n"
                f'# cache_types = ["q8_0"]              # optional\n'
                f"# extra_flags = []                    # optional: added to every server start\n"
                f'run_flags = []                        # sampling for llama-swap, e.g. ["--temp", "0.6"]\n\n'
            )
            log(f"added {name}: {p}" + (f"  (vision: {os.path.basename(mm)})" if mm else ""))
    for name, v in existing.items():
        p = resolve_model_path(cfg, v["path"])
        if os.path.exists(p) and "mmproj" not in v and find_mmproj(p):
            log(f"{name}: uses {os.path.basename(find_mmproj(p))} automatically (vision)")
    if not found:
        log("No new models found.")
    log(f"Config: {CONFIG}")


def cmd_info(args):
    cfg, models = load_config(args)
    log(
        f"{'model':<22} {'arch':<12} {'type':<6} {'layers':>6} {'experts':>8} {'attn':>5} "
        f"{'native':>8} {'size MiB':>9}  vision"
    )
    for m in models.values():
        i = m["info"]
        vis = (
            (os.path.basename(m["mmproj"]) + ("" if m["mmproj_offload"] else " (RAM)"))
            if m["mmproj"]
            else "-"
        )
        log(
            f"{m['name']:<22} {i['arch']:<12} {m['type']:<6} {i['layers']:>6} "
            f"{str(i['experts']) + '/' + str(i['experts_used']):>8} {i['attn_layers'] or '-'!s:>5} "
            f"{i['native_ctx']:>8} {i['size_mib']:>9}  {vis}"
        )
    total, used, free = vram_mb(cfg)
    log(f"\nServer: {cfg['server'] or 'not found'}  (backend {cfg['backend']})")
    log(
        f"GPU: {gpu_dir(cfg)}  VRAM total={total} MiB  free={free} MiB   RAM available={ram_free_mb()} MiB  "
        f"physical cores={physical_cores()}"
    )


class Lock:
    def __enter__(self):
        os.makedirs(RESULTS, exist_ok=True)
        self.f = open(os.path.join(RESULTS, ".lock"), "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Another llmbench sweep is running.")
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cmd_sweep(args):
    cfg, models = load_config(args)
    need_tools(cfg)
    if args.quick:
        cfg["bench_depths"], cfg["bench_reps"], cfg["bench_deep_fill_pct"] = [0], 1, 0
    if args.pct:
        cfg["accuracy_pct"] = [int(x) for x in args.pct.split(",")]
        check_config(cfg)
    if args.no_image_fit:
        cfg["image_fit"] = False
    todo = pick_models(models, args.models)
    restart = False
    with Lock():
        if not args.no_service_check and service_active(cfg["swap_service"]):
            if not args.stop_swap:
                sys.exit(f"{cfg['swap_service']} is running. Stop it, or use --stop-swap.")
            subprocess.run(["sudo", "systemctl", "stop", cfg["swap_service"]], check=True)
            restart = True
        try:
            procs = other_servers()
            if procs:
                warn("other llama-server processes use memory: " + "; ".join(procs)[:300])
            setup_devices(cfg)
            help_text = run_text([cfg["server"], "--help"], cfg["_env"])
            cfg["_fit_flag"] = "--fit" in help_text
            cfg["_build"] = build_id(cfg)
            total, used, free = vram_mb(cfg)
            if used > 1024:
                warn(f"{used} MiB VRAM is already used. Results will be lower than possible.")
            log(
                f"Run {RUN_ID}  backend {cfg['backend']}  build {cfg['_build']}  "
                f"VRAM free {free}/{total} MiB  RAM available {ram_free_mb()} MiB"
            )
            log(
                f"Server: {cfg['server']}  "
                + " ".join(f"{k}={v}" for k, v in cfg["_devenv"].items())
            )
            log(profile_summary(cfg))
            for line in override_lines(cfg):
                warn(line)
            with tracked("sweep", [m["name"] for m in todo]):
                for m in todo:
                    PROGRESS.start_model(m["name"])
                    sweep_model(cfg, m, args)
        finally:
            stop_all()
            if restart:
                subprocess.run(["sudo", "systemctl", "start", cfg["swap_service"]])
    cmd_report(args)


def sweep_model(cfg, m, args):
    i = m["info"]
    vision = bool(m["mmproj"]) and not args.skip_vision
    log(
        f"\n=== {m['name']}  {i['arch']} {m['type']} {i['layers']} layers {i['experts']} experts  "
        f"{i['size_mib']} MiB  contexts {m['contexts']}"
        + (f"  vision {os.path.basename(m['mmproj'])}" if m["mmproj"] else "")
    )
    if not m["contexts"]:
        log("  no context size left (all above the native context)")
        return
    t0 = time.time()
    state = {
        "model": m["name"],
        "run_id": RUN_ID,
        "backend": cfg["backend"],
        "build": cfg["_build"],
        "server": cfg["server"],
        "env": cfg["_devenv"],
        "info": i,
        "type": m["type"],
        "path": m["path"],
        "mmproj": m["mmproj"],
        "mmproj_offload": m["mmproj_offload"],
        "display_name": m["display_name"],
        "tools": m["tools"],
        "run_flags": m["run_flags"],
        "extra_flags": m["extra_flags"],
        "settings": [],
        "recommendation": None,
        "tuned": None,
        "accuracy": [],
        "vision": [],
        "complete": False,
        "schema": 3,
        "profile": cfg["profile"],
        "rules": {k: cfg[k] for k in PROFILES[cfg["profile"]]},
    }
    if (
        cfg["target_context"] not in m["contexts"]
        and m["contexts"]
        and max(m["contexts"]) >= cfg["target_context"]
    ):
        warn(
            f"{m['name']}: target_context {cfg['target_context']} is not in its contexts list; "
            "the largest value below it is the next test."
        )
    elif m["contexts"] and max(m["contexts"]) < cfg["target_context"]:
        warn(
            f"{m['name']}: no context in the list reaches target_context {cfg['target_context']} "
            f"(native context {i['native_ctx']})."
        )
    fitter = Fitter(cfg, m)
    PROGRESS.phase("fit", detail=f"contexts {m['contexts']}, caches {m['cache_types']}")
    settings = (sweep_moe if m["type"] == "moe" else sweep_dense)(cfg, m, fitter)
    if not settings:
        log("  no setting fits")
        return
    state["settings"] = settings
    speed_phase(cfg, m, settings)
    rec = recommend(cfg, settings, [])
    state["recommendation"] = rec
    save_state(state)  # keep the progress if the run stops later
    acc, vis = [], []
    if rec and args.accuracy:
        acc = accuracy_search(cfg, m, settings, rec)
        rec = recommend(cfg, settings, acc)
        state.update({"recommendation": rec, "accuracy": acc})
        save_state(state)
    # tune the final optimum (after the accuracy search), then test images on the tuned setting
    tuned = (
        tune_phase(cfg, m, rec["recommended"], fitter) if (not args.skip_tune and rec) else None
    )
    state["tuned"] = tuned
    if rec and vision:
        st = {
            "context": rec["recommended"]["context"],
            "setting": tuned or rec["recommended"]["setting"],
        }
        PROGRESS.phase("image", cfg["vision_runs"])
        log(f"  image test: ctx={st['context']} {sdesc(st['setting'])}")
        _, vis = served_phase(cfg, m, st, "optimum", False, True)
    state.update(
        {
            "recommendation": rec,
            "accuracy": acc,
            "vision": vis,
            "complete": True,
            "minutes": round((time.time() - t0) / 60, 1),
        }
    )
    save_state(state)
    if rec:
        r = rec["recommended"]
        vtxt = ""
        if vis:
            vtxt = f"   images {sum(1 for x in vis if x['status'] == 'PASS')}/{len(vis)} PASS"
        log(
            f"  RESULT {m['name']}: optimum ctx={r['context']} {sdesc(tuned or r['setting'])}   "
            f"maximum ctx={rec['per_cache'][rec['choice']]['maximum']['context']}{vtxt}   "
            f"({state['minutes']} min)"
        )
        if rec["target_met"]:
            log(
                f"  TARGET {cfg['target_context']}: met"
                + (
                    ""
                    if rec["verified"]
                    else " (speed only: run sweep with --accuracy to check long-prompt recall)"
                )
            )
        else:
            log(f"  TARGET {cfg['target_context']}: NOT met - {why_below_target(rec)}")


def swap_client(cfg, args):
    key = "" if args.no_key else api_key(cfg)
    return Client(args.url or cfg["swap_url"], key)


def cmd_fill(args):
    cfg, _ = load_config(args)
    client = swap_client(cfg, args)
    names = args.models
    if names == ["all"]:
        names = [x["id"] for x in client.call("/v1/models").get("data", [])]
    pcts = [int(x) for x in (args.pct or ",".join(map(str, cfg["fill_pct"]))).split(",")]
    with tracked("fill", names):
        for n in names:
            PROGRESS.start_model(n)
            log(f"=== {n}")
            PROGRESS.phase("fill", len(pcts))
            fill_series(
                cfg, client, n, pcts, args.label or f"{n} (llama-swap)", args.think, tick=True
            )


def cmd_vision(args):
    cfg, models = load_config(args)
    if args.px:
        cfg["image_px"] = args.px
    client = swap_client(cfg, args)
    names = args.models
    if names == ["all"]:
        names = [n for n, m in models.items() if m["mmproj"]]
        if not names:
            sys.exit("No model in models.toml has an mmproj file.")
    with tracked("vision", names):
        for n in names:
            PROGRESS.start_model(n)
            log(f"=== {n}")
            PROGRESS.phase("image", args.runs or cfg["vision_runs"])
            vision_series(cfg, client, n, args.label or f"{n} (llama-swap)", args.runs, tick=True)


def status_text(data, now_ts=None):
    """Text for the 'status' command from a progress.json dict."""
    now_ts = time.time() if now_ts is None else now_ts
    age = now_ts - data.get("updated", now_ts)
    alive = True
    try:
        os.kill(int(data.get("pid", 0)), 0)
    except OSError, ValueError:
        alive = False
    state = data.get("state", "?")
    lines = [
        f"Run {data.get('run_id')}  {data.get('kind')}  state: {state}",
        data.get("label", ""),
        f"Models: {data.get('model_index')}/{data.get('model_total')}  ({', '.join(data.get('models', []))})",
        f"Last update {fmt_dur(age)} ago",
    ]
    if state == "running" and not alive:
        lines.append(
            "WARNING: the process is gone. The run was killed or crashed. Finished models are saved."
        )
    elif state == "running" and age > 600:
        lines.append(
            "WARNING: no update for more than 10 minutes. A long test or model load can take this long; "
            "check the newest file in results/logs/."
        )
    return "\n".join(l for l in lines if l)


def cmd_status(args):
    path = os.path.join(RESULTS, "progress.json")
    while True:
        try:
            with open(path) as f:
                data = json.load(f)
        except OSError, ValueError:
            sys.exit(f"No progress file: {path}. No run was started with these results folder.")
        log(status_text(data))
        if not args.watch or data.get("state") != "running":
            return
        time.sleep(args.watch)
        log("")


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def system_info(cfg):
    def run(c):
        try:
            return subprocess.run(
                c, shell=True, capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception:
            return ""

    total, used, free = vram_mb(cfg)
    git = (
        run(f"git -C {shlex.quote(cfg['llama_dir'])} log -1 --format='%h %cs' 2>/dev/null")
        if cfg["llama_dir"]
        else ""
    )
    devs = "; ".join(
        f"{d['kind']}{d['index']} {d['name']} {d['mib']} MiB" for d in cfg.get("_devices", [])
    )
    cpu = run("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")
    ram = run("free -h | awk '/Mem:/ {print $2}'")
    gpu = run("lspci | grep -iE 'vga|display'")
    rocm = run(
        "PATH=$PATH:$(ls -d /opt/rocm*/bin 2>/dev/null | tr '\\n' :) hipconfig --version 2>/dev/null"
    )
    return "\n".join(
        [
            "Kernel: "
            + run("uname -r")
            + "   OS: "
            + run(". /etc/os-release && echo $PRETTY_NAME"),
            f"CPU: {cpu}  physical cores={physical_cores()}",
            f"RAM: {ram}  available now={ram_free_mb()} MiB",
            f"GPU: {gpu}",
            f"VRAM: total={total} MiB  used now={used} MiB   llama.cpp devices: {devs or '-'}",
            f"Device variables: {' '.join(f'{k}={v}' for k, v in cfg.get('_devenv', {}).items()) or '-'}",
            f"ROCm: {rocm or '-'}",
            f"llama.cpp: {cfg['llama_dir']} {git}  backend {cfg['backend']}  build {build_id(cfg) if cfg['server'] else '-'}",
            f"llama-swap: {run('command -v llama-swap >/dev/null && llama-swap -version 2>&1 | head -n 1') or '-'}",
            f"llmbench: {VERSION}",
        ]
    )


RULE_KEYS = (
    "profile",
    "target_context",
    "speed_tolerance_pct",
    "min_tg_tps",
    "needle_style",
    "needles",
    "accuracy_pct",
    "accuracy_min_pct",
)
SETTING_KEYS = RULE_KEYS + (
    "min_free_vram_mb",
    "min_free_ram_mb",
    "bench_depths",
    "bench_deep_fill_pct",
    "tune_ubatch",
    "common_flags",
    "fit_warmup",
    "image_fit",
    "image_px",
    "vision_runs",
)


def report_header(cfg, models):
    L = [
        f"# Local LLM benchmark report - {now()}",
        "",
        "## System",
        "",
        "```",
        system_info(cfg),
        "```",
        "",
        "## Rules and settings used",
        "",
        "```",
    ]
    L += [f"{k} = {cfg[k]}" for k in SETTING_KEYS]
    L += ["```", "", profile_summary(cfg), ""]
    L += [f"WARNING: {line}" for line in override_lines(cfg)]
    L += [
        "",
        "Speed values come from llama-bench (text only, no image encoder). The fit tests use "
        "llama-server with the image encoder and, for vision models, one image request.",
        "",
        "## Models",
        "",
    ]
    L.append(
        table(
            [
                dict(
                    model=n,
                    arch=m["info"]["arch"],
                    type=m["type"],
                    layers=m["info"]["layers"],
                    experts=f"{m['info']['experts']}/{m['info']['experts_used']}",
                    attn_layers=m["info"]["attn_layers"] or "-",
                    native_ctx=m["info"]["native_ctx"],
                    size_mib=m["info"]["size_mib"],
                    file=os.path.basename(m["path"]),
                    vision=(
                        os.path.basename(m["mmproj"]) + ("" if m["mmproj_offload"] else " (RAM)")
                    )
                    if m["mmproj"]
                    else "-",
                )
                for n, m in models.items()
            ],
            [
                "model",
                "arch",
                "type",
                "layers",
                "experts",
                "attn_layers",
                "native_ctx",
                "size_mib",
                "file",
                "vision",
            ],
        )
    )
    return L


def report_recommendations(cfg, states):
    rows, below = [], []
    for key, s in states.items():
        fs = final_setting(cfg, s)
        if not fs:
            continue
        o, rec = fs.rec["recommended"], fs.rec
        mx = rec["per_cache"][rec["choice"]]["maximum"]
        vis = s.get("vision", [])
        depth, deep = speed_at_depth(o)
        rows.append(
            {
                "model": key,
                "optimum_ctx": fs.context,
                "optimum_setting": sdesc(fs.setting),
                "target": "met" if rec["target_met"] else "NOT met",
                "checked": "speed + recall" if rec["verified"] else "speed only",
                "tg128": fs.speed.get("tg128@0", ""),
                "tg128_deep": f"{deep}@{depth}" if depth else "",
                "pp512": fs.speed.get("pp512@0", ""),
                "max_ctx": mx["context"],
                "max_setting": sdesc(mx["setting"]),
                "tg128_at_max": mx.get("speed", {}).get("tg128@0", ""),
                "images": f"{sum(1 for v in vis if v['status'] == 'PASS')}/{len(vis)}"
                if vis
                else "-",
                "run": s["run_id"] + ("" if s.get("complete", True) else " (incomplete)"),
            }
        )
        if not rec["target_met"]:
            below.append(
                f"- **{key}** is below the target {cfg['target_context']}: {why_below_target(rec)}"
            )
    L = [
        "",
        "## Recommendations",
        "",
        table(
            rows,
            [
                "model",
                "optimum_ctx",
                "optimum_setting",
                "target",
                "checked",
                "tg128",
                "tg128_deep",
                "pp512",
                "max_ctx",
                "max_setting",
                "tg128_at_max",
                "images",
                "run",
            ],
        ),
        "",
    ]
    if below:
        L += below + [""]
    return L


def report_decisions(cfg, s):
    """Table: why each tested context is accepted or rejected under the current rules."""
    fs = final_setting(cfg, s)
    if not fs:
        return []
    rows = []
    for cache, p in fs.rec["per_cache"].items():
        for g in p["gates"]:
            chosen = cache == fs.rec["choice"] and g["context"] == fs.context
            rows.append(
                {
                    "context": g["context"],
                    "kv": cache,
                    "tg128": g["tg128"],
                    "tg_deep": f"{g['tg_deep']}@{g['depth']}" if g["depth"] else "",
                    "recall": g["recall"] or "-",
                    "result": ("OK" if g["ok"] else "rejected: " + g["reason"])
                    + ("  <- recommended" if chosen else ""),
                }
            )
    rows.sort(key=lambda r: (r["kv"], r["context"]))
    return [
        f"### Decision per context (profile {cfg['profile']})",
        "",
        table(rows, ["context", "kv", "tg128", "tg_deep", "recall", "result"]),
        "",
    ]


def report_model(cfg, key, s):
    L = [
        f"## {key}",
        "",
        f"Run {s['run_id']}, backend {s.get('backend')}, build {s['build']}, "
        f"{s.get('minutes', '?')} minutes" + ("" if s.get("complete", True) else ", INCOMPLETE"),
        "",
    ]
    rows = []
    for st in s["settings"]:
        row = {"context": st["context"], "setting": sdesc(st["setting"])}
        row.update(st.get("speed", {}))
        rows.append(row)
    cols = ["context", "setting"] + sorted(
        {k for r in rows for k in r if "@" in k},
        key=lambda k: (k.split("@")[0], int(k.split("@")[1])),
    )
    L += ["### Speed per context (tokens/s, @ = filled context depth)", "", table(rows, cols), ""]
    L += report_decisions(cfg, s)
    sel = lambda f: [
        r for r in read_csv(f) if r.get("model") == s["model"] and r["run_id"] == s["run_id"]
    ]
    tune = sel("tune.csv")
    if tune:
        L += [
            "### Tuning (batch size and threads)",
            "",
            table(tune, ["ubatch", "threads", "pp512", "tg128"]),
            "",
        ]
    if s.get("accuracy"):
        rows = []
        for a in s["accuracy"]:
            cache, ctx = acc_key(a)
            rows.append(
                dict(
                    a,
                    context=ctx or "",
                    cache="" if cache == "*" else cache,
                    verdict=acc_verdict(cfg, a) or "-",
                )
            )
        L += [
            "### Long-prompt recall",
            "",
            table(
                rows,
                [
                    "context",
                    "cache",
                    "pct",
                    "n_ctx",
                    "prompt_tokens",
                    "prompt_tps",
                    "gen_tps",
                    "ttft_s",
                    "score",
                    "status",
                    "verdict",
                    "style",
                    "error",
                ],
            ),
            "",
        ]
    if s.get("vision"):
        L += [
            "### Image test (number + frame colour)",
            "",
            table(
                s["vision"],
                [
                    "label",
                    "image_px",
                    "prompt_tokens",
                    "prompt_tps",
                    "ttft_s",
                    "expected",
                    "score",
                    "status",
                    "answer",
                ],
            ),
            "",
        ]
    L += [
        "### Fit tests",
        "",
        table(
            sel("fit.csv"),
            [
                "context",
                "setting",
                "status",
                "load_s",
                "vram_free_mib",
                "vram_used_by_model_mib",
                "ram_free_mib",
                "ram_used_by_model_mib",
                "error",
            ],
        ),
        "",
    ]
    return L


def report_swap_runs():
    """Results of the fill and vision commands (through llama-swap)."""
    L = []
    for f, title, cols in (
        (
            "accuracy.csv",
            "Long-prompt recall through llama-swap",
            [
                "time",
                "model",
                "pct",
                "n_ctx",
                "prompt_tokens",
                "prompt_tps",
                "gen_tps",
                "ttft_s",
                "score",
                "status",
                "style",
                "error",
            ],
        ),
        (
            "vision.csv",
            "Image tests through llama-swap",
            [
                "time",
                "model",
                "image_px",
                "prompt_tokens",
                "prompt_tps",
                "ttft_s",
                "expected",
                "score",
                "status",
                "answer",
            ],
        ),
    ):
        rows = [r for r in read_csv(f) if "(llama-swap)" in r.get("label", "")]
        if rows:
            L += [f"## {title}", "", table(rows[-40:], cols), ""]
    return L


def cmd_report(args):
    cfg, models = load_config(args)
    if cfg["server"] and "_devices" not in cfg:
        cfg["_quiet_devices"] = args.cmd != "report"
        setup_devices(cfg)
    states = load_states()
    path = os.path.join(RESULTS, f"report-{RUN_ID}.md")
    L = report_header(cfg, models) + report_recommendations(cfg, states)
    for key, s in states.items():
        L += report_model(cfg, key, s)
    L += report_swap_runs()
    if os.path.exists(cfg["swap_config"]):
        try:
            current = open(cfg["swap_config"]).read()
            L += [f"## Current {cfg['swap_config']}", "", "```yaml", current, "```", ""]
        except PermissionError:
            pass
    L += [
        "## Recommended llama-swap config",
        "",
        "```yaml",
        swap_yaml(cfg, load_states(cfg["backend"])),
        "```",
        "",
    ]
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L))
    log(f"\nReport: {path}")
    for line in override_lines(cfg):
        warn(line)


def swap_yaml(cfg, states):
    sp = lambda p: (
        "${models_dir}/" + rel_model_path(cfg, p)
        if os.path.realpath(p).startswith(cfg["models_dir"].rstrip(os.sep) + os.sep)
        else p
    )
    help_text = run_text([cfg["server"], "--help"], cfg.get("_env")) if cfg["server"] else ""
    macro_flags = list(cfg["common_flags"])
    if "--fit" in help_text:
        macro_flags += ["--fit", "off"]  # use exactly the tested values
    if "--no-webui" in help_text:
        macro_flags.append("--no-webui")
    out = [
        f"# Generated by llmbench {VERSION} on {now()} (backend {cfg['backend']}, profile {cfg['profile']}).",
        "# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json",
        f"healthCheckTimeout: {cfg['swap_health_timeout']}",
        "logLevel: info",
        "logToStdout: both",
        f"startPort: {cfg['swap_start_port']}",
        f"globalTTL: {cfg['swap_ttl']}",
        "sendLoadingState: true",
        "",
    ]
    if cfg["swap_api_keys"]:
        out += ["apiKeys:", f'  - "${{env.{cfg["key_env"]}}}"', ""]
    out += [
        "macros:",
        "  server: >-",
        f"    {cfg['server']}",
        "    --host 127.0.0.1 --port ${PORT}",
        "    " + " ".join(macro_flags),
        f"  models_dir: {cfg['models_dir']}",
        "",
        "models:",
    ]
    skipped = []
    for n, s in states.items():
        fs = final_setting(cfg, s)
        if not fs:
            skipped.append(n)
            continue
        ctx, setting, speed, rec = fs
        st = {"context": ctx, "speed": speed}
        lines = ["${server}", f"-m {sp(s['path'])}"]
        if s.get("mmproj"):
            lines.append(
                f"--mmproj {sp(s['mmproj'])}"
                + ("" if s.get("mmproj_offload", True) else " --no-mmproj-offload")
            )
        lines.append(
            f"-c {st['context']} "
            + " ".join(server_flags({"extra_flags": s.get("extra_flags", [])}, setting))
        )
        if s.get("run_flags"):
            lines.append(" ".join(s["run_flags"]))
        out += [f'  "{n}":']
        if s.get("display_name"):
            out.append(f'    name: "{s["display_name"]}"')
        out += ["    cmd: |"] + [f"      {l}" for l in lines]
        env = s.get("env") or cfg.get("_devenv") or {}
        if env:
            out += ["    env:"] + [f'      - "{k}={v}"' for k, v in env.items()]
        out += [
            "    capabilities:",
            f"      in: [{'text, image' if s.get('mmproj') else 'text'}]",
            "      out: [text]",
            f"      tools: {'true' if s.get('tools', True) else 'false'}",
            f"      context: {st['context']}",
            f"    # tested {s['run_id']}: {sdesc(setting)}"
            + (f", tg128={st.get('speed', {}).get('tg128@0', '?')} t/s" if st.get("speed") else "")
            + ("" if rec["verified"] else ", long-prompt recall not checked")
            + ("" if rec["target_met"] else f", BELOW target {rec['target_context']}"),
            "",
        ]
    if skipped:
        out.append(f"# not included (no result): {', '.join(skipped)}")
    return "\n".join(out).rstrip() + "\n"


def swap_models_in(path):
    try:
        text = open(path).read()
    except OSError:
        return []
    m = re.search(r"^models:\s*\n((?:[ \t]+.*\n|\s*\n)+)", text + "\n", re.MULTILINE)
    return re.findall(r'^  "?([^"\s:#][^":]*)"?:\s*$', m.group(1), re.MULTILINE) if m else []


def cmd_swap(args):
    cfg, _ = load_config(args)
    if cfg["server"]:
        setup_devices(cfg)
    states = load_states(cfg["backend"])
    if not states:
        sys.exit(f"No results for backend '{cfg['backend']}'. Run sweep first.")
    path = os.path.join(RESULTS, "llama-swap.recommended.yaml")
    text = swap_yaml(cfg, states)
    with open(path, "w") as f:
        f.write(text)
    log(text)
    for n, st in states.items():
        fs = final_setting(cfg, st)
        if fs and not fs.rec["verified"]:
            warn(
                f"{n}: long-prompt recall was not checked for ctx {fs.context}. "
                f"Run: {sys.argv[0]} sweep {n} --accuracy --stop-swap"
            )
        if fs and not fs.rec["target_met"]:
            warn(f"{n}: below target {cfg['target_context']}: {why_below_target(fs.rec)}")
    missing = [n for n in swap_models_in(cfg["swap_config"]) if n not in states]
    if missing:
        warn(
            f"{cfg['swap_config']} has models without results: {', '.join(missing)}. "
            "They are not in the new file. Copy their blocks by hand if you need them."
        )
    log(f"Saved: {path}")
    if not args.install:
        log(f"Check it, then: {sys.argv[0]} swap --install   (backs up {cfg['swap_config']})")
        return
    if missing and not args.force:
        sys.exit("Not installed: models would be removed. Use --force to install anyway.")
    dst = cfg["swap_config"]
    if os.path.exists(dst):
        subprocess.run(["sudo", "cp", "-p", dst, f"{dst}.bak-{RUN_ID}"], check=True)
        log(f"Backup: {dst}.bak-{RUN_ID}")
    subprocess.run(["sudo", "install", "-m", "644", path, dst], check=True)
    log(
        f"Installed: {dst}. llama-swap reloads it with --watch-config; otherwise: "
        f"sudo systemctl restart {cfg['swap_service']}"
    )


def main():
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("paths (override models.toml and auto-detection)")
    g.add_argument("--config", default=argparse.SUPPRESS, help="models.toml path")
    g.add_argument("--results", default=argparse.SUPPRESS, help="results folder")
    g.add_argument(
        "--backend", default=argparse.SUPPRESS, help="rocm, vulkan, ... (build-<backend>)"
    )
    g.add_argument("--llama-dir", dest="llama_dir", default=argparse.SUPPRESS)
    g.add_argument("--models-dir", dest="models_dir", default=argparse.SUPPRESS)
    g.add_argument("--server", default=argparse.SUPPRESS, help="llama-server binary")
    g.add_argument("--bench", default=argparse.SUPPRESS, help="llama-bench binary")
    g.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        choices=sorted(PROFILES),
        help="decision rules: strict (2.2), balanced (default), long",
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common]
    )
    ap.add_argument("--version", action="version", version=f"llmbench {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    P = lambda name, h: sub.add_parser(name, help=h, parents=[common])
    P("doctor", "check paths, GPU devices, llama-swap and the API key")
    P("init", "scan the models folder and update models.toml")
    P("info", "show model details from the GGUF files")
    s = P("sweep", "fit, speed, tune, image check (and accuracy) for models")
    s.add_argument("models", nargs="*", default=["all"])
    s.add_argument("--accuracy", action="store_true", help="also run the long-prompt test (slow)")
    s.add_argument("--pct", help="accuracy fill percentages, e.g. 50,90")
    s.add_argument("--skip-tune", action="store_true")
    s.add_argument("--skip-vision", action="store_true", help="no image accuracy test")
    s.add_argument(
        "--no-image-fit", action="store_true", help="fit tests without an image request"
    )
    s.add_argument("--quick", action="store_true", help="depth 0 only, 1 repetition")
    s.add_argument(
        "--stop-swap", action="store_true", help="stop llama-swap during the test, start it after"
    )
    s.add_argument("--no-service-check", action="store_true")
    for name, h in (
        ("fill", "long-prompt accuracy test through llama-swap"),
        ("vision", "image test through llama-swap"),
    ):
        f = P(name, h)
        f.add_argument("models", nargs="+", help="model IDs or 'all'")
        f.add_argument("--url", help="default: swap_url")
        f.add_argument("--no-key", action="store_true")
        f.add_argument("--label")
        if name == "fill":
            f.add_argument("--pct")
            f.add_argument("--think", action="store_true")
        else:
            f.add_argument("--runs", type=int, help="number of images (default vision_runs)")
            f.add_argument("--px", type=int, help="image size in pixels (default image_px)")
    st = P("status", "show the progress of the running (or last) sweep, fill or vision command")
    st.add_argument("--watch", type=int, metavar="SECONDS", help="repeat until the run ends")
    P("report", "write the Markdown report")
    w = P("swap", "write a llama-swap config from the latest results")
    w.add_argument("--install", action="store_true", help="back up and replace swap_config (sudo)")
    w.add_argument("--force", action="store_true", help="install even if models would be removed")
    args = ap.parse_args()

    global CONFIG, RESULTS
    CONFIG = xp(getattr(args, "config", None) or os.environ.get("LLMBENCH_CONFIG") or CONFIG)
    RESULTS = xp(getattr(args, "results", None) or os.environ.get("LLMBENCH_RESULTS") or RESULTS)
    atexit.register(stop_all)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        {
            "doctor": cmd_doctor,
            "init": cmd_init,
            "info": cmd_info,
            "sweep": cmd_sweep,
            "fill": cmd_fill,
            "vision": cmd_vision,
            "report": cmd_report,
            "swap": cmd_swap,
            "status": cmd_status,
        }[args.cmd](args)
    except KeyboardInterrupt:
        stop_all()
        sys.exit("\nStopped. Finished models are saved in results/; run 'report' to see them.")


if __name__ == "__main__":
    main()

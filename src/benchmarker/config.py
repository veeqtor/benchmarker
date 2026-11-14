"""Configuration management for the benchmarker package.

Handles loading models.toml, merging DEFAULTS < profile < toml values,
resolving paths, and loading model metadata.
"""

import datetime
import math
import os
import shutil
import sys
import tomllib

from benchmarker.gguf import find_mmproj, model_info
from benchmarker.progress import warn
from benchmarker.utils import tilde

# ── Paths and state (module-level, like the original monolithic script) ──
VERSION = "2.3.6"
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "..", "models.toml")
RESULTS = os.path.join(HERE, "..", "..", "results")
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ── Default configuration values ──────────────────────────────────────────
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

# ── Decision rules ────────────────────────────────────────────────────────
# Order of priority: DEFAULTS < PROFILES[profile] < models.toml [defaults].
# All profiles must define the same keys (checked by tests/test_config.py).
PROFILES = {
    # Rules of version 2.2: exact recall of 3 values at 50% and 90% fill, 10% speed loss.
    "strict": {
        "target_context": 65536,
        "speed_tolerance_pct": 10,  # a context is slow if tg128 is more than this much below the best
        "min_tg_tps": 0,  # minimum generation speed (t/s) with a filled context (0 = off)
        "needle_style": "line",  # long-prompt test: "line" (Line 123: ...) or "keyed" (Project name: ...)
        "needles": 3,  # values to find in one test
        "accuracy_pct": [50, 90],  # fill levels of the long-prompt test
        "accuracy_min_pct": 100,
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


# ── Report keys ────────────────────────────────────────────────────────────

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


# ── Path resolution ───────────────────────────────────────────────────────


def xp(p):
    """Expand user vars (~, $VAR) in a path string."""
    return os.path.expanduser(os.path.expandvars(p)) if isinstance(p, str) else p


def first_existing(cands, test=os.path.isdir):
    """Return the first candidate that passes the test, or ''."""
    for c in cands:
        c = xp(c)
        if c and test(c):
            return c
    return ""


def resolve_paths(cfg):
    """Resolve all path fields in cfg (llama_dir, server, bench, models_dir, etc.)."""
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
    """Resolve a model path: absolute as-is, otherwise relative to models_dir."""
    p = xp(p)
    return p if os.path.isabs(p) else os.path.join(cfg["models_dir"], p)


def rel_model_path(cfg, p):
    """Convert an absolute model path to a path relative to models_dir (or ~-prefixed)."""
    real = os.path.realpath(p)
    base = cfg["models_dir"].rstrip(os.sep) + os.sep
    return real[len(base) :] if real.startswith(base) else tilde(p)


# ── Config loading and merging ────────────────────────────────────────────


def read_toml():
    """Load models.toml and return the [defaults] and [models] tables."""
    if not os.path.exists(CONFIG):
        return {}
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)


def merge_config(user_defaults, profile_name=None):
    """Merge DEFAULTS < profile < user_defaults. Pure function (no file access).

    Also records which values of models.toml replace a value of the profile
    (stored as '_overrides').
    """
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
    """Validate config values. Exit with a clear message on errors."""
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
    """Return a human-readable description of the current profile settings."""
    # Inline required_hits logic to avoid circular import with accuracy.py
    count = int(cfg["needles"])
    min_pct = int(cfg["accuracy_min_pct"])
    required = max(1, math.ceil(count * min_pct / 100))
    return (
        f"Profile {cfg['profile']}: target context {cfg['target_context']}, speed loss up to "
        f"{cfg['speed_tolerance_pct']}%, minimum {cfg['min_tg_tps']} t/s with a filled context, recall "
        f"{required}/{cfg['needles']} values ({cfg['needle_style']} test) "
        f"at {'/'.join(str(p) for p in cfg['accuracy_pct'])}% fill"
    )


def override_lines(cfg):
    """Warning messages for values in models.toml [defaults] that override profile values."""
    return [
        f"models.toml sets {k} = {mine}; profile '{cfg['profile']}' uses {theirs}. "
        f"Remove the line from [defaults] to use the profile."
        for k, (mine, theirs) in cfg.get("_overrides", {}).items()
    ]


def base_config(args=None, raw=None):
    """Build config from models.toml, environment variables, and CLI args."""
    raw = read_toml() if raw is None else raw
    cfg = merge_config(
        raw.get("defaults", {}),
        (getattr(args, "profile", None) if args else None)
        or os.environ.get("BENCHMARKER_PROFILE"),
    )
    check_config(cfg)
    for env, key in (
        ("BENCHMARKER_LLAMA_DIR", "llama_dir"),
        ("BENCHMARKER_MODELS_DIR", "models_dir"),
        ("BENCHMARKER_BACKEND", "backend"),
        ("BENCHMARKER_SWAP_URL", "swap_url"),
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
    """Full config loading: merge config + load model metadata from models.toml."""
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
    """Select models by name, or return all if 'all' or empty list."""
    if not names or names == ["all"]:
        return list(models.values())
    out = []
    for n in names:
        if n not in models:
            sys.exit(f"Unknown model '{n}'. Known: {', '.join(models) or '(none)'}")
        out.append(models[n])
    return out


def need_tools(cfg):
    """Verify that required tools (llama-server, llama-bench) exist and are executable."""
    for k, tool in (("server", "llama-server"), ("bench", "llama-bench")):
        if not cfg[k] or not os.access(cfg[k], os.X_OK):
            sys.exit(
                f"{tool} not found (backend '{cfg['backend']}'). Set llama_dir or {k} in "
                f"models.toml [defaults], or use --llama-dir / --{k}. Run: {sys.argv[0]} doctor"
            )


# ── Settings helpers ──────────────────────────────────────────────────────


def server_flags(m, s):
    """Build llama-server command-line flags from a setting dict."""
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
    """Build llama-bench command-line flags from a setting dict."""
    f = ["-ngl", str(s["ngl"]), "-fa", cfg["bench_fa"], "-ctk", s["cache"], "-ctv", s["cache"]]
    if s.get("ncmoe"):
        f += ["--n-cpu-moe", str(s["ncmoe"])]
    if s.get("ub"):
        f += ["-ub", str(s["ub"])]
    if s.get("threads"):
        f += ["-t", str(s["threads"])]
    return f


def sdesc(s):
    """Generate a human-readable string from a setting dict (e.g. 'ngl=48 kv=q8_0')."""
    parts = [f"ngl={s['ngl']}", f"kv={s['cache']}"]
    if s.get("ncmoe"):
        parts.append(f"n-cpu-moe={s['ncmoe']}")
    if s.get("ub"):
        parts.append(f"ub={s['ub']}")
    if s.get("threads"):
        parts.append(f"t={s['threads']}")
    return " ".join(parts)


def required_hits(cfg, count):
    """Minimum number of correct answers needed to pass an accuracy test."""
    return max(1, math.ceil(count * cfg["accuracy_min_pct"] / 100))


__all__ = [
    "ACTIVE_SERVERS",
    "CACHE_QUALITY",
    "CONFIG",
    "DEFAULTS",
    "DEVICE_ENV",
    "LLAMA_DIR_CANDIDATES",
    "MMPROJ_PREFERENCE",
    "MODELS_DIR_CANDIDATES",
    "NEEDLE_STYLES",
    "PATH_KEYS",
    "PROFILES",
    "RESULTS",
    "RUN_ID",
    "base_config",
    "bench_flags",
    "check_config",
    "load_config",
    "merge_config",
    "need_tools",
    "override_lines",
    "pick_models",
    "profile_summary",
    "read_toml",
    "rel_model_path",
    "required_hits",
    "resolve_model_path",
    "resolve_paths",
    "sdesc",
    "server_flags",
]

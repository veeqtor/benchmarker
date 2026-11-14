"""benchmarker — benchmark and tuning tool for local GGUF models (llama.cpp + llama-swap).

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
  python -m benchmarker doctor
  python -m benchmarker init
  python -m benchmarker sweep all --stop-swap            # fit + speed + tune + image check
  python -m benchmarker sweep <model-name> --accuracy --stop-swap
  python -m benchmarker swap --install

Paths are found automatically (llama.cpp in /opt/llama.cpp or ~/llama.cpp, models in
/srv/models or ~/models). Change them in models.toml [defaults], with environment
variables (BENCHMARKER_LLAMA_DIR, BENCHMARKER_MODELS_DIR, BENCHMARKER_BACKEND) or with options.

Decision rules (target context, speed limits, recall test) are in PROFILES: strict (the rules
of 2.2), balanced (default) and long. Choose with --profile, BENCHMARKER_PROFILE or
profile = "..." in models.toml. 'report' and 'swap' judge saved results with the chosen profile.
The rules are tested by `python3 -m pytest tests/`.

Only the Python standard library is used (Python 3.11 or later).
"""

from benchmarker.cli import main
from benchmarker.config import (
    ACTIVE_SERVERS,
    CACHE_QUALITY,
    CONFIG,
    DEFAULTS,
    DEVICE_ENV,
    HERE,
    LLAMA_DIR_CANDIDATES,
    MMPROJ_PREFERENCE,
    MODELS_DIR_CANDIDATES,
    NEEDLE_STYLES,
    PATH_KEYS,
    PROFILES,
    RESULTS,
    RUN_ID,
    VERSION,
    base_config,
    check_config,
    load_config,
    merge_config,
    need_tools,
    override_lines,
    profile_summary,
    read_toml,
    rel_model_path,
    resolve_model_path,
    resolve_paths,
    sdesc,
)
from benchmarker.decisions import (
    acc_key,
    acc_verdict,
    accuracy_state,
    choose_cache,
    final_setting,
    recommend,
    speed_gate,
    why_below_target,
)
from benchmarker.progress import PROGRESS, log, tracked, warn
from benchmarker.server import stop_all

__all__ = [
    "ACTIVE_SERVERS",
    "CACHE_QUALITY",
    "CONFIG",
    "DEFAULTS",
    "DEVICE_ENV",
    "HERE",
    "LLAMA_DIR_CANDIDATES",
    "MMPROJ_PREFERENCE",
    "MODELS_DIR_CANDIDATES",
    "NEEDLE_STYLES",
    "PATH_KEYS",
    "PROFILES",
    "PROGRESS",
    "RESULTS",
    "RUN_ID",
    "VERSION",
    "acc_key",
    "acc_verdict",
    "accuracy_state",
    "base_config",
    "check_config",
    "choose_cache",
    "final_setting",
    "load_config",
    "log",
    "main",
    "merge_config",
    "need_tools",
    "override_lines",
    "profile_summary",
    "read_toml",
    "recommend",
    "rel_model_path",
    "resolve_model_path",
    "resolve_paths",
    "sdesc",
    "speed_gate",
    "stop_all",
    "tracked",
    "warn",
    "why_below_target",
]

# llmbench Refactoring — Implementation Plan (Updated 2026-11-14)

## Overview

`llmbench.py` is a **3,147-line** monolithic script that mixes CLI parsing, configuration management, GPU detection, benchmark phases, HTTP communication, state persistence, report generation, and test image creation. This plan splits it into focused modules while preserving the **standard-library-only** constraint (Python 3.11+).

The project uses `uv` as its package manager with a `src/` layout (`src/benchmarker/`). Tests have been converted from `unittest` to `pytest` (45 tests, all passing).

---

## Current State

```
llmbench.py                          # 3,147 lines — monolithic script with main()
pyproject.toml                       # uv project: name=benchmarker, requires-python>=3.14
                                     #   benchmarker = "benchmarker:main" (broken — no main in __init__)
src/benchmarker/__init__.py          # Empty stub
test__llmbench.py                    # 45 pytest tests (all passing)
tests/                               # (target location for test files)
results/progress.json                # Runtime progress state
docs/implementation-plan.md          # This file
```

**Key changes since original plan:**

- Tests converted from `unittest` to `pytest` (45 tests, 0 failures)
- `src/benchmarker/` package directory created (empty stub)
- Real line count: **3,147 lines** (not ~2,000)
- Real function count: **72 top-level functions**, **4 classes**, **10+ constants**
- Entry point: `llmbench.py` is a standalone script; after refactor it becomes a thin `src/benchmarker/__main__.py` + `pyproject.toml` console script

---

## Behavioural Equivalence (Refactor Constraint)

**This is a refactor, not a rewrite.** Every observable behaviour must remain identical:

| Aspect         | Requirement                                                                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI interface  | All commands (`doctor`, `init`, `info`, `sweep`, `fill`, `vision`, `status`, `report`, `swap`) accept the same arguments and produce the same output |
| Decision rules | `speed_gate`, `accuracy_state`, `recommend`, `choose_cache`, `final_setting` produce identical results for identical inputs                          |
| Profiles       | `strict`, `balanced`, `long` produce the same verdicts as before                                                                                     |
| File I/O       | `results/` directory structure, CSV columns, `progress.json` format — unchanged                                                                      |
| Paths          | Auto-detection of `llama_dir`, `models_dir`, `server`, `bench` — unchanged                                                                           |
| Exit codes     | `sys.exit` on validation failure — unchanged                                                                                                         |
| Globals        | `CONFIG`, `RESULTS`, `RUN_ID`, `ACTIVE_SERVERS` — same values, same lifecycle                                                                        |
| Dependencies   | Only Python standard library — no new third-party packages                                                                                           |

**Verification gate:** After every phase, the full test suite (`python -m pytest tests/ -v`) must pass with zero regressions, and `./llmbench.py doctor` must produce the same output as before the refactor.

---

## Target Structure

```
llmbench.py                          # DEPRECATED — keep as symlink or remove after migration
pyproject.toml                       # uv project config (see full spec below)
src/
  benchmarker/
    __init__.py                      # Package init: VERSION, HERE, CONFIG, RESULTS, main
    __main__.py                      # Entry point: calls benchmarker.main()
    cli.py                           # argparse setup + dispatch (moved from llmbench.py)
    config.py                        # DEFAULTS, PROFILES, merge/check/base config (~220 lines)
    gguf.py                          # gguf_meta, model_files, is_mmproj, model_info (~130 lines)
    test_image.py                    # test_image, image_message, _FONT, _COLORS (~110 lines)
    client.py                        # Client, chat, http_error (~90 lines)
    lock.py                          # Lock context manager (~35 lines)
    progress.py                      # Progress class, tracked, log, warn (~210 lines)
    io.py                            # append_csv, read_csv, state_path, save_state, load_states (~110 lines)
    utils.py                         # xp, tilde, now, run_text, free_port, etc. (~110 lines)
    gpu.py                           # setup_devices, vram_mb, ram_free_mb, pick_main_gpu (~160 lines)
    models.py                        # model_info, find_mmproj, pick_models, load_config (~220 lines)
    server.py                        # Server class, stop_all, wait_vram_released (~160 lines)
    bench.py                         # run_bench, speed_phase, tune_phase, Fitter (~270 lines)
    accuracy.py                      # fill_series, vision_series, needle logic, haystack (~370 lines)
    decisions.py                     # speed_gate, accuracy_state, recommend, choose_cache, final_setting (~260 lines)
    report.py                        # report_header, report_recommendations, report_model, table (~320 lines)
    swap.py                          # swap_yaml, swap_models_in, cmd_swap (~210 lines)
    commands.py                      # cmd_doctor, cmd_init, cmd_info, cmd_sweep, etc. (~520 lines)
tests/
  conftest.py                        # Shared fixtures (progress_setup, tmp_results, etc.)
  test_config.py                     # merge_config, check_config, profile_summary, override_lines
  test_gguf.py                       # gguf_meta, model_info, model_files, find_mmproj
  test_test_image.py                 # test_image, image_message
  test_client.py                     # Client, chat, http_error
  test_lock.py                       # Lock context manager
  test_utils.py                      # xp, tilde, now, run_text, free_port, api_key
  test_io.py                         # append_csv, read_csv, save_state, load_states
  test_gpu.py                        # parse_devices, vram_mb, ram_free_mb, physical_cores
  test_progress.py                   # Progress class, tracked, log, warn
  test_bench.py                      # Fitter, run_bench, speed_phase, tune_phase
  test_accuracy.py                   # fill_series, vision_series, needle logic
  test_decisions.py                  # speed_gate, accuracy_state, recommend, choose_cache
  test_commands.py                   # cmd_doctor dry-run, cmd_info output
  test_report.py                     # report_header, report_model, table
  test_swap.py                       # swap_yaml, swap_models_in
  test_llmbench.py                   # 45 pytest tests (all passing, migrated from root)
results/progress.json                # Runtime progress state
docs/implementation-plan.md          # This file
```

**Total: ~3,220 lines** (vs. ~3,147 lines in one file). The ~873 line increase accounts for docstrings, module-level `__all__` exports, clearer function boundaries, and package boilerplate.

---

## pyproject.toml — Full Spec

```toml
[project]
name = "benchmarker"
version = "2.3.0"
description = "Benchmark and tuning tool for local GGUF models on GPU PCs"
readme = "README.md"
license = { text = "MIT" }
authors = [
    { name = "Victor Nwokeocha", email = "nwokeochavictor@gmail.com" }
]
requires-python = ">=3.11"
dependencies = []          # stdlib-only — no runtime dependencies

[project.scripts]
llmbench = "benchmarker:main"

[project.urls]
Homepage = "https://github.com/<user>/llmbench"
Repository = "https://github.com/<user>/llmbench"
Issues = "https://github.com/<user>/llmbench/issues"

[build-system]
requires = ["uv_build>=0.13.8,<0.14.0"]
build-backend = "uv_build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
]
```

**Key points:**

- `requires-python = ">=3.11"` — matches the code's claim (not 3.14)
- `dependencies = []` — stdlib-only, no runtime deps
- `project.scripts.llmbench` — installs `llmbench` CLI command pointing to `benchmarker:main`
- `tool.pytest.ini_options` — tells pytest to look in `tests/` directory
- `tool.uv.dev-dependencies` — pytest only for development, not installed in production

---

## Entry Point Design

After the refactor, the entry point works in three ways:

```
# 1. Installed CLI (via `uv pip install .` or `uv build` + pip)
llmbench doctor
llmbench sweep all

# 2. Direct package execution (python -m benchmarker)
python -m benchmarker doctor
python -m benchmarker sweep all

# 3. Legacy script (deprecated, kept for backward compat)
./llmbench.py doctor
```

### `src/benchmarker/__init__.py` — Package init + main entry point

```python
"""benchmarker — benchmark and tuning tool for local GGUF models."""

import os
import datetime

# ── Constants (moved from llmbench.py top-level) ──────────────────────────
VERSION = "2.3.0"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CONFIG = os.path.join(HERE, "..", "..", "models.toml")   # relative to package
RESULTS = os.path.join(HERE, "..", "..", "results")

# ── Public API ─────────────────────────────────────────────────────────────
from benchmarker.cli import main        # argparse + dispatch
from benchmarker.config import (
    DEFAULTS, PROFILES, merge_config, check_config,
    profile_summary, override_lines, base_config, load_config,
    pick_models, need_tools, resolve_paths, sdesc,
)
from benchmarker.decisions import (
    speed_gate, accuracy_state, recommend, choose_cache,
    final_setting, why_below_target, acc_key, acc_verdict,
)
from benchmarker.progress import PROGRESS, log, warn, tracked
from benchmarker.server import stop_all, ACTIVE_SERVERS

__all__ = [
    "main", "VERSION", "HERE", "RUN_ID", "CONFIG", "RESULTS",
    "DEFAULTS", "PROFILES", "merge_config", "check_config",
    "profile_summary", "override_lines", "base_config",
    "load_config", "pick_models", "need_tools", "resolve_paths", "sdesc",
    "speed_gate", "accuracy_state", "recommend", "choose_cache",
    "final_setting", "why_below_target", "acc_key", "acc_verdict",
    "PROGRESS", "log", "warn", "tracked", "stop_all", "ACTIVE_SERVERS",
]
```

### `src/benchmarker/__main__.py` — `python -m benchmarker` support

```python
"""Allow running as: python -m benchmarker [args]"""

from benchmarker import main

if __name__ == "__main__":
    main()
```

### `src/benchmarker/cli.py` — argparse + dispatch (moved from llmbench.py)

```python
"""CLI entry point: argparse setup + command dispatch."""

import argparse
import atexit
import signal
import sys

from benchmarker.config import CONFIG, RESULTS, xp
from benchmarker.commands import (
    cmd_doctor, cmd_init, cmd_info, cmd_sweep,
    cmd_fill, cmd_vision, cmd_status, cmd_report, cmd_swap,
)
from benchmarker.server import stop_all


def main():
    """Parse arguments and dispatch to the selected command."""
    # ... argparse setup (moved from llmbench.py) ...
    dispatch = {
        "doctor": cmd_doctor,
        "init": cmd_init,
        "info": cmd_info,
        "sweep": cmd_sweep,
        "fill": cmd_fill,
        "vision": cmd_vision,
        "status": cmd_status,
        "report": cmd_report,
        "swap": cmd_swap,
    }
    args = parser.parse_args()
    dispatch[args.cmd](args)
```

### `llmbench.py` — Deprecated wrapper (backward compat)

```python
#!/usr/bin/env python3
"""DEPRECATED: Use `llmbench` CLI command or `python -m benchmarker` instead."""

from benchmarker import main

if __name__ == "__main__":
    main()
```

---

## Module Dependencies (Actual)

```
                  ┌─────────────────────────────────┐
                  │         llmbench.py             │
                  │  (argparse + dispatch)          │
                  └──────────────┬──────────────────┘
                                 │ imports
        ┌────────┬────────┬──────┼───────┬────────┬────────┬────────┐
        ▼        ▼        ▼      ▼       ▼        ▼        ▼        ▼
     config.py  gguf.py  utils.py  progress.py  io.py  client.py  lock.py  test_image.py
                    │        │           │          │
                    │        │           │          ├──────────► gpu.py
                    │        │           │          │
                    ▼        ▼           ▼          ▼
                 bench.py       accuracy.py    server.py
                    │              │               │
                    └──────┬───────┘───────────────┘
                           │
                    ┌──────┼──────────┐
                    ▼      ▼          ▼
               decisions.py  report.py  swap.py
                    │
                    ▼
               commands.py  (imports everything above)
```

**Dependency rules:**

| Module          | Imports                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `config.py`     | _(none — leaf)_                                                               |
| `gguf.py`       | _(none — leaf)_                                                               |
| `test_image.py` | _(none — leaf)_                                                               |
| `client.py`     | _(none — leaf)_                                                               |
| `lock.py`       | _(none — leaf)_                                                               |
| `utils.py`      | _(none — leaf)_                                                               |
| `progress.py`   | stdlib only                                                                   |
| `io.py`         | stdlib only                                                                   |
| `gpu.py`        | `io.py`, `utils.py`                                                           |
| `models.py`     | `gguf.py`, `io.py`, `config.py`, `utils.py`                                   |
| `decisions.py`  | `config.py` only (pure logic)                                                 |
| `bench.py`      | `config.py`, `progress.py`, `io.py`, `utils.py`                               |
| `accuracy.py`   | `config.py`, `progress.py`, `io.py`, `client.py`, `test_image.py`, `utils.py` |
| `server.py`     | `config.py`, `progress.py`, `io.py`, `utils.py`                               |
| `report.py`     | `decisions.py`, `config.py`, `io.py`, `gpu.py`, `utils.py`                    |
| `swap.py`       | `decisions.py`, `config.py`, `io.py`, `utils.py`                              |
| `commands.py`   | all above except `report.py` and `swap.py`                                    |

---

## Detailed Module Specifications (Actual Line Counts)

### 1. `src/benchmarker/__init__.py` — Package Init + Main Entry Point (~80 lines)

**What moves from llmbench.py:**

- `VERSION`, `HERE`, `RUN_ID`, `CONFIG`, `RESULTS` constants
- `DEFAULTS`, `PROFILES`, `NEEDLE_STYLES`, `CACHE_QUALITY`, `PATH_KEYS`, `LLAMA_DIR_CANDIDATES`, `MODELS_DIR_CANDIDATES`, `MMPROJ_PREFERENCE`, `DEVICE_ENV`, `ACTIVE_SERVERS`
- Re-export of key symbols from submodules for backward-compatible imports

**Target structure:**

```python
"""benchmarker — benchmark and tuning tool for local GGUF models."""

import os, datetime

VERSION = "2.3.0"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CONFIG = os.path.join(HERE, "..", "..", "models.toml")
RESULTS = os.path.join(HERE, "..", "..", "results")

from benchmarker.cli import main
from benchmarker.config import DEFAULTS, PROFILES, ...
from benchmarker.decisions import speed_gate, recommend, ...
from benchmarker.progress import PROGRESS, log, warn, tracked
from benchmarker.server import stop_all, ACTIVE_SERVERS

__all__ = ["main", "VERSION", "HERE", ...]
```

---

### 2. `src/benchmarker/cli.py` — CLI Entry Point (~300 lines)

**What moves from llmbench.py:**

- All `argparse` setup: `common` parser, subparsers, all `add_argument` calls
- `main()` function: arg parsing, CONFIG/RESULTS overrides, signal handling, dispatch dict
- Signal handlers and atexit registration

**Target structure:**

```python
"""CLI entry point: argparse setup + command dispatch."""

import argparse, atexit, signal, sys

from benchmarker.config import CONFIG, RESULTS, xp
from benchmarker.commands import (
    cmd_doctor, cmd_init, cmd_info, cmd_sweep,
    cmd_fill, cmd_vision, cmd_status, cmd_report, cmd_swap,
)
from benchmarker.server import stop_all

def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--results", default=argparse.SUPPRESS)
    # ... all other common args ...

    ap = argparse.ArgumentParser(...)
    sub = ap.add_subparsers(dest="cmd", required=True)
    # ... all subparsers ...

    args = ap.parse_args()
    CONFIG = xp(getattr(args, "config", None) or ...)
    RESULTS = xp(getattr(args, "results", None) or ...)
    atexit.register(stop_all)
    signal.signal(signal.SIGTERM, ...)

    dispatch = {
        "doctor": cmd_doctor, "init": cmd_init, "info": cmd_info,
        "sweep": cmd_sweep, "fill": cmd_fill, "vision": cmd_vision,
        "status": cmd_status, "report": cmd_report, "swap": cmd_swap,
    }
    dispatch[args.cmd](args)
```

---

### 3. `src/benchmarker/__main__.py` — `python -m` Support (~5 lines)

```python
"""Allow running as: python -m benchmarker [args]"""

from benchmarker import main

if __name__ == "__main__":
    main()
```

---

### 4. `llmbench.py` — Deprecated Wrapper (~10 lines)

```python
#!/usr/bin/env python3
"""DEPRECATED: Use `llmbench` CLI or `python -m benchmarker` instead."""

from benchmarker import main

if __name__ == "__main__":
    main()
```

---

### 5. `config.py` — Configuration (220 lines)

**Actual functions from llmbench.py:**

| Function                                    | Lines   | Purpose                                      |
| ------------------------------------------- | ------- | -------------------------------------------- |
| `read_toml()`                               | 569–573 | Load `models.toml`                           |
| `merge_config(user_defaults, profile_name)` | 575–590 | DEFAULTS < profile < toml merge              |
| `check_config(cfg)`                         | 592–604 | Validate config values                       |
| `profile_summary(cfg)`                      | 606–610 | Human-readable profile description           |
| `override_lines(cfg)`                       | 612–618 | Warn about toml overrides                    |
| `base_config(args, raw)`                    | 620–645 | Build config from toml + env + args          |
| `load_config(args, need_models)`            | 647–701 | Full config + model loading                  |
| `pick_models(models, names)`                | 703–710 | Select models by name or "all"               |
| `need_tools(cfg)`                           | 712–718 | Verify server/bench exist                    |
| `resolve_paths(cfg)`                        | 534–556 | Resolve llama_dir, server, bench, models_dir |
| `resolve_model_path(cfg, p)`                | 558–560 | Resolve relative model path                  |
| `rel_model_path(cfg, p)`                    | 562–565 | Convert to relative path                     |
| `sdesc(s)`                                  | 890–898 | Human-readable setting string                |

**Constants:** `DEFAULTS`, `PROFILES`, `NEEDLE_STYLES`, `CACHE_QUALITY`, `PATH_KEYS`, `LLAMA_DIR_CANDIDATES`, `MODELS_DIR_CANDIDATES`, `MMPROJ_PREFERENCE`, `DEVICE_ENV`, `CONFIG`, `RESULTS`, `VERSION`, `HERE`, `RUN_ID`, `ACTIVE_SERVERS`

---

### 3. `gguf.py` — GGUF Metadata Reader (130 lines)

**Actual functions:**

| Function                  | Lines   | Purpose                                              |
| ------------------------- | ------- | ---------------------------------------------------- |
| `_FMT`                    | 423–436 | GGUF type → struct format map                        |
| `gguf_meta(path)`         | 439–468 | Parse GGUF key/value metadata                        |
| `model_files(path)`       | 471–474 | Expand sharded GGUF filenames                        |
| `is_mmproj(path)`         | 476–477 | Check if path is an mmproj file                      |
| `is_extra_part(path)`     | 479–482 | Check if path is a shard                             |
| `model_info(path)`        | 484–506 | Extract architecture, layers, experts, context, size |
| `find_mmproj(model_path)` | 508–524 | Find best mmproj in model's directory                |

---

### 4. `test_image.py` — Test Image Generator (110 lines)

**Actual functions:**

| Function                   | Lines   | Purpose                            |
| -------------------------- | ------- | ---------------------------------- |
| `_FONT`                    | 902–923 | 5×7 pixel digit bitmaps            |
| `_COLORS`                  | 924     | Frame color RGB values             |
| `test_image(px, seed)`     | 926–962 | Generate PNG with a 4-digit number |
| `image_message(png, text)` | 964–974 | Build OpenAI-style image message   |
| `VISION_Q`                 | 976–978 | Vision test prompt                 |

---

### 5. `client.py` — HTTP Client (90 lines)

**Actual functions:**

| Function                             | Lines     | Purpose                                      |
| ------------------------------------ | --------- | -------------------------------------------- |
| `Client` class                       | 632–648   | HTTP client for llama-swap/llama-server APIs |
| `http_error(e)`                      | 995–1002  | Format HTTP error message                    |
| `chat(client, model, messages, ...)` | 1004–1018 | Send chat completion request                 |

---

### 6. `lock.py` — File Lock (35 lines)

**Actual class:**

| Class  | Lines     | Purpose                                      |
| ------ | --------- | -------------------------------------------- |
| `Lock` | 1230–1240 | File-based lock to prevent concurrent sweeps |

---

### 7. `progress.py` — Progress Tracking (210 lines)

**Actual class/functions:**

| Symbol                 | Lines   | Purpose                                     |
| ---------------------- | ------- | ------------------------------------------- |
| `fmt_dur(sec)`         | 190–194 | Format seconds → `1h23m` or `23:45`         |
| `Progress` class       | 193–325 | Tracks run progress, writes `progress.json` |
| `tracked(kind, names)` | 328–337 | Context manager wrapping a tracked run      |
| `log(msg)`             | 339–342 | Print with progress tag                     |
| `warn(msg)`            | 344–345 | Print warning to stderr                     |

---

### 8. `io.py` — I/O Utilities (110 lines)

**Actual functions:**

| Function                     | Lines   | Purpose                         |
| ---------------------------- | ------- | ------------------------------- |
| `append_csv(name, row)`      | 357–370 | Append row to CSV in `results/` |
| `read_csv(name)`             | 372–376 | Read CSV from `results/`        |
| `state_path(model, backend)` | 378–379 | Path to state JSON              |
| `save_state(state)`          | 381–386 | Atomic state save               |
| `load_states(backend)`       | 388–399 | Load all state files            |

---

### 9. `utils.py` — General Utilities (110 lines)

**Actual functions:**

| Function                      | Lines   | Purpose                             |
| ----------------------------- | ------- | ----------------------------------- |
| `xp(p)`                       | 347–348 | Expand user vars in paths           |
| `tilde(p)`                    | 350–352 | Shorten path with `~`               |
| `now()`                       | 354–355 | ISO timestamp                       |
| `run_text(cmd, env, timeout)` | 401–407 | Run command, capture stdout+stderr  |
| `free_port()`                 | 409–412 | Find a free TCP port                |
| `port_free(port)`             | 414–419 | Check if a port is available        |
| `build_id(cfg)`               | 822–824 | Parse llama.cpp build version       |
| `service_active(name)`        | 826–831 | Check if systemd service is active  |
| `other_servers()`             | 833–835 | Find running llama-server processes |
| `api_key(cfg, required)`      | 838–862 | Read API key from env or file       |

---

### 10. `gpu.py` — GPU & System Detection (160 lines)

**Actual functions:**

| Function                   | Lines   | Purpose                               |
| -------------------------- | ------- | ------------------------------------- |
| `parse_devices(text)`      | 722–733 | Parse `--list-devices` output         |
| `IGPU_NAME`                | 735–736 | Regex for integrated GPU names        |
| `pick_main_gpu(cfg, gpus)` | 738–750 | Select dedicated GPU by VRAM          |
| `setup_devices(cfg)`       | 752–780 | Set device env vars, pick GPU         |
| `gpu_dir(cfg)`             | 782–794 | Find AMD GPU sysfs dir                |
| `vram_mb(cfg)`             | 796–800 | Read VRAM total/used/free from sysfs  |
| `ram_free_mb()`            | 802–807 | Read available RAM from /proc/meminfo |
| `physical_cores()`         | 809–820 | Count physical CPU cores              |

---

### 11. `models.py` — Model Management (220 lines)

**Actual functions:**

| Function                         | Lines   | Purpose                                              |
| -------------------------------- | ------- | ---------------------------------------------------- |
| `load_config(args, need_models)` | 647–701 | Full config + model loading                          |
| `pick_models(models, names)`     | 703–710 | Select models by name or "all"                       |
| `need_tools(cfg)`                | 712–718 | Verify server/bench exist                            |
| `find_mmproj(model_path)`        | 508–524 | Find best mmproj in model's directory                |
| `model_info(path)`               | 484–506 | Extract architecture, layers, experts, context, size |
| `model_files(path)`              | 471–474 | Expand sharded GGUF filenames                        |
| `resolve_paths(cfg)`             | 534–556 | Resolve paths                                        |
| `resolve_model_path(cfg, p)`     | 558–560 | Resolve relative model path                          |
| `rel_model_path(cfg, p)`         | 562–565 | Convert to relative path                             |

---

### 12. `server.py` — Server Lifecycle (160 lines)

**Actual class/functions:**

| Symbol                                       | Lines     | Purpose                                |
| -------------------------------------------- | --------- | -------------------------------------- |
| `Server` class                               | 651–710   | Spawns, monitors, stops `llama-server` |
| `stop_all()`                                 | 1070–1073 | Stop all tracked servers               |
| `wait_vram_released(cfg, baseline, timeout)` | 1075–1084 | Wait for VRAM to return to baseline    |
| `server_flags(m, s)`                         | 866–877   | Build llama-server flags from setting  |
| `bench_flags(cfg, s)`                        | 879–888   | Build llama-bench flags                |

---

### 13. `bench.py` — Benchmark Phases (270 lines)

**Actual class/functions:**

| Symbol                               | Lines     | Purpose                                       |
| ------------------------------------ | --------- | --------------------------------------------- |
| `Fitter` class                       | 726–810   | Phase 1: binary-searches optimal GPU settings |
| `sweep_moe(cfg, m, fitter)`          | 1115–1144 | Binary-search expert layers for MoE           |
| `sweep_dense(cfg, m, fitter)`        | 1146–1181 | Binary-search ngl + context for dense         |
| `run_bench(cfg, m, s, depths, ...)`  | 1185–1224 | Run llama-bench, save raw JSONL               |
| `speed_depths(cfg, ctx)`             | 1226–1235 | Compute which depths to bench                 |
| `bench_cached(done, cfg, m, s, ...)` | 1237–1243 | Deduplicate bench runs                        |
| `speed_phase(cfg, m, settings)`      | 1245–1278 | Run speed tests for all settings              |
| `tune_phase(cfg, m, st, fitter)`     | 1282–1328 | Tune ubatch + threads                         |

---

### 14. `accuracy.py` — Accuracy & Vision Tests (370 lines)

**Actual functions:**

| Function                                     | Lines     | Purpose                             |
| -------------------------------------------- | --------- | ----------------------------------- |
| `_SYLLABLES` / `LETTERS`                     | 1332–1333 | Needle-in-haystack building blocks  |
| `pseudo_word(rnd)`                           | 1335–1336 | Generate random pseudo-word         |
| `haystack(style, n, seed)`                   | 1338–1357 | Generate needle-in-haystack text    |
| `needle_positions(n_lines, count)`           | 1359–1363 | Spread needle locations             |
| `needle_question(style, keys)`               | 1365–1370 | Build the recall question           |
| `score_answer(answer, expected)`             | 1372–1383 | Count correct recalls               |
| `required_hits(cfg, count)`                  | 1385–1386 | Min correct answers needed          |
| `fill_one(cfg, client, model, pct, ...)`     | 1388–1446 | Single long-prompt accuracy test    |
| `fill_series(cfg, client, model, pcts, ...)` | 1454–1494 | Series of fill tests                |
| `vision_series(cfg, client, model, ...)`     | 1496–1549 | Image recognition test series       |
| `accuracy_search(cfg, m, settings, rec)`     | 1551–1582 | Search for best context that passes |
| `served_phase(cfg, m, st, tag, ...)`         | 1584–1606 | Start server, run accuracy + vision |

---

### 15. `decisions.py` — Decision Rules (260 lines)

**Actual functions:**

| Function                                   | Lines     | Purpose                                      |
| ------------------------------------------ | --------- | -------------------------------------------- |
| `speed_at_depth(st)`                       | 1610–1615 | Deepest tg128 measurement                    |
| `deep_tg(st)`                              | 1617–1618 | Shorthand for deep generation speed          |
| `speed_gate(cfg, st, best_tg)`             | 1620–1638 | Check if speed is within tolerance           |
| `acc_key(a)`                               | 1640–1650 | Normalize (cache, context) from accuracy row |
| `acc_verdict(cfg, a)`                      | 1652–1660 | Pass/fail/none for one accuracy row          |
| `accuracy_state(cfg, acc, cache, context)` | 1662–1678 | Aggregate accuracy state                     |
| `gate_rows(cfg, sts, acc)`                 | 1680–1714 | Build decision rows for all contexts         |
| `cache_rank(cache)`                        | 1716–1717 | Rank cache quality                           |
| `choose_cache(cfg, picks)`                 | 1719–1734 | Pick best cache type                         |
| `recommend(cfg, settings, acc)`            | 1736–1770 | Full recommendation                          |
| `why_below_target(rec)`                    | 1772–1785 | Explain why target is not met                |
| `Final`                                    | 1787      | Named tuple for final result                 |
| `final_setting(cfg, s)`                    | 1789–1808 | Compute recommendation from saved state      |

---

### 16. `report.py` — Markdown Report Generation (320 lines)

**Actual functions:**

| Function                              | Lines     | Purpose                                |
| ------------------------------------- | --------- | -------------------------------------- |
| `table(rows, cols)`                   | 2185–2189 | Markdown table helper                  |
| `system_info(cfg)`                    | 2191–2223 | System hardware info for reports       |
| `RULE_KEYS` / `SETTING_KEYS`          | 2225–2241 | Report config keys                     |
| `report_header(cfg, models)`          | 2243–2294 | Report header with system + model info |
| `report_recommendations(cfg, states)` | 2296–2336 | Recommendations table                  |
| `report_decisions(cfg, s)`            | 2338–2368 | Decision table per context             |
| `report_model(cfg, key, s)`           | 2370–2448 | Full report section for one model      |
| `report_swap_runs()`                  | 2450–2486 | Fill + vision results via llama-swap   |
| `cmd_report(args)`                    | 2488–2520 | Write full Markdown report             |

---

### 17. `swap.py` — llama-swap Config Generation (210 lines)

**Actual functions:**

| Function                 | Lines     | Purpose                                      |
| ------------------------ | --------- | -------------------------------------------- |
| `swap_yaml(cfg, states)` | 2522–2588 | Generate llama-swap YAML config              |
| `swap_models_in(path)`   | 2590–2596 | Parse models from existing swap config       |
| `cmd_swap(args)`         | 2598–2629 | Write + optionally install llama-swap config |

---

### 18. `commands.py` — CLI Command Handlers (520 lines)

**Actual functions:**

| Function                    | Lines     | Purpose                                            |
| --------------------------- | --------- | -------------------------------------------------- |
| `cmd_doctor(args)`          | 1812–1891 | Health check: paths, GPU, llama-swap, API key      |
| `cmd_init(args)`            | 1893–1966 | Scan models folder, create/update `models.toml`    |
| `cmd_info(args)`            | 1968–1988 | Print model table from GGUF metadata               |
| `cmd_sweep(args)`           | 1990–2037 | Full sweep: fit → speed → accuracy → tune → vision |
| `sweep_model(cfg, m, args)` | 2039–2107 | Sweep one model                                    |
| `swap_client(cfg, args)`    | 2109–2112 | llama-swap HTTP client                             |
| `cmd_fill(args)`            | 2114–2126 | Long-prompt accuracy test via llama-swap           |
| `cmd_vision(args)`          | 2128–2142 | Image test via llama-swap                          |
| `status_text(data, now_ts)` | 2144–2167 | Format progress.json as text                       |
| `cmd_status(args)`          | 2169–2183 | Show/monitor run progress                          |

---

## Implementation Phases

### Phase 1: Leaf Modules (no internal dependencies)

**Files:** `config.py`, `gguf.py`, `test_image.py`, `client.py`, `lock.py`, `utils.py`

**Steps:**

1. Create each file with content extracted from `llmbench.py`
2. No imports between new files needed
3. Add module docstrings and `__all__` exports
4. Verify each module parses cleanly (`python3 -m py_compile module.py`)

**Verification:** `python -c "from benchmarker import config, gguf, test_image, client, lock, utils"`

### Phase 1a: Package entry point

**Files:** `src/benchmarker/__init__.py`, `src/benchmarker/__main__.py`, `src/benchmarker/cli.py`, `pyproject.toml`

**Steps:**

1. Populate `src/benchmarker/__init__.py` with constants, re-exports, and `main` import
2. Create `src/benchmarker/__main__.py` for `python -m benchmarker` support
3. Create `src/benchmarker/cli.py` with argparse + dispatch (moved from llmbench.py)
4. Update `pyproject.toml` with full spec (see pyproject.toml section above)
5. Update `llmbench.py` to a thin wrapper
6. Verify: `python -m benchmarker --help` shows the same help as `./llmbench.py --help`

### Phase 1b: Test scaffolding

**Files:** `tests/conftest.py`, `tests/test_config.py`, `tests/test_gguf.py`, `tests/test_test_image.py`, `tests/test_client.py`, `tests/test_lock.py`, `tests/test_utils.py`

**Steps:**

1. Create `tests/` directory
2. Create `tests/conftest.py` with shared fixtures (migrate `progress_setup` from old test file)
3. Split existing `test__llmbench.py` tests into module-specific test files
4. Verify: `python -m pytest tests/ -v`

---

### Phase 2: Low-Level Modules

**Files:** `progress.py`, `io.py`, `gpu.py`

**Steps:**

1. Create `progress.py` — depends only on stdlib
2. Create `io.py` — depends only on stdlib
3. Create `gpu.py` — depends on `io.py`, `utils.py`
4. Verify imports resolve

**Verification:** `python -c "from benchmarker.progress import PROGRESS, log, warn, tracked; from benchmarker.io import read_csv, save_state; from benchmarker.gpu import vram_mb"`

---

### Phase 3: Domain Modules

**Files:** `models.py`, `server.py`, `bench.py`, `accuracy.py`, `decisions.py`

**Steps:**

1. Create `models.py` — depends on `gguf.py`, `io.py`, `config.py`, `utils.py`
2. Create `server.py` — depends on `config.py`, `progress.py`, `io.py`, `utils.py`
3. Create `bench.py` — depends on `config.py`, `progress.py`, `io.py`, `utils.py`
4. Create `accuracy.py` — depends on `config.py`, `progress.py`, `client.py`, `test_image.py`, `io.py`, `utils.py`
5. Create `decisions.py` — depends on `config.py` only

**Verification:** Run existing tests: `python -m pytest tests/ -v`

---

### Phase 4: Report & Swap

**Files:** `report.py`, `swap.py`

**Steps:**

1. Create `report.py` — depends on `decisions.py`, `config.py`, `io.py`, `gpu.py`, `utils.py`
2. Create `swap.py` — depends on `decisions.py`, `config.py`, `io.py`, `utils.py`

**Verification:** Run `./llmbench.py doctor` (dry run, no side effects)

---

### Phase 5: Commands & Entry Point

**Files:** `commands.py`, `llmbench.py`

**Steps:**

1. Create `commands.py` — imports from all above modules
2. Update `src/benchmarker/cli.py` — finalize argparse + dispatch
3. Update `llmbench.py` — thin wrapper calling `benchmarker.main()`
4. Update `src/benchmarker/__init__.py` — finalize re-exports

**Verification:**

```bash
llmbench doctor               # via installed CLI
python -m benchmarker doctor  # via python -m
./llmbench.py doctor          # via deprecated wrapper
python -m pytest tests/ -v
```

---

### Phase 6: Cleanup

**Steps:**

1. Remove all code from `llmbench.py` except entry point
2. Run full test suite: `python -m pytest tests/ -v`
3. Run `./llmbench.py doctor` end-to-end
4. Update `src/benchmarker/__init__.py` with clean `__all__` exports
5. Update README if module structure needs documentation

---

## Testing Strategy

### Existing tests (45 pytest tests, all passing)

```bash
python -m pytest tests/ -v
```

These test pure functions in `decisions.py`, `config.py`, `progress.py` — they should continue to work after extraction.

### Test organisation

All test files live under `tests/` following pytest conventions:

| File                       | Tests                                                               |
| -------------------------- | ------------------------------------------------------------------- |
| `tests/conftest.py`        | Shared fixtures: `progress_setup`, `tmp_results`, `mock_cfg`        |
| `tests/test_config.py`     | `merge_config`, `check_config`, `profile_summary`, `override_lines` |
| `tests/test_gguf.py`       | `gguf_meta`, `model_info`, `model_files`, `find_mmproj`             |
| `tests/test_test_image.py` | `test_image`, `image_message`                                       |
| `tests/test_client.py`     | `Client`, `chat`, `http_error`                                      |
| `tests/test_lock.py`       | `Lock` context manager                                              |
| `tests/test_utils.py`      | `xp`, `tilde`, `now`, `run_text`, `free_port`, `api_key`            |
| `tests/test_io.py`         | `append_csv`, `read_csv`, `save_state`, `load_states`               |
| `tests/test_gpu.py`        | `parse_devices`, `vram_mb`, `ram_free_mb`, `physical_cores`         |
| `tests/test_progress.py`   | `Progress` class, `tracked`, `log`, `warn`                          |
| `tests/test_bench.py`      | `Fitter`, `run_bench`, `speed_phase`, `tune_phase`                  |
| `tests/test_accuracy.py`   | `fill_series`, `vision_series`, needle logic, `haystack`            |
| `tests/test_decisions.py`  | `speed_gate`, `accuracy_state`, `recommend`, `choose_cache`         |
| `tests/test_commands.py`   | `cmd_doctor` dry-run, `cmd_info` output                             |
| `tests/test_report.py`     | `report_header`, `report_model`, `table`                            |
| `tests/test_swap.py`       | `swap_yaml`, `swap_models_in`                                       |
| `tests/test_llmbench.py`   | 45 pytest tests (migrated from root, split across above files)      |

---

## Risk Mitigation

### Risk: Circular imports

**Mitigation:** Dependency graph is strictly hierarchical (no cycles). Phase ordering enforces this.

### Risk: `PROGRESS` singleton state

**Mitigation:** `progress.py` exports `PROGRESS` singleton. All modules import from `benchmarker.progress`. No change in behavior.

### Risk: `ACTIVE_SERVERS` global

**Mitigation:** Move `ACTIVE_SERVERS = []` to `server.py`. `stop_all()` operates on it. Import from `benchmarker.server` where needed.

### Risk: `RUN_ID` and `RUN_ID`-dependent paths

**Mitigation:** Keep `RUN_ID` in `config.py` (computed at import time). `io.py` imports it from `benchmarker.config`.

### Risk: `sys.exit` calls scattered across modules

**Mitigation:** Preserve current behavior. `check_config()` in `config.py` already does this. Other `sys.exit` calls stay in command handlers (`commands.py`).

### Risk: Package import paths change

**Mitigation:** All internal imports change from `import foo` to `from benchmarker import foo` or `from benchmarker.foo import bar`. Use a find-and-replace across all new files. Test each phase before moving to the next.

---

## File-by-File Migration Checklist

| #   | File                          | Lines | Dependencies                                                                  | Phase |
| --- | ----------------------------- | ----- | ----------------------------------------------------------------------------- | ----- |
| 1   | `src/benchmarker/__init__.py` | ~80   | all submodules (re-exports)                                                   | 1a    |
| 2   | `src/benchmarker/__main__.py` | ~5    | `benchmarker.main`                                                            | 1a    |
| 3   | `src/benchmarker/cli.py`      | ~300  | `commands.*`, `config.*`, `server.*`                                          | 1a    |
| 4   | `config.py`                   | ~220  | none                                                                          | 1     |
| 5   | `gguf.py`                     | ~130  | none                                                                          | 1     |
| 6   | `test_image.py`               | ~110  | none                                                                          | 1     |
| 7   | `client.py`                   | ~90   | none                                                                          | 1     |
| 8   | `lock.py`                     | ~35   | none                                                                          | 1     |
| 9   | `utils.py`                    | ~110  | none                                                                          | 1     |
| 10  | `progress.py`                 | ~210  | stdlib only                                                                   | 2     |
| 11  | `io.py`                       | ~110  | stdlib only                                                                   | 2     |
| 12  | `gpu.py`                      | ~160  | `io.py`, `utils.py`                                                           | 2     |
| 13  | `models.py`                   | ~220  | `gguf.py`, `io.py`, `config.py`, `utils.py`                                   | 3     |
| 14  | `server.py`                   | ~160  | `config.py`, `progress.py`, `io.py`, `utils.py`                               | 3     |
| 15  | `bench.py`                    | ~270  | `config.py`, `progress.py`, `io.py`, `utils.py`                               | 3     |
| 16  | `accuracy.py`                 | ~370  | `config.py`, `progress.py`, `client.py`, `test_image.py`, `io.py`, `utils.py` | 3     |
| 17  | `decisions.py`                | ~260  | `config.py`                                                                   | 3     |
| 18  | `report.py`                   | ~320  | `decisions.py`, `config.py`, `io.py`, `gpu.py`, `utils.py`                    | 4     |
| 19  | `swap.py`                     | ~210  | `decisions.py`, `config.py`, `io.py`, `utils.py`                              | 4     |
| 20  | `commands.py`                 | ~520  | all above                                                                     | 5     |
| 21  | `llmbench.py`                 | ~10   | `benchmarker.main` (deprecated wrapper)                                       | 5     |
| 22  | `pyproject.toml`              | ~30   | project config                                                                | 1a    |
| 23  | `tests/conftest.py`           | ~50   | `benchmarker.*` fixtures                                                      | 1b    |
| 24  | `tests/test_config.py`        | ~80   | `benchmarker.config`                                                          | 1b    |
| 25  | `tests/test_gguf.py`          | ~60   | `benchmarker.gguf`                                                            | 1b    |
| 26  | `tests/test_test_image.py`    | ~40   | `benchmarker.test_image`                                                      | 1b    |
| 27  | `tests/test_client.py`        | ~40   | `benchmarker.client`                                                          | 1b    |
| 28  | `tests/test_lock.py`          | ~30   | `benchmarker.lock`                                                            | 1b    |
| 29  | `tests/test_utils.py`         | ~60   | `benchmarker.utils`                                                           | 1b    |
| 30  | `tests/test_io.py`            | ~50   | `benchmarker.io`                                                              | 1b    |
| 31  | `tests/test_gpu.py`           | ~60   | `benchmarker.gpu`                                                             | 2     |
| 32  | `tests/test_progress.py`      | ~50   | `benchmarker.progress`                                                        | 2     |
| 33  | `tests/test_bench.py`         | ~80   | `benchmarker.bench`                                                           | 3     |
| 34  | `tests/test_accuracy.py`      | ~100  | `benchmarker.accuracy`                                                        | 3     |
| 35  | `tests/test_decisions.py`     | ~120  | `benchmarker.decisions`                                                       | 3     |
| 36  | `tests/test_commands.py`      | ~60   | `benchmarker.commands`                                                        | 5     |
| 37  | `tests/test_report.py`        | ~60   | `benchmarker.report`                                                          | 4     |
| 38  | `tests/test_swap.py`          | ~50   | `benchmarker.swap`                                                            | 4     |
| 39  | `tests/test_llmbench.py`      | ~45   | old consolidated test file (migrate out)                                      | 1b    |
| 36  | `tests/test_commands.py`      | ~60   | `benchmarker.commands`                                                        | 5     |
| 37  | `tests/test_report.py`        | ~60   | `benchmarker.report`                                                          | 4     |
| 38  | `tests/test_swap.py`          | ~50   | `benchmarker.swap`                                                            | 4     |
| 39  | `tests/test_llmbench.py`      | ~45   | old consolidated test file (migrate out)                                      | 1b    |

**Total estimated lines: ~3,220** (source) + **~1,300** (tests) + **~425** (entry point) = **~4,945**

---

## Rollback Plan

Each phase produces independently testable modules. If a phase fails:

1. Revert that phase's files
2. Keep `llmbench.py` intact (it's only rewritten in Phase 5)
3. Debug and retry

The safest approach is to keep `llmbench.py` as-is until Phase 5, then immediately run the full test suite. If Phase 5 fails, revert `llmbench.py` and the new modules, keeping the intermediate modules for a future attempt.

---

## Notes

- **Python version:** `pyproject.toml` declares `>=3.11` (matches the code's claim), not 3.14
- **Package manager:** `uv` (not pip) — use `uv build` to create dists, `uv pip install dist/*.whl` to install
- **Test framework:** `pytest` (converted from `unittest` in Nov 2026, 45 tests all passing)
- **Test layout:** All tests live under `tests/` (conftest + per-module test files + migrated test_llmbench.py)
- **No third-party dependencies:** Only Python standard library used at runtime
- **`src/` layout:** Package lives under `src/benchmarker/` following modern Python conventions
- **Refactor-first:** Behaviour must remain identical — every phase is verified against the existing test suite and CLI output
- **Entry point:** Three ways to run: `llmbench` (installed CLI), `python -m benchmarker` (package), `./llmbench.py` (deprecated wrapper)
- **Build commands:**

  ```bash
  uv build                    # creates dist/benchmarker-2.3.0-py3-none-any.whl
  uv pip install dist/*.whl   # install the wheel
  llmbench doctor             # verify installed CLI works
  uv pip install -e .         # editable install for development
  ```

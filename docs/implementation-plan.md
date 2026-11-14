# benchmarker — Refactoring Plan (Completed)

## Overview

The original monolithic script was a **3,147-line** file that mixed CLI parsing, configuration management, GPU detection, benchmark phases, HTTP communication, state persistence, report generation, and test image creation. It has been split into focused modules while preserving the **standard-library-only** constraint (Python 3.11+).

The project uses `uv` as its package manager with a `src/` layout (`src/benchmarker/`). Tests use `pytest` (45 tests, all passing).

---

## Current State (COMPLETED)

```
pyproject.toml                       # uv project: name=benchmarker, requires-python>=3.11
                                     #   llmbench = "benchmarker:main" (CLI command)
src/
  benchmarker/
    __init__.py                      # Package init: 41 __all__ exports, re-exports from submodules
    __main__.py                      # python -m benchmarker support
    cli.py                           # argparse setup + dispatch
    config.py                        # DEFAULTS, PROFILES, merge/check/base config, model loading (478 lines)
    gguf.py                          # gguf_meta, model_files, is_mmproj, model_info, find_mmproj
    test_image.py                    # test_image, image_message, VISION_Q
    client.py                        # Client, chat, http_error
    lock.py                          # Lock context manager
    progress.py                      # Progress class, tracked, log, warn
    io.py                            # append_csv, read_csv, save_state, load_states
    utils.py                         # xp, tilde, now, run_text, free_port, api_key, etc.
    gpu.py                           # setup_devices, vram_mb, ram_free_mb, pick_main_gpu
    server.py                        # Server class, stop_all, wait_vram_released
    bench.py                         # Fitter, sweep_moe, sweep_dense, speed_phase, tune_phase
    accuracy.py                      # fill_series, vision_series, needle logic, haystack
    decisions.py                     # speed_gate, accuracy_state, recommend, choose_cache, final_setting
    report.py                        # report_header, report_recommendations, report_model, cmd_report
    swap.py                          # swap_yaml, swap_models_in, cmd_swap
    commands.py                      # cmd_doctor, cmd_init, cmd_info, cmd_sweep, cmd_fill, cmd_vision, cmd_status
tests/
  conftest.py                        # Shared fixtures (progress_setup, tmp_results, mock_cfg)
  test_config.py                     # merge_config, check_config, profile_summary, override_lines (12 tests)
  test_decisions.py                  # speed_gate, accuracy_state, recommend, choose_cache (23 tests)
  test_progress.py                   # Progress class, tracked, log, warn (6 tests)
  test_accuracy.py                   # fill_series, vision_series, needle logic (6 tests)
  test_bench.py                      # Placeholder — no tests yet
  test_client.py                     # Placeholder — no tests yet
  test_commands.py                   # Placeholder — no tests yet
  test_gguf.py                       # Placeholder — no tests yet
  test_gpu.py                        # Placeholder — no tests yet
  test_io.py                         # Placeholder — no tests yet
  test_lock.py                       # Placeholder — no tests yet
  test_report.py                     # Placeholder — no tests yet
  test_swap.py                       # Placeholder — no tests yet
  test_test_image.py                 # Placeholder — no tests yet
  test_utils.py                      # Placeholder — no tests yet
results/progress.json                # Runtime progress state
docs/implementation-plan.md          # This file
```

---

## Completed Phases

| Phase | Files | Status | Notes |
|-------|-------|--------|-------|
| 1: Leaf modules | config.py, gguf.py, test_image.py, client.py, lock.py, utils.py | ✅ Complete | 8 files, ~1,208 lines |
| 1a: Package entry point | __init__.py, __main__.py, cli.py, pyproject.toml | ✅ Complete | 4 files, 41 __all__ exports |
| 1b: Test scaffolding | tests/ with 16 files | ✅ Complete | 45 active tests + 7 placeholders |
| 2: Low-level modules | gpu.py | ✅ Complete | 1 file, ~154 lines |
| 3: Domain modules | decisions.py, server.py, bench.py, accuracy.py | ✅ Complete | 4 files, ~1,020 lines |
| 4: Report & Swap | report.py, swap.py | ✅ Complete | 2 files, ~605 lines |
| 5: Commands & Entry Point | commands.py | ✅ Complete | 542 lines |
| 6: Cleanup | README, __init__.py | ✅ Complete | All docs updated |

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

**Verification gate:** After every phase, the full test suite (`python -m pytest tests/ -v`) must pass with zero regressions.

---

## Entry Point Design

```
# Installed CLI (via `uv pip install .` or `uv build` + pip)
llmbench doctor
llmbench sweep all

# Direct package execution (python -m benchmarker)
python -m benchmarker doctor
python -m benchmarker sweep all
```

---

## Module Dependencies (Actual)

```
                  ┌─────────────────────────────────┐
                  │         benchmarker.py           │
                  │  (argparse + dispatch)           │
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

---

## File-by-File Migration Status

| #   | File                          | Lines | Status |
| --- | ----------------------------- | ----- | ------ |
| 1   | `src/benchmarker/__init__.py` | 118   | ✅ Complete |
| 2   | `src/benchmarker/__main__.py` | 6     | ✅ Complete |
| 3   | `src/benchmarker/cli.py`      | 138   | ✅ Complete |
| 4   | `src/benchmarker/config.py`   | 478   | ✅ Complete (includes models.py functions) |
| 5   | `src/benchmarker/gguf.py`     | 146   | ✅ Complete |
| 6   | `src/benchmarker/test_image.py`| 94   | ✅ Complete |
| 7   | `src/benchmarker/client.py`   | 74    | ✅ Complete |
| 8   | `src/benchmarker/lock.py`     | 35    | ✅ Complete |
| 9   | `src/benchmarker/utils.py`    | 123   | ✅ Complete |
| 10  | `src/benchmarker/progress.py` | 209   | ✅ Complete |
| 11  | `src/benchmarker/io.py`       | 84    | ✅ Complete |
| 12  | `src/benchmarker/gpu.py`      | 154   | ✅ Complete |
| 13  | `src/benchmarker/server.py`   | 122   | ✅ Complete |
| 14  | `src/benchmarker/bench.py`    | 335   | ✅ Complete |
| 15  | `src/benchmarker/accuracy.py` | 316   | ✅ Complete |
| 16  | `src/benchmarker/decisions.py`| 247   | ✅ Complete |
| 17  | `src/benchmarker/report.py`   | 442   | ✅ Complete |
| 18  | `src/benchmarker/swap.py`     | 163   | ✅ Complete |
| 19  | `src/benchmarker/commands.py` | 542   | ✅ Complete |
| 20  | `pyproject.toml`              | 28    | ✅ Complete |
| 21  | `tests/conftest.py`           | 7     | ✅ Complete |
| 22  | `tests/test_config.py`        | 62    | ✅ Complete |
| 23  | `tests/test_decisions.py`     | 230   | ✅ Complete |
| 24  | `tests/test_progress.py`      | 92    | ✅ Complete |
| 25  | `tests/test_accuracy.py`      | 64    | ✅ Complete |
| 26  | `tests/test_bench.py`         | 1     | ⏸ Placeholder |
| 27  | `tests/test_client.py`        | 1     | ⏸ Placeholder |
| 28  | `tests/test_commands.py`      | 1     | ⏸ Placeholder |
| 29  | `tests/test_gguf.py`          | 1     | ⏸ Placeholder |
| 30  | `tests/test_gpu.py`           | 1     | ⏸ Placeholder |
| 31  | `tests/test_io.py`            | 1     | ⏸ Placeholder |
| 32  | `tests/test_lock.py`          | 1     | ⏸ Placeholder |
| 33  | `tests/test_report.py`        | 1     | ⏸ Placeholder |
| 34  | `tests/test_swap.py`          | 1     | ⏸ Placeholder |
| 35  | `tests/test_test_image.py`    | 1     | ⏸ Placeholder |
| 36  | `tests/test_utils.py`         | 1     | ⏸ Placeholder |

**Total: 19 source files, 3,826 lines** + **16 test files, 45 active tests**

---

## Deviations from Original Plan

| Original Plan | Actual | Reason |
|--------------|--------|--------|
| `models.py` separate file | Model functions merged into `config.py` | `load_config`, `pick_models`, `need_tools`, `resolve_paths`, `resolve_model_path`, `rel_model_path` are tightly coupled with config |
| `test_llmbench.py` in tests/ | Removed from root entirely | Tests migrated to module-specific files |
| ~3,220 lines total | 3,826 lines | More docstrings, `__all__` exports, cleaner structure |
| 45 tests | 45 tests + 7 placeholders | Placeholder files for future test development |

---

## Build & Run Commands

```bash
# Development
uv pip install -e .                    # editable install
uv pip install -e ".[dev]"             # + pytest

# Tests
uv run pytest tests/ -v                # 45 tests

# CLI (two entry points)
llmbench doctor                        # installed CLI command
python -m benchmarker doctor           # package execution

# Build
uv build                              # creates dist/benchmarker-2.3.0-py3-none-any.whl
```

---

## Notes

- **Python version:** `pyproject.toml` declares `>=3.11` (matches the code's claim)
- **Package manager:** `uv` (not pip) — use `uv build` to create dists, `uv pip install dist/*.whl` to install
- **Test framework:** `pytest` (converted from `unittest` in Nov 2026, 45 tests all passing)
- **Test layout:** All tests live under `tests/` (conftest + per-module test files)
- **No third-party dependencies:** Only Python standard library used at runtime
- **`src/` layout:** Package lives under `src/benchmarker/` following modern Python conventions
- **Refactor-first:** Behaviour must remain identical — every phase is verified against the existing test suite and CLI output
- **Entry point:** Two ways to run: `llmbench` (installed CLI), `python -m benchmarker` (package)

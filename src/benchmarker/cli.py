"""benchmarker - benchmark and tuning tool for local GGUF models (llama.cpp + llama-swap).

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
variables (LLMBENCH_LLAMA_DIR, LLMBENCH_MODELS_DIR, LLMBENCH_BACKEND) or with options.

Decision rules (target context, speed limits, recall test) are in PROFILES: strict (the rules
of 2.2), balanced (default) and long. Choose with --profile, LLMBENCH_PROFILE or
profile = "..." in models.toml. 'report' and 'swap' judge saved results with the chosen profile.
The rules are tested by `python3 -m pytest tests/`.

Only the Python standard library is used (Python 3.11 or later).
"""

import argparse
import atexit
import os
import signal
import sys

from benchmarker.commands import (
    cmd_doctor,
    cmd_fill,
    cmd_info,
    cmd_init,
    cmd_report,
    cmd_status,
    cmd_swap,
    cmd_sweep,
    cmd_vision,
)
from benchmarker.config import CONFIG, PROFILES, RESULTS, VERSION, xp
from benchmarker.server import stop_all


def main():
    """Parse arguments and dispatch to the selected command."""
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
    ap.add_argument("--version", action="version", version=f"benchmarker {VERSION}")
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

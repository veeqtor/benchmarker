"""CLI command handlers.

Each command handler is a function that receives the parsed args namespace
and dispatches to the appropriate workflow (doctor, init, sweep, etc.).
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

from benchmarker.accuracy import accuracy_search, fill_series, served_phase, vision_series
from benchmarker.bench import Fitter, speed_phase, sweep_dense, sweep_moe, tune_phase
from benchmarker.client import Client, http_error
from benchmarker.config import (
    CONFIG,
    DEFAULTS,
    PROFILES,
    RESULTS,
    RUN_ID,
    VERSION,
    base_config,
    check_config,
    load_config,
    need_tools,
    override_lines,
    pick_models,
    profile_summary,
    read_toml,
    rel_model_path,
    resolve_model_path,
    sdesc,
)
from benchmarker.decisions import recommend, why_below_target
from benchmarker.gguf import find_mmproj, is_extra_part, is_mmproj, model_info
from benchmarker.gpu import (
    gpu_dir,
    physical_cores,
    ram_free_mb,
    setup_devices,
    vram_mb,
)
from benchmarker.io import save_state
from benchmarker.lock import Lock
from benchmarker.progress import PROGRESS, log, tracked, warn
from benchmarker.report import cmd_report
from benchmarker.server import stop_all
from benchmarker.swap import cmd_swap
from benchmarker.utils import (
    api_key,
    build_id,
    fmt_dur,
    other_servers,
    run_text,
    service_active,
    tilde,
)

__all__ = [
    "cmd_doctor",
    "cmd_fill",
    "cmd_info",
    "cmd_init",
    "cmd_report",
    "cmd_status",
    "cmd_swap",
    "cmd_sweep",
    "cmd_vision",
    "status_text",
]


# ---------------------------------------------------------------- commands
def cmd_doctor(args):
    raw = read_toml()
    cfg = base_config(args, raw)
    ok = True

    def line(status, what, detail=""):
        nonlocal ok
        ok = ok and status != "FAIL"
        log(f"[{status:<4}] {what:<22} {detail}")

    log(f"benchmarker {VERSION}   config {CONFIG}{'' if raw else ' (missing: run init)'}")
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
                "# benchmarker model list. Change values here; delete a block to remove a model.\n"
                "# Any value of DEFAULTS or of the profile in benchmarker.config can be set in [defaults].\n"
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
    save_state(state)
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


def cmd_status(args):
    path = os.path.join(RESULTS, "progress.json")
    while True:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            sys.exit(f"No progress file: {path}. No run was started with these results folder.")
        log(status_text(data))
        if not args.watch or data.get("state") != "running":
            return
        time.sleep(args.watch)
        log("")


def status_text(data, now_ts=None):
    """Text for the 'status' command from a progress.json dict."""
    now_ts = time.time() if now_ts is None else now_ts
    age = now_ts - data.get("updated", now_ts)
    alive = True
    try:
        os.kill(int(data.get("pid", 0)), 0)
    except (OSError, ValueError):
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
            "WARNING: no update for 10 minutes. A long test or model load can take this long; "
            "check the newest file in results/logs/."
        )
    return "\n".join(l for l in lines if l)

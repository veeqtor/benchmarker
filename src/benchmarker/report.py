"""Markdown report generation.

Produces a full benchmark report with system info, model details, speed
tables, decision tables, accuracy results, and fit test summaries.
"""

import os
import shlex
import subprocess

from benchmarker.config import (
    RESULTS,
    RUN_ID,
    SETTING_KEYS,
    VERSION,
    load_config,
    override_lines,
    profile_summary,
    sdesc,
)
from benchmarker.decisions import (
    acc_key,
    acc_verdict,
    final_setting,
    speed_at_depth,
    why_below_target,
)
from benchmarker.gpu import physical_cores, ram_free_mb, vram_mb
from benchmarker.io import load_states, read_csv
from benchmarker.progress import log, warn
from benchmarker.swap import swap_yaml
from benchmarker.utils import build_id, now


# ------------------------------------------------------------------ helpers
def table(rows, cols):
    """Format rows as a Markdown table."""
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |")
    return "\n".join(out)


def system_info(cfg):
    """System hardware info for reports."""

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
            f"benchmarker: {VERSION}",
        ]
    )


# ------------------------------------------------------------------ report functions
def report_header(cfg, models):
    """Report header with system + model info."""
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
        (
            "Speed values come from llama-bench (text only, no image encoder). "
            "The fit tests use llama-server with the image encoder and, "
            "for vision models, one image request."
        ),
        "",
        "## Models",
        "",
    ]
    L.append(
        table(
            [
                {
                    "model": n,
                    "arch": m["info"]["arch"],
                    "type": m["type"],
                    "layers": m["info"]["layers"],
                    "experts": f"{m['info']['experts']}/{m['info']['experts_used']}",
                    "attn_layers": m["info"]["attn_layers"] or "-",
                    "native_ctx": m["info"]["native_ctx"],
                    "size_mib": m["info"]["size_mib"],
                    "file": os.path.basename(m["path"]),
                    "vision": (
                        os.path.basename(m["mmproj"]) + ("" if m["mmproj_offload"] else " (RAM)")
                    )
                    if m["mmproj"]
                    else "-",
                }
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
    """Recommendations table from saved results."""
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
                    "result": (
                        ("OK" if g["ok"] else "rejected: " + g["reason"])
                        + ("  <- recommended" if chosen else "")
                    ),
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
    """Full report section for one model."""
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
    cols = [
        "context",
        "setting",
    ] + sorted(
        {k for r in rows for k in r if "@" in k},
        key=lambda k: (k.split("@")[0], int(k.split("@")[1])),
    )
    L += [
        "### Speed per context (tokens/s, @ = filled context depth)",
        "",
        table(rows, cols),
        "",
    ]
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
    """Write full Markdown report."""
    cfg, models = load_config(args)
    if cfg["server"] and "_devices" not in cfg:
        cfg["_quiet_devices"] = args.cmd != "report"
        from benchmarker.gpu import setup_devices

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

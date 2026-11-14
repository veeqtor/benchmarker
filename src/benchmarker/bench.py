"""Benchmark phases: fit (Fitter), speed, tune.

Fitter class handles GPU memory fitting tests.  speed_phase runs llama-bench
at multiple depths.  tune_phase searches for optimal ubatch + threads.
"""

import json
import os
import shlex
import subprocess
import time

from benchmarker.client import chat, http_error
from benchmarker.config import RESULTS, RUN_ID, bench_flags, sdesc
from benchmarker.gpu import (
    physical_cores,
    ram_free_mb,
    vram_mb,
    wait_vram_released,
)
from benchmarker.io import append_csv
from benchmarker.progress import PROGRESS, log
from benchmarker.server import Server
from benchmarker.utils import now


# ------------------------------------------------------------------ Fitter
class Fitter:
    """Phase 1: binary-searches optimal GPU settings by fitting models."""

    def __init__(self, cfg, m):
        self.cfg, self.m, self.cache = cfg, m, {}
        _, self.baseline_used, _ = vram_mb(cfg)

    def warmup(self, srv):
        """Run the work a real client does, so all buffers are allocated before measuring."""
        c, m, t = srv.client(), self.m, self.cfg["request_timeout_s"]
        if self.cfg["fit_warmup"]:
            chat(c, m["name"], [{"role": "user", "content": "Say OK."}], 4, t)

        if m["mmproj"] and self.cfg["image_fit"]:
            from benchmarker.vision_test import generate_test_image, image_message

            png, _, _ = generate_test_image(self.cfg["image_px"], seed=1)
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


# ------------------------------------------------------------------ sweep helpers
def sweep_moe(cfg, m, fitter):
    """Binary-search expert layers for MoE models."""
    L = m["info"]["layers"] or 64
    settings = []
    for cache in m["cache_types"]:
        lo = 0
        mk = lambda n, c=cache: {"ngl": L + 1, "ncmoe": n, "cache": c}
        for ctx in m["contexts"]:
            a, b = lo, L
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
    """Binary-search ngl + context for dense models."""
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
            while a < b:
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
        while a < b:
            mid = (a + b + 1) // 2
            if fitter.fits(cands[mid], mk(ngl)):
                a = mid
            else:
                b = mid - 1
        settings.append({"context": cands[a], "setting": mk(ngl)})
    return settings


# ------------------------------------------------------------------ speed phase
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
    """(depths, deep). depths = filled-context depths that fit. deep = one more depth for
    contexts >= target_context (bench_deep_fill_pct fill), because that is the speed a long
    session has. None if not needed."""
    depths = sorted(d for d in cfg["bench_depths"] if d + 1024 <= ctx) or [0]
    pct = cfg["bench_deep_fill_pct"]
    if pct and ctx >= cfg["target_context"]:
        deep = int(ctx * pct / 100) // 1024 * 1024
        if deep > max(depths):
            return depths, deep
    return depths, None


def bench_cached(done, cfg, m, s, depths, **kw):
    """run_bench, but the same flags and depths run only once."""
    key = json.dumps([bench_flags(cfg, s), depths, kw], sort_keys=True)
    if key not in done:
        done[key] = run_bench(cfg, m, s, depths, **kw)
    return done[key]


def speed_phase(cfg, m, settings):
    """Run speed tests for all settings."""
    done = {}
    PROGRESS.phase("speed", len(settings))
    for st in settings:
        s, ctx = st["setting"], st["context"]
        PROGRESS.tick(f"llama-bench: ctx={ctx} {sdesc(s)}")
        depths, deep = speed_depths(cfg, ctx)
        rows = list(bench_cached(done, cfg, m, s, depths))
        if deep:
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


# ------------------------------------------------------------------ tune phase
def tune_phase(cfg, m, st, fitter):
    """Tune ubatch + threads for the optimal setting."""
    s = dict(st["setting"])
    ubs = cfg["tune_ubatch"]
    threads = [None]
    if m["type"] == "moe" or s["ngl"] <= m["info"]["layers"]:
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
    for (ub, t), _ in ranked:
        cand = dict(s, ub=ub, threads=t if threads != [None] else None)
        if fitter.fits(st["context"], cand):
            log(f"  tune  best: {sdesc(cand)}")
            return cand
    return None

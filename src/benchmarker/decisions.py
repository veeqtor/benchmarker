"""Decision rules for benchmarker.

Pure-logic functions that judge benchmark results: speed gates, accuracy
verdicts, cache selection, and final recommendations.  Depends only on
config.py (constants, PROFILES, etc.).
"""

import collections
import re

from benchmarker.config import CACHE_QUALITY, PROFILES, required_hits, sdesc


# ------------------------------------------------------------------ helpers
def speed_at_depth(st):
    """(depth, t/s) of the deepest tg128 measurement of a setting."""
    tg = [
        (int(k.split("@")[1]), v)
        for k, v in st.get("speed", {}).items()
        if k.startswith("tg128@")
    ]
    return max(tg) if tg else (0, 0)


def deep_tg(st):
    return speed_at_depth(st)[1]


# ------------------------------------------------------------------ speed_gate
def speed_gate(cfg, st, best_tg):
    """(ok, note). Slow = more than speed_tolerance_pct below the best tg128,
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


# ------------------------------------------------------------------ accuracy
def acc_key(a):
    """(cache, context) of a saved long-prompt result."""
    ctx, cache = a.get("context"), a.get("cache")
    if not ctx:
        c = re.search(r"ctx=(\d+)", a.get("label", ""))
        ctx = c.group(1) if c else 0
    if not cache:
        k = re.search(r"kv=(\S+)", a.get("label", ""))
        cache = k.group(1) if k else "*"
    return cache, int(ctx)


def acc_verdict(cfg, a):
    """'pass', 'fail' or None (request error, no valid result)."""
    mt = re.fullmatch(r"(\d+)/(\d+)", str(a.get("score", "")))
    if a.get("status") in ("FAIL", "HTTP") or not mt:
        return None
    hits, n = int(mt.group(1)), int(mt.group(2))
    return "pass" if hits >= required_hits(cfg, n) else "fail"


def accuracy_state(cfg, acc, cache, context):
    """('PASS' | 'FAIL' | 'UNTESTED', 'score@pct% ...') for one cache type and context."""
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
        f"{a['score']}@{a['pct']}%"
        for a, _ in sorted(rows, key=lambda r: int(r[0]["pct"]))
    )
    return ("FAIL" if any(v == "fail" for _, v in rows) else "PASS"), detail


def gate_rows(cfg, sts, acc):
    """One decision row per context for one cache type."""
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
            r["accuracy"] == "PASS"
            or (r["accuracy"] == "UNTESTED" and not accuracy_ran)
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


# ------------------------------------------------------------------ cache selection
def cache_rank(cache):
    return CACHE_QUALITY.index(cache) if cache in CACHE_QUALITY else len(CACHE_QUALITY)


def choose_cache(cfg, picks):
    """Among the cache types that reach target_context: the best quality that keeps
    at least 85% of the fastest one's speed with a filled context."""
    target = cfg["target_context"]
    reach = [
        c for c, p in picks.items() if p["reliable"] and p["optimum"]["context"] >= target
    ]
    if reach:
        fastest = max(deep_tg(picks[c]["optimum"]) for c in reach)
        return min(
            (c for c in reach if deep_tg(picks[c]["optimum"]) >= 0.85 * fastest),
            key=cache_rank,
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


# ------------------------------------------------------------------ recommend
def recommend(cfg, settings, acc):
    """Optimum = the largest context that passes the speed rule and the long-prompt rule."""
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
        else:
            maybe = [
                g["context"]
                for g in gates
                if g["speed_ok"] and g["accuracy"] != "FAIL"
            ]
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
    """One line: which rule keeps the chosen cache type under target_context."""
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
            f"no tested context >= {target} fits in memory "
            f"(largest fit: {chosen['maximum']['context']}). "
            f"Try: {hint}, or lower min_free_vram_mb"
        )
    return f"ctx {above[0]['context']}: {above[0]['reason'] or 'no result'}"


# ------------------------------------------------------------------ final_setting
Final = collections.namedtuple("Final", "context setting speed rec")


def final_setting(cfg, s):
    """Recommendation of a saved result, computed with the current rules."""
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
        else:
            if tuned.get("threads"):
                setting["threads"] = tuned["threads"]
            if tuned.get("ub") and tuned["ub"] <= 512:
                setting["ub"] = tuned["ub"]
    return Final(st["context"], setting, st.get("speed", {}), rec)

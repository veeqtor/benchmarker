"""Accuracy & vision tests.

fill_series / fill_one: long-prompt needle-in-haystack tests.
vision_series: image recognition tests.
accuracy_search: search for best context that passes accuracy tests.
served_phase: start a server and run accuracy + vision tests on it.
"""

import random
import re
import urllib.error

from benchmarker.client import chat, http_error
from benchmarker.config import RUN_ID, sdesc
from benchmarker.io import append_csv
from benchmarker.progress import PROGRESS, log
from benchmarker.server import Server
from benchmarker.test_image import VISION_Q, image_message, test_image
from benchmarker.utils import now

# ------------------------------------------------------------------ haystack helpers
_SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
LETTERS = "ABCDEFGHIJ"


def pseudo_word(rnd):
    return "".join(rnd.choice(_SYLLABLES) for _ in range(3))


def haystack(style, n, seed):
    """n distinct (key, value, text) entries for the long-prompt test."""
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
    """Count the correct values. Returns (hits, indices of the wrong values)."""
    got = {l.upper(): v for l, v in re.findall(r"\b([A-Ja-j])\s*[=:]\s*(\d+)", answer)}
    hits, missed = 0, []
    for i, value in enumerate(expected):
        ok = got[LETTERS[i]] == str(value) if LETTERS[i] in got else str(value) in answer
        if ok:
            hits += 1
        else:
            missed.append(i)
    return hits, missed


# ------------------------------------------------------------------ fill tests
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
    need = cfg["accuracy_min_pct"]
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


def fill_series(
    cfg, client, model, pcts, label, think=False, meta=None, stop_on_fail=False, tick=False
):
    """Long-prompt tests at the given fill levels, smallest first."""
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


# ------------------------------------------------------------------ vision tests
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


# ------------------------------------------------------------------ accuracy search
def accuracy_search(cfg, m, settings, rec):
    """Per KV cache type: test the largest context that is fast enough."""
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
            from benchmarker.progress import warn

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

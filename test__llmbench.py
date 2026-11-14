"""Unit tests for the decision rules of llmbench (no GPU, no llama.cpp, no network).

Run:  pytest test__llmbench.py -v      (in the llmbench folder)

Run these tests after each change of the rules (PROFILES, speed_gate, accuracy_state,
recommend, choose_cache).
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llmbench as lb


def cfg_for(profile="balanced", **over):
    cfg = lb.merge_config(over, profile)
    return cfg


def setting(ctx, cache="q8_0", tg0=50.0, deep=None, ncmoe=20, ngl=41):
    speed = {"tg128@0": tg0}
    if deep:
        speed[f"tg128@{deep[0]}"] = deep[1]
    return {
        "context": ctx,
        "setting": {"ngl": ngl, "ncmoe": ncmoe, "cache": cache},
        "speed": speed,
    }


def fill(ctx, pct, score, cache="q8_0", status=None):
    hits, n = map(int, score.split("/"))
    return {
        "context": ctx,
        "cache": cache,
        "pct": pct,
        "score": score,
        "status": status or ("PASS" if hits == n else "PARTIAL" if hits else "MISS"),
    }


# ─── Profile Tests ───────────────────────────────────────────────────────────


def test_all_profiles_define_the_same_keys():
    keys = {frozenset(p) for p in lb.PROFILES.values()}
    assert len(keys) == 1


def test_default_profile_exists():
    assert lb.DEFAULTS["profile"] in lb.PROFILES


def test_precedence_defaults_profile_user():
    cfg = lb.merge_config({"target_context": 200000, "backend": "vulkan"}, "balanced")
    assert cfg["target_context"] == 200000  # user beats profile
    assert cfg["speed_tolerance_pct"] == lb.PROFILES["balanced"]["speed_tolerance_pct"]
    assert cfg["backend"] == "vulkan"  # user beats DEFAULTS
    assert cfg["bench_reps"] == lb.DEFAULTS["bench_reps"]


def test_profile_from_user_table():
    assert lb.merge_config({"profile": "long"})["profile"] == "long"
    assert lb.merge_config({"profile": "long"}, "strict")["profile"] == "strict"  # option wins


def test_override_is_reported():
    cfg = lb.merge_config({"speed_tolerance_pct": 10}, "balanced")
    assert cfg["_overrides"] == {"speed_tolerance_pct": (10, 25)}
    assert len(lb.override_lines(cfg)) == 1
    assert lb.merge_config({"speed_tolerance_pct": 25}, "balanced")["_overrides"] == {}


def test_unknown_profile_stops():
    with pytest.raises(SystemExit):
        lb.merge_config({}, "fast")


@pytest.mark.parametrize(
    "key,bad",
    [
        ("needle_style", "x"),
        ("needles", 0),
        ("accuracy_min_pct", 0),
        ("accuracy_pct", []),
        ("accuracy_pct", [99]),
        ("speed_tolerance_pct", 100),
    ],
)
def test_check_config_rejects_bad_values(key, bad):
    with pytest.raises(SystemExit):
        lb.check_config(cfg_for(**{key: bad}))
    # unreachable for this iteration, but the loop above covers all cases


# ─── Needle Tests ────────────────────────────────────────────────────────────


def test_haystack_is_repeatable_and_keys_are_unique():
    a, b = lb.haystack("keyed", 3000, seed=7), lb.haystack("keyed", 3000, seed=7)
    assert a == b
    assert len({x[0] for x in a}) == 3000
    assert a != lb.haystack("keyed", 3000, seed=8)


def test_line_style_is_the_old_format():
    items = lb.haystack("line", 10, seed=1)
    assert items[3][2] == "Line 3: the value is %d." % ((3 * 7919 + 13) % 100000)


def test_positions_are_spread():
    assert lb.needle_positions(1000, 3) == [100, 500, 900]
    assert lb.needle_positions(1000, 5) == [100, 300, 500, 700, 900]
    assert lb.needle_positions(1000, 1) == [500]


def test_question_text():
    assert lb.needle_question("line", ["1", "5", "9"]) == (
        "What are the values on line 1, line 5 and line 9? "
        "Answer only in this form: A=<number>, B=<number>, C=<number>."
    )
    q = lb.needle_question("keyed", ["kelmora", "tuvrin"])
    assert "project kelmora and project tuvrin" in q
    assert "A=<number>, B=<number>" in q


def test_score_answer():
    assert lb.score_answer("A=11111, B=22222, C=33333", [11111, 22222, 33333]) == (3, [])
    assert lb.score_answer("A=11111, B=22223, C=33333", [11111, 22222, 33333]) == (2, [1])
    # the letter must match: the right number under the wrong letter is wrong
    assert lb.score_answer("A=22222, B=11111", [11111, 22222])[0] == 0
    # no letters: the number anywhere counts
    assert lb.score_answer("The values are 11111 and 22222.", [11111, 22222])[0] == 2
    assert lb.score_answer("", [1, 2])[0] == 0


def test_required_hits():
    assert lb.required_hits(cfg_for("balanced"), 5) == 4
    assert lb.required_hits(cfg_for("strict"), 3) == 3
    assert lb.required_hits(cfg_for("balanced", accuracy_min_pct=1), 5) == 1


# ─── Verdict Tests ───────────────────────────────────────────────────────────


def test_verdict_uses_the_current_rules():
    a = fill(131072, 50, "4/5")
    assert lb.acc_verdict(cfg_for("balanced"), a) == "pass"
    assert lb.acc_verdict(cfg_for("strict"), a) == "fail"


def test_request_errors_are_not_a_verdict():
    assert lb.acc_verdict(cfg_for(), {"status": "FAIL", "score": ""}) is None
    assert lb.acc_verdict(cfg_for(), {"status": "HTTP", "score": "0/5"}) is None


def test_acc_key_new_and_old_rows():
    assert lb.acc_key({"context": 65536, "cache": "f16"}) == ("f16", 65536)
    old = {"label": "m accuracy ctx=98304 ngl=66 kv=q4_0 ub=512"}
    assert lb.acc_key(old) == ("q4_0", 98304)
    assert lb.acc_key({"label": "m (llama-swap)"}) == ("*", 0)


def test_only_configured_fill_levels_count():
    cfg = cfg_for("balanced")  # [50, 75]
    acc = [fill(131072, 50, "5/5"), fill(131072, 90, "1/5")]
    assert lb.accuracy_state(cfg, acc, "q8_0", 131072)[0] == "PASS"
    cfg = cfg_for("balanced", accuracy_pct=[50, 90])
    assert lb.accuracy_state(cfg, acc, "q8_0", 131072)[0] == "FAIL"


def test_untested():
    assert lb.accuracy_state(cfg_for(), [], "q8_0", 65536) == ("UNTESTED", "")
    err = [{"context": 65536, "cache": "q8_0", "pct": 50, "score": "", "status": "FAIL"}]
    assert lb.accuracy_state(cfg_for(), err, "q8_0", 65536)[0] == "UNTESTED"


# ─── Speed Tests ─────────────────────────────────────────────────────────────


def test_relative_limit():
    cfg = cfg_for("balanced")  # 25%
    assert lb.speed_gate(cfg, setting(1, tg0=40), 50)[0]
    ok, why = lb.speed_gate(cfg, setting(1, tg0=36), 50)
    assert not ok
    assert "28% below" in why
    assert lb.speed_gate(cfg_for("strict"), setting(1, tg0=46), 50)[0]
    assert not lb.speed_gate(cfg_for("strict"), setting(1, tg0=44), 50)[0]


def test_floor_uses_the_deepest_measurement():
    cfg = cfg_for("balanced")  # 15 t/s
    assert lb.speed_gate(cfg, setting(1, tg0=50, deep=(65536, 16)), 50)[0]
    ok, why = lb.speed_gate(cfg, setting(1, tg0=50, deep=(65536, 12)), 50)
    assert not ok
    assert "65536" in why


def test_missing_speed_does_not_block():
    assert lb.speed_gate(cfg_for(), {"context": 1, "setting": {}}, 0)[0]


def test_depths():
    cfg = cfg_for("balanced")
    assert lb.speed_depths(cfg, 32768) == ([0, 8192], None)  # below target: no extra depth
    assert lb.speed_depths(cfg, 131072) == ([0, 8192, 32768], 65536)
    assert lb.speed_depths(cfg, 196608) == ([0, 8192, 32768], 98304)
    assert lb.speed_depths(dict(cfg, bench_deep_fill_pct=0), 131072) == ([0, 8192, 32768], None)


# ─── Recommend Tests ─────────────────────────────────────────────────────────


def _three(tg=(52, 50, 47), deeps=((32768, 40), (65536, 30), (98304, 24))):
    return [setting(c, tg0=t, deep=d) for c, t, d in zip((65536, 131072, 196608), tg, deeps)]


def test_balanced_reaches_131k_where_strict_does_not():
    """Measured case: 131K recalls 4/5 at 50% and 75% fill, but not at 90%."""
    settings = _three()
    acc = [
        fill(131072, 50, "5/5"),
        fill(131072, 75, "4/5"),
        fill(131072, 90, "2/5"),
        fill(196608, 50, "4/5"),
        fill(196608, 75, "2/5"),
    ]
    rec = lb.recommend(cfg_for("balanced"), settings, acc)
    assert rec["recommended"]["context"] == 131072
    assert rec["target_met"]
    assert rec["verified"]
    rec = lb.recommend(cfg_for("strict"), settings, acc)
    assert rec["recommended"]["context"] < 131072
    assert not rec["target_met"]


def test_long_profile_checks_only_50_percent():
    acc = [fill(196608, 50, "4/5"), fill(196608, 75, "2/5")]
    rec = lb.recommend(cfg_for("long"), _three(), acc)
    assert rec["recommended"]["context"] == 196608


def test_speed_only_when_accuracy_was_not_run():
    rec = lb.recommend(cfg_for("balanced"), _three(), [])
    assert rec["recommended"]["context"] == 196608
    assert not rec["verified"]
    assert rec["target_met"]


def test_speed_rule_removes_slow_contexts():
    settings = _three(tg=(52, 50, 30))  # 196K is 42% below the best
    rec = lb.recommend(cfg_for("balanced"), settings, [])
    assert rec["recommended"]["context"] == 131072
    gate = next(g for g in rec["per_cache"]["q8_0"]["gates"] if g["context"] == 196608)
    assert not gate["ok"]
    assert "too slow" in gate["reason"]


def test_error_at_the_top_does_not_win_over_a_passed_context():
    """A context whose tests only gave request errors must not beat a context that passed."""
    acc = [
        {"context": 196608, "cache": "q8_0", "pct": 50, "score": "", "status": "FAIL"},
        fill(131072, 50, "5/5"),
        fill(131072, 75, "5/5"),
    ]
    rec = lb.recommend(cfg_for("balanced"), _three(), acc)
    assert rec["recommended"]["context"] == 131072


def test_nothing_passes():
    acc = [fill(c, 50, "1/5", status="MISS") for c in (65536, 131072, 196608)]
    rec = lb.recommend(cfg_for("balanced"), _three(), acc)
    assert not rec["target_met"]
    assert not rec["per_cache"]["q8_0"]["reliable"]
    assert rec["recommended"]["context"] == 65536  # smallest: nothing is known to be good
    assert "long-prompt test failed" in lb.why_below_target(rec)


def test_fallback_is_the_largest_context_not_known_to_fail():
    acc = [fill(196608, 50, "1/5"), fill(131072, 50, "1/5")]  # 65536 was never tested
    rec = lb.recommend(cfg_for("balanced"), _three(), acc)
    assert rec["recommended"]["context"] == 65536
    assert not rec["verified"]
    assert "not run" in lb.why_below_target(dict(rec, target_context=65536))


def test_nothing_fits_the_target():
    rec = lb.recommend(cfg_for("balanced"), [setting(32768), setting(65536)], [])
    assert not rec["target_met"]
    text = lb.why_below_target(rec)
    assert "fits in memory" in text
    assert "q4_0" in text


def test_cache_choice_prefers_quality_when_speed_is_close():
    settings = [
        setting(131072, "q8_0", tg0=50, deep=(65536, 30)),
        setting(131072, "f16", tg0=50, deep=(65536, 28)),
    ]
    assert lb.recommend(cfg_for(), settings, [])["choice"] == "f16"


def test_cache_choice_prefers_speed_when_quality_costs_too_much():
    settings = [
        setting(131072, "q8_0", tg0=50, deep=(65536, 30)),
        setting(131072, "f16", tg0=50, deep=(65536, 20)),
    ]  # f16 below 85% of 30
    assert lb.recommend(cfg_for(), settings, [])["choice"] == "q8_0"


def test_cache_choice_without_target_takes_largest_context():
    settings = [setting(65536, "f16", tg0=50), setting(98304, "q4_0", tg0=50)]
    rec = lb.recommend(cfg_for(), settings, [])
    assert (rec["choice"], rec["recommended"]["context"]) == ("q4_0", 98304)


def test_final_setting_keeps_the_tuned_values():
    s = {
        "settings": _three(),
        "accuracy": [],
        "type": "moe",
        "tuned": {"ngl": 41, "ncmoe": 20, "cache": "q8_0", "ub": 1024, "threads": 8},
    }
    fs = lb.final_setting(cfg_for(), s)
    assert fs.context == 196608
    assert fs.setting["ub"] == 1024
    assert lb.final_setting(cfg_for(), {"settings": []}) is None


# ─── Progress Tests ──────────────────────────────────────────────────────────


@pytest.fixture
def progress_setup():
    """Fixture that mimics the old setUp/tearDown pattern."""
    now = [1000.0]  # mutable container so the clock lambda sees updates
    p = lb.Progress(clock=lambda: now[0])
    old_results = lb.RESULTS
    tmp = tempfile.mkdtemp()
    lb.RESULTS = tmp
    yield p, tmp, now
    lb.RESULTS = old_results
    p.end()


def test_fmt_dur():
    assert lb.fmt_dur(5) == "0:05"
    assert lb.fmt_dur(125) == "2:05"
    assert lb.fmt_dur(3725) == "1h02m"
    assert lb.fmt_dur(-3) == "0:00"


def test_inactive_progress_does_nothing(progress_setup):
    p, tmp, now = progress_setup
    p.tick("x")
    p.note("y")
    assert not os.path.exists(os.path.join(tmp, "progress.json"))


def test_tag_and_label(progress_setup):
    p, tmp, now = progress_setup
    p.begin("sweep", ["a", "b", "c"])
    p.start_model("a")
    assert p.tag() == "[1/3 +0:00]"
    p.phase("fit")
    now[0] += 65
    p.tick("fit test: ctx=1")
    assert p.tag() == "[1/3 fit #1 +1:05]"
    p.phase("speed", 4)
    p.tick("bench")
    assert p.tag() == "[1/3 speed 1/4 +1:05]"
    label = p.label()
    for text in ("sweep model 1/3: a", "speed 1/4", "bench", "run 1:05"):
        assert text in label


def test_eta_uses_finished_models(progress_setup):
    p, tmp, now = progress_setup
    p.begin("sweep", ["a", "b", "c"])
    p.start_model("a")
    assert p.eta_rest_s() is None
    now[0] += 600
    p.start_model("b")  # a took 600 s, one model after b
    assert p.eta_rest_s() == 600
    assert "later models ~10:00" in p.label()
    now[0] += 300
    p.start_model("c")  # average 450 s, nothing after c
    assert p.eta_rest_s() is None


def test_progress_file_and_status_text(progress_setup):
    p, tmp, now = progress_setup
    p.begin("sweep", ["a", "b"])
    p.start_model("a")
    p.phase("accuracy", 3)
    p.tick("long-prompt tests: ctx=131072")
    with open(os.path.join(tmp, "progress.json")) as f:
        data = json.load(f)
    assert (data["model"], data["phase"], data["step"], data["total"]) == ("a", "accuracy", 1, 3)
    text = lb.status_text(data, now_ts=data["updated"] + 5)
    assert "state: running" in text
    assert "accuracy 1/3" in text
    assert "WARNING" not in text  # this process is alive
    data["pid"] = 2**22 + 12345  # no such process
    assert "process is gone" in lb.status_text(data)
    p.end("stopped")
    with open(os.path.join(tmp, "progress.json")) as f:
        assert json.load(f)["state"] == "stopped"


def test_old_run_is_not_reported_as_dead():
    data = {"state": "finished", "pid": 2**22 + 12345, "updated": 0, "label": "x"}
    assert "WARNING" not in lb.status_text(data, now_ts=99999)

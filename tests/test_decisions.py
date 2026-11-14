"""Tests for decision rules — imported from benchmarker.decisions.

These tests exercise the pure logic functions: speed_gate, accuracy_state,
recommend, choose_cache, final_setting, why_below_target, acc_key, acc_verdict.
"""

from benchmarker.bench import speed_depths
from benchmarker.config import merge_config
from benchmarker.decisions import (
    acc_key,
    acc_verdict,
    accuracy_state,
    final_setting,
    recommend,
    speed_gate,
    why_below_target,
)


def cfg_for(profile="balanced", **over):
    return merge_config(over, profile)


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


# ─── Verdict Tests ───────────────────────────────────────────────────────────


def test_verdict_uses_the_current_rules():
    a = fill(131072, 50, "4/5")
    assert acc_verdict(cfg_for("balanced"), a) == "pass"
    assert acc_verdict(cfg_for("strict"), a) == "fail"


def test_request_errors_are_not_a_verdict():
    assert acc_verdict(cfg_for(), {"status": "FAIL", "score": ""}) is None
    assert acc_verdict(cfg_for(), {"status": "HTTP", "score": "0/5"}) is None


def test_acc_key_new_and_old_rows():
    assert acc_key({"context": 65536, "cache": "f16"}) == ("f16", 65536)
    old = {"label": "m accuracy ctx=98304 ngl=66 kv=q4_0 ub=512"}
    assert acc_key(old) == ("q4_0", 98304)
    assert acc_key({"label": "m (llama-swap)"}) == ("*", 0)


def test_only_configured_fill_levels_count():
    cfg = cfg_for("balanced")  # [50, 75]
    acc = [fill(131072, 50, "5/5"), fill(131072, 90, "1/5")]
    assert accuracy_state(cfg, acc, "q8_0", 131072)[0] == "PASS"
    cfg = cfg_for("balanced", accuracy_pct=[50, 90])
    assert accuracy_state(cfg, acc, "q8_0", 131072)[0] == "FAIL"


def test_untested():
    assert accuracy_state(cfg_for(), [], "q8_0", 65536) == ("UNTESTED", "")
    err = [{"context": 65536, "cache": "q8_0", "pct": 50, "score": "", "status": "FAIL"}]
    assert accuracy_state(cfg_for(), err, "q8_0", 65536)[0] == "UNTESTED"


# ─── Speed Tests ─────────────────────────────────────────────────────────────


def test_relative_limit():
    cfg = cfg_for("balanced")  # 25%
    assert speed_gate(cfg, setting(1, tg0=40), 50)[0]
    ok, why = speed_gate(cfg, setting(1, tg0=36), 50)
    assert not ok
    assert "28% below" in why
    assert speed_gate(cfg_for("strict"), setting(1, tg0=46), 50)[0]
    assert not speed_gate(cfg_for("strict"), setting(1, tg0=44), 50)[0]


def test_floor_uses_the_deepest_measurement():
    cfg = cfg_for("balanced")  # 15 t/s
    assert speed_gate(cfg, setting(1, tg0=50, deep=(65536, 16)), 50)[0]
    ok, why = speed_gate(cfg, setting(1, tg0=50, deep=(65536, 12)), 50)
    assert not ok
    assert "65536" in why


def test_missing_speed_does_not_block():
    assert speed_gate(cfg_for(), {"context": 1, "setting": {}}, 0)[0]


def test_depths():
    cfg = cfg_for("balanced")
    assert speed_depths(cfg, 32768) == ([0, 8192], None)  # below target: no extra depth
    assert speed_depths(cfg, 131072) == ([0, 8192, 32768], 65536)
    assert speed_depths(cfg, 196608) == ([0, 8192, 32768], 98304)
    assert speed_depths(dict(cfg, bench_deep_fill_pct=0), 131072) == ([0, 8192, 32768], None)


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
    rec = recommend(cfg_for("balanced"), settings, acc)
    assert rec["recommended"]["context"] == 131072
    assert rec["target_met"]
    assert rec["verified"]
    rec = recommend(cfg_for("strict"), settings, acc)
    assert rec["recommended"]["context"] < 131072
    assert not rec["target_met"]


def test_long_profile_checks_only_50_percent():
    acc = [fill(196608, 50, "4/5"), fill(196608, 75, "2/5")]
    rec = recommend(cfg_for("long"), _three(), acc)
    assert rec["recommended"]["context"] == 196608


def test_speed_only_when_accuracy_was_not_run():
    rec = recommend(cfg_for("balanced"), _three(), [])
    assert rec["recommended"]["context"] == 196608
    assert not rec["verified"]
    assert rec["target_met"]


def test_speed_rule_removes_slow_contexts():
    settings = _three(tg=(52, 50, 30))  # 196K is 42% below the best
    rec = recommend(cfg_for("balanced"), settings, [])
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
    rec = recommend(cfg_for("balanced"), _three(), acc)
    assert rec["recommended"]["context"] == 131072


def test_nothing_passes():
    acc = [fill(c, 50, "1/5", status="MISS") for c in (65536, 131072, 196608)]
    rec = recommend(cfg_for("balanced"), _three(), acc)
    assert not rec["target_met"]
    assert not rec["per_cache"]["q8_0"]["reliable"]
    assert rec["recommended"]["context"] == 65536  # smallest: nothing is known to be good
    assert "long-prompt test failed" in why_below_target(rec)


def test_fallback_is_the_largest_context_not_known_to_fail():
    acc = [fill(196608, 50, "1/5"), fill(131072, 50, "1/5")]  # 65536 was never tested
    rec = recommend(cfg_for("balanced"), _three(), acc)
    assert rec["recommended"]["context"] == 65536
    assert not rec["verified"]
    assert "not run" in why_below_target(dict(rec, target_context=65536))


def test_nothing_fits_the_target():
    rec = recommend(cfg_for("balanced"), [setting(32768), setting(65536)], [])
    assert not rec["target_met"]
    text = why_below_target(rec)
    assert "fits in memory" in text
    assert "q4_0" in text


def test_cache_choice_prefers_quality_when_speed_is_close():
    settings = [
        setting(131072, "q8_0", tg0=50, deep=(65536, 30)),
        setting(131072, "f16", tg0=50, deep=(65536, 28)),
    ]
    assert recommend(cfg_for(), settings, [])["choice"] == "f16"


def test_cache_choice_prefers_speed_when_quality_costs_too_much():
    settings = [
        setting(131072, "q8_0", tg0=50, deep=(65536, 30)),
        setting(131072, "f16", tg0=50, deep=(65536, 20)),
    ]  # f16 below 85% of 30
    assert recommend(cfg_for(), settings, [])["choice"] == "q8_0"


def test_cache_choice_without_target_takes_largest_context():
    settings = [setting(65536, "f16", tg0=50), setting(98304, "q4_0", tg0=50)]
    rec = recommend(cfg_for(), settings, [])
    assert (rec["choice"], rec["recommended"]["context"]) == ("q4_0", 98304)


def test_final_setting_keeps_the_tuned_values():
    s = {
        "settings": _three(),
        "accuracy": [],
        "type": "moe",
        "tuned": {"ngl": 41, "ncmoe": 20, "cache": "q8_0", "ub": 1024, "threads": 8},
    }
    fs = final_setting(cfg_for(), s)
    assert fs.context == 196608
    assert fs.setting["ub"] == 1024
    assert final_setting(cfg_for(), {"settings": []}) is None

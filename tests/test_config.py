import pytest

from benchmarker.config import (
    DEFAULTS, PROFILES, merge_config, check_config, override_lines,
)


def cfg_for(profile="balanced", **over):
    return merge_config(over, profile)


# ─── Profile Tests ───────────────────────────────────────────────────────────


def test_all_profiles_define_the_same_keys():
    keys = {frozenset(p) for p in PROFILES.values()}
    assert len(keys) == 1


def test_default_profile_exists():
    assert DEFAULTS["profile"] in PROFILES


def test_precedence_defaults_profile_user():
    cfg = merge_config({"target_context": 200000, "backend": "vulkan"}, "balanced")
    assert cfg["target_context"] == 200000  # user beats profile
    assert cfg["speed_tolerance_pct"] == PROFILES["balanced"]["speed_tolerance_pct"]
    assert cfg["backend"] == "vulkan"  # user beats DEFAULTS
    assert cfg["bench_reps"] == DEFAULTS["bench_reps"]


def test_profile_from_user_table():
    assert merge_config({"profile": "long"})["profile"] == "long"
    assert merge_config({"profile": "long"}, "strict")["profile"] == "strict"  # option wins


def test_override_is_reported():
    cfg = merge_config({"speed_tolerance_pct": 10}, "balanced")
    assert cfg["_overrides"] == {"speed_tolerance_pct": (10, 25)}
    assert len(override_lines(cfg)) == 1
    assert merge_config({"speed_tolerance_pct": 25}, "balanced")["_overrides"] == {}


def test_unknown_profile_stops():
    with pytest.raises(SystemExit):
        merge_config({}, "fast")


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
        check_config(cfg_for(**{key: bad}))

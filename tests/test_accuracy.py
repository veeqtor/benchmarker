"""Tests for accuracy/needle functions — imported from benchmarker.accuracy.

These tests exercise the needle-in-haystack logic: haystack, needle_positions,
needle_question, score_answer, required_hits.
"""

from benchmarker.accuracy import (
    haystack,
    needle_positions,
    needle_question,
    score_answer,
)
from benchmarker.config import merge_config, required_hits


def cfg_for(profile="balanced", **over):
    return merge_config(over, profile)


# ─── Needle Tests ────────────────────────────────────────────────────────────


def test_haystack_is_repeatable_and_keys_are_unique():
    a, b = haystack("keyed", 3000, seed=7), haystack("keyed", 3000, seed=7)
    assert a == b
    assert len({x[0] for x in a}) == 3000
    assert a != haystack("keyed", 3000, seed=8)


def test_line_style_is_the_old_format():
    items = haystack("line", 10, seed=1)
    assert items[3][2] == "Line 3: the value is %d." % ((3 * 7919 + 13) % 100000)


def test_positions_are_spread():
    assert needle_positions(1000, 3) == [100, 500, 900]
    assert needle_positions(1000, 5) == [100, 300, 500, 700, 900]
    assert needle_positions(1000, 1) == [500]


def test_question_text():
    assert needle_question("line", ["1", "5", "9"]) == (
        "What are the values on line 1, line 5 and line 9? "
        "Answer only in this form: A=<number>, B=<number>, C=<number>."
    )
    q = needle_question("keyed", ["kelmora", "tuvrin"])
    assert "project kelmora and project tuvrin" in q
    assert "A=<number>, B=<number>" in q


def test_score_answer():
    assert score_answer("A=11111, B=22222, C=33333", [11111, 22222, 33333]) == (3, [])
    assert score_answer("A=11111, B=22223, C=33333", [11111, 22222, 33333]) == (2, [1])
    # the letter must match: the right number under the wrong letter is wrong
    assert score_answer("A=22222, B=11111", [11111, 22222])[0] == 0
    # no letters: the number anywhere counts
    assert score_answer("The values are 11111 and 22222.", [11111, 22222])[0] == 2
    assert score_answer("", [1, 2])[0] == 0


def test_required_hits():
    assert required_hits(cfg_for("balanced"), 5) == 4
    assert required_hits(cfg_for("strict"), 3) == 3
    assert required_hits(cfg_for("balanced", accuracy_min_pct=1), 5) == 1

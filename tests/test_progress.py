import json
import os
import tempfile

import pytest

import benchmarker.config as cfg_mod
from benchmarker.commands import status_text
from benchmarker.progress import Progress, fmt_dur


@pytest.fixture
def progress_setup():
    """Fixture that mimics the old setUp/tearDown pattern."""
    now = [1000.0]  # mutable container so the clock lambda sees updates
    p = Progress(clock=lambda: now[0])
    old_results = cfg_mod.RESULTS
    tmp = tempfile.mkdtemp()
    cfg_mod.RESULTS = tmp
    yield p, tmp, now
    cfg_mod.RESULTS = old_results
    p.end()


def test_fmt_dur():
    assert fmt_dur(5) == "0:05"
    assert fmt_dur(125) == "2:05"
    assert fmt_dur(3725) == "1h02m"
    assert fmt_dur(-3) == "0:00"


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
    text = status_text(data, now_ts=data["updated"] + 5)
    assert "state: running" in text
    assert "accuracy 1/3" in text
    assert "WARNING" not in text  # this process is alive
    data["pid"] = 2**22 + 12345  # no such process
    assert "process is gone" in status_text(data)
    p.end("stopped")
    with open(os.path.join(tmp, "progress.json")) as f:
        assert json.load(f)["state"] == "stopped"


def test_old_run_is_not_reported_as_dead():
    data = {"state": "finished", "pid": 2**22 + 12345, "updated": 0, "label": "x"}
    assert "WARNING" not in status_text(data, now_ts=99999)

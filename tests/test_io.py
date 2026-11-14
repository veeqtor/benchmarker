"""Production-critical tests for io.py."""

import json
from unittest.mock import patch

import pytest

from benchmarker.io import append_csv, load_states, read_csv, save_state, state_path


@pytest.fixture
def tmp_results(tmp_path):
    """Provide a temporary RESULTS directory."""
    with patch("benchmarker.io.RESULTS", str(tmp_path)):
        with patch("benchmarker.config.RESULTS", str(tmp_path)):
            yield tmp_path


class TestAppendCsv:
    def test_creates_file_with_headers(self, tmp_results):
        append_csv("test.csv", {"a": "1", "b": "2"})
        path = tmp_results / "test.csv"
        assert path.exists()
        content = path.read_text()
        assert "a" in content
        assert "b" in content
        assert "1" in content

    def test_appends_row(self, tmp_results):
        append_csv("test.csv", {"a": "1"})
        append_csv("test.csv", {"a": "2"})
        rows = read_csv("test.csv")
        assert len(rows) == 2
        assert rows[0]["a"] == "1"
        assert rows[1]["a"] == "2"

    def test_renames_on_format_mismatch(self, tmp_results):
        append_csv("test.csv", {"a": "1"})
        append_csv("test.csv", {"b": "2"})  # different fields
        files = list(tmp_results.glob("*.csv"))
        old_files = list(tmp_results.glob("*.old-*.csv"))
        assert len(old_files) >= 1  # old format renamed


class TestReadCsv:
    def test_empty_file(self):
        assert read_csv("nonexistent_file.csv") == []


class TestStatePath:
    def test_returns_correct_path(self, tmp_results):
        with patch("benchmarker.io.RESULTS", str(tmp_results)):
            path = state_path("my-model", "rocm")
            assert "my-model" in path
            assert "rocm" in path


class TestSaveState:
    def test_saves_and_loads(self, tmp_results):
        state = {"model": "test", "backend": "rocm", "run_id": 123}
        with patch("benchmarker.io.RESULTS", str(tmp_results)):
            save_state(state)
            path = tmp_results / "state-test@rocm.json"
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["run_id"] == 123


class TestLoadStates:
    def test_loads_all(self, tmp_results):
        state1 = {"model": "m1", "backend": "rocm", "run_id": 1}
        state2 = {"model": "m2", "backend": "cuda", "run_id": 2}
        with patch("benchmarker.io.RESULTS", str(tmp_results)):
            save_state(state1)
            save_state(state2)
            states = load_states()
            assert len(states) == 2

    def test_filters_by_backend(self, tmp_results):
        state1 = {"model": "m1", "backend": "rocm", "run_id": 1}
        state2 = {"model": "m2", "backend": "cuda", "run_id": 2}
        with patch("benchmarker.io.RESULTS", str(tmp_results)):
            save_state(state1)
            save_state(state2)
            states = load_states("rocm")
            assert len(states) == 1
            assert "m1" in states or "m1@rocm" in states

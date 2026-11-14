"""Production-critical tests for bench.py."""

import json
from unittest.mock import MagicMock, patch

from benchmarker.bench import bench_cached, run_bench, speed_depths


class TestRunBench:
    def test_parses_jsonl_output(self):
        mock_result = MagicMock()
        mock_result.stdout = (
            json.dumps({"n_prompt": 512, "n_depth": 0, "avg_ts": 45.2, "stddev_ts": 1.1}) + "\n"
        )
        mock_result.stderr = ""
        with patch("benchmarker.bench.subprocess.run", return_value=mock_result):
            rows = run_bench(
                {"bench": "llama-bench", "backend": "rocm", "bench_reps": 1, "_env": None, "bench_fa": "1"},
                {"name": "test", "path": "/tmp/test.gguf"},
                {"ngl": 33, "cache": "q8_0"},
                [0],
            )
            assert len(rows) == 1
            assert rows[0]["avg_ts"] == 45.2

    def test_returns_empty_on_no_output(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "some error"
        with patch("benchmarker.bench.subprocess.run", return_value=mock_result):
            rows = run_bench(
                {"bench": "llama-bench", "backend": "rocm", "bench_reps": 1, "_env": None, "bench_fa": "1"},
                {"name": "test", "path": "/tmp/test.gguf"},
                {"ngl": 33, "cache": "q8_0"},
                [0],
            )
            assert rows == []


class TestSpeedDepths:
    def test_basic_depths(self):
        cfg = {
            "bench_depths": [0, 8192, 32768],
            "bench_deep_fill_pct": 50,
            "target_context": 131072,
        }
        depths, deep = speed_depths(cfg, 8192)
        assert 0 in depths  # depth 0 always included
        assert deep is None  # ctx not >= target_context

    def test_deep_depth_included(self):
        cfg = {
            "bench_depths": [512, 1024, 2048],
            "bench_deep_fill_pct": 50,
            "target_context": 4096,
        }
        depths, deep = speed_depths(cfg, 8192)
        assert deep is not None
        assert deep > max(depths)


class TestBenchCached:
    def test_caches_same_flags(self):
        call_count = 0

        def mock_run(*a, **kw):
            nonlocal call_count
            call_count += 1
            return []

        with patch("benchmarker.bench.run_bench", side_effect=mock_run):
            cfg = {"backend": "rocm", "_env": None, "bench_reps": 1, "bench_fa": "1"}
            m = {"name": "test", "path": "/tmp/test.gguf"}
            s = {"ngl": 33, "cache": "q8_0"}
            done = {}
            bench_cached(done, cfg, m, s, [0], reps=1, prompt=512)
            bench_cached(done, cfg, m, s, [0], reps=1, prompt=512)  # same key
            assert call_count == 1  # only called once

    def test_different_keys_not_cached(self):
        call_count = 0

        def mock_run(*a, **kw):
            nonlocal call_count
            call_count += 1
            return []

        with patch("benchmarker.bench.run_bench", side_effect=mock_run):
            cfg = {"backend": "rocm", "_env": None, "bench_reps": 1, "bench_fa": "1"}
            m = {"name": "test", "path": "/tmp/test.gguf"}
            s = {"ngl": 33, "cache": "q8_0"}
            done = {}
            bench_cached(done, cfg, m, s, [0], reps=1, prompt=512)
            bench_cached(done, cfg, m, s, [0], reps=2, prompt=512)  # different reps
            assert call_count == 2  # called twice

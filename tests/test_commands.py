"""Production-critical tests for commands.py."""

import argparse
from unittest.mock import patch

import pytest

from benchmarker.commands import cmd_doctor


@pytest.fixture
def mock_args():
    return argparse.Namespace(
        config=None,
        results=None,
        profile=None,
        llama_dir=None,
        models_dir=None,
        backend=None,
        swap_url=None,
        server=None,
        bench=None,
        visible_devices=None,
        gpu_sysfs=None,
        swap_config=None,
        swap_api_keys=None,
        key_env=None,
        key_file=None,
        swap_health_timeout=None,
        swap_ttl=None,
        swap_start_port=None,
        port=None,
        common_flags=None,
        load_timeout_s=None,
        request_timeout_s=None,
        min_free_vram_mb=None,
        min_free_ram_mb=None,
        swap_service=None,
        swap_api_keys_file=None,
        no_key=False,
        install=False,
        force=False,
    )


class TestCmdDoctor:
    def test_no_crash_with_empty_config(self, mock_args):
        """cmd_doctor should not crash when no models.toml exists."""
        with patch("benchmarker.commands.read_toml", return_value={}):
            with patch("benchmarker.commands.base_config") as mock_base:
                mock_base.return_value = {
                    "llama_dir": "",
                    "server": "",
                    "bench": "",
                    "models_dir": "/tmp/nonexistent",
                    "backend": "rocm",
                    "swap_service": "llama-swap",
                    "swap_url": "http://127.0.0.1:9931",
                    "key_env": "LLAMA_SWAP_API_KEY",
                    "key_file": "/etc/llama-swap.env",
                    "swap_config": "/etc/llama-swap.yaml",
                    "swap_api_keys": True,
                    "swap_health_timeout": 600,
                    "swap_ttl": 1800,
                    "swap_start_port": 10001,
                    "port": 0,
                    "common_flags": [],
                    "load_timeout_s": 900,
                    "request_timeout_s": 7200,
                    "min_free_vram_mb": 512,
                    "min_free_ram_mb": 4096,
                    "needles": 5,
                    "needle_style": "keyed",
                    "accuracy_pct": [50, 75],
                    "accuracy_min_pct": 80,
                    "target_context": 131072,
                    "speed_tolerance_pct": 25,
                    "min_tg_tps": 15,
                    "fill_pct": [25, 50, 75, 90],
                    "accuracy_max_steps": 4,
                    "vision_runs": 3,
                    "bench_depths": [0, 8192, 32768],
                    "bench_deep_fill_pct": 50,
                    "bench_reps": 2,
                    "bench_fa": "1",
                    "tune_ubatch": [256, 512, 1024, 2048],
                    "moe_contexts": [32768, 65536],
                    "dense_contexts": [16384, 32768],
                    "moe_cache_types": ["q8_0"],
                    "dense_cache_types": ["q8_0", "q4_0"],
                    "fit_warmup": True,
                    "image_fit": True,
                    "image_px": 896,
                    "profile": "balanced",
                    "visible_devices": "auto",
                    "gpu_sysfs": "",
                }
                with patch("benchmarker.commands.ram_free_mb", return_value=8192):
                    with patch("benchmarker.commands.physical_cores", return_value=8):
                        with patch("benchmarker.commands.run_text", return_value=""):
                            with patch("benchmarker.commands.service_active", return_value=False):
                                with patch("benchmarker.commands.other_servers", return_value=[]):
                                    # Should not raise
                                    cmd_doctor(mock_args)

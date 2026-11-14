"""Production-critical tests for swap.py."""

import tempfile

from benchmarker.swap import swap_models_in


class TestSwapModelsIn:
    def test_parses_model_names(self):
        config = """
models:
  "my-model":
    cmd: |
      --model /path/to/model.gguf
  "other-model":
    cmd: |
      --model /path/to/other.gguf
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config)
            f.flush()
            models = swap_models_in(f.name)
            assert "my-model" in models
            assert "other-model" in models

    def test_returns_empty_on_missing_file(self):
        models = swap_models_in("/nonexistent/config.yaml")
        assert models == []

    def test_returns_empty_on_no_models_section(self):
        config = """
healthCheckTimeout: 600
logLevel: info
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config)
            f.flush()
            models = swap_models_in(f.name)
            assert models == []

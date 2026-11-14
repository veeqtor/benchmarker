"""Production-critical tests for gpu.py."""

from benchmarker.gpu import parse_devices


class TestParseDevices:
    def test_parses_single_gpu(self):
        text = "AMDGPU0: 7900 XTX (48 MiB)"
        devs = parse_devices(text)
        assert len(devs) == 1
        assert devs[0]["kind"] == "AMDGPU"
        assert devs[0]["index"] == 0
        assert devs[0]["mib"] == 48

    def test_parses_multiple_gpus(self):
        text = "AMDGPU0: 7900 XTX (48 MiB)\nAMDGPU1: Vega 56 (8 MiB)"
        devs = parse_devices(text)
        assert len(devs) == 2
        assert devs[0]["index"] == 0
        assert devs[1]["index"] == 1

    def test_skips_cpu(self):
        text = "AMDGPU0: 7900 XTX (48 MiB)\nCPU"
        devs = parse_devices(text)
        assert len(devs) == 1
        assert devs[0]["kind"] == "AMDGPU"

    def test_empty(self):
        assert parse_devices("") == []

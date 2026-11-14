"""Production-critical tests for utils.py."""

import socket
from unittest.mock import mock_open, patch

from benchmarker.utils import build_id, free_port, port_free, run_text


class TestFmtDur:
    def test_zero(self):
        assert run_text(["echo", "hello"]) == "hello\n"

    def test_mixed(self):
        from benchmarker.utils import fmt_dur

        assert fmt_dur(3725) == "1h02m"

    def test_negative(self):
        from benchmarker.utils import fmt_dur

        assert fmt_dur(-10) == "0:00"


class TestPort:
    def test_free_port_is_free(self):
        port = free_port()
        assert port_free(port) is True

    def test_port_free_busy(self):
        port = free_port()
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
            assert port_free(port) is False


class TestBuildId:
    def test_parse_version(self):
        cfg = {"server": "echo", "_env": None}
        with patch("benchmarker.utils.run_text", return_value="llama.cpp version: abc1234\n"):
            assert build_id(cfg) == "abc1234"

    def test_missing_version(self):
        cfg = {"server": "echo", "_env": None}
        with patch("benchmarker.utils.run_text", return_value="no version here"):
            assert build_id(cfg) == "?"


class TestRunText:
    def test_command_not_found(self):
        result = run_text(["nonexistent_command_xyz"])
        assert "ERROR:" in result


class TestApiKey:
    def test_from_env(self):
        from benchmarker.utils import api_key

        cfg = {"key_env": "TEST_API_KEY", "key_file": "/tmp/should_not_read"}
        with patch.dict(__import__("os").environ, {"TEST_API_KEY": "mykey123"}):
            assert api_key(cfg, required=False) == "mykey123"

    def test_from_file_no_equals(self):
        from benchmarker.utils import api_key

        cfg = {"key_env": "TEST_API_KEY", "key_file": "/tmp/test_key_file"}
        with patch.dict(__import__("os").environ, clear=True):
            with patch("builtins.open", mock_open(read_data="secretkey123")):
                assert api_key(cfg, required=False) == "secretkey123"


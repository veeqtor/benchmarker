"""Production-critical tests for lock.py."""

from unittest.mock import mock_open, patch

import pytest

from benchmarker.lock import Lock


@pytest.fixture
def tmp_results(tmp_path):
    with patch("benchmarker.lock.RESULTS", str(tmp_path)):
        yield tmp_path


class TestLock:
    def test_context_manager_creates_directory(self, tmp_results):
        lock_dir = tmp_results / "locktest"
        with patch("benchmarker.lock.RESULTS", str(lock_dir)):
            m = mock_open()
            with patch("builtins.open", m), patch("fcntl.flock"):
                lock = Lock()
                lock.__enter__()
                assert lock_dir.exists()

    def test_context_manager_exits_cleanly(self, tmp_results):
        with patch("benchmarker.lock.RESULTS", str(tmp_results)):
            m = mock_open()
            with patch("builtins.open", m), patch("fcntl.flock"), Lock():
                pass  # should exit cleanly

    def test_double_lock_raises(self, tmp_results):
        """Two locks on same file should not both acquire."""
        with patch("benchmarker.lock.RESULTS", str(tmp_results)):
            f = open(tmp_results / ".lock", "w")
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(SystemExit), Lock():
                pass
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

"""File-based lock to prevent concurrent sweep runs."""

import fcntl
import os
import sys

# Results directory for lock file
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


class Lock:
    """Context manager that acquires an exclusive file lock to prevent concurrent sweeps.

    Usage::

        with Lock():
            # only one sweep can be running at a time
            run_sweep()
    """

    def __enter__(self):
        os.makedirs(RESULTS, exist_ok=True)
        self.f = open(os.path.join(RESULTS, ".lock"), "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("Another benchmarker sweep is running.")
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


__all__ = ["Lock"]

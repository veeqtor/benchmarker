"""Progress tracking for benchmark runs.

Provides the Progress class for tracking run state, writing progress.json,
and helpers for logging and warnings.
"""

import contextlib
import json
import os
import sys
import threading
import time

from benchmarker import config
from benchmarker.utils import fmt_dur


def log(msg=""):
    """Print a log message with the current progress tag."""
    if PROGRESS.active and msg.strip() and not msg.lstrip().startswith("==="):
        msg = f"{PROGRESS.tag()} {msg}"
    PROGRESS.last_output = PROGRESS.clock()
    print(msg, flush=True)


def warn(msg):
    """Print a warning message to stderr."""
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


class Progress:
    """Track run progress: model x of y, phase, step, elapsed time.

    Every log line gets a short tag, a banner starts each model and phase,
    a heartbeat line appears when nothing was printed for a minute, and
    results/progress.json is updated for the 'status' command.
    All methods do nothing until begin() is called.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self.active = False
        self.last_output = 0.0
        self._stop = threading.Event()

    def begin(self, kind, models):
        self.active = True
        self.kind = kind
        self.models = list(models)
        self.t0 = self.model_t0 = self.last_output = self.clock()
        self.done_secs, self.mi, self.model, self._open = [], 0, "", False
        self.phase_name, self.step, self.total, self.detail, self.state = "", 0, 0, "", "running"
        self._write()

    def start_model(self, name):
        if not self.active:
            return
        self.finish_model()
        self.mi, self.model, self.model_t0, self._open = self.mi + 1, name, self.clock(), True
        self.phase_name, self.step, self.total, self.detail = "", 0, 0, ""
        self._write()

    def finish_model(self):
        if self._open:
            self.done_secs.append(self.clock() - self.model_t0)
            self._open = False

    def phase(self, name, total=0, detail=""):
        if not self.active:
            return
        self.phase_name, self.step, self.total, self.detail = name, 0, total, detail
        self.say(f"---- {self.label()} ----")
        self._write()

    def tick(self, detail=""):
        if not self.active:
            return
        self.step += 1
        self.detail = detail or self.detail
        self._write()

    def note(self, detail):
        if not self.active:
            return
        self.detail = detail
        self._write()

    def step_text(self):
        if self.total:
            return f"{self.step}/{self.total}"
        return f"#{self.step}" if self.step else ""

    def eta_rest_s(self):
        """Seconds for remaining models, estimated from models finished so far (None = unknown)."""
        left = len(self.models) - self.mi
        if not self.done_secs or left <= 0:
            return None
        return sum(self.done_secs) / len(self.done_secs) * left

    def tag(self):
        parts = [
            f"{self.mi}/{len(self.models)}",
            self.phase_name,
            self.step_text(),
            f"+{fmt_dur(self.clock() - self.t0)}",
        ]
        return "[" + " ".join(p for p in parts if p) + "]"

    def label(self):
        eta = self.eta_rest_s()
        parts = [
            f"{self.kind} model {self.mi}/{len(self.models)}: {self.model}",
            f"{self.phase_name} {self.step_text()}".strip(),
            self.detail,
            f"run {fmt_dur(self.clock() - self.t0)}",
            f"model {fmt_dur(self.clock() - self.model_t0)}",
        ]
        if eta is not None:
            parts.append(f"later models ~{fmt_dur(eta)}")
        return " | ".join(p for p in parts if p)

    def say(self, text):
        self.last_output = self.clock()
        print(text, flush=True)

    def _write(self):
        """Write progress.json for 'benchmarker status' in a second terminal."""
        if not self.active:
            return
        try:
            os.makedirs(config.RESULTS, exist_ok=True)
            data = {
                "run_id": config.RUN_ID,
                "kind": self.kind,
                "state": self.state,
                "pid": os.getpid(),
                "model": self.model,
                "model_index": self.mi,
                "model_total": len(self.models),
                "models": self.models,
                "phase": self.phase_name,
                "step": self.step,
                "total": self.total,
                "detail": self.detail,
                "elapsed_s": round(self.clock() - self.t0),
                "model_elapsed_s": round(self.clock() - self.model_t0),
                "eta_rest_s": None if self.eta_rest_s() is None else round(self.eta_rest_s()),
                "updated": time.time(),
                "label": self.label(),
            }
            path = os.path.join(config.RESULTS, "progress.json")
            with open(path + ".tmp", "w") as f:
                json.dump(data, f, indent=2)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def start_heartbeat(self, every=60):
        """Print one line when nothing was printed for 'every' seconds."""

        def loop():
            while not self._stop.wait(min(10, every)):
                if self.active and self.clock() - self.last_output >= every:
                    self.say(f"  ... still working: {self.label()}")
                    self._write()

        self._stop.clear()
        threading.Thread(target=loop, daemon=True).start()

    def end(self, state="finished"):
        if self.active:
            self.finish_model()
            self.state = self.phase_name = self.step = self.total = state
            self.detail = f"{self.mi} of {len(self.models)} models"
            self._write()
            self.say(f"---- {self.kind} {state} after {fmt_dur(self.clock() - self.t0)} ----")
        self._stop.set()
        self.active = False


PROGRESS = Progress()


@contextlib.contextmanager
def tracked(kind, names):
    """Context manager for a command run: shows and records progress.

    The final state is 'finished', 'stopped', or 'failed'.
    """
    PROGRESS.begin(kind, names)
    PROGRESS.start_heartbeat()
    state = "failed"
    try:
        yield
        state = "finished"
    except KeyboardInterrupt:
        state = "stopped"
        raise
    finally:
        PROGRESS.end(state)


__all__ = [
    "PROGRESS",
    "Progress",
    "log",
    "tracked",
    "warn",
]

"""I/O utilities for CSV data and state persistence.

Handles appending/reading CSV files in the results directory,
and atomic save/load of JSON state files.
"""

import csv
import glob
import json
import os

from benchmarker.config import RESULTS, RUN_ID


def append_csv(name, row):
    """Append a row to a CSV file in the results directory.

    Creates the file with headers if it doesn't exist. Renames old files
    on format mismatch.
    """
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fields = list(row.keys())
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:  # older format: keep it, start a new file
            os.rename(path, path[:-4] + f".old-{RUN_ID}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def read_csv(name):
    """Read all rows from a CSV file in the results directory."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def state_path(model, backend):
    """Return the path to a state JSON file for the given model and backend."""
    return os.path.join(RESULTS, f"state-{model}@{backend}.json")


def save_state(state):
    """Atomically save a state dict to a JSON file in the results directory."""
    os.makedirs(RESULTS, exist_ok=True)
    tmp = state_path(state["model"], state["backend"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path(state["model"], state["backend"]))


def load_states(backend=None):
    """Load all state files, optionally filtered by backend.

    Returns a dict keyed by model name (or model@backend).
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "state-*.json"))):
        with open(p) as f:
            s = json.load(f)
        s.setdefault("backend", "rocm")  # files from version 1
        if backend and s["backend"] != backend:
            continue
        key = s["model"] if backend else f"{s['model']}@{s['backend']}"
        if key not in out or s["run_id"] > out[key]["run_id"]:
            out[key] = s
    return out


__all__ = [
    "append_csv",
    "load_states",
    "read_csv",
    "save_state",
    "state_path",
]

"""GPU & system detection for AMD GPUs on Linux.

Parses --list-devices output, reads VRAM from sysfs, detects integrated vs
discrete GPUs, and sets device visibility environment variables.
"""

import glob
import os
import re
import sys
import time

from benchmarker.config import DEVICE_ENV
from benchmarker.progress import log, warn
from benchmarker.utils import run_text

# Regex matching integrated GPU names (skip these when picking the main GPU)
IGPU_NAME = re.compile(
    r"Radeon( \d{3,4}M| Graphics| \d{4}S)?$|Radeon\(TM\) Graphics|Vega \d+$", re.IGNORECASE
)


def parse_devices(text):
    """Parse the --list-devices output from llama-server into a list of GPU dicts."""
    devs = []
    for m in re.finditer(r"^\s*([A-Za-z]+)(\d+):\s*(.+?)\s*\((\d+) MiB", text, re.MULTILINE):
        devs.append(
            {
                "kind": m.group(1),
                "index": int(m.group(2)),
                "name": m.group(3).strip(),
                "mib": int(m.group(4)),
            }
        )
    return devs


def pick_main_gpu(cfg, gpus):
    """Select the dedicated GPU. Integrated GPUs report shared RAM, so their size is not useful.

    Uses the dedicated VRAM size from sysfs and skips typical iGPU names.
    """
    try:
        total, _, _ = vram_mb(cfg)
    except SystemExit:
        total = 0
    if total:
        near = [d for d in gpus if abs(d["mib"] - total) <= max(1024, total * 0.05)]
        if len(near) == 1:
            return near[0]
    dgpu = [d for d in gpus if not IGPU_NAME.search(d["name"])]
    return max(dgpu or gpus, key=lambda d: d["mib"])


def setup_devices(cfg):
    """Make only the wanted GPU visible to llama.cpp. Sets cfg['_env'], cfg['_devenv'], cfg['_devices']."""
    env = dict(os.environ)
    var = DEVICE_ENV.get(cfg["backend"].lower(), "")
    devenv = {}
    for v in set(DEVICE_ENV.values()) | {"ROCR_VISIBLE_DEVICES"}:
        if os.environ.get(v):
            devenv[v] = os.environ[v]
    vis = str(cfg["visible_devices"])
    if var and vis not in ("auto", "") and var not in devenv:
        devenv[var] = vis
        env[var] = vis
    devs = parse_devices(run_text([cfg["server"], "--list-devices"], env)) if cfg["server"] else []
    gpus = [d for d in devs if d["kind"].lower() != "cpu"]
    if var and vis == "auto" and var not in devenv and len(gpus) > 1:
        best = pick_main_gpu(cfg, gpus)
        env[var] = devenv[var] = str(best["index"])
        if not cfg.get("_quiet_devices"):
            log(f"{len(gpus)} GPUs found. Using only {best['name']} ({var}={best['index']}).")
        devs = parse_devices(run_text([cfg["server"], "--list-devices"], env))
        gpus = [d for d in devs if d["kind"].lower() != "cpu"]
    if len(gpus) > 1:
        warn(
            f"{len(gpus)} GPUs are visible: {', '.join(d['name'] for d in gpus)}. "
            f"Set visible_devices in models.toml or {var or 'the device variable'}."
        )
    cfg["_env"] = env
    cfg["_devenv"] = devenv
    cfg["_devices"] = gpus
    return gpus


def gpu_dir(cfg):
    """Find the AMD GPU sysfs directory (the one with the most VRAM)."""
    if cfg["gpu_sysfs"]:
        return cfg["gpu_sysfs"]
    best = None
    for d in glob.glob("/sys/class/drm/card*/device"):
        p = os.path.join(d, "mem_info_vram_total")
        if os.path.exists(p):
            total = int(open(p).read())
            if not best or total > best[0]:
                best = (total, d)
    if not best:
        sys.exit("No AMD GPU found in /sys/class/drm (set gpu_sysfs in models.toml)")
    cfg["gpu_sysfs"] = best[1]
    return best[1]


def vram_mb(cfg):
    """Read VRAM total/used/free from AMD GPU sysfs (in MiB)."""
    d = gpu_dir(cfg)
    total = int(open(os.path.join(d, "mem_info_vram_total")).read()) // 1048576
    used = int(open(os.path.join(d, "mem_info_vram_used")).read()) // 1048576
    return total, used, total - used


def ram_free_mb():
    """Read available RAM from /proc/meminfo (in MiB)."""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def physical_cores():
    """Count physical CPU cores from /proc/cpuinfo."""
    cores, phys = set(), None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                cores.add((phys, line.split(":")[1].strip()))
    except OSError:
        pass
    return len(cores) or max(1, (os.cpu_count() or 2) // 2)


def wait_vram_released(cfg, baseline_used, timeout=30):
    """Wait for VRAM to return to baseline_used + 256 MiB, or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, used, _ = vram_mb(cfg)
        if used <= baseline_used + 256:
            return
        time.sleep(0.5)
    warn(f"VRAM not released after {timeout}s (used {used} MiB, before {baseline_used} MiB)")


__all__ = [
    "IGPU_NAME",
    "gpu_dir",
    "parse_devices",
    "physical_cores",
    "pick_main_gpu",
    "ram_free_mb",
    "setup_devices",
    "vram_mb",
]

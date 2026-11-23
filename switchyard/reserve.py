"""System reserve: VRAM and host RAM kept back for the desktop and the OS.

Pure helpers; the arbiter decides what to do with the numbers. Linux only for RAM
(/proc/meminfo, /proc/pressure/memory). Both procfs files are host-wide, so they
read the same inside a container as on the host. Elsewhere they return None and
the RAM guard does nothing.
"""
from __future__ import annotations


def meminfo(path: str = "/proc/meminfo") -> dict | None:
    """{"total_mb", "available_mb"} from /proc/meminfo, or None if unreadable."""
    try:
        info = {}
        with open(path, encoding="ascii") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        if total is None or avail is None:
            return None
        return {"total_mb": total // 1024, "available_mb": avail // 1024}
    except Exception:
        return None


def memory_pressure(path: str = "/proc/pressure/memory") -> dict | None:
    """PSI for memory: {"some_avg10", "full_avg10"} (percent of time stalled), or None.

    `full` is the share of time every runnable task was stalled on memory: that is
    the stutter you feel on the desktop.
    """
    try:
        out = {}
        with open(path, encoding="ascii") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                fields = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
                if "avg10" in fields:
                    out[f"{parts[0]}_avg10"] = float(fields["avg10"])
        return out or None
    except Exception:
        return None


def level1_fits(available_mb: int | None, backup_mb: int, reserve_mb: int) -> bool:
    """Would backing `backup_mb` of weights up to RAM (sleep level 1) keep the reserve?

    Unknown RAM or no reserve -> True (keep the configured level).
    """
    if not reserve_mb or available_mb is None:
        return True
    return available_mb - int(backup_mb or 0) >= reserve_mb


def ram_low(available_mb: int | None, reserve_mb: int) -> bool:
    return bool(reserve_mb) and available_mb is not None and available_mb < reserve_mb

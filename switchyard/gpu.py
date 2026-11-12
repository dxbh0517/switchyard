"""VRAM probe: NVML, then nvidia-smi, then ComfyUI /system_stats, then unknown.

pynvml / nvidia-ml-py is an optional extra. nvidia-smi and ComfyUI use the stdlib.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass

from .config import Settings

SOURCES = ("auto", "nvml", "nvidia-smi", "comfy", "off")
_AUTO = ("nvml", "nvidia-smi", "comfy")
_TTL_SEC = 1.0


@dataclass(frozen=True)
class GpuReading:
    source: str
    index: int
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    name: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _mb(nbytes: int) -> int:
    return round(nbytes / 2**20)


def _read_nvml(index: int) -> GpuReading | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
    except Exception:
        return None
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return GpuReading(
        source="nvml", index=index, name=name or None,
        total_mb=_mb(mem.total), used_mb=_mb(mem.used), free_mb=_mb(mem.free),
    )


def _read_smi(index: int) -> GpuReading | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-i", str(index),
             "--query-gpu=memory.total,memory.used,memory.free,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",", 3)]
    if len(parts) < 4:
        return None
    try:
        total, used, free = (int(float(parts[i])) for i in range(3))
    except ValueError:
        return None
    return GpuReading(source="nvidia-smi", index=index, total_mb=total, used_mb=used,
                      free_mb=free, name=parts[3] or None)


def _read_comfy(url: str, index: int) -> GpuReading | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=5) as r:
            data = json.load(r)
    except Exception:
        return None
    devs = data.get("devices") or []
    cuda = [d for d in devs if d.get("type") == "cuda"] or list(devs)
    if index >= len(cuda):
        return None
    d = cuda[index]
    try:
        total = _mb(d["vram_total"])
        free = _mb(d["vram_free"])
    except (KeyError, TypeError):
        return None
    name = re.sub(r"^cuda:\d+\s+", "", str(d.get("name") or "").split(" : ")[0]) or None
    return GpuReading(source="comfy", index=index, total_mb=total, used_mb=total - free,
                      free_mb=free, name=name)


class GpuProbe:
    def __init__(self, settings: Settings, ttl_sec: float = _TTL_SEC):
        self.s = settings
        self.ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._cached: tuple[float, tuple, GpuReading] | None = None

    def read(self) -> GpuReading:
        index = int(self.s.get("gpu_index"))
        source = str(self.s.get("gpu_probe"))
        key = (source, index)
        with self._lock:
            now = time.monotonic()
            if (self._cached is not None and self.ttl_sec > 0
                    and self._cached[1] == key and now - self._cached[0] < self.ttl_sec):
                return self._cached[2]
            reading = self._probe(source, index)
            self._cached = (now, key, reading)
            return reading

    def _probe(self, source: str, index: int) -> GpuReading:
        if source == "off":
            return GpuReading(source="off", index=index)
        order = _AUTO if source == "auto" else (source,)
        url = str(self.s.get("comfy_url") or "")
        for src in order:
            if src == "nvml":
                got = _read_nvml(index)
            elif src == "nvidia-smi":
                got = _read_smi(index)
            elif src == "comfy":
                got = _read_comfy(url, index)
            else:
                got = None
            if got is not None:
                return got
        return GpuReading(source="unknown", index=index)

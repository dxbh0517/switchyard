"""Shared activity log (stdout + an in-memory ring the dashboard reads) and counters."""
from __future__ import annotations

import collections
import itertools
import threading
import time

_ring: collections.deque = collections.deque(maxlen=500)
_ids = itertools.count(1)
_lock = threading.Lock()
STARTED = time.time()


def log(component: str, msg: str, level: str = "info") -> None:
    entry = {"id": next(_ids), "t": time.time(), "c": component, "level": level, "msg": msg}
    with _lock:
        _ring.append(entry)
    print(f"[{component}] {msg}", flush=True)


def since(last_id: int = 0, limit: int = 200) -> list[dict]:
    with _lock:
        items = [e for e in _ring if e["id"] > last_id]
    return items[-limit:]


class Counters:
    """Thread-safe named counters plus a few 'last value' gauges."""

    def __init__(self):
        self._lock = threading.Lock()
        self.data: dict = collections.defaultdict(float)

    def inc(self, key: str, by: float = 1) -> None:
        with self._lock:
            self.data[key] += by

    def set(self, key: str, value) -> None:
        with self._lock:
            self.data[key] = value

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.data)

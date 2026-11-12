"""GPU occupant backends: load/wake, unload/sleep, health, estimate_vram.

vLLM is the engine implementation; ComfyUI is the image-side occupant. Behavior of
the HTTP sequences is the same as when they lived on Arbiter.

To add another occupant (e.g. llama.cpp later): implement health, wake, sleep,
estimate_vram (and is_sleeping if the occupant has a sleep mode) and register it
in BACKENDS.
"""
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.request

from .events import log

C = "arbiter"


def _ok(status_raw, what: str) -> None:
    status, raw = status_raw
    if status != 200:
        raise RuntimeError(f"{what} -> HTTP {status}: {raw[:300]!r}")


class VllmBackend:
    """One vLLM OpenAI server with sleep mode (`VLLM_SERVER_DEV_MODE=1`)."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port

    def _simple(self, method, path, body=None, timeout=600.0) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json"} if body is not None else None
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def health(self) -> bool | None:
        sleeping = self.is_sleeping()
        return None if sleeping is None else True

    def is_sleeping(self) -> bool | None:
        try:
            status, raw = self._simple("GET", "/is_sleeping", timeout=10)
            if status != 200:
                return None
            data = json.loads(raw)
            return bool(data if isinstance(data, bool) else data.get("is_sleeping", False))
        except Exception:
            return None

    def wake(self, level: int = 2) -> None:
        """Load weights / leave sleep. Level 2 reloads weights from disk."""
        if level >= 2:
            _ok(self._simple("POST", "/wake_up?tags=weights"), "wake weights")
            _ok(self._simple("POST", "/collective_rpc",
                             json.dumps({"method": "reload_weights"}).encode(), timeout=900),
                "reload_weights")
            _ok(self._simple("POST", "/wake_up?tags=kv_cache"), "wake kv_cache")
            self._simple("POST", "/reset_prefix_cache", timeout=60)
        else:
            _ok(self._simple("POST", "/wake_up"), "wake_up")

    def sleep(self, level: int = 2) -> None:
        """Unload / enter vLLM sleep mode."""
        _ok(self._simple("POST", f"/sleep?level={level}", timeout=300), "sleep")

    def estimate_vram(self) -> int | None:
        return None  # vLLM does not report a footprint; the budget layer estimates


class ComfyBackend:
    """ComfyUI occupant: models load on the next prompt; unload is POST /free."""

    def __init__(self, settings):
        self.s = settings

    def url(self) -> str:
        return self.s.get("comfy_url").rstrip("/")

    def _get(self, path: str):
        with urllib.request.urlopen(f"{self.url()}{path}", timeout=5) as r:
            return json.load(r)

    def health(self) -> bool | None:
        try:
            self._get("/queue")
            return True
        except Exception:
            return None

    def wake(self, level: int = 2) -> None:  # noqa: ARG002
        return  # models load on demand

    def sleep(self, level: int = 2) -> None:  # noqa: ARG002
        self.unload(0)

    def estimate_vram(self) -> int | None:
        _, snap = self.vram()
        if not snap:
            return None
        return max(0, int(snap["total_mb"] - snap["free_mb"]))

    def vram(self) -> tuple[float | None, dict | None]:
        """(free MiB as float, {free_mb, total_mb, name}) from /system_stats."""
        try:
            devs = self._get("/system_stats").get("devices", [])
            cuda = [d for d in devs if d.get("type") == "cuda"] or devs
            if not cuda:
                return None, None
            free = cuda[0]["vram_free"] / 2**20
            snap = {
                "free_mb": round(free),
                "total_mb": round(cuda[0]["vram_total"] / 2**20),
                "name": re.sub(r"^cuda:\d+\s+", "", cuda[0].get("name", "").split(" : ")[0]),
            }
            return free, snap
        except Exception:
            return None, None

    def queue(self) -> tuple[bool | None, dict]:
        try:
            q = self._get("/queue")
            info = {"running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", []))}
            return bool(q.get("queue_running")), info
        except Exception:
            return None, {"running": None, "pending": None}

    def unload(self, target_mb: float, require_idle: bool = False,
               running=None, free_mb=None) -> None:
        """POST /free and wait until free_mb() >= target_mb. No-op if ComfyUI is down.

        `running` and `free_mb` are callables so the arbiter can keep its queue/gpu
        snapshots in sync while we wait.
        """
        running = running or (lambda: self.queue()[0])
        free_mb = free_mb or (lambda: self.vram()[0])
        if require_idle:
            while running():
                time.sleep(5)
        try:
            req = urllib.request.Request(
                f"{self.url()}/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            log(C, f"ComfyUI /free skipped: {e}")
            return
        deadline = time.time() + self.s.get("comfy_free_wait_sec")
        free, t = None, time.time()
        while time.time() < deadline:
            free = free_mb()
            if free is None or free >= target_mb:
                break
            time.sleep(0.25)
        log(C, f"ComfyUI unloaded in {time.time() - t:.1f}s (GPU free {free and round(free)} MiB, want {target_mb})")


# kind -> class. llama.cpp is not implemented; add it here when it is.
BACKENDS = {"vllm": VllmBackend, "comfy": ComfyBackend}

"""GPU arbiter: several vLLM engines and ComfyUI take turns on one NVIDIA GPU.

Default (`shared_vram=false`): exactly one of {each vLLM engine, ComfyUI} owns the
VRAM at a time. Engines that don't own it sleep (vLLM sleep mode; level 2 drops the
weights and leaves under 1 GiB each). Inference is serialized.

With `shared_vram=true`, several engines may stay awake while their VRAM costs fit
in the card budget minus a safety margin. Resident engines serve in parallel. A wake
that does not fit evicts least-recently-used residents, waiting only for those
victims' in-flight requests. ComfyUI still takes the whole GPU for a prompt.

Occupancy HTTP (sleep/wake/ComfyUI) runs outside the condition lock. `switching`
and `evicting` mark a transition so new requests wait without holding the lock
across the network.

  Engines : each engine gets an OpenAI-compatible listen port. After `idle_sleep_sec`
            without traffic an idle resident goes back to sleep.
  ComfyUI : the gpu_arbiter custom node (comfyui_node/) calls POST /gpu/acquire
            before each prompt and POST /gpu/release after it.
  Startup : an engine container calls POST /gpu/loading?engine=NAME before `vllm serve`.
            Engine loads are serialized; once an engine is up it is put to sleep.

  Health  : a background thread probes vLLM /health and ComfyUI /system_stats without
            holding the arbiter lock. Failed wakes retry with exponential backoff; an
            engine that stays down is marked unhealthy, ownership is released, and
            requests fail fast with HTTP 503 until a later probe recovers it.

GET /gpu/status on any engine port returns the state as JSON. Responses are streamed.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

from . import analytics
from .backends import ComfyBackend, VllmBackend
from .budget import lru_victims, resolve_cost, usable_mb
from .config import Settings
from .estimate import estimate_from_engine
from .events import Counters, log
from .gpu import GpuProbe
from .httputil import Handler, Server

INFER_PREFIXES = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses",
                  "/v1/audio", "/v1/messages", "/tokenize", "/detokenize")
IMAGE_PART_TYPES = {"image_url", "input_image", "image", "video_url", "input_video"}
C = "arbiter"


class EngineUnavailable(RuntimeError):
    """Engine is marked down; fail the request without waiting for the GPU."""

    def __init__(self, engine: str, health: dict, retry_after: int):
        self.engine = engine
        self.health = health["status"]
        self.retry_after = retry_after
        err = health.get("last_error")
        msg = f"{engine} is {health['status']}"
        if err:
            msg += f": {err}"
        super().__init__(msg)


def _blank_health() -> dict:
    return {"status": "unknown", "last_error": None, "consecutive_failures": 0,
            "checked_at": None, "seen_up": False}


def _health_view(h: dict) -> dict:
    return {"status": h["status"], "last_error": h["last_error"],
            "consecutive_failures": h["consecutive_failures"],
            "checked_at": None if not h["checked_at"] else round(h["checked_at"], 3)}


def strip_images(body: bytes | None) -> tuple[bytes | None, int]:
    """Replace image parts in a chat request with a text note (for text-only engines)."""
    if not body:
        return body, 0
    try:
        req = json.loads(body)
    except ValueError:
        return body, 0
    n = 0
    for msg in req.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            new = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES:
                    n += 1
                    new.append({"type": "text", "text": "[image omitted: this model is text-only]"})
                else:
                    new.append(part)
            msg["content"] = new
    return (json.dumps(req).encode(), n) if n else (body, 0)


class Arbiter:
    def __init__(self, settings: Settings):
        self.s = settings
        # Backends and listen ports are fixed for the life of the process.
        # Disabled engines are omitted: no listen port, no wake/sleep.
        self.engines = {e["name"]: dict(e) for e in settings.get("engines") if e.get("enabled", True)}
        for e in self.engines.values():
            host, _, port = e["backend"].rpartition(":")
            e["host"], e["port"] = host, int(port)
        names = list(self.engines)
        self.ctl = {n: VllmBackend(e["host"], e["port"]) for n, e in self.engines.items()}
        self.comfy = ComfyBackend(settings)
        self.cond = threading.Condition()
        now = time.time()
        self.state = {
            "owner": None,                                  # None | "comfy" | engine name
            "comfy_active": 0,                              # prompts running or waiting
            "inflight": {e: 0 for e in names},
            "last_infer": {e: now for e in names},
            "sleep_level": {e: settings.get("sleep_level") for e in names},
            "loading_until": {e: 0.0 for e in names},
            "switching": None,                              # engine name | "comfy" | None
            "evicting": set(),                              # engines being slept
        }
        self.sleeping: dict = {e: None for e in names}      # cached; occupancy HTTP updates it
        self.health = {e: _blank_health() for e in names}
        self.comfy_health = _blank_health()
        self.waking: set[str] = set()                       # skip failed probes during a wake
        self.measured_mb: dict = {}                         # engine -> MiB observed after a wake
        self.probe = GpuProbe(settings)
        self.gpu: dict = {"free_mb": None, "used_mb": None, "total_mb": None,
                          "name": None, "source": "unknown", "index": settings.get("gpu_index")}
        self.comfy_queue: dict = {"running": None, "pending": None}
        self.stats = {e: Counters() for e in names}
        self.servers: list = []

    # ------------------------------------------------------------ settings
    def shared(self) -> bool:
        return bool(self.s.get("shared_vram"))

    def free_mb(self, engine: str) -> int:
        live = self.s.engine(engine)
        return int((live or self.engines[engine])["free_mb"])

    def live_engine(self, engine: str) -> dict:
        return self.s.engine(engine) or self.engines[engine]

    def text_only(self, engine: str) -> bool:
        return engine in {x.strip() for x in self.s.get("text_only_engines").split(",")}

    def is_enabled(self, engine: str) -> bool:
        live = self.s.engine(engine)
        return bool(live and live.get("enabled", True))

    def wants_boot(self, engine: str) -> bool:
        live = self.s.engine(engine)
        return bool(live and live.get("enabled", True) and live.get("boot"))

    def comfy_url(self) -> str:
        return self.s.get("comfy_url").rstrip("/")

    def cost_of(self, engine: str) -> tuple[int, str]:
        cfg = self.live_engine(engine)
        try:
            estimated = estimate_from_engine(cfg)
        except (ValueError, TypeError):  # e.g. an unknown quant name; fall back, don't 500
            estimated = None
        return resolve_cost(
            self.measured_mb.get(engine),
            cfg.get("vram_mb"),
            estimated,
            cfg.get("free_mb") or 0,
        )

    def cost_mb(self, engine: str) -> int:
        return self.cost_of(engine)[0]

    def usable(self) -> int | None:
        budget = int(self.s.get("vram_budget_mb") or 0) or self.gpu.get("total_mb")
        return usable_mb(budget, self.s.get("vram_safety_mb"))

    # ------------------------------------------------------------ helpers
    def any_inflight(self, exclude: str | None = None) -> bool:
        return any(n for e, n in self.state["inflight"].items() if e != exclude)

    def loading_engines(self, exclude: str | None = None) -> list[str]:
        now = time.time()
        return [e for e, t in self.state["loading_until"].items() if e != exclude and now < t]

    def backend(self, engine, method, path, body=None, headers=None, timeout=600.0):
        cfg = self.engines[engine]
        conn = http.client.HTTPConnection(cfg["host"], cfg["port"], timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            return conn, conn.getresponse()
        except Exception:
            conn.close()
            raise

    def is_sleeping(self, engine: str) -> bool | None:
        """Probe the backend (HTTP). Callers must not hold `self.cond`."""
        val = self.ctl[engine].is_sleeping()
        self.sleeping[engine] = val
        return val

    def retry_after(self) -> int:
        return max(1, int(self.s.get("health_interval_sec")))

    def unavailable(self, engine: str) -> dict | None:
        """Health snapshot if this engine must not receive traffic, else None."""
        h = self.health[engine]
        if h["status"] == "unhealthy":
            return h
        if (h["status"] == "unreachable" and h.get("seen_up")
                and time.time() >= self.state["loading_until"][engine]):
            return h
        return None

    def _raise_if_unavailable(self, engine: str) -> None:
        h = self.unavailable(engine)
        if h:
            raise EngineUnavailable(engine, h, self.retry_after())

    def _record_health(self, slot: dict, name: str, status: str, error: str | None) -> None:
        prev = slot["status"]
        slot["checked_at"] = time.time()
        if status == "healthy":
            slot["consecutive_failures"] = 0
            slot["last_error"] = None
            slot["seen_up"] = True
            slot["status"] = "healthy"
        else:
            slot["consecutive_failures"] += 1
            slot["last_error"] = (error or status)[:300]
            slot["status"] = status
        if prev != slot["status"]:
            extra = f": {slot['last_error']}" if slot["last_error"] else ""
            log(C, f"{name} health {prev} → {slot['status']}{extra}",
                "info" if status == "healthy" else "error")
            # Transitions only: a row per probe would be thousands a day per engine.
            analytics.record_health_check(
                "comfy" if name == "ComfyUI" else name, ok=status == "healthy",
                details={"from": prev, "to": slot["status"], "error": slot["last_error"]})

    def _mark_unhealthy(self, engine: str, error: str) -> None:
        self._record_health(self.health[engine], engine, "unhealthy", error)

    def probe_engine_health(self, engine: str) -> tuple[str, str | None]:
        """GET /health. Callers must not hold `self.cond`."""
        return self.ctl[engine].probe_health(self.s.get("health_timeout_sec"))

    def probe_comfy_health(self) -> tuple[str, str | None]:
        """GET ComfyUI /system_stats. Callers must not hold `self.cond`."""
        return self.comfy.probe_health(self.s.get("health_timeout_sec"))

    def _release_owner(self, engine: str) -> None:
        if self.state["owner"] == engine:
            self.state["owner"] = next(
                (e for e in self.engines if e != engine and self.sleeping[e] is False), None)

    def _admit(self, engine: str) -> None:
        self.state["owner"] = engine
        self.state["inflight"][engine] += 1
        self.state["last_infer"][engine] = time.time()

    def _begin_switch(self, who, victims) -> None:
        self.state["switching"] = who
        self.state["evicting"] = set(victims)
        self.cond.notify_all()

    def _end_switch(self, who) -> None:
        if self.state["switching"] == who:
            self.state["switching"] = None
        self.state["evicting"] = set()
        self.cond.notify_all()

    def _victims_locked(self, engine: str) -> list[str]:
        """Who must sleep so `engine` can wake. Uses cached `sleeping`, no HTTP."""
        awake = [e for e in self.engines if e != engine and self.sleeping[e] is False]
        if not self.shared():
            return awake
        room = self.usable()
        if room is None:
            return awake
        resident = {e: self.state["last_infer"][e] for e in awake}
        costs = {e: self.cost_mb(e) for e in awake}
        return lru_victims(resident, costs, engine, self.cost_mb(engine), room)

    # ------------------------------------------------------------ engine power (no cond held)
    def wake(self, engine: str) -> None:
        if self.sleeping[engine]:
            level = self.state["sleep_level"][engine]
            log(C, f"waking {engine} (from level {level})")
            t, vram_b, ram_b = time.time(), self.vram_used_mb(), analytics.ram_used_mb()
            self.ctl[engine].wake(level)
            took = time.time() - t
            self.stats[engine].inc("wakes")
            self.stats[engine].set("last_wake_sec", round(took, 1))
            log(C, f"{engine} awake in {took:.1f}s")
            self._emit("wake", engine, t0=t, vram_before=vram_b, ram_before=ram_b,
                       details={"level": level})
        self.sleeping[engine] = False
        status, err = self.probe_engine_health(engine)
        if status != "healthy":
            raise RuntimeError(f"{engine} not healthy after wake: {err or status}")
        with self.cond:
            self._record_health(self.health[engine], engine, "healthy", None)

    def sleep(self, engine: str, reason: str) -> None:
        if self.sleeping[engine] is False:
            level = self.s.get("sleep_level")
            log(C, f"sleeping {engine} level={level} ({reason})")
            t, vram_b, ram_b = time.time(), self.vram_used_mb(), analytics.ram_used_mb()
            try:
                self.ctl[engine].sleep(level)
            except Exception as e:
                self._emit("error", engine, t0=t, vram_before=vram_b, ram_before=ram_b,
                           details={"op": "sleep", "reason": reason, "error": str(e)})
                raise
            self.state["sleep_level"][engine] = level
            self.stats[engine].inc("sleeps")
            log(C, f"{engine} asleep in {time.time() - t:.1f}s")
            self._emit("sleep", engine, t0=t, vram_before=vram_b, ram_before=ram_b,
                       details={"level": level, "reason": reason})
        self.sleeping[engine] = True

    def _sleep_list(self, engines: list[str], reason: str) -> None:
        for e in engines:
            try:
                self.sleep(e, reason)
            except Exception as ex:
                log(C, f"sleep {e} failed: {ex}", "error")

    def _apply_wake(self, engine: str, victims: list[str], reason: str, unload_comfy: bool) -> None:
        while self.comfy_running():
            time.sleep(0.5)
        self._sleep_list(victims, reason)
        target = self.free_mb(engine)
        if unload_comfy:
            self.comfy_unload(target)
        attempts = self.s.get("wake_attempts")
        delay = self.s.get("wake_backoff_sec")
        for attempt in range(1, attempts + 1):
            try:
                before = self.gpu_free_mb()
                if before is None and self.sleeping[engine]:
                    log(C, f"waking {engine} with VRAM unknown", "warn")
                self.wake(engine)
                self._note_measured(engine, before)
                return
            except Exception as e:
                log(C, f"wake {engine} failed (attempt {attempt}/{attempts}): {e}", "error")
                if attempt == attempts:
                    analytics.record_wake_failure(
                        engine, error=str(e), details={"attempt": attempt, "attempts": attempts})
                    raise RuntimeError(f"{engine} could not wake: {e}") from e
                analytics.record_wake_retry(
                    engine, attempt=attempt, error=str(e),
                    details={"attempts": attempts, "backoff_sec": delay})
                self.comfy_unload(target)
                time.sleep(delay)
                delay *= 2

    def _note_measured(self, engine: str, before: float | None) -> None:
        after = self.gpu_free_mb()
        if before is None or after is None:
            return
        used = int(before - after)
        if used > 0:
            self.measured_mb[engine] = used

    # ------------------------------------------------------------ ComfyUI
    def gpu_free_mb(self) -> float | None:
        self.gpu = self.probe.read().as_dict()
        free = self.gpu.get("free_mb")
        return None if free is None else float(free)

    def vram_used_mb(self) -> int | None:
        """Cached used VRAM (MiB). None when the poller has not filled gpu yet."""
        if self.gpu.get("used_mb") is not None:  # NVML / nvidia-smi report it directly
            return int(self.gpu["used_mb"])
        total, free = self.gpu.get("total_mb"), self.gpu.get("free_mb")
        if total is None or free is None:
            return None
        try:
            return max(0, int(total) - int(free))
        except (TypeError, ValueError):
            return None

    def _emit(self, kind: str, engine: str | None = None, *, t0: float | None = None,
              vram_before: int | None = None, ram_before: int | None = None,
              details: dict | None = None) -> None:
        """Queue a telemetry event. Safe under self.cond: no sqlite I/O."""
        duration_ms = None if t0 is None else int(round((time.time() - t0) * 1000))
        analytics.emit(
            kind, engine=engine, duration_ms=duration_ms, details=details,
            vram_before_mb=vram_before, vram_after_mb=self.vram_used_mb(),
            ram_before_mb=ram_before, ram_after_mb=analytics.ram_used_mb(),
        )

    def comfy_running(self) -> bool | None:
        running, info = self.comfy.queue()
        self.comfy_queue = info
        return running

    def comfy_unload(self, target_mb: float, require_idle: bool = False) -> None:
        """Ask ComfyUI to drop its models and wait for target_mb free; no-op if it is down."""
        if require_idle:
            while self.comfy_running():  # prompts the hook didn't register
                time.sleep(5)
        t, vram_b, ram_b = time.time(), self.vram_used_mb(), analytics.ram_used_mb()
        result = self.comfy.unload(target_mb, running=self.comfy_running, free_mb=self.gpu_free_mb)
        self._emit("unload", "comfy", t0=t, vram_before=vram_b, ram_before=ram_b,
                   details={"target_mb": target_mb, **result})

    # ------------------------------------------------------------ ownership
    def engine_enter(self, engine: str) -> None:
        if engine not in self.engines:
            raise RuntimeError(f"unknown engine {engine}")
        if not self.is_enabled(engine):
            raise RuntimeError(f"{engine} is disabled")
        st = self.state
        self._raise_if_unavailable(engine)
        with self.cond:
            while True:
                self._raise_if_unavailable(engine)
                if st["comfy_active"] or self.loading_engines() or engine in st["evicting"]:
                    self.cond.wait(timeout=5)
                    continue
                resident = self.sleeping[engine] is False
                if self.shared():
                    if resident:
                        self._admit(engine)
                        return
                    if st["switching"]:
                        self.cond.wait(timeout=5)
                        continue
                else:
                    if self.any_inflight(engine) or st["switching"]:
                        self.cond.wait(timeout=5)
                        continue
                    if st["owner"] == engine:
                        self._admit(engine)
                        return
                break

            # No ComfyUI claim, load or switch is pending here. Reserve the switch before
            # draining the victims: cond.wait drops the lock, and without the reservation
            # another switch, ComfyUI prompt or load could start in between. New requests
            # to a victim see it in `evicting` and queue instead of starving this wake.
            victims = self._victims_locked(engine)
            prev = st["owner"]
            unload_comfy = prev not in self.engines
            reason = f"LRU for {engine}" if self.shared() else f"switching to {engine}"
            self._begin_switch(engine, victims)
            self.waking.add(engine)
            while any(st["inflight"][v] for v in victims):
                self.cond.wait(timeout=2)

        woke, error = False, None
        try:
            self._apply_wake(engine, victims, reason, unload_comfy)
            woke = True
        except Exception as e:
            error = e
            raise
        finally:
            with self.cond:
                self.waking.discard(engine)
                self._end_switch(engine)
                if woke:
                    self._admit(engine)
                    if prev and prev != engine:
                        self._emit("swap", engine, details={"from": prev, "to": engine})
                else:
                    if error is not None:
                        self._mark_unhealthy(engine, str(error))
                    # The failed engine never owns the GPU, even if vLLM woke but /health failed.
                    st["owner"] = next((e for e in self.engines
                                        if e != engine and self.sleeping[e] is False), None)

    def engine_exit(self, engine: str) -> None:
        with self.cond:
            self.state["inflight"][engine] -= 1
            self.state["last_infer"][engine] = time.time()
            self.cond.notify_all()

    def comfy_acquire(self) -> None:
        st = self.state
        with self.cond:
            st["comfy_active"] += 1
            while self.any_inflight() or self.loading_engines() or st["switching"]:
                self.cond.wait(timeout=5)
            if st["owner"] == "comfy":
                return
            victims = [e for e in self.engines if self.sleeping[e] is False]
            prev = st["owner"]
            self._begin_switch("comfy", victims)
        try:
            self._sleep_list(victims, "ComfyUI prompt")
            with self.cond:
                st["owner"] = "comfy"
                log(C, "ComfyUI owns the GPU")
                if prev:
                    self._emit("swap", "comfy", details={"from": prev, "to": "comfy"})
        finally:
            with self.cond:
                self._end_switch("comfy")

    def comfy_release(self) -> None:
        with self.cond:
            self.state["comfy_active"] = max(0, self.state["comfy_active"] - 1)
            self.cond.notify_all()

    def engine_loading(self, engine: str) -> None:
        """Called by an engine container's entrypoint before `vllm serve`."""
        st = self.state
        with self.cond:
            # If this engine still holds the claim (vLLM crashed and restarted), waiting
            # for the parked ComfyUI prompts/engines would deadlock, so just renew it.
            renew = time.time() < st["loading_until"][engine]
            if not renew:
                while not (not self.loading_engines(engine) and st["comfy_active"] == 0
                           and not self.any_inflight(engine) and not st["switching"]):
                    self.cond.wait(timeout=5)
            st["loading_until"][engine] = time.time() + self.s.get("loading_max_sec")
            st["owner"] = None
            log(C, f"{engine} is loading")
            victims = [e for e in self.engines if e != engine and self.sleeping[e] is False]
            self._begin_switch(engine, victims)
        try:
            self._sleep_list(victims, f"{engine} loading")
        finally:
            with self.cond:
                self._end_switch(engine)
        self._emit("load", engine, details={"renew": renew, "free_mb": self.free_mb(engine)})
        self.comfy_unload(self.free_mb(engine), require_idle=not renew)

    # ------------------------------------------------------------ dashboard actions
    def action(self, name: str, target: str | None) -> str:
        if name in ("sleep", "warm") and target not in self.engines:
            raise ValueError(f"unknown engine {target}")
        if name == "warm" and not self.is_enabled(target):
            raise ValueError(f"{target} is disabled")
        if name == "sleep":
            def run():
                with self.cond:
                    while self.state["switching"]:  # never overwrite another transition
                        self.cond.wait(timeout=2)
                    if self.state["inflight"][target]:
                        log(C, f"not sleeping {target}: requests in flight", "warn")
                        return
                    if self.sleeping[target] is not False:
                        return
                    self._begin_switch(target, [target])
                try:
                    self.sleep(target, "dashboard")
                finally:
                    with self.cond:
                        self._release_owner(target)
                        self._end_switch(target)
        elif name == "warm":
            def run():
                self.engine_enter(target)
                self.engine_exit(target)
        elif name == "comfy_unload":
            def run():
                with self.cond:
                    if self.state["comfy_active"]:
                        log(C, "not unloading ComfyUI: a prompt is running", "warn")
                        return
                self.comfy_unload(0)
                with self.cond:
                    if self.state["owner"] == "comfy":
                        self.state["owner"] = None
                    self.cond.notify_all()
        elif name == "comfy_release":
            def run():
                with self.cond:
                    log(C, "ComfyUI claim released from the dashboard", "warn")
                    self.state["comfy_active"] = 0
                    self.cond.notify_all()
        else:
            raise ValueError(f"unknown action {name}")
        threading.Thread(target=self._safe, args=(run, name), daemon=True).start()
        return "started"

    @staticmethod
    def _safe(fn, name):
        try:
            fn()
        except Exception as e:
            log(C, f"{name} failed: {e}", "error")
            analytics.emit("error", details={"op": name, "error": str(e)})

    def status(self) -> dict:
        st, now = self.state, time.time()
        engines = {}
        resident = []
        for live in self.s.get("engines"):
            e = live["name"]
            enabled, boot = bool(live.get("enabled", True)), bool(live.get("boot"))
            if e not in self.engines:
                engines[e] = {
                    "listen": live["listen"], "backend": live["backend"],
                    "free_mb": int(live.get("free_mb") or 0), "text_only": self.text_only(e),
                    "state": "disabled" if not enabled else "down",
                    "enabled": enabled, "boot": boot, "inflight": 0, "idle_sec": 0,
                }
                continue
            cfg = self.engines[e]
            loading = now < st["loading_until"][e]
            s = self.sleeping[e]
            cost, src = self.cost_of(e)
            if s is False:
                resident.append(e)
            engines[e] = {
                "listen": cfg["listen"], "backend": cfg["backend"], "free_mb": self.free_mb(e),
                "text_only": self.text_only(e), "enabled": enabled, "boot": boot,
                "state": "loading" if loading else {True: "asleep", False: "awake", None: "down"}[s],
                "inflight": st["inflight"][e],
                "idle_sec": round(now - st["last_infer"][e]),
                "vram_mb": cost, "vram_source": src,
                "health": _health_view(self.health[e]),
                **self.stats[e].snapshot(),
            }
        room = self.usable()
        return {"owner": st["owner"], "comfy_active": st["comfy_active"], "comfy_queue": self.comfy_queue,
                "comfy_health": _health_view(self.comfy_health),
                "gpu": self.gpu, "engines": engines, "idle_sleep_sec": self.s.get("idle_sleep_sec"),
                "placement": "shared" if self.shared() else "exclusive",
                "resident": resident,
                "switching": st["switching"],
                "budget": {
                    "total_mb": int(self.s.get("vram_budget_mb") or 0) or self.gpu.get("total_mb"),
                    "safety_mb": self.s.get("vram_safety_mb"),
                    "usable_mb": room,
                    "resident_mb": sum(self.cost_mb(e) for e in resident),
                }}

    # ------------------------------------------------------------ background threads
    def idle_watcher(self) -> None:
        stale_since = None
        while True:
            time.sleep(10)
            with self.cond:
                comfy_active = self.state["comfy_active"]
                busy = self.any_inflight()
            if comfy_active and not busy:
                if self.comfy_running() is not True:
                    stale_since = stale_since or time.time()
                    if time.time() - stale_since > 60:
                        with self.cond:
                            log(C, "ComfyUI claim looks stale; releasing", "warn")
                            self.state["comfy_active"] = 0
                            self.cond.notify_all()
                        stale_since = None
                else:
                    stale_since = None
            else:
                stale_since = None
            to_sleep = []
            with self.cond:
                st = self.state
                if st["switching"]:
                    continue
                idle_limit = self.s.get("idle_sleep_sec")
                now = time.time()
                if self.shared():
                    candidates = list(self.engines)
                else:
                    owner = st["owner"]
                    candidates = [owner] if owner in self.engines else []
                for e in candidates:
                    if (self.sleeping[e] is False and st["inflight"][e] == 0
                            and now - st["last_infer"][e] >= idle_limit):
                        to_sleep.append(e)
                if to_sleep:
                    self._begin_switch(to_sleep[0], to_sleep)
            if not to_sleep:
                continue
            try:
                self._sleep_list(to_sleep, "idle")
            finally:
                with self.cond:
                    for e in to_sleep:
                        self._release_owner(e)
                    self._end_switch(to_sleep[0])

    def on_backend_ready(self, engine: str) -> None:
        """Park a newly-up engine, or wake it if it is marked to load on boot."""
        with self.cond:
            self.state["loading_until"][engine] = 0.0
            busy = self.state["owner"] == engine or self.state["inflight"][engine]
            boot = self.wants_boot(engine)
            self.cond.notify_all()
        if busy:
            return
        if boot:
            try:
                self.engine_enter(engine)
                self.engine_exit(engine)
                log(C, f"{engine} loaded on boot")
            except Exception as e:
                log(C, f"boot load {engine} failed: {e}", "error")
            return
        should_sleep = False
        with self.cond:
            if (self.state["owner"] != engine and self.state["inflight"][engine] == 0
                    and not self.state["switching"] and self.sleeping[engine] is False):
                should_sleep = True
                self._begin_switch(engine, [engine])
        if should_sleep:
            try:
                self.sleep(engine, "startup")
            except Exception as e:
                log(C, f"startup sleep {engine} failed: {e}", "error")
            finally:
                with self.cond:
                    self._end_switch(engine)

    def startup_watcher(self, engine: str) -> None:
        """Once an engine is up, park it (or boot-load it); repeat after restarts."""
        while True:
            while self.is_sleeping(engine) is None:
                time.sleep(5)
            log(C, f"{engine} backend ready")
            self.on_backend_ready(engine)
            # /is_sleeping can time out during a level-2 wake, so require several misses.
            misses = 0
            while misses < 4:
                time.sleep(15)
                misses = misses + 1 if self.is_sleeping(engine) is None else 0
            log(C, f"{engine} backend went away; waiting for it to come back", "warn")
            with self.cond:
                if self.state["owner"] == engine:
                    self.state["owner"] = None
                self.cond.notify_all()

    def poller(self) -> None:
        """Keep GPU/ComfyUI numbers fresh for the dashboard."""
        while True:
            self.gpu_free_mb()
            self.comfy_running()
            time.sleep(3)

    def health_monitor(self) -> None:
        """Probe engines and ComfyUI. HTTP runs outside `self.cond`."""
        while True:
            results = {e: self.probe_engine_health(e) for e in self.engines}
            comfy = self.probe_comfy_health()
            with self.cond:
                for e, (status, err) in results.items():
                    if status != "healthy" and e in self.waking:
                        continue
                    self._record_health(self.health[e], e, status, err)
                    if (self.unavailable(e) and self.state["owner"] == e
                            and self.state["inflight"][e] == 0):
                        self._release_owner(e)
                self._record_health(self.comfy_health, "ComfyUI", *comfy)
                self.cond.notify_all()
            time.sleep(self.s.get("health_interval_sec"))

    # ------------------------------------------------------------ HTTP
    def make_handler(self, engine: str):
        arb = self

        class EngineHandler(Handler):
            def _gpu_api(self, body: bytes) -> bool:
                u = urlparse(self.path)
                p = u.path
                if p == "/gpu/status":
                    self.send_json(200, arb.status())
                elif p == "/gpu/acquire" and self.command == "POST":
                    arb.comfy_acquire()
                    self.send_json(200, {"owner": "comfy"})
                elif p == "/gpu/release" and self.command == "POST":
                    arb.comfy_release()
                    self.send_json(200, {"released": True})
                elif p in ("/gpu/loading", "/gpu/qwen-loading") and self.command == "POST":
                    default = "qwen" if p == "/gpu/qwen-loading" else engine
                    name = (parse_qs(u.query).get("engine") or [default])[0]
                    if name not in arb.engines:
                        self.send_json(404, {"error": f"unknown engine {name}"})
                    else:
                        arb.engine_loading(name)
                        self.send_json(200, {"loading": name, "max_sec": arb.s.get("loading_max_sec")})
                else:
                    return False
                return True

            def proxy(self) -> None:
                body = self.body() or None
                if self.path.startswith("/gpu/") and self._gpu_api(body or b""):
                    return
                path = urlparse(self.path).path
                infer = path.startswith(INFER_PREFIXES)
                if arb.text_only(engine) and path.startswith("/v1/chat/completions"):
                    body, n = strip_images(body)
                    if n:
                        log(C, f"stripped {n} image part(s) from a request to text-only {engine}")
                down = arb.unavailable(engine)
                if down and not infer:
                    err = EngineUnavailable(engine, down, arb.retry_after())
                    arb.stats[engine].inc("errors")
                    self.send_json(503, {"error": str(err), "engine": engine, "health": err.health},
                                   extra={"Retry-After": str(err.retry_after)})
                    return
                if infer:
                    try:
                        arb.engine_enter(engine)
                    except EngineUnavailable as e:
                        arb.stats[engine].inc("errors")
                        self.send_json(503, {"error": str(e), "engine": e.engine, "health": e.health},
                                       extra={"Retry-After": str(e.retry_after)})
                        return
                    except Exception as e:
                        arb.stats[engine].inc("errors")
                        self.send_json(503, {"error": str(e)},
                                       extra={"Retry-After": str(arb.retry_after())})
                        return
                    arb.stats[engine].inc("requests")
                try:
                    try:
                        conn, resp = arb.backend(engine, self.command, self.path, body=body,
                                                 headers=self.forward_headers(), timeout=900)
                    except Exception as e:
                        if infer:
                            arb.stats[engine].inc("errors")
                        self.send_json(502, {"error": f"{engine} backend unavailable: {e}"})
                        return
                    try:
                        if infer and resp.status >= 400:
                            arb.stats[engine].inc("errors")
                        self.relay(resp)
                    finally:
                        conn.close()
                finally:
                    if infer:
                        arb.engine_exit(engine)

            do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = proxy  # noqa: N815

        return EngineHandler

    def start(self) -> None:
        host = self.s.get("listen_host")
        mode = "shared" if self.shared() else "exclusive"
        if not self.engines:
            log(C, "no enabled engines; GPU API listen ports are not bound", "warn")
        for e, cfg in self.engines.items():
            boot = "boot" if cfg.get("boot") else "parked"
            log(C, f"{e}: {host}:{cfg['listen']} -> {cfg['backend']} "
                   f"(wake needs {cfg['free_mb']} MiB free, {mode}, {boot})")
        for target in (self.idle_watcher, self.poller, self.health_monitor):
            threading.Thread(target=target, daemon=True).start()
        for e, cfg in self.engines.items():
            threading.Thread(target=self.startup_watcher, args=(e,), daemon=True).start()
            srv = Server((host, cfg["listen"]), self.make_handler(e))
            self.servers.append(srv)
            threading.Thread(target=srv.serve_forever, daemon=True).start()

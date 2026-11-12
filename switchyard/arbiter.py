"""GPU arbiter: several vLLM engines and ComfyUI take turns on one NVIDIA GPU.

Default (`shared_vram=false`): exactly one of {each vLLM engine, ComfyUI} owns the
VRAM at a time. Engines that don't own it sleep (vLLM sleep mode; level 2 drops the
weights and leaves under 1 GiB each).

With `shared_vram=true`, several engines may stay awake while their VRAM costs fit
in the card budget minus a safety margin. A new wake evicts least-recently-used
residents until it fits. Inference is still serialized (one request owner at a
time). ComfyUI still takes the whole GPU for a prompt.

  Engines : each engine gets an OpenAI-compatible listen port. Before an inference
            request is forwarded, the arbiter waits for ComfyUI and the other engines
            to finish, then wakes this engine (sleeping others only when exclusive,
            or when the budget requires it). After `idle_sleep_sec` without traffic
            an idle resident goes back to sleep.
  ComfyUI : the gpu_arbiter custom node (comfyui_node/) calls POST /gpu/acquire
            before each prompt and POST /gpu/release after it.
  Startup : an engine container calls POST /gpu/loading?engine=NAME before `vllm serve`.
            Engine loads are serialized; once an engine is up it is put to sleep.

GET /gpu/status on any engine port returns the state as JSON. Responses are streamed.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

from .backends import ComfyBackend, VllmBackend
from .budget import lru_victims, resolve_cost, usable_mb
from .config import Settings
from .estimate import estimate_from_engine
from .events import Counters, log
from .httputil import Handler, Server

INFER_PREFIXES = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses",
                  "/v1/audio", "/v1/messages", "/tokenize", "/detokenize")
IMAGE_PART_TYPES = {"image_url", "input_image", "image", "video_url", "input_video"}
C = "arbiter"


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
        self.engines = {e["name"]: dict(e) for e in settings.get("engines")}
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
        }
        self.sleeping: dict = {e: None for e in names}      # polled, for the dashboard
        self.measured_mb: dict = {}                         # engine -> MiB observed after a wake
        self.gpu: dict = {"free_mb": None, "total_mb": None, "name": None}
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

    def comfy_url(self) -> str:
        return self.s.get("comfy_url").rstrip("/")

    def cost_of(self, engine: str) -> tuple[int, str]:
        cfg = self.live_engine(engine)
        return resolve_cost(
            self.measured_mb.get(engine),
            cfg.get("vram_mb"),
            estimate_from_engine(cfg),
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
        conn.request(method, path, body=body, headers=headers or {})
        return conn, conn.getresponse()

    def is_sleeping(self, engine: str) -> bool | None:
        val = self.ctl[engine].is_sleeping()
        self.sleeping[engine] = val
        return val

    def _release_owner(self, engine: str) -> None:
        if self.state["owner"] == engine:
            self.state["owner"] = next(
                (e for e in self.engines if e != engine and self.sleeping[e] is False), None)

    # ------------------------------------------------------------ engine power
    def wake(self, engine: str) -> None:
        if self.ctl[engine].is_sleeping():
            level = self.state["sleep_level"][engine]
            log(C, f"waking {engine} (from level {level})")
            t = time.time()
            self.ctl[engine].wake(level)
            took = time.time() - t
            self.stats[engine].inc("wakes")
            self.stats[engine].set("last_wake_sec", round(took, 1))
            log(C, f"{engine} awake in {took:.1f}s")
        self.sleeping[engine] = False

    def sleep(self, engine: str, reason: str) -> None:
        if self.ctl[engine].is_sleeping() is False:
            level = self.s.get("sleep_level")
            log(C, f"sleeping {engine} level={level} ({reason})")
            t = time.time()
            self.ctl[engine].sleep(level)
            self.state["sleep_level"][engine] = level
            self.stats[engine].inc("sleeps")
            log(C, f"{engine} asleep in {time.time() - t:.1f}s")
        self.sleeping[engine] = True

    def sleep_others(self, keep: str | None, reason: str) -> None:
        for e in self.engines:
            if e != keep and self.is_sleeping(e) is False:
                try:
                    self.sleep(e, reason)
                except Exception as ex:
                    log(C, f"sleep {e} failed: {ex}", "error")

    def _evict_for(self, engine: str) -> None:
        """Sleep whoever must move so `engine` can wake. Exclusive: everyone else."""
        if not self.shared():
            self.sleep_others(engine, f"switching to {engine}")
            return
        if self.gpu.get("total_mb") is None:
            self.gpu_free_mb()
        room = self.usable()
        if room is None:
            self.sleep_others(engine, f"switching to {engine} (budget unknown)")
            return
        resident = {e: self.state["last_infer"][e]
                    for e in self.engines if e != engine and self.is_sleeping(e) is False}
        costs = {e: self.cost_mb(e) for e in resident}
        for name in lru_victims(resident, costs, engine, self.cost_mb(engine), room):
            try:
                self.sleep(name, f"LRU for {engine}")
            except Exception as ex:
                log(C, f"sleep {name} failed: {ex}", "error")

    def _note_measured(self, engine: str, before: float | None) -> None:
        after = self.gpu_free_mb()
        if before is None or after is None:
            return
        used = int(before - after)
        if used > 0:
            self.measured_mb[engine] = used

    # ------------------------------------------------------------ ComfyUI
    def gpu_free_mb(self) -> float | None:
        free, snap = self.comfy.vram()
        if snap:
            self.gpu.update(snap)
        return free

    def comfy_running(self) -> bool | None:
        running, info = self.comfy.queue()
        self.comfy_queue = info
        return running

    def comfy_unload(self, target_mb: float, require_idle: bool = False) -> None:
        """Ask ComfyUI to drop its models and wait for target_mb free; no-op if it is down."""
        self.comfy.unload(target_mb, require_idle=require_idle,
                          running=self.comfy_running, free_mb=self.gpu_free_mb)

    # ------------------------------------------------------------ ownership
    def engine_enter(self, engine: str) -> None:
        st = self.state
        with self.cond:
            # ComfyUI has priority while running or waiting; other engines finish their
            # in-flight requests first; nothing wakes while an engine is loading.
            while not (st["comfy_active"] == 0 and not self.any_inflight(engine) and not self.loading_engines()):
                self.cond.wait(timeout=5)
            needs_wake = (self.is_sleeping(engine) is not False) if self.shared() else st["owner"] != engine
            if needs_wake:
                while st["comfy_active"] or self.any_inflight(engine) or self.comfy_running():
                    self.cond.wait(timeout=2)
                prev = st["owner"]
                self._evict_for(engine)
                target = self.free_mb(engine)
                if prev not in self.engines:  # ComfyUI or nobody held the GPU
                    self.comfy_unload(target)
                attempts = self.s.get("wake_attempts")
                for attempt in range(1, attempts + 1):
                    try:
                        before = self.gpu_free_mb()
                        self.wake(engine)
                        self._note_measured(engine, before)
                        break
                    except Exception as e:
                        log(C, f"wake {engine} failed (attempt {attempt}/{attempts}): {e}", "error")
                        if attempt == attempts:
                            st["owner"] = None
                            raise RuntimeError(f"{engine} could not wake: {e}") from e
                        self.comfy_unload(target)
                        time.sleep(2)
                st["owner"] = engine
            elif self.shared():
                st["owner"] = engine
            st["inflight"][engine] += 1
            st["last_infer"][engine] = time.time()

    def engine_exit(self, engine: str) -> None:
        with self.cond:
            self.state["inflight"][engine] -= 1
            self.state["last_infer"][engine] = time.time()
            self.cond.notify_all()

    def comfy_acquire(self) -> None:
        with self.cond:
            self.state["comfy_active"] += 1
            while self.any_inflight() or self.loading_engines():
                self.cond.wait(timeout=5)
            if self.state["owner"] != "comfy":
                self.sleep_others(None, "ComfyUI prompt")
                self.state["owner"] = "comfy"
                log(C, "ComfyUI owns the GPU")

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
                           and not self.any_inflight(engine)):
                    self.cond.wait(timeout=5)
            st["loading_until"][engine] = time.time() + self.s.get("loading_max_sec")
            st["owner"] = None
            log(C, f"{engine} is loading")
            self.sleep_others(engine, f"{engine} loading")
        self.comfy_unload(self.free_mb(engine), require_idle=not renew)

    # ------------------------------------------------------------ dashboard actions
    def action(self, name: str, target: str | None) -> str:
        if name in ("sleep", "warm") and target not in self.engines:
            raise ValueError(f"unknown engine {target}")
        if name == "sleep":
            def run():
                with self.cond:
                    if self.state["inflight"][target]:
                        log(C, f"not sleeping {target}: requests in flight", "warn")
                        return
                    self.sleep(target, "dashboard")
                    self._release_owner(target)
        elif name == "warm":
            def run():
                self.engine_enter(target)
                self.engine_exit(target)
        elif name == "comfy_unload":
            def run():
                if self.state["comfy_active"]:
                    log(C, "not unloading ComfyUI: a prompt is running", "warn")
                    return
                self.comfy_unload(0)
                if self.state["owner"] == "comfy":
                    self.state["owner"] = None
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

    def status(self) -> dict:
        st, now = self.state, time.time()
        engines = {}
        resident = []
        for e, cfg in self.engines.items():
            loading = now < st["loading_until"][e]
            s = self.sleeping[e]
            cost, src = self.cost_of(e)
            if s is False:
                resident.append(e)
            engines[e] = {
                "listen": cfg["listen"], "backend": cfg["backend"], "free_mb": self.free_mb(e),
                "text_only": self.text_only(e),
                "state": "loading" if loading else {True: "asleep", False: "awake", None: "down"}[s],
                "inflight": st["inflight"][e],
                "idle_sec": round(now - st["last_infer"][e]),
                "vram_mb": cost, "vram_source": src,
                **self.stats[e].snapshot(),
            }
        room = self.usable()
        return {"owner": st["owner"], "comfy_active": st["comfy_active"], "comfy_queue": self.comfy_queue,
                "gpu": self.gpu, "engines": engines, "idle_sleep_sec": self.s.get("idle_sleep_sec"),
                "placement": "shared" if self.shared() else "exclusive",
                "resident": resident,
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
            st = self.state
            # Safety valve: ComfyUI died mid-prompt and never released.
            if st["comfy_active"] and not self.any_inflight() and self.comfy_running() is not True:
                stale_since = stale_since or time.time()
                if time.time() - stale_since > 60:
                    with self.cond:
                        log(C, "ComfyUI claim looks stale; releasing", "warn")
                        st["comfy_active"] = 0
                        self.cond.notify_all()
                    stale_since = None
            else:
                stale_since = None
            with self.cond:
                idle_limit = self.s.get("idle_sleep_sec")
                now = time.time()
                if self.shared():
                    candidates = list(self.engines)
                else:
                    owner = st["owner"]
                    candidates = [owner] if owner in self.engines else []
                for e in candidates:
                    if (self.is_sleeping(e) is False and st["inflight"][e] == 0
                            and now - st["last_infer"][e] >= idle_limit):
                        try:
                            self.sleep(e, "idle")
                            self._release_owner(e)
                        except Exception as ex:
                            log(C, f"idle sleep {e} failed: {ex}", "error")

    def startup_watcher(self, engine: str) -> None:
        """Once an engine is up, put it to sleep so the GPU is free; repeat after restarts."""
        while True:
            while self.is_sleeping(engine) is None:
                time.sleep(5)
            log(C, f"{engine} backend ready")
            with self.cond:
                self.state["loading_until"][engine] = 0.0
                if self.state["owner"] != engine and self.state["inflight"][engine] == 0:
                    try:
                        self.sleep(engine, "startup")
                    except Exception as e:
                        log(C, f"startup sleep {engine} failed: {e}", "error")
                self.cond.notify_all()
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
                if infer:
                    try:
                        arb.engine_enter(engine)
                    except Exception as e:
                        self.send_json(503, {"error": str(e)})
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
        for e, cfg in self.engines.items():
            log(C, f"{e}: {host}:{cfg['listen']} -> {cfg['backend']} "
                   f"(wake needs {cfg['free_mb']} MiB free, {mode})")
        for target in (self.idle_watcher, self.poller):
            threading.Thread(target=target, daemon=True).start()
        for e, cfg in self.engines.items():
            threading.Thread(target=self.startup_watcher, args=(e,), daemon=True).start()
            srv = Server((host, cfg["listen"]), self.make_handler(e))
            self.servers.append(srv)
            threading.Thread(target=srv.serve_forever, daemon=True).start()

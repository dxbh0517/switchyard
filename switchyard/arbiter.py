"""GPU arbiter: several vLLM engines and ComfyUI take turns on one NVIDIA GPU.

Exactly one of {each vLLM engine, ComfyUI} owns the VRAM at a time. Engines that don't own
it sleep (vLLM sleep mode; level 2 drops the weights and leaves under 1 GiB each).

  Engines : each engine gets an OpenAI-compatible listen port. Before an inference request
            is forwarded, the arbiter waits for ComfyUI and the other engines to finish,
            puts the current owner to sleep (or asks ComfyUI to unload), then wakes this
            engine. After `idle_sleep_sec` without traffic the owner goes back to sleep.
  ComfyUI : the gpu_arbiter custom node (comfyui_node/) calls POST /gpu/acquire before each
            prompt and POST /gpu/release after it.
  Startup : an engine container calls POST /gpu/loading?engine=NAME before `vllm serve`.
            Engine loads are serialized; once an engine is up it is put to sleep.

GET /gpu/status on any engine port returns the state as JSON. Responses are streamed.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .events import Counters, log
from .gpu import GpuProbe
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
        # Disabled engines are omitted: no listen port, no wake/sleep.
        self.engines = {e["name"]: dict(e) for e in settings.get("engines") if e.get("enabled", True)}
        for e in self.engines.values():
            host, _, port = e["backend"].rpartition(":")
            e["host"], e["port"] = host, int(port)
        names = list(self.engines)
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
        self.probe = GpuProbe(settings)
        self.gpu: dict = {"free_mb": None, "used_mb": None, "total_mb": None,
                          "name": None, "source": "unknown", "index": settings.get("gpu_index")}
        self.comfy_queue: dict = {"running": None, "pending": None}
        self.stats = {e: Counters() for e in names}
        self.servers: list = []

    # ------------------------------------------------------------ settings
    def free_mb(self, engine: str) -> int:
        live = self.s.engine(engine)
        return int((live or self.engines[engine])["free_mb"])

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

    def backend_simple(self, engine, method, path, body=None, timeout=600.0) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json"} if body is not None else None
        conn, resp = self.backend(engine, method, path, body=body, headers=headers, timeout=timeout)
        try:
            return resp.status, resp.read()
        finally:
            conn.close()

    def is_sleeping(self, engine: str) -> bool | None:
        try:
            status, raw = self.backend_simple(engine, "GET", "/is_sleeping", timeout=10)
            if status != 200:
                return None
            data = json.loads(raw)
            val = bool(data if isinstance(data, bool) else data.get("is_sleeping", False))
        except Exception:
            val = None
        self.sleeping[engine] = val
        return val

    @staticmethod
    def _ok(status_raw, what: str) -> None:
        status, raw = status_raw
        if status != 200:
            raise RuntimeError(f"{what} -> HTTP {status}: {raw[:300]!r}")

    # ------------------------------------------------------------ engine power
    def wake(self, engine: str) -> None:
        if self.is_sleeping(engine):
            if self.gpu_free_mb() is None:
                log(C, f"waking {engine} with VRAM unknown", "warn")
            level = self.state["sleep_level"][engine]
            log(C, f"waking {engine} (from level {level})")
            t = time.time()
            if level >= 2:
                # Level 2 discarded the weights: re-allocate, reload from disk (page cache
                # when warm), then restore the KV cache and drop stale prefixes.
                self._ok(self.backend_simple(engine, "POST", "/wake_up?tags=weights"), "wake weights")
                self._ok(self.backend_simple(engine, "POST", "/collective_rpc",
                                             json.dumps({"method": "reload_weights"}).encode(), timeout=900),
                         "reload_weights")
                self._ok(self.backend_simple(engine, "POST", "/wake_up?tags=kv_cache"), "wake kv_cache")
                self.backend_simple(engine, "POST", "/reset_prefix_cache", timeout=60)
            else:
                self._ok(self.backend_simple(engine, "POST", "/wake_up"), "wake_up")
            took = time.time() - t
            self.stats[engine].inc("wakes")
            self.stats[engine].set("last_wake_sec", round(took, 1))
            log(C, f"{engine} awake in {took:.1f}s")
        self.sleeping[engine] = False

    def sleep(self, engine: str, reason: str) -> None:
        if self.is_sleeping(engine) is False:
            level = self.s.get("sleep_level")
            log(C, f"sleeping {engine} level={level} ({reason})")
            t = time.time()
            self._ok(self.backend_simple(engine, "POST", f"/sleep?level={level}", timeout=300), "sleep")
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

    # ------------------------------------------------------------ ComfyUI
    def _comfy_get(self, path: str):
        with urllib.request.urlopen(f"{self.comfy_url()}{path}", timeout=5) as r:
            return json.load(r)

    def gpu_free_mb(self) -> float | None:
        self.gpu = self.probe.read().as_dict()
        free = self.gpu.get("free_mb")
        return None if free is None else float(free)

    def comfy_running(self) -> bool | None:
        try:
            q = self._comfy_get("/queue")
            self.comfy_queue = {"running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", []))}
            return bool(q.get("queue_running"))
        except Exception:
            self.comfy_queue = {"running": None, "pending": None}
            return None

    def comfy_unload(self, target_mb: float, require_idle: bool = False) -> None:
        """Ask ComfyUI to drop its models and wait for target_mb free; no-op if it is down."""
        if require_idle:
            while self.comfy_running():  # prompts the hook didn't register
                time.sleep(5)
        try:
            req = urllib.request.Request(f"{self.comfy_url()}/free",
                                         data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            log(C, f"ComfyUI /free skipped: {e}")
            if self.gpu_free_mb() is None:
                log(C, "VRAM unknown; proceeding without a free-memory check", "warn")
            return
        deadline = time.time() + self.s.get("comfy_free_wait_sec")
        free, t = None, time.time()
        while time.time() < deadline:
            free = self.gpu_free_mb()
            if free is None:
                log(C, "VRAM unknown after ComfyUI unload; proceeding without a free-memory check", "warn")
                break
            if free >= target_mb:
                break
            time.sleep(0.25)
        log(C, f"ComfyUI unloaded in {time.time() - t:.1f}s (GPU free {free and round(free)} MiB, want {target_mb})")

    # ------------------------------------------------------------ ownership
    def engine_enter(self, engine: str) -> None:
        if engine not in self.engines:
            raise RuntimeError(f"unknown engine {engine}")
        if not self.is_enabled(engine):
            raise RuntimeError(f"{engine} is disabled")
        st = self.state
        with self.cond:
            # ComfyUI has priority while running or waiting; other engines finish their
            # in-flight requests first; nothing wakes while an engine is loading.
            while not (st["comfy_active"] == 0 and not self.any_inflight(engine) and not self.loading_engines()):
                self.cond.wait(timeout=5)
            if st["owner"] != engine:
                while st["comfy_active"] or self.any_inflight(engine) or self.comfy_running():
                    self.cond.wait(timeout=2)
                prev = st["owner"]
                self.sleep_others(engine, f"switching to {engine}")
                target = self.free_mb(engine)
                if prev not in self.engines:  # ComfyUI or nobody held the GPU
                    self.comfy_unload(target)
                attempts = self.s.get("wake_attempts")
                for attempt in range(1, attempts + 1):
                    try:
                        self.wake(engine)
                        break
                    except Exception as e:
                        log(C, f"wake {engine} failed (attempt {attempt}/{attempts}): {e}", "error")
                        if attempt == attempts:
                            st["owner"] = None
                            raise RuntimeError(f"{engine} could not wake: {e}") from e
                        self.comfy_unload(target)
                        time.sleep(2)
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
        if name == "warm" and not self.is_enabled(target):
            raise ValueError(f"{target} is disabled")
        if name == "sleep":
            def run():
                with self.cond:
                    if self.state["inflight"][target]:
                        log(C, f"not sleeping {target}: requests in flight", "warn")
                        return
                    self.sleep(target, "dashboard")
                    if self.state["owner"] == target:
                        self.state["owner"] = None
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
        for cfg in self.s.get("engines"):
            e = cfg["name"]
            enabled, boot = bool(cfg.get("enabled", True)), bool(cfg.get("boot"))
            if e not in self.engines:
                engines[e] = {
                    "listen": cfg["listen"], "backend": cfg["backend"],
                    "free_mb": int(cfg.get("free_mb") or 0), "text_only": self.text_only(e),
                    "state": "disabled" if not enabled else "down",
                    "enabled": enabled, "boot": boot, "inflight": 0, "idle_sec": 0,
                }
                continue
            loading = now < st["loading_until"][e]
            s = self.sleeping[e]
            engines[e] = {
                "listen": cfg["listen"], "backend": cfg["backend"], "free_mb": self.free_mb(e),
                "text_only": self.text_only(e), "enabled": enabled, "boot": boot,
                "state": "loading" if loading else {True: "asleep", False: "awake", None: "down"}[s],
                "inflight": st["inflight"][e],
                "idle_sec": round(now - st["last_infer"][e]),
                **self.stats[e].snapshot(),
            }
        return {"owner": st["owner"], "comfy_active": st["comfy_active"], "comfy_queue": self.comfy_queue,
                "gpu": self.gpu, "engines": engines, "idle_sleep_sec": self.s.get("idle_sleep_sec")}

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
                owner = st["owner"]
                if (owner in self.engines and st["inflight"][owner] == 0
                        and time.time() - st["last_infer"][owner] >= self.s.get("idle_sleep_sec")):
                    try:
                        self.sleep(owner, "idle")
                        st["owner"] = None
                    except Exception as e:
                        log(C, f"idle sleep {owner} failed: {e}", "error")

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
        with self.cond:
            try:
                self.sleep(engine, "startup")
            except Exception as e:
                log(C, f"startup sleep {engine} failed: {e}", "error")

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
        if not self.engines:
            log(C, "no enabled engines; GPU API listen ports are not bound", "warn")
        for e, cfg in self.engines.items():
            boot = "boot" if cfg.get("boot") else "parked"
            log(C, f"{e}: {host}:{cfg['listen']} -> {cfg['backend']} "
                   f"(wake needs {cfg['free_mb']} MiB free, {boot})")
        for target in (self.idle_watcher, self.poller):
            threading.Thread(target=target, daemon=True).start()
        for e, cfg in self.engines.items():
            threading.Thread(target=self.startup_watcher, args=(e,), daemon=True).start()
            srv = Server((host, cfg["listen"]), self.make_handler(e))
            self.servers.append(srv)
            threading.Thread(target=srv.serve_forever, daemon=True).start()

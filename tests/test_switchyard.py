"""Stdlib-only tests: python -m unittest discover -s tests

The arbiter tests run against fake vLLM engines and a fake ComfyUI on localhost.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from switchyard import __version__, analytics, config, events
from switchyard.arbiter import Arbiter, strip_images
from switchyard.backends import BACKENDS, ComfyBackend, VllmBackend
from switchyard.budget import lru_victims, resolve_cost, usable_mb
from switchyard.estimate import estimate_from_engine, estimate_vram_mb, gpu_layers_that_fit, quant_bytes
from switchyard.dashboard import Dashboard, EXPOSED_NO_TOKEN, is_loopback_host
from switchyard.httputil import Server
from switchyard.ovms_shim import OvmsShim, StreamThinkFilter, split_think
from switchyard.stt import Stt, parse_multipart


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(handler) -> tuple[Server, int]:
    port = free_port()
    srv = Server(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def post(url, body=None):
    # X-Switchyard: a non-browser client, as the dashboard's CSRF check expects.
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Switchyard": "test"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def wait_until(pred, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def this_thread_holds(cond) -> bool:
    """True if the calling thread owns the arbiter condition (RLock has no public locked())."""
    return cond._lock._is_owned()


def wrap_occupancy_http(arb, held: list):
    """Record whether the caller still holds the arbiter lock when wake/sleep/is_sleeping run."""
    def wrap(fn):
        def inner(*a, **k):
            held.append(this_thread_holds(arb.cond))
            return fn(*a, **k)
        return inner
    for b in arb.ctl.values():
        b.wake = wrap(b.wake)
        b.sleep = wrap(b.sleep)
        b.is_sleeping = wrap(b.is_sleeping)


def fake_engine():
    """A vLLM-like backend with sleep mode."""
    st = {"sleeping": True, "calls": [], "serving": 0, "hold": None, "is_sleeping_gets": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj):
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/is_sleeping":
                st["is_sleeping_gets"] += 1
                self._send({"is_sleeping": st["sleeping"]})
            else:
                self._send({"data": []})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            st["calls"].append(self.path)
            if self.path.startswith("/sleep"):
                st["sleeping"] = True
            elif self.path.startswith("/wake_up?tags=kv_cache") or self.path == "/wake_up":
                st["sleeping"] = False
            if self.path.startswith("/v1/chat/completions"):
                st["last_body"] = json.loads(body)
                st["serving"] += 1
                try:
                    hold = st["hold"]
                    if hold is not None:
                        hold.wait(timeout=10)
                    self._send({"id": "x", "awake_when_served": not st["sleeping"]})
                finally:
                    st["serving"] -= 1
            else:
                self._send({})

    return st, H


def fake_comfy():
    st = {"freed": 0}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj):
            raw = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/system_stats":
                self._send({"devices": [{"type": "cuda", "name": "cuda:0 Fake GPU : x",
                                         "vram_free": 30 * 2**30, "vram_total": 32 * 2**30}]})
            else:
                self._send({"queue_running": [], "queue_pending": []})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            st["freed"] += 1
            self._send({})

    return st, H


class ConfigTests(unittest.TestCase):
    def test_parse_engines(self):
        e = config.parse_engines("qwen=127.0.0.1:8002@8000:21400,vision=127.0.0.1:8003@8005:17500")
        self.assertEqual(e[1], {"name": "vision", "backend": "127.0.0.1:8003", "listen": 8005,
                                "free_mb": 17500, "enabled": True, "boot": False})
        flags = config.parse_engines("qwen=127.0.0.1:8002@8000:21400:boot,vision=127.0.0.1:8003@8005:17500:off")
        self.assertTrue(flags[0]["boot"] and flags[0]["enabled"])
        self.assertFalse(flags[1]["enabled"])
        extra = config.parse_engines("qwen=127.0.0.1:8002@8000:21400:18000:boot")
        self.assertEqual(extra[0]["free_mb"], 21400)
        self.assertEqual(extra[0]["vram_mb"], 18000)
        self.assertTrue(extra[0]["boot"])
        kept = config.parse_engines([{"name": "a", "backend": "127.0.0.1:1", "listen": 2,
                                      "free_mb": 3, "vram_mb": 9000, "quant": "q4_k"}])
        self.assertEqual(kept[0]["vram_mb"], 9000)
        self.assertEqual(kept[0]["quant"], "q4_k")
        self.assertTrue(kept[0]["enabled"])
        self.assertFalse(kept[0]["boot"])

    def test_shared_vram_defaults_off(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            self.assertFalse(s.get("shared_vram"))
            self.assertEqual(s.get("vram_budget_mb"), 0)
            self.assertEqual(s.get("vram_safety_mb"), 1024)

    def test_env_file_precedence_and_restart_flag(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            os.environ["IDLE_SLEEP_SEC"] = "300"
            try:
                s = config.Settings(path)
                self.assertEqual(s.get("idle_sleep_sec"), 300)
                r = s.update({"idle_sleep_sec": 45, "stt_port": 9999})
                self.assertEqual(r["restart_pending"], ["stt_port"])
                self.assertEqual(config.Settings(path).get("idle_sleep_sec"), 45)  # file beats env
            finally:
                del os.environ["IDLE_SLEEP_SEC"]

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            with self.assertRaises(ValueError):
                s.update({"sleep_level": 3})
            with self.assertRaises(ValueError):
                s.update({"engines": [{"name": "a", "backend": "nohostport", "listen": 1}]})
            with self.assertRaises(ValueError):
                s.update({"nope": 1})
            s.update({"gpu_probe": "nvidia-smi", "gpu_index": 1})
            self.assertEqual(s.get("gpu_probe"), "nvidia-smi")
            self.assertEqual(s.get("gpu_index"), 1)
            with self.assertRaises(ValueError):
                s.update({"gpu_probe": "cuda"})
            with self.assertRaises(ValueError):
                s.update({"engines": "a=127.0.0.1:1@2:3:nope"})
            with self.assertRaises(ValueError):
                s.update({"engines": "a=127.0.0.1:1@2:3:boot,b=127.0.0.1:3@4:5:boot"})
            r = s.update({"engines": "a=127.0.0.1:1@2:3:boot,b=127.0.0.1:3@4:5:off"})
            self.assertIn("engines", r["changed"])
            self.assertTrue(s.get("engines")[0]["boot"])
            self.assertFalse(s.get("engines")[1]["enabled"])

    def test_reset_creates_parent_and_drops_saved_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "s.json")
            s = config.Settings(path)
            s.update({"idle_sleep_sec": 45, "stt_port": 9999})
            os.remove(path)
            os.rmdir(os.path.dirname(path))
            r = s.reset("idle_sleep_sec")
            self.assertEqual(r, {"reset": "idle_sleep_sec"})
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(os.path.join(d, "nested", "s.tmp")))
            with open(path) as f:
                saved = json.loads(f.read())
            self.assertNotIn("idle_sleep_sec", saved)
            self.assertEqual(saved["stt_port"], 9999)

    def test_health_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            self.assertEqual(s.get("wake_attempts"), 3)
            self.assertEqual(s.get("wake_backoff_sec"), 1.0)
            self.assertEqual(s.get("health_interval_sec"), 15.0)
            self.assertEqual(s.get("health_timeout_sec"), 5.0)


class HelperTests(unittest.TestCase):
    def test_strip_images(self):
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:"}}]}]}).encode()
        out, n = strip_images(body)
        self.assertEqual(n, 1)
        self.assertNotIn("image_url", out.decode())
        self.assertEqual(strip_images(b'{"messages":[]}')[1], 0)

    def test_multipart(self):
        b = "XyZ"
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n"
                f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n").encode() + b"\x00\x01RIFF" + f"\r\n--{b}--\r\n".encode()
        form = parse_multipart(f"multipart/form-data; boundary={b}", body)
        self.assertEqual(form["model"], b"m1")
        self.assertEqual(form["file"], b"\x00\x01RIFF")


class ThinkTests(unittest.TestCase):
    def test_split_non_stream(self):
        m = {"content": "Sourdough?</think>\U0001F35E How to Bake Sourdough"}
        split_think(m)
        self.assertEqual(m["content"], "\U0001F35E How to Bake Sourdough")
        self.assertEqual(m["reasoning_content"], "Sourdough?")
        clean = {"content": "plain answer"}
        split_think(clean)
        self.assertEqual(clean, {"content": "plain answer"})

    def test_stream(self):
        f = StreamThinkFilter()
        out = [f.feed({"choices": [{"delta": {"role": "assistant"}}]}),
               f.feed({"choices": [{"delta": {"content": "thought"}}]}),
               f.feed({"choices": [{"delta": {"content": "</think> Ans"}}]}),
               f.feed({"choices": [{"delta": {"content": "wer"}}]})]
        self.assertEqual(out[1], None)
        self.assertEqual(out[2]["choices"][0]["delta"]["content"], "Ans")
        self.assertEqual(out[3]["choices"][0]["delta"]["content"], "wer")
        g = StreamThinkFilter()
        g.feed({"choices": [{"delta": {"content": "hi"}}]})
        last = g.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        self.assertEqual(last["choices"][0]["delta"]["content"], "hi")


class ArbiterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a_state, ha = fake_engine()
        self.b_state, hb = fake_engine()
        self.c_state, hc = fake_comfy()
        self.srv = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = self.srv
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": f"a=127.0.0.1:{pa}@{self.la}:1000,b=127.0.0.1:{pb}@{self.lb}:1000",
                  "text_only_engines": "a", "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                  "gpu_probe": "comfy"})
        self.arb = Arbiter(s)
        self.arb.start()
        time.sleep(0.3)

    def tearDown(self):
        analytics.reset()
        for srv in [s for s, _ in self.srv] + self.arb.servers:
            srv.shutdown()
            srv.server_close()
        self.tmp.cleanup()

    def chat(self, port, content="hi"):
        return post(f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": content}]})

    def test_wakes_one_engine_at_a_time(self):
        self.assertTrue(self.chat(self.la)["awake_when_served"])
        self.assertFalse(self.a_state["sleeping"])
        self.assertTrue(self.chat(self.lb)["awake_when_served"])
        self.assertTrue(self.a_state["sleeping"], "a must sleep before b wakes")
        self.assertFalse(self.b_state["sleeping"])
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["owner"], "b")
        self.assertEqual(st["engines"]["a"]["requests"], 1)
        self.assertEqual(st["gpu"]["source"], "comfy")
        self.assertEqual(st["gpu"]["free_mb"], round(30 * 2**30 / 2**20))
        self.assertEqual(st["gpu"]["used_mb"], st["gpu"]["total_mb"] - st["gpu"]["free_mb"])
        self.assertEqual(st["gpu"]["name"], "Fake GPU")

    def test_comfy_acquire_sleeps_engines(self):
        self.chat(self.la)
        post(f"http://127.0.0.1:{self.la}/gpu/acquire")
        self.assertTrue(self.a_state["sleeping"])
        self.assertEqual(self.arb.state["owner"], "comfy")
        post(f"http://127.0.0.1:{self.la}/gpu/release")
        self.chat(self.lb)
        self.assertGreaterEqual(self.c_state["freed"], 1, "ComfyUI must be unloaded before an engine wakes")

    def test_text_only_strips_images(self):
        self.chat(self.la, [{"type": "image_url", "image_url": {"url": "data:x"}}])
        self.assertEqual(self.a_state["last_body"]["messages"][0]["content"][0]["type"], "text")

    def test_wake_warns_when_vram_unknown(self):
        self.arb.s.update({"gpu_probe": "off"})
        self.arb.probe._cached = None
        before = max((e["id"] for e in events.since(0)), default=0)
        self.chat(self.la)
        msgs = [e["msg"] for e in events.since(before)]
        self.assertTrue(any("VRAM unknown" in m for m in msgs))
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["gpu"]["source"], "off")
        self.assertIsNone(st["gpu"]["free_mb"])

    def test_boot_load_wakes_marked_engine(self):
        engines = json.loads(json.dumps(self.arb.s.get("engines")))
        engines[0]["boot"] = True
        self.arb.s.update({"engines": engines})
        self.assertTrue(self.a_state["sleeping"])
        self.arb.on_backend_ready("a")
        self.assertFalse(self.a_state["sleeping"])
        self.assertEqual(self.arb.state["owner"], "a")
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertTrue(st["engines"]["a"]["boot"])
        self.assertEqual(st["engines"]["a"]["state"], "awake")

    def test_backend_ready_parks_without_boot(self):
        self.arb.on_backend_ready("a")
        self.assertTrue(self.a_state["sleeping"])
        self.assertIsNone(self.arb.state["owner"])

    def test_disabled_live_refuses_wake(self):
        engines = json.loads(json.dumps(self.arb.s.get("engines")))
        engines[0]["enabled"] = False
        self.arb.s.update({"engines": engines})
        with self.assertRaises(RuntimeError):
            self.arb.engine_enter("a")
        with self.assertRaises(ValueError):
            self.arb.action("warm", "a")

    def test_wake_records_analytics(self):
        analytics.configure(self.arb.s, path=os.path.join(self.tmp.name, "analytics.sqlite"))
        self.chat(self.la)
        analytics.get().flush()
        kinds = [e["kind"] for e in analytics.history()["events"]]
        self.assertIn("wake", kinds)
        self.assertIn("unload", kinds)

    def test_status_reports_exclusive_placement(self):
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["placement"], "exclusive")
        self.assertIn("budget", st)
        self.assertIn("vram_source", st["engines"]["a"])

    def test_exclusive_serializes_other_engine(self):
        gate = threading.Event()
        self.a_state["hold"] = gate
        ta = threading.Thread(target=lambda: self.chat(self.la))
        ta.start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"]))
        tb_done = []
        tb = threading.Thread(target=lambda: tb_done.append(self.chat(self.lb)))
        tb.start()
        time.sleep(0.3)
        self.assertFalse(self.b_state["serving"], "b must wait while a is in flight")
        self.assertFalse(tb_done)
        gate.set()
        ta.join(5)
        tb.join(5)
        self.assertTrue(tb_done)
        self.assertTrue(self.a_state["sleeping"])
        self.assertFalse(self.b_state["sleeping"])

    def test_occupancy_http_drops_the_lock(self):
        held = []
        wrap_occupancy_http(self.arb, held)
        self.chat(self.la)
        self.chat(self.lb)
        self.assertTrue(held, "wake/sleep should have run")
        self.assertFalse(any(held), "occupancy HTTP must not run under the arbiter lock")


class BackendTests(unittest.TestCase):
    def test_vllm_health_wake_sleep(self):
        st, handler = fake_engine()
        srv, port = serve(handler)
        try:
            b = VllmBackend("127.0.0.1", port)
            self.assertTrue(b.health())
            self.assertTrue(b.is_sleeping())
            self.assertIsNone(b.estimate_vram())
            b.wake(2)
            self.assertFalse(b.is_sleeping())
            self.assertIn("/wake_up?tags=weights", st["calls"])
            self.assertIn("/collective_rpc", st["calls"])
            self.assertTrue(any(c.startswith("/wake_up?tags=kv_cache") for c in st["calls"]))
            b.sleep(2)
            self.assertTrue(b.is_sleeping())
            self.assertTrue(any(c.startswith("/sleep?level=") for c in st["calls"]))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_vllm_level1_wake_skips_reload(self):
        st, handler = fake_engine()
        srv, port = serve(handler)
        try:
            VllmBackend("127.0.0.1", port).wake(1)
            self.assertEqual(st["calls"], ["/wake_up"])
            self.assertFalse(st["sleeping"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_comfy_health_vram_unload(self):
        st, handler = fake_comfy()
        srv, port = serve(handler)
        try:
            with tempfile.TemporaryDirectory() as d:
                s = config.Settings(os.path.join(d, "s.json"))
                s.update({"comfy_url": f"http://127.0.0.1:{port}", "comfy_free_wait_sec": 1})
                b = ComfyBackend(s)
                self.assertTrue(b.health())
                free, snap = b.vram()
                self.assertGreater(free, 0)
                self.assertEqual(snap["total_mb"], 32768)
                self.assertIn("Fake GPU", snap["name"])
                self.assertEqual(b.estimate_vram(), snap["total_mb"] - snap["free_mb"])
                b.sleep()
                self.assertGreaterEqual(st["freed"], 1)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_backends_registry_is_the_extension_point(self):
        self.assertEqual(set(BACKENDS), {"vllm", "comfy"})
        self.assertIs(BACKENDS["vllm"], VllmBackend)
        self.assertIs(BACKENDS["comfy"], ComfyBackend)


class EstimateTests(unittest.TestCase):
    def test_fp16_heavier_than_q4_and_context_grows_kv(self):
        fp16 = estimate_vram_mb(7_000_000_000, quant="fp16", context_length=4096)
        q4 = estimate_vram_mb(7_000_000_000, quant="q4_k", context_length=4096)
        q4_long = estimate_vram_mb(7_000_000_000, quant="q4_k", context_length=16384)
        self.assertGreater(fp16, 12_000)
        self.assertLess(fp16, 18_000)
        self.assertGreater(fp16, q4 * 2)
        self.assertLess(q4, 8_000)
        self.assertGreater(q4_long, q4)

    def test_quant_aliases(self):
        self.assertEqual(quant_bytes("float16"), quant_bytes("fp16"))
        self.assertEqual(quant_bytes("q4_k_m"), quant_bytes("q4_k"))
        self.assertEqual(quant_bytes("awq"), quant_bytes("int4"))
        with self.assertRaises(ValueError):
            quant_bytes("nope")

    def test_gpu_layers_that_fit(self):
        full = gpu_layers_that_fit(24_000, 7_000_000_000, quant="fp16", context_length=4096)
        partial = gpu_layers_that_fit(8_192, 7_000_000_000, quant="fp16", context_length=4096)
        quant = gpu_layers_that_fit(8_192, 7_000_000_000, quant="q4_k", context_length=4096)
        self.assertEqual(full, 32)
        self.assertGreater(partial, 0)
        self.assertLess(partial, full)
        self.assertEqual(quant, 32)

    def test_head_dim_from_hidden_size_uses_attention_heads(self):
        # 8B GQA: hidden 4096, 32 attention heads, 8 KV heads -> head_dim 128, not 512.
        explicit = estimate_vram_mb(8_000_000_000, n_kv_heads=8, head_dim=128)
        derived = estimate_vram_mb(8_000_000_000, n_kv_heads=8, hidden_size=4096, n_heads=32)
        self.assertEqual(derived, explicit)

    def test_unknown_quant_rejected_by_settings(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            with self.assertRaises(ValueError):
                s.update({"engines": [{"name": "a", "backend": "127.0.0.1:1", "listen": 2,
                                       "free_mb": 1, "n_params": 7_000_000_000, "quant": "q9_zz"}]})

    def test_estimate_from_engine_needs_params(self):
        self.assertIsNone(estimate_from_engine({"free_mb": 1000}))
        mb = estimate_from_engine({"n_params": 7_000_000_000, "quant": "q4_k", "context_length": 4096})
        self.assertEqual(mb, estimate_vram_mb(7_000_000_000, quant="q4_k", context_length=4096))


class BudgetTests(unittest.TestCase):
    def test_usable_and_resolve_cost(self):
        self.assertEqual(usable_mb(8192, 1024), 7168)
        self.assertIsNone(usable_mb(0, 1024))
        self.assertEqual(resolve_cost(100, 200, 300, 400), (100, "measured"))
        self.assertEqual(resolve_cost(None, 200, 300, 400), (200, "configured"))
        self.assertEqual(resolve_cost(None, None, 300, 400), (300, "estimate"))
        self.assertEqual(resolve_cost(None, None, None, 400), (400, "free_mb"))
        self.assertEqual(resolve_cost(0, 0, 0, 0), (0, "free_mb"))

    def test_lru_evicts_oldest_until_it_fits(self):
        resident = {"a": 1.0, "b": 2.0}
        costs = {"a": 4000, "b": 4000}
        self.assertEqual(lru_victims(resident, costs, "c", 4000, 12_000), [])
        self.assertEqual(lru_victims(resident, costs, "c", 4000, 9000), ["a"])
        self.assertEqual(lru_victims(resident, costs, "c", 4000, 7000), ["a", "b"])
        self.assertEqual(lru_victims(resident, costs, "c", 8000, 7000), ["a", "b"])


class ArbiterSharedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a_state, ha = fake_engine()
        self.b_state, hb = fake_engine()
        self.c_state, hc = fake_comfy()
        self.srv = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = self.srv
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": f"a=127.0.0.1:{pa}@{self.la}:1000:2000,b=127.0.0.1:{pb}@{self.lb}:1000:2000",
                  "text_only_engines": "a", "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                  "shared_vram": True, "vram_budget_mb": 8000, "vram_safety_mb": 1000, "idle_sleep_sec": 3600})
        self.arb = Arbiter(s)
        self.arb.start()
        time.sleep(0.3)

    def tearDown(self):
        for srv in [s for s, _ in self.srv] + self.arb.servers:
            srv.shutdown()
            srv.server_close()
        self.tmp.cleanup()

    def chat(self, port, content="hi"):
        return post(f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": content}]})

    def test_keeps_both_resident_when_they_fit(self):
        self.chat(self.la)
        self.chat(self.lb)
        self.assertFalse(self.a_state["sleeping"])
        self.assertFalse(self.b_state["sleeping"])
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["placement"], "shared")
        self.assertEqual(st["owner"], "b")
        self.assertCountEqual(st["resident"], ["a", "b"])

    def test_lru_evicts_when_the_budget_is_tight(self):
        self.arb.s.update({"engines": [
            {**self.arb.s.engine("a"), "vram_mb": 4000},
            {**self.arb.s.engine("b"), "vram_mb": 4000},
        ]})
        self.chat(self.la)
        self.chat(self.lb)
        self.assertTrue(self.a_state["sleeping"], "a is LRU and must move for b")
        self.assertFalse(self.b_state["sleeping"])
        self.chat(self.la)
        self.assertFalse(self.a_state["sleeping"])
        self.assertTrue(self.b_state["sleeping"], "b is LRU and must move for a")

    def test_comfy_still_sleeps_every_engine(self):
        self.chat(self.la)
        self.chat(self.lb)
        post(f"http://127.0.0.1:{self.la}/gpu/acquire")
        self.assertTrue(self.a_state["sleeping"])
        self.assertTrue(self.b_state["sleeping"])
        self.assertEqual(self.arb.state["owner"], "comfy")
        post(f"http://127.0.0.1:{self.la}/gpu/release")

    def test_cost_falls_back_to_estimate(self):
        self.arb.s.update({"engines": [
            {**self.arb.s.engine("a"), "vram_mb": 0, "n_params": 7_000_000_000, "quant": "q4_k",
             "context_length": 4096, "free_mb": 1000},
            self.arb.s.engine("b"),
        ]})
        cost, src = self.arb.cost_of("a")
        self.assertEqual(src, "estimate")
        self.assertEqual(cost, estimate_vram_mb(7_000_000_000, quant="q4_k", context_length=4096))

    def test_resident_engines_serve_in_parallel(self):
        self.chat(self.la)
        self.chat(self.lb)
        gate = threading.Event()
        self.a_state["hold"] = self.b_state["hold"] = gate
        ra, rb = {}, {}
        ta = threading.Thread(target=lambda: ra.update(self.chat(self.la)))
        tb = threading.Thread(target=lambda: rb.update(self.chat(self.lb)))
        ta.start()
        tb.start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"] and self.b_state["serving"]),
                        "both resident engines should be serving at once")
        gate.set()
        ta.join(5)
        tb.join(5)
        self.assertTrue(ra.get("awake_when_served"))
        self.assertTrue(rb.get("awake_when_served"))

    def test_eviction_waits_for_victim_inflight(self):
        self.arb.s.update({"engines": [
            {**self.arb.s.engine("a"), "vram_mb": 4000},
            {**self.arb.s.engine("b"), "vram_mb": 4000},
        ]})
        self.chat(self.la)
        gate = threading.Event()
        self.a_state["hold"] = gate
        ta = threading.Thread(target=lambda: self.chat(self.la))
        ta.start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"]))
        tb_done = []
        tb = threading.Thread(target=lambda: tb_done.append(self.chat(self.lb)))
        tb.start()
        time.sleep(0.3)
        self.assertTrue(self.a_state["serving"])
        self.assertFalse(self.a_state["sleeping"], "cannot evict a while it still has a request")
        self.assertFalse(tb_done)
        self.assertFalse(self.b_state["serving"])
        gate.set()
        ta.join(5)
        tb.join(5)
        self.assertTrue(tb_done)
        self.assertTrue(self.a_state["sleeping"])
        self.assertFalse(self.b_state["sleeping"])

    def test_draining_victim_does_not_starve_the_wake(self):
        """While b waits for a to drain, new requests to a must queue, not join a."""
        self.arb.s.update({"engines": [
            {**self.arb.s.engine("a"), "vram_mb": 4000},
            {**self.arb.s.engine("b"), "vram_mb": 4000},
        ]})
        self.chat(self.la)
        gate = threading.Event()
        self.a_state["hold"] = gate
        ta = threading.Thread(target=lambda: self.chat(self.la))
        ta.start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"]))
        tb = threading.Thread(target=lambda: self.chat(self.lb))
        tb.start()
        self.assertTrue(wait_until(lambda: self.arb.state["switching"] == "b"))
        self.assertIn("a", self.arb.state["evicting"])
        ta2 = threading.Thread(target=lambda: self.chat(self.la))
        ta2.start()
        time.sleep(0.3)
        self.assertEqual(self.arb.state["inflight"]["a"], 1, "late request to a must not be admitted")
        gate.set()
        for t in (ta, tb, ta2):
            t.join(10)
        self.assertEqual(self.arb.state["inflight"], {"a": 0, "b": 0})

    def test_comfy_cannot_interleave_with_a_draining_switch(self):
        self.arb.s.update({"engines": [
            {**self.arb.s.engine("a"), "vram_mb": 4000},
            {**self.arb.s.engine("b"), "vram_mb": 4000},
        ]})
        self.chat(self.la)
        gate = threading.Event()
        self.a_state["hold"] = gate
        ta = threading.Thread(target=lambda: self.chat(self.la))
        ta.start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"]))
        tb = threading.Thread(target=lambda: self.chat(self.lb))
        tb.start()
        self.assertTrue(wait_until(lambda: self.arb.state["switching"] == "b"))
        tc = threading.Thread(target=lambda: post(f"http://127.0.0.1:{self.la}/gpu/acquire"))
        tc.start()
        time.sleep(0.3)
        self.assertEqual(self.arb.state["switching"], "b", "ComfyUI must not take over a reserved switch")
        gate.set()
        for t in (ta, tb, tc):
            t.join(10)
        self.assertEqual(self.arb.state["owner"], "comfy")
        self.assertTrue(self.b_state["sleeping"])
        post(f"http://127.0.0.1:{self.la}/gpu/release")

    def test_fast_path_does_not_probe_is_sleeping(self):
        self.chat(self.la)
        n = self.a_state["is_sleeping_gets"]
        self.chat(self.la)
        self.assertEqual(self.a_state["is_sleeping_gets"], n,
                         "already-resident enter must use cached sleep state, not HTTP")

    def test_occupancy_http_drops_the_lock(self):
        held = []
        wrap_occupancy_http(self.arb, held)
        self.chat(self.la)
        self.chat(self.lb)
        self.assertTrue(held, "wake/sleep should have run")
        self.assertFalse(any(held), "occupancy HTTP must not run under the arbiter lock")


class OvmsShimTests(unittest.TestCase):
    def test_adds_id_and_wraps_rerank(self):
        seen = {}

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                seen[self.path] = body
                obj = ({"results": [{"index": 0, "relevance_score": 0.9}]} if "rerank" in self.path
                       else {"choices": [{"message": {"content": "ok"}}]})
                raw = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        with tempfile.TemporaryDirectory() as d:
            up, port = serve(H)
            s = config.Settings(os.path.join(d, "s.json"))
            shim_port = free_port()
            s.update({"ovms_upstream": f"http://127.0.0.1:{port}", "ovms_port": shim_port,
                      "listen_host": "127.0.0.1"})
            shim = OvmsShim(s)
            shim.start()
            time.sleep(0.2)
            r = post(f"http://127.0.0.1:{shim_port}/v1/chat/completions", {"messages": []})
            self.assertTrue(r["id"].startswith("chatcmpl-"))
            self.assertIn("/v3/chat/completions", seen)
            r = post(f"http://127.0.0.1:{shim_port}/v1/rerank",
                     {"query": "q", "documents": ["d"], "return_documents": True})
            self.assertEqual(r["results"][0]["document"]["text"], "d")
            self.assertIn("<Document>: d", seen["/v3/rerank"]["documents"][0])
            shim.server.shutdown()
            shim.server.server_close()
            up.shutdown()
            up.server_close()

    def test_honors_listen_host(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            port = free_port()
            s.update({"listen_host": "127.0.0.1", "ovms_port": port})
            shim = OvmsShim(s)
            shim.start()
            try:
                self.assertEqual(shim.server.server_address[0], "127.0.0.1")
                self.assertEqual(shim.server.server_address[1], port)
            finally:
                shim.server.shutdown()
                shim.server.server_close()


class SttBindTests(unittest.TestCase):
    def test_honors_listen_host(self):
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            port = free_port()
            s.update({"listen_host": "127.0.0.1", "stt_port": port, "stt_model_path": os.path.join(d, "no-model")})
            stt = Stt(s)
            stt.reload = lambda: None
            stt.start()
            try:
                self.assertEqual(stt.server.server_address[0], "127.0.0.1")
                self.assertEqual(stt.server.server_address[1], port)
            finally:
                stt.server.shutdown()
                stt.server.server_close()


class CliTests(unittest.TestCase):
    def test_help_lists_settings_and_dashboard_warning(self):
        from switchyard.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        out = buf.getvalue()
        self.assertIn("SWITCHYARD_SETTINGS", out)
        self.assertIn("ARBITER_ENABLED", out)
        self.assertIn("STT_ENABLED", out)
        self.assertIn("DASHBOARD_HOST", out)
        self.assertIn("no auth", out)
        self.assertIn("localhost", out)
        self.assertIn(__version__, out)

    def test_version(self):
        from switchyard.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())

    def test_no_args_does_not_exit(self):
        from switchyard.__main__ import _parser
        ns = _parser().parse_args([])
        self.assertIsNone(ns.command)


def httpjson(url, method="GET", body=None, headers=None):
    h = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            obj = json.loads(raw) if raw else {}
        except ValueError:
            obj = {}
        return e.code, obj


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dash = None

    def tearDown(self):
        if self.dash and self.dash.server:
            self.dash.server.shutdown()
            self.dash.server.server_close()
        self.tmp.cleanup()

    def start_dash(self, host="127.0.0.1", token="", **updates):
        port = free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"dashboard_host": host, "dashboard_port": port, "dashboard_token": token, **updates})
        logged = events.since(0)
        self.last_id = logged[-1]["id"] if logged else 0
        self.dash = Dashboard(s, {"arbiter": None, "ovms": None, "stt": None})
        self.dash.start()
        time.sleep(0.15)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        return self.dash

    def test_is_loopback_host(self):
        for h in ("127.0.0.1", "127.0.0.2", "::1", "localhost", "[::1]"):
            self.assertTrue(is_loopback_host(h), h)
        for h in ("0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "gpu.local"):
            self.assertFalse(is_loopback_host(h), h)

    def test_info_and_html_without_token(self):
        self.start_dash()
        code, info = httpjson(self.base + "/api/info")
        self.assertEqual(code, 200)
        self.assertEqual(info["auth_required"], False)
        self.assertIn("version", info)
        self.assertTrue(info["host"].endswith(f":{self.port}"))
        code, _ = httpjson(self.base + "/api/status")
        self.assertEqual(code, 200)
        req = urllib.request.Request(self.base + "/")
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertIn(b"Switchyard", r.read())

    def test_auth_missing_wrong_correct(self):
        self.start_dash(token="s3cret")
        code, info = httpjson(self.base + "/api/info")
        self.assertEqual(code, 200)
        self.assertEqual(info["auth_required"], True)
        req = urllib.request.Request(self.base + "/")
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)

        code, body = httpjson(self.base + "/api/status")
        self.assertEqual(code, 401)
        self.assertEqual(body["error"], "unauthorized")
        code, body = httpjson(self.base + "/api/status",
                              headers={"Authorization": "Bearer wrong"})
        self.assertEqual(code, 401)
        code, body = httpjson(self.base + "/api/status",
                              headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)
        self.assertIn("version", body)
        code, _ = httpjson(self.base + "/api/settings",
                           headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)

    def test_token_is_write_only_in_settings(self):
        self.start_dash(token="s3cret")
        code, body = httpjson(self.base + "/api/settings",
                              headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)
        self.assertEqual(body["values"]["dashboard_token"], "")
        self.assertIn("dashboard_token", body["secrets_set"])
        self.assertNotIn("s3cret", json.dumps(body))

    def test_rebinding_host_refused_without_token(self):
        self.start_dash()
        code, body = httpjson(self.base + "/api/status", headers={"Host": f"attacker.example:{self.port}"})
        self.assertEqual(code, 403)
        self.assertIn("Host", body["error"])
        origin = f"http://attacker.example:{self.port}"
        code, _ = httpjson(self.base + "/api/settings", method="POST", body={"idle_sleep_sec": 90},
                           headers={"Host": f"attacker.example:{self.port}", "Origin": origin})
        self.assertEqual(code, 403)
        for host in (f"localhost:{self.port}", f"127.0.0.1:{self.port}", f"[::1]:{self.port}"):
            code, _ = httpjson(self.base + "/api/status", headers={"Host": host})
            self.assertEqual(code, 200, host)

    def test_any_host_allowed_with_token(self):
        self.start_dash(token="s3cret")
        code, _ = httpjson(self.base + "/api/status",
                           headers={"Host": f"gpu-box.lan:{self.port}", "Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)

    def test_bad_logs_since_is_400(self):
        self.start_dash()
        code, _ = httpjson(self.base + "/api/logs?since=abc")
        self.assertEqual(code, 400)

    def test_post_required_on_mutating(self):
        self.start_dash()
        for path in ("/api/action", "/api/restart"):
            code, body = httpjson(self.base + path)
            self.assertEqual(code, 405, path)
            self.assertEqual(body["error"], "POST required")

    def test_csrf_origin_or_custom_header(self):
        self.start_dash(token="s3cret")
        url = self.base + "/api/settings"
        auth = {"Authorization": "Bearer s3cret"}
        # No Origin and no X-Switchyard: refused.
        code, body = httpjson(url, method="POST", body={"idle_sleep_sec": 90}, headers=auth)
        self.assertEqual(code, 403)
        self.assertEqual(body["error"], "cross-origin request refused")
        # Foreign Origin, even with the custom header: refused.
        code, _ = httpjson(url, method="POST", body={"idle_sleep_sec": 90},
                           headers={**auth, "Origin": "http://evil.test", "X-Switchyard": "1"})
        self.assertEqual(code, 403)
        # Custom header, no Origin (curl / scripts).
        code, body = httpjson(url, method="POST", body={"idle_sleep_sec": 91},
                              headers={**auth, "X-Switchyard": "1"})
        self.assertEqual(code, 200)
        self.assertIn("idle_sleep_sec", body["changed"])
        # Matching Origin, no custom header (browser).
        origin = f"http://127.0.0.1:{self.port}"
        code, body = httpjson(url, method="POST", body={"idle_sleep_sec": 92},
                              headers={**auth, "Origin": origin})
        self.assertEqual(code, 200)

    def test_warning_non_loopback_without_token(self):
        self.start_dash(host="0.0.0.0")
        msgs = [e["msg"] for e in events.since(self.last_id)]
        self.assertTrue(any("SWITCHYARD_TOKEN" in m and "not loopback" in m for m in msgs), msgs)
        self.assertIn("0.0.0.0", EXPOSED_NO_TOKEN.format(host="0.0.0.0"))
        code, info = httpjson(self.base + "/api/info")
        self.assertEqual(code, 200)
        self.assertEqual(info["auth_required"], False)
        self.assertTrue(info["host"].startswith("0.0.0.0:"))

    def test_no_warning_loopback_or_token(self):
        self.start_dash(host="127.0.0.1")
        msgs = [e["msg"] for e in events.since(self.last_id)]
        self.assertFalse(any("SWITCHYARD_TOKEN is empty" in m for m in msgs), msgs)
        if self.dash.server:
            self.dash.server.shutdown()
            self.dash.server.server_close()
        self.start_dash(host="0.0.0.0", token="s3cret")
        msgs = [e["msg"] for e in events.since(self.last_id)]
        self.assertFalse(any("SWITCHYARD_TOKEN is empty" in m for m in msgs), msgs)


class EngineSetupTests(unittest.TestCase):
    """Disabled engines are not bound; boot-load runs when the backend comes up."""

    def tearDown(self):
        for srv in getattr(self, "servers", []):
            srv.shutdown()
            srv.server_close()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def _start(self, spec, **extra):
        self.tmp = tempfile.TemporaryDirectory()
        a_state, ha = fake_engine()
        b_state, hb = fake_engine()
        c_state, hc = fake_comfy()
        served = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = served
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": spec.format(pa=pa, pb=pb, la=self.la, lb=self.lb),
                  "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                  "gpu_probe": "comfy", **extra})
        self.arb = Arbiter(s)
        self.arb.start()
        self.servers = [srv for srv, _ in served] + self.arb.servers
        self.a_state, self.b_state = a_state, b_state
        time.sleep(0.3)
        return s

    def test_disabled_engine_is_not_bound(self):
        self._start("a=127.0.0.1:{pa}@{la}:1000,b=127.0.0.1:{pb}@{lb}:1000:off")
        self.assertIn("a", self.arb.engines)
        self.assertNotIn("b", self.arb.engines)
        self.assertEqual(len(self.arb.servers), 1)
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["engines"]["b"]["state"], "disabled")
        self.assertFalse(st["engines"]["b"]["enabled"])
        with self.assertRaises(urllib.error.URLError):
            get(f"http://127.0.0.1:{self.lb}/gpu/status")
        post(f"http://127.0.0.1:{self.la}/v1/chat/completions",
             {"messages": [{"role": "user", "content": "hi"}]})
        self.assertFalse(self.a_state["sleeping"])

    def test_boot_flag_wakes_on_startup(self):
        self._start("a=127.0.0.1:{pa}@{la}:1000:boot,b=127.0.0.1:{pb}@{lb}:1000")
        deadline = time.time() + 2
        while time.time() < deadline and self.a_state["sleeping"]:
            time.sleep(0.05)
        self.assertFalse(self.a_state["sleeping"])
        self.assertTrue(self.b_state["sleeping"])
        self.assertEqual(self.arb.state["owner"], "a")


class DashboardSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.port = free_port()
        self.s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        self.s.update({"dashboard_host": "127.0.0.1", "dashboard_port": self.port,
                       "engines": "qwen=127.0.0.1:8002@8000:21400:boot,vision=127.0.0.1:8003@8005:17500"})
        self.arb = _FakeArbiter()
        self.dash = Dashboard(self.s, {"arbiter": self.arb, "ovms": None, "stt": None})
        self.dash.start()
        time.sleep(0.15)
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.dash.server.shutdown()
        self.dash.server.server_close()
        self.tmp.cleanup()

    def test_binds_localhost(self):
        self.assertEqual(self.dash.server.server_address[0], "127.0.0.1")

    def test_status_and_settings_schema(self):
        st = get(self.base + "/api/status")
        self.assertEqual(st["arbiter"]["owner"], None)
        self.assertTrue(st["arbiter"]["engines"]["qwen"]["boot"])
        schema = get(self.base + "/api/settings")
        self.assertIn("engines", schema["sections"])
        self.assertIn("switching", schema["sections"])
        self.assertIn("health", schema["sections"])
        self.assertIn("vram", schema["sections"])
        self.assertIn("budget", schema["sections"])
        qwen = next(e for e in schema["values"]["engines"] if e["name"] == "qwen")
        self.assertTrue(qwen["boot"] and qwen["enabled"])
        self.assertEqual(schema["values"]["dashboard_host"], "127.0.0.1")
        html = urllib.request.urlopen(self.base + "/", timeout=5).read()
        self.assertIn(b'data-f="boot"', html)
        self.assertIn(b"sec-h", html)

    def test_settings_persist_and_validate(self):
        engines = [{"name": "qwen", "backend": "127.0.0.1:8002", "listen": 8000,
                    "free_mb": 20000, "enabled": True, "boot": False, "vram_mb": 9000},
                   {"name": "vision", "backend": "127.0.0.1:8003", "listen": 8005,
                    "free_mb": 17500, "enabled": False, "boot": False}]
        r = post(self.base + "/api/settings", {"engines": engines, "idle_sleep_sec": 45,
                                               "gpu_probe": "nvidia-smi", "gpu_index": 1})
        self.assertIn("engines", r["changed"])
        saved = json.loads(Path(self.s.path).read_text())
        self.assertEqual(saved["idle_sleep_sec"], 45)
        self.assertEqual(saved["engines"][0]["vram_mb"], 9000)
        self.assertFalse(saved["engines"][1]["enabled"])
        self.assertFalse(os.path.exists(str(self.s.path) + ".tmp") or
                         os.path.exists(str(self.s.path.with_suffix(".tmp"))))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            post(self.base + "/api/settings",
                 {"engines": "a=127.0.0.1:1@2:3:boot,b=127.0.0.1:3@4:5:boot"})
        self.assertEqual(cm.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            post(self.base + "/api/settings", {"sleep_level": 9})
        self.assertEqual(cm.exception.code, 400)

    def test_action_and_cross_origin_refused(self):
        r = post(self.base + "/api/action", {"component": "arbiter", "action": "warm", "target": "qwen"})
        self.assertEqual(r["result"], "started")
        self.assertEqual(self.arb.actions[-1], ("warm", "qwen"))
        req = urllib.request.Request(self.base + "/api/settings", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Origin": "http://evil.example", "Host": f"127.0.0.1:{self.port}"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 403)


class _FakeArbiter:
    def __init__(self):
        self.actions = []

    def status(self):
        return {"owner": None, "comfy_active": 0, "comfy_queue": {"running": 0, "pending": 0},
                "gpu": {"name": "Fake GPU", "free_mb": 20000, "used_mb": 4000, "total_mb": 24000,
                        "source": "nvml", "index": 0},
                "engines": {"qwen": {"listen": 8000, "backend": "127.0.0.1:8002", "free_mb": 21400,
                                     "text_only": True, "state": "asleep", "inflight": 0, "idle_sec": 0,
                                     "enabled": True, "boot": True, "requests": 0, "wakes": 0}},
                "idle_sleep_sec": 120}

    def action(self, name, target=None):
        self.actions.append((name, target))
        return "started"


if __name__ == "__main__":
    unittest.main()

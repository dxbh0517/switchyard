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
import urllib.request
from http.server import BaseHTTPRequestHandler

from switchyard import __version__, config
from switchyard.arbiter import Arbiter, strip_images
from switchyard.backends import BACKENDS, ComfyBackend, VllmBackend
from switchyard.budget import lru_victims, resolve_cost, usable_mb
from switchyard.estimate import estimate_from_engine, estimate_vram_mb, gpu_layers_that_fit, quant_bytes
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
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def fake_engine():
    """A vLLM-like backend with sleep mode."""
    st = {"sleeping": True, "calls": []}

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
                self._send({"id": "x", "awake_when_served": not st["sleeping"]})
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
        self.assertEqual(e[1], {"name": "vision", "backend": "127.0.0.1:8003", "listen": 8005, "free_mb": 17500})
        with_cost = config.parse_engines("qwen=127.0.0.1:8002@8000:21400:18000")
        self.assertEqual(with_cost[0]["vram_mb"], 18000)
        self.assertEqual(with_cost[0]["free_mb"], 21400)

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
                  "text_only_engines": "a", "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1"})
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

    def test_wakes_one_engine_at_a_time(self):
        self.assertTrue(self.chat(self.la)["awake_when_served"])
        self.assertFalse(self.a_state["sleeping"])
        self.assertTrue(self.chat(self.lb)["awake_when_served"])
        self.assertTrue(self.a_state["sleeping"], "a must sleep before b wakes")
        self.assertFalse(self.b_state["sleeping"])
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["owner"], "b")
        self.assertEqual(st["engines"]["a"]["requests"], 1)

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

    def test_status_reports_exclusive_placement(self):
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["placement"], "exclusive")
        self.assertIn("budget", st)
        self.assertIn("vram_source", st["engines"]["a"])


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
        self.assertEqual(ns, argparse.Namespace())


if __name__ == "__main__":
    unittest.main()

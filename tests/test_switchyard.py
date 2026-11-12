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

from switchyard import __version__, config, events
from switchyard.arbiter import Arbiter, strip_images
from switchyard.dashboard import Dashboard
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
        self.assertEqual(e[1], {"name": "vision", "backend": "127.0.0.1:8003", "listen": 8005,
                                "free_mb": 17500, "enabled": True, "boot": False})
        flags = config.parse_engines("qwen=127.0.0.1:8002@8000:21400:boot,vision=127.0.0.1:8003@8005:17500:off")
        self.assertTrue(flags[0]["boot"] and flags[0]["enabled"])
        self.assertFalse(flags[1]["enabled"])
        extra = config.parse_engines("qwen=127.0.0.1:8002@8000:21400:18000:boot")  # extra int reserved
        self.assertEqual(extra[0]["free_mb"], 21400)
        self.assertTrue(extra[0]["boot"])
        kept = config.parse_engines([{"name": "a", "backend": "127.0.0.1:1", "listen": 2,
                                      "free_mb": 3, "vram_mb": 9000, "quant": "q4_k"}])
        self.assertEqual(kept[0]["vram_mb"], 9000)
        self.assertEqual(kept[0]["quant"], "q4_k")
        self.assertTrue(kept[0]["enabled"])
        self.assertFalse(kept[0]["boot"])

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


class DashboardTests(unittest.TestCase):
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
        self.assertIn("vram", schema["sections"])
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

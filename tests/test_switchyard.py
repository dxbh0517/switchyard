"""Stdlib-only tests: python -m unittest discover -s tests

The arbiter tests run against fake vLLM engines and a fake ComfyUI on localhost.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler

from switchyard import analytics, config
from switchyard.arbiter import Arbiter, strip_images
from switchyard.httputil import Server
from switchyard.ovms_shim import OvmsShim, StreamThinkFilter, split_think
from switchyard.stt import parse_multipart


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

    def test_wake_records_analytics(self):
        analytics.configure(self.arb.s, path=os.path.join(self.tmp.name, "analytics.sqlite"))
        self.chat(self.la)
        analytics.get().flush()
        kinds = [e["kind"] for e in analytics.history()["events"]]
        self.assertIn("wake", kinds)
        self.assertIn("unload", kinds)


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
            s.update({"ovms_upstream": f"http://127.0.0.1:{port}", "ovms_port": shim_port})
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
            up.shutdown()
            up.server_close()


if __name__ == "__main__":
    unittest.main()

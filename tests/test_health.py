"""Health monitor, wake retry/backoff, and fail-fast 503s.

Self-contained so it does not depend on helpers in test_switchyard.py (other arbiter
PRs edit that file).
"""
from __future__ import annotations

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

from switchyard import config
from switchyard.arbiter import Arbiter
from switchyard.events import since
from switchyard.httputil import Server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(handler) -> tuple[Server, int]:
    port = free_port()
    srv = Server(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def fake_engine():
    """vLLM-like backend. Flip keys on the returned state dict to drive failures."""
    st = {"sleeping": True, "calls": [], "health_status": 200, "wake_fails": 0, "wake_times": [],
          "health_block": None, "health_delay": 0.0, "unhealthy_when_awake": False}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, obj, status=200):
            raw = json.dumps(obj).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/is_sleeping":
                self._send({"is_sleeping": st["sleeping"]})
                return
            if self.path == "/health":
                block = st.get("health_block")
                if block:
                    entered, release = block
                    entered.set()
                    release.wait(timeout=10)
                delay = st.get("health_delay") or 0
                if delay:
                    time.sleep(delay)
                if (not st["sleeping"]) and st.get("unhealthy_when_awake"):
                    self._send({"error": "unhealthy"}, status=500)
                    return
                code = st.get("health_status", 200)
                self._send({"status": "ok"} if code == 200 else {"error": "unhealthy"}, status=code)
                return
            self._send({"data": []})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            st["calls"].append(self.path)
            if self.path.startswith("/wake_up?tags=weights") or self.path == "/wake_up":
                st["wake_times"].append(time.time())
                if st.get("wake_fails", 0) > 0:
                    st["wake_fails"] -= 1
                    self._send({"error": "wake failed"}, status=500)
                    return
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
            self._send({})

    return {}, H


def http_json(url, body=None, timeout=30):
    data = json.dumps({} if body is None else body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return r.status, json.loads(raw) if raw else None, hdrs
    except urllib.error.HTTPError as e:
        raw = e.read()
        hdrs = {k.lower(): v for k, v in e.headers.items()}
        try:
            obj = json.loads(raw) if raw else None
        except ValueError:
            obj = raw.decode("utf-8", "replace")
        return e.code, obj, hdrs


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class HealthTests(unittest.TestCase):
    def start(self, extra=None, wait=0.4):
        self.tmp = tempfile.TemporaryDirectory()
        self.a_state, ha = fake_engine()
        self.b_state, hb = fake_engine()
        _, hc = fake_comfy()
        self.srv = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = self.srv
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": f"a=127.0.0.1:{pa}@{self.la}:1000,b=127.0.0.1:{pb}@{self.lb}:1000",
                  "text_only_engines": "a", "comfy_url": f"http://127.0.0.1:{pc}",
                  "listen_host": "127.0.0.1", "health_interval_sec": 60,
                  "health_timeout_sec": 0.5, "wake_attempts": 3, "wake_backoff_sec": 0.05,
                  **(extra or {})})
        self.arb = Arbiter(s)
        self.arb.start()
        if wait:
            time.sleep(wait)

    def tearDown(self):
        block = getattr(self, "a_state", None) and self.a_state.get("health_block")
        if block:
            block[1].set()
        servers = [s for s, _ in getattr(self, "srv", [])]
        if getattr(self, "arb", None):
            servers.extend(self.arb.servers)
        for srv in servers:
            srv.shutdown()
            srv.server_close()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, port, timeout=30):
        return http_json(f"http://127.0.0.1:{port}/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hi"}]}, timeout=timeout)

    def wait_health(self, engine, status, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.arb.health[engine]["status"] == status:
                return
            time.sleep(0.05)
        self.fail(f"{engine} health stayed {self.arb.health[engine]['status']!r}, wanted {status!r}")

    def test_status_exposes_health(self):
        self.start()
        self.wait_health("a", "healthy")
        st = get(f"http://127.0.0.1:{self.la}/gpu/status")
        self.assertEqual(st["engines"]["a"]["health"]["status"], "healthy")
        self.assertEqual(st["engines"]["a"]["health"]["consecutive_failures"], 0)
        self.assertIsNone(st["engines"]["a"]["health"]["last_error"])
        self.assertEqual(st["comfy_health"]["status"], "healthy")

    def test_failed_health_marks_unhealthy_and_503s(self):
        self.start()
        self.wait_health("a", "healthy")
        self.a_state["health_status"] = 500
        self.arb._record_health(self.arb.health["a"], "a", *self.arb.probe_engine_health("a"))
        self.assertEqual(self.arb.health["a"]["status"], "unhealthy")
        self.assertGreaterEqual(self.arb.health["a"]["consecutive_failures"], 1)
        t0 = time.time()
        code, body, hdrs = self.chat(self.la, timeout=5)
        self.assertLess(time.time() - t0, 1.0, "unhealthy engine must 503 without waiting")
        self.assertEqual(code, 503)
        self.assertIn("unhealthy", body["error"])
        self.assertEqual(body["health"], "unhealthy")
        self.assertIn("retry-after", hdrs)
        self.assertFalse(any(p.startswith("/wake_up") for p in self.a_state["calls"]),
                         "must not try to wake an unhealthy engine")

    def test_health_timeout_is_unreachable(self):
        self.start()
        self.wait_health("a", "healthy")
        self.a_state["health_delay"] = 1.0
        status, err = self.arb.probe_engine_health("a")
        self.assertEqual(status, "unreachable")
        self.assertTrue(err)

    def test_wake_retries_then_succeeds_with_backoff(self):
        self.start({"wake_attempts": 3, "wake_backoff_sec": 0.08})
        self.wait_health("a", "healthy")
        self.a_state["wake_fails"] = 2
        code, body, _ = self.chat(self.la, timeout=10)
        self.assertEqual(code, 200)
        self.assertTrue(body["awake_when_served"])
        times = self.a_state["wake_times"]
        self.assertEqual(len(times), 3)
        g0, g1 = times[1] - times[0], times[2] - times[1]
        self.assertGreaterEqual(g0, 0.05)
        self.assertGreaterEqual(g1, 0.12)
        self.assertGreater(g1, g0)

    def test_wake_exhaustion_marks_unhealthy_releases_owner_and_503s(self):
        self.start({"wake_attempts": 2, "wake_backoff_sec": 0.02})
        self.wait_health("a", "healthy")
        self.a_state["wake_fails"] = 10
        code, body, hdrs = self.chat(self.la, timeout=10)
        self.assertEqual(code, 503)
        self.assertIn("could not wake", body["error"])
        self.assertIn("retry-after", hdrs)
        self.assertEqual(self.arb.health["a"]["status"], "unhealthy")
        self.assertIsNone(self.arb.state["owner"])
        self.assertEqual(self.arb.state["inflight"]["a"], 0)
        t0 = time.time()
        code2, _, _ = self.chat(self.la, timeout=5)
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual(code2, 503)
        code_b, body_b, _ = self.chat(self.lb, timeout=10)
        self.assertEqual(code_b, 200)
        self.assertTrue(body_b["awake_when_served"])
        self.assertEqual(self.arb.state["owner"], "b")

    def test_not_healthy_after_wake_counts_as_failure(self):
        self.start({"wake_attempts": 2, "wake_backoff_sec": 0.02})
        self.wait_health("a", "healthy")
        self.a_state["unhealthy_when_awake"] = True
        code, body, _ = self.chat(self.la, timeout=10)
        self.assertEqual(code, 503)
        self.assertIn("could not wake", body["error"])
        self.assertEqual(self.arb.health["a"]["status"], "unhealthy")
        self.assertIsNone(self.arb.state["owner"])

    def test_recovers_when_health_passes(self):
        self.start({"health_interval_sec": 0.15, "wake_attempts": 1, "wake_backoff_sec": 0})
        self.wait_health("a", "healthy")
        self.a_state["health_status"] = 500
        deadline = time.time() + 2
        while time.time() < deadline and self.arb.health["a"]["status"] == "healthy":
            time.sleep(0.05)
        self.assertEqual(self.arb.health["a"]["status"], "unhealthy")
        self.assertEqual(self.chat(self.la, timeout=5)[0], 503)
        self.a_state["health_status"] = 200
        self.wait_health("a", "healthy", timeout=3)
        code, body, _ = self.chat(self.la, timeout=10)
        self.assertEqual(code, 200)
        self.assertTrue(body["awake_when_served"])

    def test_health_probe_does_not_hold_the_lock(self):
        entered, release = threading.Event(), threading.Event()
        self.start(wait=0)
        self.a_state["health_block"] = (entered, release)
        t = threading.Thread(target=lambda: self.arb.probe_engine_health("a"), daemon=True)
        t.start()
        self.assertTrue(entered.wait(2), "/health never blocked")
        got = self.arb.cond.acquire(timeout=1.0)
        self.assertTrue(got, "health HTTP must not hold the arbiter lock")
        if got:
            self.arb.cond.release()
        release.set()
        t.join(timeout=2)

    def test_health_monitor_drops_the_lock(self):
        entered, release = threading.Event(), threading.Event()
        self.tmp = tempfile.TemporaryDirectory()
        self.a_state, ha = fake_engine()
        _, hb = fake_engine()
        _, hc = fake_comfy()
        self.a_state["health_block"] = (entered, release)
        self.srv = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = self.srv
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": f"a=127.0.0.1:{pa}@{self.la}:1000,b=127.0.0.1:{pb}@{self.lb}:1000",
                  "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                  "health_interval_sec": 30, "health_timeout_sec": 8})
        self.arb = Arbiter(s)
        self.arb.start()
        self.assertTrue(entered.wait(3), "health monitor never called /health")
        got = self.arb.cond.acquire(timeout=1.0)
        self.assertTrue(got, "health monitor must not hold the lock during HTTP")
        if got:
            self.arb.cond.release()
        release.set()

    def test_health_transitions_are_logged(self):
        self.start()
        self.wait_health("a", "healthy")
        before = max((e["id"] for e in since(0)), default=0)
        self.arb._record_health(self.arb.health["a"], "a", "unhealthy", "HTTP 500")
        self.arb._record_health(self.arb.health["a"], "a", "healthy", None)
        msgs = [e["msg"] for e in since(before)]
        self.assertTrue(any("a health" in m and "unhealthy" in m for m in msgs))
        self.assertTrue(any("a health" in m and "healthy" in m for m in msgs))


if __name__ == "__main__":
    unittest.main()

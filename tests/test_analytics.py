"""Analytics recorder, retention, and GET /api/history."""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from switchyard import analytics, config
from switchyard.dashboard import Dashboard
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


def get(url):
    with urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _iso(days_ago: float = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "analytics.sqlite")
        self._env = {k: os.environ.get(k) for k in ("SWITCHYARD_ANALYTICS", "SWITCHYARD_ANALYTICS_DB")}
        os.environ.pop("SWITCHYARD_ANALYTICS", None)
        os.environ["SWITCHYARD_ANALYTICS_DB"] = self.db
        analytics.reset()

    def tearDown(self):
        analytics.reset()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def _rec(self, **kwargs) -> analytics.Recorder:
        rec = analytics.Recorder(self.db, **kwargs)
        self.addCleanup(rec.close)
        return rec

    def test_record_and_history(self):
        rec = self._rec()
        rec.emit("wake", engine="qwen", duration_ms=1500, vram_before_mb=800, vram_after_mb=21000,
                 ram_before_mb=12000, ram_after_mb=14000, details={"level": 2})
        rec.emit("sleep", engine="qwen", duration_ms=80, vram_after_mb=900, ram_after_mb=11000,
                 details={"reason": "idle"})
        rec.flush()
        h = rec.history()
        kinds = [e["kind"] for e in h["events"]]
        self.assertEqual(kinds, ["sleep", "wake"])  # newest first
        wake = h["events"][1]
        self.assertEqual(wake["engine"], "qwen")
        self.assertEqual(wake["duration_ms"], 1500)
        self.assertEqual(wake["vram_before_mb"], 800)
        self.assertEqual(wake["vram_after_mb"], 21000)
        self.assertEqual(wake["details"]["level"], 2)
        self.assertTrue(wake["ts"].endswith("Z"))
        self.assertEqual(h["summary"]["counts"]["wake"], 1)
        self.assertEqual(h["summary"]["counts"]["sleep"], 1)
        self.assertEqual(h["summary"]["peak_vram_mb"], 21000)
        self.assertEqual(h["summary"]["peak_ram_mb"], 14000)
        self.assertEqual(h["summary"]["errors"], 0)

    def test_filters(self):
        rec = self._rec()
        rec.emit("wake", engine="qwen", ram_after_mb=1, vram_after_mb=10)
        time.sleep(0.02)
        cutoff = analytics.utc_now()
        rec.emit("sleep", engine="qwen", ram_after_mb=1, vram_after_mb=10)
        rec.emit("wake", engine="vision", ram_after_mb=1, vram_after_mb=20)
        rec.emit("error", engine="vision", ram_after_mb=1, details={"op": "sleep", "error": "x"})
        rec.flush()
        wakes = rec.history(kind="wake")
        self.assertEqual({e["engine"] for e in wakes["events"]}, {"qwen", "vision"})
        qwen = rec.history(engine="qwen")
        self.assertEqual({e["kind"] for e in qwen["events"]}, {"wake", "sleep"})
        recent = rec.history(since=cutoff)
        self.assertNotIn("qwen", {e["engine"] for e in recent["events"] if e["kind"] == "wake"})
        self.assertEqual(len(rec.history(limit=1)["events"]), 1)
        self.assertEqual(rec.history(limit=1)["events"][0]["kind"], "error")
        self.assertEqual(recent["summary"]["errors"], 1)
        unix = rec.history(since=time.time() - 60)
        self.assertGreaterEqual(len(unix["events"]), 3)

    def test_retention_pruning(self):
        rec = self._rec(max_age_days=14, max_rows=3, start_writer=False)
        rec.emit("wake", engine="old", ts=_iso(20), ram_after_mb=1, vram_after_mb=99)
        for i in range(5):
            rec.emit("sleep", engine="qwen", details={"i": i}, ram_after_mb=1, vram_after_mb=i)
        rec.flush()
        h = rec.history()
        self.assertEqual(len(h["events"]), 3)
        self.assertNotIn("old", [e["engine"] for e in h["events"]])
        self.assertEqual([e["details"]["i"] for e in h["events"]], [4, 3, 2])  # newest kept
        self.assertEqual(h["summary"]["peak_vram_mb"], 99)  # peak survives prune

    def test_disabled(self):
        os.environ["SWITCHYARD_ANALYTICS"] = "off"
        rec = analytics.configure(path=self.db)
        rec.emit("wake", engine="qwen", vram_after_mb=1)
        rec.flush()
        self.assertFalse(rec.enabled)
        self.assertEqual(rec.history()["events"], [])
        self.assertEqual(rec.history()["summary"]["errors"], 0)
        self.assertFalse(os.path.exists(self.db))

    def test_emit_queues_without_sqlite(self):
        rec = self._rec(start_writer=False)
        rec.emit("wake", engine="qwen", ram_after_mb=1, vram_after_mb=2)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
        rec.flush()
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_no_sqlite_io_under_lock(self):
        rec = self._rec(start_writer=False)
        lock = threading.Lock()  # stand-in for Arbiter.cond
        with lock:
            rec.emit("wake", engine="qwen", ram_after_mb=1, vram_after_mb=2)
            rec.emit("error", engine="qwen", details={"op": "sleep", "error": "x"})
            self.assertGreater(rec._q.qsize(), 0)
            with sqlite3.connect(self.db) as c:
                self.assertEqual(c.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)
        rec.flush()
        self.assertEqual(len(rec.history()["events"]), 2)

    def test_emit_never_raises(self):
        rec = self._rec()
        rec.close()
        rec.emit("wake", engine="qwen")  # writer gone; still must not raise
        analytics.emit("not-a-kind", engine=None, details={"x": object()})

    def test_hooks_and_error_summary(self):
        rec = analytics.configure(path=self.db)
        analytics.record_health_check("qwen", ok=True, duration_ms=5)
        analytics.record_wake_retry("qwen", attempt=1, error="timeout")
        analytics.record_wake_failure("qwen", error="still down")
        analytics.record_oom("qwen", details={"where": "wake"})
        analytics.record_evict("vision", reason="budget")
        rec.flush()
        h = rec.history()
        kinds = {e["kind"] for e in h["events"]}
        self.assertEqual(kinds, {"health_check", "wake_retry", "wake_failure", "oom", "evict"})
        self.assertEqual(h["summary"]["errors"], 2)  # oom + wake_failure

    def test_history_endpoint(self):
        rec = analytics.configure(path=self.db)
        rec.emit("wake", engine="qwen", duration_ms=12, vram_after_mb=100, ram_after_mb=50,
                 details={"level": 2}, ts="2026-01-01T00:00:00.000Z")
        rec.emit("error", engine="qwen", details={"op": "sleep", "error": "boom"},
                 ts="2026-06-01T00:00:00.000Z")
        rec.flush()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        dash = Dashboard(s, {"arbiter": None, "ovms": None, "stt": None})
        srv, port = serve(dash.make_handler())
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        time.sleep(0.05)
        data = get(f"http://127.0.0.1:{port}/api/history?limit=10")
        self.assertEqual([e["kind"] for e in data["events"]], ["error", "wake"])
        self.assertEqual(data["summary"]["errors"], 1)
        self.assertEqual(data["summary"]["peak_vram_mb"], 100)
        self.assertEqual(data["summary"]["counts"]["wake"], 1)
        q = urlencode({"kind": "wake", "engine": "qwen", "limit": 2})
        wakes = get(f"http://127.0.0.1:{port}/api/history?{q}")
        self.assertEqual(len(wakes["events"]), 1)
        self.assertEqual(wakes["events"][0]["duration_ms"], 12)
        since = urlencode({"since": "2026-03-01T00:00:00.000Z"})
        only_new = get(f"http://127.0.0.1:{port}/api/history?{since}")
        self.assertEqual(len(only_new["events"]), 1)
        self.assertEqual(only_new["events"][0]["kind"], "error")
        try:
            get(f"http://127.0.0.1:{port}/api/history?limit=nope")
            self.fail("expected 400")
        except HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_ram_used_mb_linux(self):
        used = analytics.ram_used_mb()
        if os.path.exists("/proc/meminfo"):
            self.assertIsInstance(used, int)
            self.assertGreater(used, 0)
        else:
            self.assertIsNone(used)


class HistoryEndpointDisabledTests(unittest.TestCase):
    def test_disabled_endpoint_is_empty(self):
        saved = os.environ.get("SWITCHYARD_ANALYTICS")
        os.environ["SWITCHYARD_ANALYTICS"] = "off"
        analytics.reset()
        try:
            with tempfile.TemporaryDirectory() as d:
                s = config.Settings(os.path.join(d, "s.json"))
                dash = Dashboard(s, {"arbiter": None, "ovms": None, "stt": None})
                srv, port = serve(dash.make_handler())
                try:
                    time.sleep(0.05)
                    data = get(f"http://127.0.0.1:{port}/api/history")
                    self.assertEqual(data["events"], [])
                    self.assertEqual(data["summary"]["errors"], 0)
                finally:
                    srv.shutdown()
                    srv.server_close()
        finally:
            analytics.reset()
            if saved is None:
                os.environ.pop("SWITCHYARD_ANALYTICS", None)
            else:
                os.environ["SWITCHYARD_ANALYTICS"] = saved

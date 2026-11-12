"""CLI tests: fake dashboard HTTP plus the real Dashboard handler."""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

from switchyard.ctl import (
    Client, CtlError, budget_view, format_budget, format_status,
    format_table, main, resolve_base, resolve_token,
)
from switchyard.dashboard import Dashboard
from switchyard.httputil import Server
from switchyard import config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(handler) -> tuple[Server, int]:
    port = free_port()
    srv = Server(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return srv, port
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"server did not start on {port}")


ARB = {
    "owner": "qwen",
    "comfy_active": 0,
    "comfy_queue": {"running": 0, "pending": 1},
    "gpu": {"free_mb": 2048, "total_mb": 32768, "name": "Fake GPU"},
    "engines": {
        "qwen": {"listen": 8000, "backend": "127.0.0.1:8002", "free_mb": 21400,
                 "text_only": True, "state": "awake", "inflight": 0, "idle_sec": 12,
                 "requests": 3, "wakes": 1},
        "vision": {"listen": 8005, "backend": "127.0.0.1:8003", "free_mb": 17500,
                   "text_only": False, "state": "asleep", "inflight": 0, "idle_sec": 40},
    },
    "idle_sleep_sec": 120,
}

STATUS = {"version": "0.1.0", "uptime": 90, "restart_pending": [],
          "arbiter": ARB, "ovms": None, "stt": None}


def fake_dashboard(status=None, action_result="started", error=None, code=None):
    """Minimal dashboard: records Authorization and /api/action bodies."""
    st = {"status": json.loads(json.dumps(status if status is not None else STATUS)),
          "calls": [], "headers": [], "action_result": action_result,
          "error": error, "code": code}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, http_code, obj):
            raw = json.dumps(obj).encode()
            self.send_response(http_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            st["headers"].append(dict(self.headers))
            if self.path == "/api/status":
                if st["code"]:
                    self._send(st["code"], {"error": st["error"] or "fail"})
                else:
                    self._send(200, st["status"])
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            st["headers"].append(dict(self.headers))
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            st["calls"].append((self.path, json.loads(body) if body else {}))
            if st["code"]:
                self._send(st["code"], {"error": st["error"] or "fail"})
            elif self.path == "/api/action":
                self._send(200, {"result": st["action_result"]})
            else:
                self._send(404, {"error": "not found"})

    return st, H


def run_main(argv, env=None) -> tuple[int, str, str]:
    env = env or {}
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with patch.dict(os.environ, env, clear=False):
        for k in ("SWITCHYARD_URL", "SWITCHYARD_TOKEN"):
            if k not in env:
                os.environ.pop(k, None)
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            code = main(argv, prog="switchyardctl")
    return code, buf_out.getvalue(), buf_err.getvalue()


class ResolveTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWITCHYARD_URL", None)
            self.assertEqual(resolve_base(None, None), "http://127.0.0.1:8100")
            self.assertEqual(resolve_base("10.0.0.5", 9000), "http://10.0.0.5:9000")
            os.environ["SWITCHYARD_URL"] = "http://box:8100"
            self.assertEqual(resolve_base(None, None), "http://box:8100")
            self.assertEqual(resolve_base(None, 9100), "http://box:9100")
            os.environ["SWITCHYARD_URL"] = "https://box"
            self.assertEqual(resolve_base(None, None), "https://box:443")
            os.environ["SWITCHYARD_URL"] = "box:8123"
            self.assertEqual(resolve_base("127.0.0.1", None), "http://127.0.0.1:8123")

    def test_token_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWITCHYARD_TOKEN", None)
            self.assertIsNone(resolve_token(None))
            os.environ["SWITCHYARD_TOKEN"] = " env-tok "
            self.assertEqual(resolve_token(None), "env-tok")
            self.assertEqual(resolve_token("cli"), "cli")
            self.assertIsNone(resolve_token(""))


class FormatTests(unittest.TestCase):
    def test_status_table_and_missing_keys(self):
        text = format_status(STATUS)
        self.assertIn("Switchyard 0.1.0", text)
        self.assertIn("Fake GPU", text)
        self.assertIn("qwen", text)
        self.assertIn("awake", text)
        self.assertIn("Owner  qwen", text)
        self.assertIn("ComfyUI", text)
        sparse = format_status({"version": "0.1.0", "uptime": 3,
                                "arbiter": {"owner": None, "engines": {"a": {"state": "asleep"}},
                                            "gpu": {}}})
        self.assertIn("VRAM unknown", sparse)
        self.assertIn("Owner  (idle)", sparse)
        self.assertIn("a", sparse)
        self.assertIn("GPU arbiter is disabled", format_status({"version": "1", "uptime": 0}))

    def test_budget_tolerates_future_keys(self):
        status = json.loads(json.dumps(STATUS))
        status["arbiter"]["gpu"]["used_mb"] = 100
        status["arbiter"]["gpu"]["source"] = "nvml"
        status["arbiter"]["budget"] = {"total_mb": 32768, "safety_mb": 1024}
        status["arbiter"]["engines"]["qwen"]["vram_mb"] = 18000
        view = budget_view(status)
        self.assertEqual(view["gpu"]["used_mb"], 100)
        self.assertEqual(view["budget"]["safety_mb"], 1024)
        self.assertEqual(view["engines"]["qwen"]["vram_mb"], 18000)
        text = format_budget(view)
        self.assertIn("nvml", text)
        self.assertIn("safety_mb", text)
        # no crash when almost everything is missing
        view2 = budget_view({"arbiter": {"engines": {"x": {}}}})
        self.assertEqual(view2["engines"]["x"], {})
        format_budget(view2)

    def test_format_table(self):
        out = format_table(["A", "B"], [["qwen", None], ["v", 1]])
        self.assertIn("qwen", out)
        self.assertIn("-", out)


class FakeHttpTests(unittest.TestCase):
    def setUp(self):
        self.st, h = fake_dashboard()
        self.srv, self.port = serve(h)
        self.env = {"SWITCHYARD_URL": f"http://127.0.0.1:{self.port}"}

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_status_engines_budget_json(self):
        code, out, err = run_main(["status"], self.env)
        self.assertEqual(code, 0, err)
        self.assertIn("qwen", out)
        self.assertIn("awake", out)
        code, out, err = run_main(["--json", "engines"], self.env)
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["owner"], "qwen")
        self.assertIn("vision", data["engines"])
        code, out, err = run_main(["budget", "--json"], self.env)
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["gpu"]["total_mb"], 32768)
        self.assertEqual(data["gpu"]["used_mb"], 32768 - 2048)
        self.assertEqual(data["engines"]["qwen"]["free_mb"], 21400)

    def test_wake_sleep_swap_comfy_unload(self):
        cases = [
            (["wake", "qwen"], "warm", "qwen"),
            (["load", "vision"], "warm", "vision"),
            (["swap", "qwen"], "warm", "qwen"),
            (["sleep", "qwen"], "sleep", "qwen"),
            (["unload", "vision"], "sleep", "vision"),
            (["comfy-unload"], "comfy_unload", None),
        ]
        for argv, action, target in cases:
            self.st["calls"].clear()
            code, out, err = run_main(argv, self.env)
            self.assertEqual(code, 0, err)
            self.assertEqual(self.st["calls"][-1][0], "/api/action")
            body = self.st["calls"][-1][1]
            self.assertEqual(body["component"], "arbiter")
            self.assertEqual(body["action"], action)
            self.assertEqual(body.get("target"), target)
            self.assertIn("started", out)

    def test_bearer_token_and_errors(self):
        code, out, err = run_main(["--token", "s3cret", "status"], self.env)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.st["headers"][-1].get("Authorization"), "Bearer s3cret")
        env = dict(self.env, SWITCHYARD_TOKEN="from-env")
        code, out, err = run_main(["status"], env)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.st["headers"][-1].get("Authorization"), "Bearer from-env")

        self.st["code"], self.st["error"] = 400, "unknown engine nope"
        code, out, err = run_main(["wake", "nope"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("unknown engine nope", err)
        self.assertEqual(out, "")

        code, out, err = run_main(["status"], {"SWITCHYARD_URL": "http://127.0.0.1:1"})
        self.assertEqual(code, 1)
        self.assertIn("cannot reach", err)

        code, out, err = run_main(["wake"])
        self.assertEqual(code, 1)
        self.assertIn("requires an engine name", err)
        code, out, err = run_main(["status", "qwen"])
        self.assertEqual(code, 1)
        self.assertIn("does not take", err)

    def test_disabled_arbiter(self):
        self.st["status"]["arbiter"] = None
        code, out, err = run_main(["engines"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("not running", err)
        code, out, err = run_main(["status"], self.env)
        self.assertEqual(code, 0, err)
        self.assertIn("disabled", out)


class RealDashboardTests(unittest.TestCase):
    """Drive the real Dashboard handler so action names stay in sync."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeArbiter()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        self.dash = Dashboard(s, {"arbiter": self.fake, "ovms": None, "stt": None})
        self.srv, self.port = serve(self.dash.make_handler())
        self.env = {"SWITCHYARD_URL": f"http://127.0.0.1:{self.port}"}

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def test_round_trip(self):
        code, out, err = run_main(["status"], self.env)
        self.assertEqual(code, 0, err)
        self.assertIn("qwen", out)
        self.assertIn("Fake GPU", out)
        code, out, err = run_main(["swap", "qwen"], self.env)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.fake.calls, [("warm", "qwen")])
        code, out, err = run_main(["sleep", "vision"], self.env)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.fake.calls[-1], ("sleep", "vision"))
        code, out, err = run_main(["comfy-unload"], self.env)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.fake.calls[-1], ("comfy_unload", None))
        code, out, err = run_main(["wake", "missing"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("unknown engine missing", err)

    def test_client_direct(self):
        c = Client(f"http://127.0.0.1:{self.port}")
        st = c.status()
        self.assertEqual(st["arbiter"]["owner"], "qwen")
        with self.assertRaises(CtlError):
            Client("http://127.0.0.1:1").status()


class FakeArbiter:
    def __init__(self):
        self.calls = []
        self._status = json.loads(json.dumps(ARB))

    def status(self):
        return self._status

    def action(self, name, target):
        self.calls.append((name, target))
        if name in ("sleep", "warm") and target not in self._status["engines"]:
            raise ValueError(f"unknown engine {target}")
        return "started"


def _console_script(name: str) -> str:
    bindir = Path(sys.executable).resolve().parent
    for cand in (bindir / name, bindir / f"{name}.exe", Path.home() / ".local/bin" / name,
                 Path.home() / ".local/bin" / f"{name}.exe"):
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(name)


class EntryPointTests(unittest.TestCase):
    def test_help_and_dispatch(self):
        r = subprocess.run([sys.executable, "-m", "switchyard", "ctl", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("status", r.stdout)
        r = subprocess.run([sys.executable, "-m", "switchyard.ctl", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("budget", r.stdout)
        r = subprocess.run([_console_script("switchyardctl"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("wake", r.stdout)


if __name__ == "__main__":
    unittest.main()

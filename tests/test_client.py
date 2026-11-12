"""Dashboard / engine-port HTTP client: mocked urllib, no GPU."""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.parse import urlparse

from switchyard.client import (
    ApiError,
    Client,
    arbiter_block,
    default_url,
    engines_map,
    fmt_mib,
    owner_of,
    shared_enabled,
    shared_supported,
    vram_used_total,
)


def _headers(d=None):
    m = Message()
    for k, v in (d or {"Content-Type": "application/json"}).items():
        m[k] = str(v)
    return m


class FakeResp:
    def __init__(self, payload, status=200):
        if isinstance(payload, bytes):
            self._raw = payload
        else:
            self._raw = json.dumps(payload).encode()
        self.status = status
        self.headers = _headers()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener(routes):
    """routes: {(METHOD, path): (status, payload)|Exception}."""
    calls = []

    def open_url(req, timeout=None):
        method = req.get_method()
        url = req.full_url
        path = urlparse(url).path
        body = json.loads(req.data) if req.data else None
        calls.append({"method": method, "url": url, "path": path, "body": body, "timeout": timeout})
        spec = routes.get((method, path), routes.get(path))
        if spec is None:
            raise urllib.error.URLError(f"no route for {method} {path}")
        if isinstance(spec, Exception):
            raise spec
        status, payload = spec
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "err", _headers(), io.BytesIO(raw))
        return FakeResp(payload, status)

    open_url.calls = calls
    return open_url


DASH = {
    "version": "0.1.0",
    "arbiter": {
        "owner": "qwen",
        "engines": {
            "qwen": {"listen": 8000, "state": "awake", "backend": "127.0.0.1:8002"},
            "vision": {"listen": 8005, "state": "asleep", "backend": "127.0.0.1:8003"},
        },
        "gpu": {"free_mb": 20000, "total_mb": 24576, "name": "RTX"},
    },
}


class FieldTests(unittest.TestCase):
    def test_vram_derives_used_when_missing(self):
        used, total, source = vram_used_total(DASH)
        self.assertEqual(used, 4576)
        self.assertEqual(total, 24576)
        self.assertIsNone(source)

    def test_vram_prefers_used_mb_and_source(self):
        st = {"gpu": {"used_mb": 1024, "total_mb": 8192, "free_mb": 1, "source": "nvml", "index": 0}}
        used, total, source = vram_used_total(st)
        self.assertEqual((used, total, source), (1024, 8192, "nvml"))

    def test_vram_unknown_when_empty(self):
        self.assertEqual(vram_used_total({}), (None, None, None))
        self.assertEqual(vram_used_total({"arbiter": None}), (None, None, None))
        self.assertEqual(vram_used_total(None), (None, None, None))

    def test_arbiter_block_accepts_gpu_status_shape(self):
        raw = {"owner": "comfy", "engines": {"a": {"listen": 1}}, "gpu": {"total_mb": 8}}
        self.assertEqual(arbiter_block(raw)["owner"], "comfy")
        self.assertEqual(engines_map(raw)["a"]["listen"], 1)
        self.assertEqual(owner_of(raw), "comfy")

    def test_engines_list_shape(self):
        st = {"engines": [{"name": "qwen", "listen": 8000}]}
        self.assertEqual(list(engines_map(st)), ["qwen"])

    def test_shared_from_settings_schema(self):
        schema = {"fields": [{"key": "shared_vram"}], "values": {"shared_vram": True}}
        self.assertTrue(shared_supported(schema, {}))
        self.assertTrue(shared_enabled(schema, {}))
        schema["values"]["shared_vram"] = False
        self.assertFalse(shared_enabled(schema, {}))

    def test_shared_from_placement_when_settings_missing(self):
        self.assertTrue(shared_supported(None, {"placement": "shared"}))
        self.assertTrue(shared_enabled(None, {"placement": "shared"}))
        self.assertFalse(shared_enabled(None, {"placement": "exclusive"}))
        self.assertFalse(shared_supported(None, DASH))

    def test_fmt_mib(self):
        self.assertEqual(fmt_mib(None, None), "—")
        self.assertIn("GiB", fmt_mib(1024, 2048))


class UrlTests(unittest.TestCase):
    def test_default_url_env(self):
        with patch.dict(os.environ, {"SWITCHYARD_URL": "http://box:9/"}):
            self.assertEqual(default_url(), "http://box:9")
        with patch.dict(os.environ, {"DASHBOARD_HOST": "0.0.0.0", "DASHBOARD_PORT": "8101",
                                     "SWITCHYARD_URL": ""}):
            self.assertEqual(default_url(), "http://127.0.0.1:8101")


class ClientTests(unittest.TestCase):
    def test_status_prefers_dashboard(self):
        o = opener({("GET", "/api/status"): (200, DASH)})
        self.assertEqual(Client("http://127.0.0.1:8100", opener=o).status()["version"], "0.1.0")
        self.assertEqual(o.calls[0]["path"], "/api/status")

    def test_status_falls_back_to_gpu_status_on_404(self):
        gpu = {"owner": "qwen", "gpu": {"used_mb": 1, "total_mb": 2, "source": "nvidia-smi"}}
        o = opener({
            ("GET", "/api/status"): (404, {"error": "not found"}),
            ("GET", "/gpu/status"): (200, gpu),
        })
        st = Client("http://127.0.0.1:8000", opener=o).status()
        self.assertEqual(vram_used_total(st), (1, 2, "nvidia-smi"))

    def test_unreachable(self):
        o = opener({("GET", "/api/status"): urllib.error.URLError("down")})
        with self.assertRaises(ApiError) as cm:
            Client("http://127.0.0.1:8100", opener=o).status()
        self.assertIn("unreachable", str(cm.exception))

    def test_action_and_settings(self):
        o = opener({
            ("POST", "/api/action"): (200, {"result": "started"}),
            ("POST", "/api/settings"): (200, {"changed": ["shared_vram"]}),
        })
        c = Client("http://127.0.0.1:8100", opener=o)
        self.assertEqual(c.action("arbiter", "warm", "qwen")["result"], "started")
        self.assertEqual(o.calls[0]["body"], {"component": "arbiter", "action": "warm", "target": "qwen"})
        c.set_shared(True)
        self.assertEqual(o.calls[1]["body"], {"shared_vram": True})

    def test_set_shared_falls_back_to_env_style_key(self):
        calls = {"n": 0}

        def open_url(req, timeout=None):
            calls["n"] += 1
            body = json.loads(req.data)
            if "shared_vram" in body:
                raise urllib.error.HTTPError(req.full_url, 400, "err", _headers(),
                                             io.BytesIO(b'{"error":"unknown setting shared_vram"}'))
            return FakeResp({"changed": ["SHARED_VRAM"]})

        c = Client("http://127.0.0.1:8100", opener=open_url)
        self.assertEqual(c.set_shared(False)["changed"], ["SHARED_VRAM"])

    def test_give_comfy_falls_back_to_engine_acquire(self):
        o = opener({
            ("POST", "/api/action"): (400, {"error": "unknown action comfy_acquire"}),
            ("POST", "/gpu/acquire"): (200, {"owner": "comfy"}),
        })
        c = Client("http://192.168.1.5:8100", opener=o)
        r = c.give_gpu("comfy", DASH)
        self.assertEqual(r["owner"], "comfy")
        acquire = [x for x in o.calls if x["path"] == "/gpu/acquire"][0]
        self.assertEqual(acquire["url"], "http://192.168.1.5:8000/gpu/acquire")

    def test_give_engine_warms(self):
        o = opener({("POST", "/api/action"): (200, {"result": "started"})})
        Client("http://127.0.0.1:8100", opener=o).give_gpu("vision", DASH)
        self.assertEqual(o.calls[0]["body"]["action"], "warm")
        self.assertEqual(o.calls[0]["body"]["target"], "vision")

    def test_clear_all_sleeps_engines_and_unloads_comfy(self):
        o = opener({("POST", "/api/action"): (200, {"result": "started"})})
        errors = Client("http://127.0.0.1:8100", opener=o).clear_all(DASH)
        self.assertEqual(errors, [])
        actions = [(c["body"]["action"], c["body"].get("target")) for c in o.calls]
        self.assertEqual(sorted(a for a in actions if a[0] == "sleep"),
                         [("sleep", "qwen"), ("sleep", "vision")])
        self.assertIn(("comfy_unload", None), actions)

    def test_clear_all_keeps_going_after_a_failure(self):
        def open_url(req, timeout=None):
            body = json.loads(req.data)
            if body.get("target") == "qwen":
                raise urllib.error.HTTPError(req.full_url, 400, "err", _headers(),
                                             io.BytesIO(b'{"error":"busy"}'))
            return FakeResp({"result": "started"})

        errors = Client("http://127.0.0.1:8100", opener=open_url).clear_all(DASH)
        self.assertTrue(any("qwen" in e for e in errors))

    def test_http_error_message(self):
        o = opener({("POST", "/api/action"): (400, {"error": "unknown action x"})})
        with self.assertRaises(ApiError) as cm:
            Client("http://127.0.0.1:8100", opener=o).action("arbiter", "x")
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("unknown action", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

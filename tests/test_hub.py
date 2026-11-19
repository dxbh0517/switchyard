"""Hub: guides matched to a fake ComfyUI inventory, skills, MCP server and MCP proxy."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import _ports
from switchyard import config
from switchyard.httputil import Server
from switchyard.hub import Hub, parse_front_matter

INVENTORY = {
    "diffusion_models": ["flux-2-klein-9b.safetensors", "mystery_model.safetensors"],
    "loras": ["flux 2 klein/style.safetensors", "other/thing.safetensors"],
    "vae": ["flux2-vae.safetensors"],
    "configs": ["v1.yaml"],
    "mirror_loras": ["flux 2 klein/style.safetensors"],
    "voices": ["Model/.cache/x.lock", "Model/model.safetensors", "Model/config.json"],
}

KLEIN = """---
title: FLUX.2 Klein 9B
summary: Fast text-to-image and editing.
tags: [image, edit]
files:
  - diffusion_models/flux-2-klein*.safetensors
  - loras/flux 2 klein/*
  - vae/flux2-vae.safetensors
---
# FLUX.2 Klein 9B

Use euler with 4 steps. Write prose prompts, not tag lists.
"""

GONE = """---
title: Removed Model
files: [diffusion_models/removed.safetensors]
---
Nothing here.
"""

PICKER = """# Which model to use

Pick Klein for fast pictures and upscale afterwards.
"""

SKILL = """---
name: poster-maker
description: >
  Make posters with ComfyUI.
  Uses Klein.
---
# Poster maker

Steps go here.
"""


def serve(handler) -> tuple[Server, int]:
    port = _ports.free_port()
    srv = Server(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


class FakeComfy(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):  # noqa: N802
        from urllib.parse import unquote
        path = unquote(self.path)
        if path == "/models":
            body = list(INVENTORY)
        elif path.startswith("/models/") and path[8:] in INVENTORY:
            body = INVENTORY[path[8:]]
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class FakeMcp(BaseHTTPRequestHandler):
    """Echoes the request so the test can check what the proxy forwarded."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        raw = json.dumps({"path": self.path, "body": json.loads(body),
                          "auth": self.headers.get("Authorization"),
                          "x_key": self.headers.get("X-Key"),
                          "session": self.headers.get("Mcp-Session-Id")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "abc")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class HubTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comfy, cport = serve(FakeComfy)
        cls.mcp_srv, mport = serve(FakeMcp)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "hub"
        (root / "models").mkdir(parents=True)
        (root / "models" / "flux2-klein.md").write_text(KLEIN)
        (root / "models" / "removed.md").write_text(GONE)
        (root / "models" / "aaa-audio.md").write_text("---\ntitle: Audio\nfiles: [audio_encoders/*]\n---\nTTS.\n")
        (root / "models" / "00-picker.md").write_text(PICKER)
        (root / "ignore.txt").write_text("# mirrors\nmirror_loras/*\n*.json\n")
        sk = root / "skills" / "poster"
        (sk / "refs").mkdir(parents=True)
        (sk / "SKILL.md").write_text(SKILL)
        (sk / "refs" / "layout.md").write_text("grid notes")
        (Path(cls.tmp.name) / "secret.txt").write_text("nope")
        extra = Path(cls.tmp.name) / "extra" / "cat" / "other-skill"
        extra.mkdir(parents=True)
        (extra / "SKILL.md").write_text("---\nname: other-skill\ndescription: Another one.\n---\nbody\n")
        (root / "mcp.json").write_text(json.dumps({"mcpServers": {
            "echo": {"url": f"http://127.0.0.1:{mport}/mcp", "description": "echo server",
                     "headers": {"X-Key": "server-secret"}},
            "audio": {"command": "node", "args": ["audio.js"], "env": {"TOKEN": "x"},
                      "description": "stdio one"},
        }}))
        cls.port = _ports.free_port()
        env = {"SWITCHYARD_SETTINGS": str(Path(cls.tmp.name) / "settings.json"),
               "COMFY_URL": f"http://127.0.0.1:{cport}", "HUB_DIR": str(root),
               "HUB_SKILLS_DIRS": str(Path(cls.tmp.name) / "extra"),
               "HUB_PORT": str(cls.port), "HUB_HOST": "127.0.0.1", "LISTEN_HOST": "0.0.0.0"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            cls.settings = config.Settings()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        cls.hub = Hub(cls.settings)
        cls.hub.start()

    @classmethod
    def tearDownClass(cls):
        cls.hub.server.shutdown()
        cls.comfy.shutdown()
        cls.mcp_srv.shutdown()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers=headers or {})
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw.decode()

    def rpc(self, method, params=None, mid=1, path="/mcp", headers=None):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if mid is not None:
            msg["id"] = mid
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(msg).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None), r.headers

    def tool(self, tool_name, **args):
        _, out, _ = self.rpc("tools/call", {"name": tool_name, "arguments": args})
        res = out["result"]
        return res["content"][0]["text"], res["isError"]

    # -- parsing ---------------------------------------------------------------
    def test_front_matter(self):
        meta, body = parse_front_matter(KLEIN)
        self.assertEqual(meta["title"], "FLUX.2 Klein 9B")
        self.assertEqual(meta["tags"], ["image", "edit"])
        self.assertEqual(len(meta["files"]), 3)
        self.assertTrue(body.startswith("# FLUX.2"))
        meta, _ = parse_front_matter(SKILL)
        self.assertEqual(meta["description"], "Make posters with ComfyUI. Uses Klein.")
        self.assertEqual(parse_front_matter("no front matter")[0], {})

    # -- inventory ---------------------------------------------------------------
    def test_models_grouped_by_guide(self):
        m = self.get("/models")
        self.assertTrue(m["comfy_reachable"])
        self.assertNotIn("configs", m["folders"])
        by = {g["guide"]: g for g in m["models"]}
        self.assertEqual(sorted(by["flux2-klein"]["files"]), [
            "diffusion_models/flux-2-klein-9b.safetensors", "loras/flux 2 klein/style.safetensors",
            "vae/flux2-vae.safetensors"])
        self.assertFalse(by["removed"]["installed"])
        self.assertNotIn("mirror_loras", m["folders"])
        self.assertEqual(sorted(m["undocumented"]), ["diffusion_models/mystery_model.safetensors",
                                                     "loras/other/thing.safetensors",
                                                     "voices/Model/model.safetensors"])
        self.assertEqual([g["slug"] for g in m["general_guides"]], ["00-picker"])
        loras = self.get("/models?folder=loras")
        self.assertEqual(loras["undocumented"], ["loras/other/thing.safetensors"])

    def test_guide_lookup(self):
        self.assertEqual(self.get("/guides/flux2-klein")["title"], "FLUX.2 Klein 9B")
        self.assertIn("euler", self.get("/guides/flux2-klein?format=md"))
        self.assertEqual(self.hub.guide("flux-2-klein-9b.safetensors").slug, "flux2-klein")
        self.assertEqual(self.hub.guide("FLUX.2 Klein 9B").slug, "flux2-klein")
        # A catch-all pattern in another guide must not claim an unrelated file name.
        self.assertEqual(self.hub.guide("flux2-vae.safetensors").slug, "flux2-klein")
        self.assertIsNone(self.hub.guide("not-a-model.safetensors"))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/guides/nope")
        self.assertEqual(cm.exception.code, 404)

    # -- skills ----------------------------------------------------------------
    def test_skills(self):
        names = [s["name"] for s in self.get("/skills")]
        self.assertEqual(names, ["other-skill", "poster-maker"])
        sk = self.get("/skills/poster-maker")
        self.assertIn("refs/layout.md", sk["files"])
        self.assertEqual(self.get("/skills/poster-maker/refs/layout.md"), "grid notes")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/skills/poster-maker/..%2F..%2F..%2Fsecret.txt")
        self.assertIn(cm.exception.code, (403, 404))
        text, err = self.tool("get_skill", name="poster-maker", file="../../../secret.txt")
        self.assertTrue(err)
        self.assertNotIn("nope", text)

    def test_search(self):
        hits = self.get("/search?q=klein+prompts")
        self.assertEqual(hits[0]["id"], "flux2-klein")
        self.assertIn("poster-maker", [h["id"] for h in self.get("/search?q=posters")])

    # -- MCP ---------------------------------------------------------------------
    def test_mcp_handshake_and_tools(self):
        status, out, _ = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                 "clientInfo": {"name": "t", "version": "1"}})
        self.assertEqual(status, 200)
        self.assertEqual(out["result"]["protocolVersion"], "2025-06-18")
        status, out, _ = self.rpc("notifications/initialized", mid=None)
        self.assertEqual(status, 202)
        _, out, _ = self.rpc("tools/list")
        self.assertIn("model_guide", [t["name"] for t in out["result"]["tools"]])
        text, err = self.tool("model_guide", name="flux-2-klein-9b.safetensors")
        self.assertFalse(err)
        self.assertIn("Installed files covered", text)
        self.assertIn("vae/flux2-vae.safetensors", text)
        text, err = self.tool("model_guide", name="unknown")
        self.assertTrue(err)
        inv = json.loads(self.tool("comfy_models")[0])
        self.assertIn("undocumented", inv)
        self.assertIn("poster-maker", self.tool("list_skills", filter="poster")[0])
        servers = json.loads(self.tool("list_mcp_servers")[0])
        by = {s["name"]: s for s in servers}
        self.assertTrue(by["echo"]["url"].endswith("/mcp/echo"))
        self.assertEqual(by["audio"]["env_keys"], ["TOKEN"])
        self.assertNotIn("env", by["audio"])
        _, out, _ = self.rpc("resources/read", {"uri": "switchyard://guides/flux2-klein"})
        self.assertIn("euler", out["result"]["contents"][0]["text"])
        _, out, _ = self.rpc("bogus/method")
        self.assertEqual(out["error"]["code"], -32601)

    def test_mcp_proxy(self):
        status, out, headers = self.rpc("tools/list", path="/mcp/echo",
                                        headers={"Mcp-Session-Id": "s1", "Authorization": "Bearer client"})
        self.assertEqual(status, 200)
        self.assertEqual(out["path"], "/mcp")
        self.assertEqual(out["body"]["method"], "tools/list")
        self.assertEqual(out["session"], "s1")
        self.assertEqual(out["x_key"], "server-secret")
        self.assertIsNone(out["auth"])  # the hub's client credential is not passed upstream
        self.assertEqual(headers.get("Mcp-Session-Id"), "abc")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.rpc("tools/list", path="/mcp/audio")
        self.assertEqual(cm.exception.code, 404)

    def test_token(self):
        self.settings.update({"hub_token": "s3cret"})
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get("/guides")
            self.assertEqual(cm.exception.code, 401)
            with self.assertRaises(urllib.error.HTTPError):
                self.rpc("tools/list")
            self.assertTrue(self.get("/guides", headers={"Authorization": "Bearer s3cret"}))
        finally:
            self.settings.update({"hub_token": ""})

    def test_hub_host_overrides_listen_host(self):
        self.assertEqual(self.hub.server.server_address[0], "127.0.0.1")

    def test_status(self):
        st = self.hub.status()
        self.assertEqual(st["guides"], 4)
        self.assertEqual(st["skills"], 2)
        self.assertEqual(st["mcp_servers"], 2)


if __name__ == "__main__":
    unittest.main()

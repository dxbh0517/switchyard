"""Dashboard: one page (static/index.html) plus a small JSON API.

GET  /api/info                  version, bind host, whether a token is required (no auth)
GET  /api/status                live state of every component
GET  /api/settings              schema + current values
POST /api/settings              {"key": value, ...}
POST /api/action                {"component": "arbiter|stt", "action": "...", "target": "..."}
GET  /api/logs?since=ID         activity log
POST /api/restart               exit; the container's restart policy starts it again

When SWITCHYARD_TOKEN is set, every /api/* path except /api/info needs
Authorization: Bearer <token>. Mutating paths also need POST plus a CSRF check
(matching Origin, or the X-Switchyard header for non-browser clients).
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, events
from .config import Settings
from .events import log
from .httputil import Handler, Server

INDEX = (Path(__file__).parent / "static" / "index.html").read_bytes()
CSRF_HEADER = "X-Switchyard"
MUTATING = {"/api/settings", "/api/action", "/api/restart"}
EXPOSED_NO_TOKEN = (
    "DASHBOARD_HOST is {host} (not loopback) and SWITCHYARD_TOKEN is empty; "
    "anyone who can reach the dashboard can change settings. Set SWITCHYARD_TOKEN, "
    "or bind 127.0.0.1 and use an SSH tunnel."
)


def is_loopback_host(host: str) -> bool:
    """True for localhost / loopback bind addresses (not 0.0.0.0 or a LAN IP)."""
    h = (host or "").strip().lower()
    if h.startswith("[") and "]" in h:
        h = h[1:h.index("]")]
    if "%" in h:
        h = h.split("%", 1)[0]
    if h in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def hostport(host: str, port: int) -> str:
    h = host or ""
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return f"{h}:{port}"


def _bearer_ok(expected: str, header: str | None) -> bool:
    """Constant-time compare of Authorization: Bearer <token> to the configured token."""
    got = ""
    if header:
        kind, _, rest = header.partition(" ")
        if kind.lower() == "bearer":
            got = rest.strip()
    exp = expected.encode("utf-8")
    got_b = got.encode("utf-8")
    if len(got_b) != len(exp):
        hmac.compare_digest(exp, exp)
        return False
    return hmac.compare_digest(got_b, exp)


def _csrf_ok(handler: Handler) -> bool:
    """Same-origin Origin, or X-Switchyard (so a <form> on another site cannot POST)."""
    origin = handler.headers.get("Origin")
    host = (handler.headers.get("Host") or "").lower()
    if origin:
        return urlparse(origin).netloc.lower() == host
    return bool(handler.headers.get(CSRF_HEADER))


class Dashboard:
    def __init__(self, settings: Settings, components: dict):
        self.s = settings
        self.c = components   # name -> component (or None when disabled)
        self.server: Server | None = None

    def token(self) -> str:
        return (self.s.get("dashboard_token") or "").strip()

    def info(self) -> dict:
        host, port = self.s.get("dashboard_host"), self.s.get("dashboard_port")
        return {
            "version": __version__,
            "host": hostport(host, port),
            "auth_required": bool(self.token()),
        }

    def status(self) -> dict:
        out = {"version": __version__, "uptime": round(time.time() - events.STARTED),
               "restart_pending": self.s.restart_pending()}
        for name, comp in self.c.items():
            out[name] = comp.status() if comp else None
        return out

    def action(self, req: dict) -> dict:
        comp_name, action = req.get("component"), req.get("action")
        comp = self.c.get(comp_name)
        if comp is None:
            raise ValueError(f"{comp_name} is not running")
        if comp_name == "arbiter":
            return {"result": comp.action(action, req.get("target"))}
        if comp_name == "stt" and action == "reload":
            comp.reload()
            return {"result": "started"}
        raise ValueError(f"unknown action {comp_name}.{action}")

    def make_handler(self):
        dash = self

        class DashHandler(Handler):
            def _auth_ok(self) -> bool:
                token = dash.token()
                if token and not _bearer_ok(token, self.headers.get("Authorization")):
                    self.send_json(401, {"error": "unauthorized"},
                                   extra={"WWW-Authenticate": "Bearer"})
                    return False
                return True

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                if u.path in ("/", "/index.html"):
                    self.send_bytes(200, INDEX, "text/html; charset=utf-8")
                    return
                if u.path == "/api/info":
                    self.send_json(200, dash.info())
                    return
                if u.path in ("/api/action", "/api/restart"):
                    self.send_json(405, {"error": "POST required"}, extra={"Allow": "POST"})
                    return
                if not self._auth_ok():
                    return
                if u.path == "/api/status":
                    self.send_json(200, dash.status())
                elif u.path == "/api/settings":
                    self.send_json(200, dash.s.schema())
                elif u.path == "/api/logs":
                    since = int((parse_qs(u.query).get("since") or ["0"])[0])
                    self.send_json(200, events.since(since))
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                u = urlparse(self.path)
                if u.path.startswith("/api/") and u.path != "/api/info" and not self._auth_ok():
                    return
                if u.path in MUTATING and not _csrf_ok(self):
                    self.send_json(403, {"error": "cross-origin request refused"})
                    return
                try:
                    req = json.loads(self.body() or b"{}")
                    if u.path == "/api/settings":
                        result = dash.s.update(req)
                        if result["changed"]:
                            log("dashboard", "settings changed: " + ", ".join(result["changed"]))
                        self.send_json(200, result)
                    elif u.path == "/api/action":
                        self.send_json(200, dash.action(req))
                    elif u.path == "/api/restart":
                        log("dashboard", "restart requested", "warn")
                        self.send_json(200, {"restarting": True})
                        threading.Timer(0.5, lambda: os._exit(0)).start()
                    else:
                        self.send_json(404, {"error": "not found"})
                except (ValueError, TypeError, KeyError) as e:
                    self.send_json(400, {"error": str(e)})

        return DashHandler

    def start(self) -> None:
        host, port = self.s.get("dashboard_host"), self.s.get("dashboard_port")
        if not is_loopback_host(host) and not self.token():
            log("dashboard", EXPOSED_NO_TOKEN.format(host=host), "warn")
        log("dashboard", f"http://{hostport(host, port)}/")
        self.server = Server((host, port), self.make_handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

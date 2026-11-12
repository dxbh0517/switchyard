"""Dashboard: one page (static/index.html) plus a small JSON API.

GET  /api/status                live state of every component
GET  /api/settings              schema + current values
POST /api/settings              {"key": value, ...}
POST /api/action                {"component": "arbiter|stt", "action": "...", "target": "..."}
GET  /api/logs?since=ID         activity log
GET  /api/history?limit=&since=&kind=&engine=   analytics events + summary
POST /api/restart               exit; the container's restart policy starts it again
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, analytics, events
from .config import Settings
from .events import log
from .httputil import Handler, Server

INDEX = (Path(__file__).parent / "static" / "index.html").read_bytes()


class Dashboard:
    def __init__(self, settings: Settings, components: dict):
        self.s = settings
        self.c = components   # name -> component (or None when disabled)
        analytics.configure_if_needed(settings)

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
            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                if u.path in ("/", "/index.html"):
                    self.send_bytes(200, INDEX, "text/html; charset=utf-8")
                elif u.path == "/api/status":
                    self.send_json(200, dash.status())
                elif u.path == "/api/settings":
                    self.send_json(200, dash.s.schema())
                elif u.path == "/api/logs":
                    since = int((parse_qs(u.query).get("since") or ["0"])[0])
                    self.send_json(200, events.since(since))
                elif u.path == "/api/history":
                    qs = parse_qs(u.query)
                    def q(name, default=None):
                        return (qs.get(name) or [default])[0]
                    try:
                        limit = int(q("limit", "100"))
                        payload = analytics.get().history(
                            limit=limit, since=q("since") or None,
                            kind=q("kind") or None, engine=q("engine") or None)
                    except (ValueError, TypeError) as e:
                        self.send_json(400, {"error": str(e)})
                        return
                    self.send_json(200, payload)
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                u = urlparse(self.path)
                # Same-origin only: a page on another site must not be able to drive this API.
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
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
        log("dashboard", f"http://{host}:{port}/")
        srv = Server((host, port), self.make_handler())
        threading.Thread(target=srv.serve_forever, daemon=True).start()

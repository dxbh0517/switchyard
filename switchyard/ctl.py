"""CLI for a running Switchyard: `switchyard ctl` / `switchyardctl`.

Talks to the dashboard JSON API (GET /api/status, POST /api/action). No local
arbiter logic — the server owns wake/sleep/unload. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import __version__

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100
TIMEOUT = 30

# Dashboard POST /api/action names. load/wake/swap all give the GPU to an engine.
ACTION = {
    "wake": ("warm", True),
    "load": ("warm", True),
    "swap": ("warm", True),
    "sleep": ("sleep", True),
    "unload": ("sleep", True),
    "comfy-unload": ("comfy_unload", False),
}


class CtlError(Exception):
    """User-facing failure; main() prints the message and exits 1."""


# ------------------------------------------------------------ HTTP
def resolve_base(host: str | None, port: int | None) -> str:
    """--host/--port override SWITCHYARD_URL (http://127.0.0.1:8100 by default)."""
    scheme, env_host, env_port = "http", DEFAULT_HOST, DEFAULT_PORT
    raw = (os.environ.get("SWITCHYARD_URL") or "").strip()
    if raw:
        if "://" not in raw:
            raw = "http://" + raw
        u = urlparse(raw)
        scheme = u.scheme or "http"
        env_host = u.hostname or env_host
        if u.port is not None:
            env_port = u.port
        elif scheme == "https":
            env_port = 443
    return f"{scheme}://{host or env_host}:{port if port is not None else env_port}"


def resolve_token(explicit: str | None) -> str | None:
    tok = explicit if explicit is not None else os.environ.get("SWITCHYARD_TOKEN")
    tok = (tok or "").strip()
    return tok or None


class Client:
    def __init__(self, base: str, token: str | None = None, timeout: float = TIMEOUT):
        self.base = base.rstrip("/")
        self.token = token
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        h = {"Accept": "application/json", "User-Agent": f"switchyard-ctl/{__version__}"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def request(self, method: str, path: str, body: dict | None = None) -> object:
        data = None
        headers = self.headers()
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            raw = e.read() if e.fp else b""
            raise CtlError(_http_error(e.code, raw)) from e
        except urllib.error.URLError as e:
            raise CtlError(f"cannot reach {self.base} ({_reason(e)})") from e
        except TimeoutError as e:
            raise CtlError(f"cannot reach {self.base} (timed out)") from e
        except OSError as e:
            raise CtlError(f"cannot reach {self.base} ({e})") from e
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as e:
            raise CtlError(f"HTTP {status}: response is not JSON") from e

    def status(self) -> dict:
        data = self.request("GET", "/api/status")
        if not isinstance(data, dict):
            raise CtlError("HTTP 200: expected a JSON object from /api/status")
        return data

    def action(self, name: str, target: str | None = None) -> dict:
        data = self.request("POST", "/api/action",
                            {"component": "arbiter", "action": name, "target": target})
        if not isinstance(data, dict):
            raise CtlError("HTTP 200: expected a JSON object from /api/action")
        return data


def _reason(exc: urllib.error.URLError) -> str:
    r = exc.reason
    return str(getattr(r, "strerror", None) or r or exc)


def _http_error(code: int, raw: bytes) -> str:
    msg = raw.decode("utf-8", "replace").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("error"):
            msg = str(obj["error"])
    except ValueError:
        pass
    return f"HTTP {code}: {msg or 'request failed'}"


# ------------------------------------------------------------ formatting
def _cell(v) -> str:
    if v is None or v == "":
        return "-"
    return str(v)


def format_table(headers: list[str], rows: list[list]) -> str:
    widths = [len(h) for h in headers]
    cells = [[_cell(c) for c in row] for row in rows]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    align = []
    for h, w in zip(headers, widths):
        align.append(h.ljust(w))
    lines = ["  ".join(align), "  ".join("-" * w for w in widths)]
    for row in cells:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def _fmt_dur(sec) -> str:
    try:
        s = int(sec)
    except (TypeError, ValueError):
        return _cell(sec)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def _fmt_mib(v) -> str:
    if v is None or v == "":
        return "-"
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return str(v)
    return f"{n:,} MiB"


def _arbiter(status: dict) -> dict:
    a = status.get("arbiter")
    if not isinstance(a, dict):
        raise CtlError("GPU arbiter is not running")
    return a


def budget_view(status: dict) -> dict:
    """VRAM numbers that exist today, plus keys later PRs may add."""
    a = _arbiter(status)
    gpu = dict(a["gpu"]) if isinstance(a.get("gpu"), dict) else {}
    if gpu.get("used_mb") is None and gpu.get("total_mb") is not None and gpu.get("free_mb") is not None:
        try:
            gpu["used_mb"] = gpu["total_mb"] - gpu["free_mb"]
        except TypeError:
            pass
    engines = {}
    for name, e in (a.get("engines") or {}).items():
        if not isinstance(e, dict):
            engines[name] = {}
            continue
        row = {k: e[k] for k in ("free_mb", "vram_mb", "vram_source") if k in e}
        engines[name] = row
    out = {"gpu": gpu, "engines": engines}
    for k in ("budget", "placement", "resident"):
        if k in a:
            out[k] = a[k]
    return out


def format_status(status: dict) -> str:
    ver = status.get("version") or "?"
    lines = [f"Switchyard {ver}  up {_fmt_dur(status.get('uptime'))}"]
    a = status.get("arbiter")
    if not isinstance(a, dict):
        lines.append("GPU arbiter is disabled")
        return "\n".join(lines)
    gpu = a.get("gpu") if isinstance(a.get("gpu"), dict) else {}
    gpu_name = gpu.get("name") or "GPU"
    free, total = gpu.get("free_mb"), gpu.get("total_mb")
    used = gpu.get("used_mb")
    if used is None and total is not None and free is not None:
        try:
            used = total - free
        except TypeError:
            used = None
    vram = []
    if used is not None or total is not None:
        vram.append(f"{_fmt_mib(used)} used / {_fmt_mib(total)} total")
    if free is not None:
        vram.append(f"{_fmt_mib(free)} free")
    extra = "  ".join(f"{k}={_cell(gpu[k])}" for k in ("source", "index") if k in gpu)
    lines.append(f"GPU  {gpu_name}" + (f"  {'  '.join(vram)}" if vram else "  VRAM unknown"))
    if extra:
        lines.append(f"     {extra}")
    owner = a.get("owner")
    lines.append(f"Owner  {owner}" if owner else "Owner  (idle)")
    lines.append("")
    lines.append(format_engines(a))
    return "\n".join(lines)


def format_engines(arbiter: dict) -> str:
    owner = arbiter.get("owner")
    rows = []
    for name, e in (arbiter.get("engines") or {}).items():
        e = e if isinstance(e, dict) else {}
        rows.append([
            name,
            e.get("state"),
            "*" if owner == name else "",
            e.get("inflight"),
            _fmt_dur(e.get("idle_sec")) if e.get("idle_sec") is not None else "-",
            e.get("listen"),
            _fmt_mib(e.get("free_mb")),
        ])
    table = format_table(
        ["ENGINE", "STATE", "OWNER", "IN-FLIGHT", "IDLE", "LISTEN", "WAKE-NEED"],
        rows,
    )
    q = arbiter.get("comfy_queue") if isinstance(arbiter.get("comfy_queue"), dict) else {}
    running, pending = q.get("running"), q.get("pending")
    claim = arbiter.get("comfy_active")
    comfy = "ComfyUI  "
    if owner == "comfy":
        comfy += "holds GPU"
    elif running is None:
        comfy += "down"
    else:
        comfy += "idle"
    bits = []
    if running is not None or pending is not None:
        bits.append(f"queue {_cell(running)}/{_cell(pending)}")
    if claim is not None:
        bits.append(f"claims {claim}")
    if bits:
        comfy += "  " + "  ".join(bits)
    return table + "\n\n" + comfy


def format_budget(view: dict) -> str:
    gpu = view.get("gpu") if isinstance(view.get("gpu"), dict) else {}
    lines = [
        f"GPU     {_cell(gpu.get('name') or 'GPU')}",
        f"total   {_fmt_mib(gpu.get('total_mb'))}",
        f"used    {_fmt_mib(gpu.get('used_mb'))}",
        f"free    {_fmt_mib(gpu.get('free_mb'))}",
    ]
    extra = [f"{k}  {_cell(gpu[k])}" for k in ("source", "index") if k in gpu]
    lines.extend(extra)
    budget = view.get("budget")
    if isinstance(budget, dict):
        lines.append("")
        for k, v in budget.items():
            lines.append(f"{k}  {_cell(v)}")
    elif budget is not None:
        lines.append(f"budget  {_cell(budget)}")
    rows = []
    for name, e in (view.get("engines") or {}).items():
        e = e if isinstance(e, dict) else {}
        rows.append([name, _fmt_mib(e.get("free_mb")), _fmt_mib(e.get("vram_mb")) if "vram_mb" in e else "-"])
    if rows:
        headers = ["ENGINE", "WAKE-NEED"]
        if any(isinstance(e, dict) and "vram_mb" in e for e in (view.get("engines") or {}).values()):
            headers.append("VRAM")
            data = rows
        else:
            data = [r[:2] for r in rows]
        lines.append("")
        lines.append(format_table(headers, data))
    return "\n".join(lines)


def _emit(as_json: bool, obj, text: str) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(text)


# ------------------------------------------------------------ commands
def run(args: argparse.Namespace) -> None:
    client = Client(resolve_base(args.host, args.port), resolve_token(args.token))
    cmd = args.command
    if cmd in ACTION:
        name, needs = ACTION[cmd]
        if needs and not args.engine:
            raise CtlError(f"{cmd} requires an engine name")
        if not needs and args.engine:
            raise CtlError(f"{cmd} does not take an engine name")
        result = client.action(name, args.engine if needs else None)
        payload = {"action": name, "target": args.engine if needs else None, **result}
        who = args.engine or "ComfyUI"
        _emit(args.json, payload, f"{cmd} {who}: {_cell(result.get('result'))}")
        return
    if args.engine:
        raise CtlError(f"{cmd} does not take an engine name")
    status = client.status()
    if cmd == "status":
        _emit(args.json, status, format_status(status))
        return
    arb = _arbiter(status)
    if cmd == "engines":
        _emit(args.json, {"owner": arb.get("owner"), "engines": arb.get("engines") or {}},
              format_engines(arb))
        return
    if cmd == "budget":
        view = budget_view(status)
        _emit(args.json, view, format_budget(view))
        return
    raise CtlError(f"unknown command {cmd}")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "switchyardctl",
        description="Control a running Switchyard over its dashboard HTTP API.",
        epilog="""commands:
  status              owner, engines, sleeping/awake, VRAM
  engines             list engines
  wake|load ENGINE    wake an engine (dashboard action: warm)
  sleep|unload ENGINE put an engine to sleep
  swap ENGINE         give the GPU to an engine (same as wake)
  comfy-unload        unload ComfyUI models from VRAM
  budget              memory / VRAM numbers the server exposes today

Environment: SWITCHYARD_URL (default http://127.0.0.1:8100), SWITCHYARD_TOKEN
(sent as Authorization: Bearer; ignored if the server has no auth).""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default=None, help=f"dashboard host (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=None, help=f"dashboard port (default {DEFAULT_PORT})")
    p.add_argument("--token", default=None, help="Bearer token (env SWITCHYARD_TOKEN)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("command", choices=[
        "status", "engines", "wake", "load", "sleep", "unload", "swap", "comfy-unload", "budget",
    ], help="command to run")
    p.add_argument("engine", nargs="?", help="engine name for wake/sleep/swap")
    return p


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
        run(args)
        return 0
    except CtlError as e:
        print(f"{parser.prog}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

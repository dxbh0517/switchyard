"""HTTP client for a running Switchyard dashboard (or engine port). Stdlib only.

Talks to GET /api/status, GET /api/settings, POST /api/action, POST /api/settings.
Falls back to GET /gpu/status and POST /gpu/acquire when the dashboard is absent
or an older server lacks a field. Missing keys are treated as unknown, never as errors.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import __version__

DEFAULT_TIMEOUT = 8
ACTION_TIMEOUT = 30
ACQUIRE_TIMEOUT = 120


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def default_url() -> str:
    """Dashboard URL: SWITCHYARD_URL, else DASHBOARD_HOST/PORT, else localhost:8100."""
    if os.environ.get("SWITCHYARD_URL"):
        return os.environ["SWITCHYARD_URL"].rstrip("/")
    host = os.environ.get("DASHBOARD_HOST") or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = os.environ.get("DASHBOARD_PORT") or "8100"
    return f"http://{host}:{port}"


def _num(value) -> int | None:
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def arbiter_block(status: dict | None) -> dict:
    """Dashboard wraps the arbiter; GET /gpu/status *is* the arbiter block."""
    if not isinstance(status, dict):
        return {}
    inner = status.get("arbiter")
    if isinstance(inner, dict):
        return inner
    if inner is None and "arbiter" in status:
        return {}
    if any(k in status for k in ("engines", "gpu", "owner", "placement")):
        return status
    return {}


def gpu_block(status: dict | None) -> dict:
    block = arbiter_block(status)
    gpu = block.get("gpu")
    if isinstance(gpu, dict):
        return gpu
    gpu = (status or {}).get("gpu") if isinstance(status, dict) else None
    return gpu if isinstance(gpu, dict) else {}


def vram_used_total(status: dict | None) -> tuple[int | None, int | None, str | None]:
    """(used_mb, total_mb, source). Derives used from total-free when used_mb is absent."""
    gpu = gpu_block(status)
    total = _num(gpu.get("total_mb"))
    used = _num(gpu.get("used_mb"))
    free = _num(gpu.get("free_mb"))
    if used is None and total is not None and free is not None:
        used = total - free
    source = gpu.get("source")
    return used, total, str(source) if source not in (None, "") else None


def engines_map(status: dict | None) -> dict:
    engines = arbiter_block(status).get("engines")
    if isinstance(engines, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in engines.items()}
    if isinstance(engines, list):
        out = {}
        for item in engines:
            if isinstance(item, dict) and item.get("name"):
                out[str(item["name"])] = item
        return out
    return {}


def owner_of(status: dict | None) -> str | None:
    owner = arbiter_block(status).get("owner")
    return str(owner) if owner else None


def _field_map(settings: dict | None) -> dict:
    empty = {"values": {}, "keys": set()}
    if not isinstance(settings, dict):
        return empty
    values = settings.get("values") if isinstance(settings.get("values"), dict) else {}
    fields = settings.get("fields") if isinstance(settings.get("fields"), list) else []
    keys = {f.get("key") for f in fields if isinstance(f, dict)}
    return {"values": values, "keys": keys}


def shared_supported(settings: dict | None, status: dict | None = None) -> bool:
    meta = _field_map(settings)
    if "shared_vram" in meta["keys"] or "shared_vram" in meta["values"]:
        return True
    block = arbiter_block(status)
    return any(k in block for k in ("placement", "resident", "budget"))


def shared_enabled(settings: dict | None, status: dict | None = None) -> bool:
    meta = _field_map(settings)
    if "shared_vram" in meta["values"]:
        return bool(meta["values"]["shared_vram"])
    placement = arbiter_block(status).get("placement")
    if isinstance(placement, str):
        return placement.lower() in ("shared", "pack", "budget")
    if isinstance(placement, dict):
        mode = placement.get("mode") or placement.get("kind")
        return str(mode).lower() in ("shared", "pack", "budget")
    return False


def fmt_mib(used: int | None, total: int | None) -> str:
    if total is None:
        return "—"
    if used is None:
        return f"?/{total / 1024:.1f} GiB"
    return f"{used / 1024:.1f}/{total / 1024:.1f} GiB"


class Client:
    def __init__(self, base: str | None = None, opener=None):
        self.base = (base or default_url()).rstrip("/")
        self._opener = opener  # tests inject a fake urlopen

    def request(self, method: str, path_or_url: str, body=None, timeout: float = DEFAULT_TIMEOUT):
        url = path_or_url if path_or_url.startswith("http") else self.base + path_or_url
        headers = {
            "User-Agent": f"switchyard/{__version__}",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        open_url = self._opener or urllib.request.urlopen
        try:
            with open_url(req, timeout=timeout) as resp:
                raw = resp.read()
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as e:
            raw = e.read() if e.fp else b""
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {}
            msg = payload.get("error") if isinstance(payload, dict) else None
            raise ApiError(str(msg or e.reason or f"HTTP {e.code}"), e.code) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ApiError(f"unreachable ({e})") from e
        if not raw:
            return status, None
        try:
            return status, json.loads(raw)
        except ValueError as e:
            raise ApiError(f"non-JSON response from {url}", status) from e

    def get(self, path: str, timeout: float = DEFAULT_TIMEOUT):
        return self.request("GET", path, timeout=timeout)[1]

    def post(self, path: str, body: dict | None = None, timeout: float = ACTION_TIMEOUT):
        return self.request("POST", path, body if body is not None else {}, timeout=timeout)[1]

    def status(self) -> dict:
        """Prefer /api/status; an engine port only has /gpu/status."""
        try:
            data = self.get("/api/status")
            if isinstance(data, dict):
                return data
        except ApiError as e:
            if e.status != 404:
                raise
        data = self.get("/gpu/status")
        return data if isinstance(data, dict) else {}

    def settings(self) -> dict | None:
        try:
            data = self.get("/api/settings")
            return data if isinstance(data, dict) else None
        except ApiError:
            return None

    def action(self, component: str, action: str, target: str | None = None) -> dict:
        body = {"component": component, "action": action}
        if target is not None:
            body["target"] = target
        data = self.post("/api/action", body)
        return data if isinstance(data, dict) else {"result": data}

    def update_settings(self, changes: dict) -> dict:
        data = self.post("/api/settings", changes)
        return data if isinstance(data, dict) else {}

    def set_shared(self, enabled: bool) -> dict:
        try:
            return self.update_settings({"shared_vram": bool(enabled)})
        except ApiError as e:
            if e.status == 400 or "unknown" in str(e).lower():
                return self.update_settings({"SHARED_VRAM": bool(enabled)})
            raise

    def give_gpu(self, name: str, status: dict | None = None) -> dict:
        if name == "comfy":
            return self.give_comfy(status)
        return self.action("arbiter", "warm", name)

    def give_comfy(self, status: dict | None = None) -> dict:
        try:
            return self.action("arbiter", "comfy_acquire")
        except ApiError as e:
            if e.status in (400, 404) or "unknown" in str(e).lower():
                return self.acquire_via_engine(status)
            raise

    def acquire_via_engine(self, status: dict | None) -> dict:
        scheme, host = _host_of(self.base)
        for eng in engines_map(status).values():
            port = _num(eng.get("listen"))
            if not port:
                continue
            url = f"{scheme}://{host}:{port}/gpu/acquire"
            data = self.request("POST", url, {}, timeout=ACQUIRE_TIMEOUT)[1]
            return data if isinstance(data, dict) else {"owner": "comfy"}
        raise ApiError("no engine listen port to POST /gpu/acquire")

    def clear_all(self, status: dict | None = None) -> list[str]:
        """Sleep every engine and unload ComfyUI. Continues past individual failures."""
        errors = []
        for name in engines_map(status):
            try:
                self.action("arbiter", "sleep", name)
            except ApiError as e:
                errors.append(f"{name}: {e}")
        try:
            self.action("arbiter", "comfy_unload")
        except ApiError as e:
            errors.append(f"comfy: {e}")
        return errors


def _host_of(base: str) -> tuple[str, str]:
    u = urlparse(base)
    host = u.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return u.scheme or "http", host

"""Settings: defaults < environment variables < saved file (written by the dashboard).

Every setting is declared once in FIELDS with its type, group and whether a change takes
effect immediately ("live") or after a restart. Components read settings on each use
(`settings.get(key)`), so live changes need no extra wiring.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Field:
    key: str
    group: str
    label: str
    type: str            # int | float | str | bool | choice | engines
    default: object
    env: str
    live: bool = True
    help: str = ""
    choices: tuple = ()
    min: float | None = None
    max: float | None = None


GROUPS = {
    "arbiter": "GPU arbiter",
    "ovms": "OVMS shim",
    "stt": "Speech to text",
    "dashboard": "Dashboard",
}

FIELDS: list[Field] = [
    # GPU arbiter ---------------------------------------------------------------
    Field("arbiter_enabled", "arbiter", "Enabled", "bool", True, "ARBITER_ENABLED", live=False),
    Field("engines", "arbiter", "Engines", "engines",
          "qwen=127.0.0.1:8002@8000:21400,vision=127.0.0.1:8003@8005:17500", "ENGINES", live=False,
          help="vLLM engines sharing the GPU. Name, backend and listen port changes need a "
               "restart; the free-VRAM target applies live."),
    Field("text_only_engines", "arbiter", "Text-only engines", "str", "qwen", "TEXT_ONLY_ENGINES",
          help="Comma-separated. Image parts sent to these engines are replaced with a note."),
    Field("comfy_url", "arbiter", "ComfyUI URL", "str", "http://127.0.0.1:8188", "COMFY_URL",
          help="Queue, model unload, and last-resort VRAM probe via /system_stats."),
    Field("gpu_probe", "arbiter", "VRAM probe", "choice", "auto", "GPU_PROBE",
          choices=("auto", "nvml", "nvidia-smi", "comfy", "off"),
          help="auto tries NVML, then nvidia-smi, then ComfyUI. off disables probing."),
    Field("gpu_index", "arbiter", "GPU index", "int", 0, "GPU_INDEX", min=0,
          help="Which NVIDIA GPU to read. Single-card hosts leave this at 0."),
    Field("listen_host", "arbiter", "Listen host", "str", "0.0.0.0", "LISTEN_HOST", live=False),
    Field("idle_sleep_sec", "arbiter", "Sleep after idle (s)", "int", 120, "IDLE_SLEEP_SEC", min=10,
          help="An awake engine with no traffic for this long goes back to sleep."),
    Field("sleep_level", "arbiter", "Sleep level", "choice", 2, "SLEEP_LEVEL", choices=(1, 2),
          help="1 keeps weights in CPU RAM (fast wake, uses RAM). 2 drops them (reload from disk)."),
    Field("wake_attempts", "arbiter", "Wake attempts", "int", 3, "WAKE_ATTEMPTS", min=1, max=10),
    Field("comfy_free_wait_sec", "arbiter", "Wait for ComfyUI unload (s)", "float", 30.0,
          "COMFY_FREE_WAIT_SEC", min=0),
    Field("loading_max_sec", "arbiter", "Engine startup claim (s)", "int", 900, "LOADING_MAX_SEC", min=60,
          help="How long a starting engine may hold the GPU before its claim expires."),
    # OVMS shim -----------------------------------------------------------------
    Field("ovms_enabled", "ovms", "Enabled", "bool", True, "OVMS_SHIM_ENABLED", live=False),
    Field("ovms_port", "ovms", "Listen port", "int", 8012, "OVMS_SHIM_PORT", live=False),
    Field("ovms_upstream", "ovms", "OVMS URL", "str", "http://127.0.0.1:8010", "OVMS_UPSTREAM"),
    Field("rerank_model", "ovms", "Rerank model", "str", "qwen3-reranker-0.6b", "RERANK_MODEL"),
    Field("rerank_instruction", "ovms", "Rerank instruction", "str",
          "Given a web search query, retrieve relevant passages that answer the query", "RERANK_INSTRUCTION"),
    # STT -----------------------------------------------------------------------
    Field("stt_enabled", "stt", "Enabled", "bool", True, "STT_ENABLED", live=False),
    Field("stt_port", "stt", "Listen port", "int", 8091, "STT_PORT", live=False),
    Field("stt_model_path", "stt", "Model path", "str", "/models/granite-speech-5.0-470m-turboctc",
          "STT_MODEL_PATH", help="Changing it reloads the model."),
    Field("stt_model_id", "stt", "Model name", "str", "granite-speech-5.0-470m-turboctc", "STT_MODEL_ID"),
    Field("stt_threads", "stt", "CPU threads", "int", 8, "STT_THREADS", min=1, max=64),
    Field("stt_chunk_sec", "stt", "Chunk length (s)", "float", 30.0, "STT_CHUNK_SEC", min=5, max=120,
          help="Long audio is cut at the quietest point near each boundary."),
    # Dashboard -----------------------------------------------------------------
    Field("dashboard_host", "dashboard", "Listen host", "str", "127.0.0.1", "DASHBOARD_HOST", live=False,
          help="The dashboard can change settings and has no login; keep it on localhost."),
    Field("dashboard_port", "dashboard", "Listen port", "int", 8100, "DASHBOARD_PORT", live=False),
]
BY_KEY = {f.key: f for f in FIELDS}


# ------------------------------------------------------------------ engines spec
def parse_engines(spec: str | list) -> list[dict]:
    """'name=host:port@listen:free_mb,...' -> [{name, backend, listen, free_mb}]"""
    if isinstance(spec, list):
        return [dict(e) for e in spec]
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, rest = item.split("=", 1)
        backend, rest = rest.split("@", 1)
        listen, free_mb = (rest.split(":", 1) + ["0"])[:2]
        out.append({"name": name, "backend": backend, "listen": int(listen), "free_mb": int(free_mb)})
    return out


def _coerce(f: Field, value):
    if f.type == "engines":
        engines = parse_engines(value)
        seen = set()
        for e in engines:
            name = str(e.get("name", "")).strip()
            host, _, port = str(e.get("backend", "")).rpartition(":")
            if not name or name in seen or not host or not port.isdigit():
                raise ValueError(f"bad engine entry: {e}")
            seen.add(name)
            e.update(name=name, listen=int(e["listen"]), free_mb=int(e.get("free_mb") or 0))
        return engines
    if f.type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    numeric_choice = f.type == "choice" and f.choices and isinstance(next(iter(f.choices)), int)
    if f.type == "int" or numeric_choice:
        value = int(value)
    elif f.type == "float":
        value = float(value)
    else:
        value = str(value)
    if f.choices and value not in f.choices:
        raise ValueError(f"{f.key}: must be one of {f.choices}")
    if f.min is not None and value < f.min:
        raise ValueError(f"{f.key}: must be >= {f.min}")
    if f.max is not None and value > f.max:
        raise ValueError(f"{f.key}: must be <= {f.max}")
    return value


class Settings:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or os.environ.get("SWITCHYARD_SETTINGS", "/data/settings.json"))
        self._lock = threading.Lock()
        self._values: dict = {}
        self._listeners: list = []
        for f in FIELDS:
            raw = os.environ.get(f.env)
            self._values[f.key] = _coerce(f, raw if raw not in (None, "") else f.default)
        self.saved: dict = {}
        if self.path.exists():
            try:
                self.saved = json.loads(self.path.read_text())
            except ValueError:
                self.saved = {}
            for k, v in self.saved.items():
                if k in BY_KEY:
                    try:
                        self._values[k] = _coerce(BY_KEY[k], v)
                    except (ValueError, TypeError):
                        pass
        # Values in effect at start; a non-live field that differs needs a restart.
        self.boot = json.loads(json.dumps(self._values))

    def get(self, key: str):
        with self._lock:
            return self._values[key]

    def engine(self, name: str) -> dict | None:
        return next((e for e in self.get("engines") if e["name"] == name), None)

    def on_change(self, fn) -> None:
        self._listeners.append(fn)

    def schema(self) -> dict:
        with self._lock:
            values = json.loads(json.dumps(self._values))
        return {
            "groups": GROUPS,
            "fields": [asdict(f) for f in FIELDS],
            "values": values,
            "restart_pending": self.restart_pending(),
        }

    def restart_pending(self) -> list[str]:
        with self._lock:
            return [f.key for f in FIELDS if not f.live and self._values[f.key] != self.boot[f.key]]

    def update(self, changes: dict) -> dict:
        clean = {}
        for k, v in changes.items():
            if k not in BY_KEY:
                raise ValueError(f"unknown setting {k}")
            clean[k] = _coerce(BY_KEY[k], v)
        with self._lock:
            old = {k: self._values[k] for k in clean}
            self._values.update(clean)
            self.saved.update(clean)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.saved, indent=2))
            tmp.replace(self.path)
        changed = {k: v for k, v in clean.items() if old[k] != v}
        for fn in self._listeners:
            try:
                fn(changed)
            except Exception:  # a listener must not break saving
                pass
        return {"changed": sorted(changed), "restart_pending": self.restart_pending()}

    def reset(self, key: str) -> dict:
        """Forget a saved override (falls back to env/default after restart)."""
        with self._lock:
            self.saved.pop(key, None)
            self.path.write_text(json.dumps(self.saved, indent=2))
        return {"reset": key}

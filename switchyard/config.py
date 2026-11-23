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

from .estimate import quant_bytes


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
    section: str = ""    # optional heading inside a group; unknown sections just render their key


GROUPS = {
    "arbiter": "GPU arbiter",
    "ovms": "OVMS shim",
    "stt": "Speech to text",
    "hub": "Hub",
    "dashboard": "Dashboard",
}

# Headings inside a settings group.
SECTIONS = {
    "engines": "Engines",
    "switching": "Switching & swapping",
    "health": "Health checks",
    "vram": "VRAM probe",
    "budget": "Shared VRAM budget",
    "reserve": "System reserve",
}

FIELDS: list[Field] = [
    # GPU arbiter ---------------------------------------------------------------
    Field("arbiter_enabled", "arbiter", "Enabled", "bool", True, "ARBITER_ENABLED", live=False),
    Field("listen_host", "arbiter", "Listen host", "str", "0.0.0.0", "LISTEN_HOST", live=False),
    Field("engines", "arbiter", "Engines", "engines",
          "qwen=127.0.0.1:8002@8000:21400,vision=127.0.0.1:8003@8005:17500", "ENGINES", live=False,
          help="name=host:port@listen:free_mb[:vram_mb][:boot|:off]. On/off, boot, name, backend "
               "and listen need a restart; free MiB and VRAM cost apply live. At most one On engine "
               "may load on boot.",
          section="engines"),
    Field("text_only_engines", "arbiter", "Text-only engines", "str", "qwen", "TEXT_ONLY_ENGINES",
          help="Comma-separated. Image parts sent to these engines are replaced with a note.",
          section="engines"),
    Field("idle_sleep_sec", "arbiter", "Sleep after idle (s)", "int", 120, "IDLE_SLEEP_SEC", min=10,
          help="An awake engine with no traffic for this long goes back to sleep.",
          section="switching"),
    Field("sleep_level", "arbiter", "Sleep level", "choice", 2, "SLEEP_LEVEL", choices=(1, 2),
          help="1 keeps weights in CPU RAM (fast wake, uses RAM). 2 drops them (reload from disk).",
          section="switching"),
    Field("wake_attempts", "arbiter", "Wake attempts", "int", 3, "WAKE_ATTEMPTS", min=1, max=10,
          help="How many times to try waking an engine before marking it unhealthy and returning HTTP 503.",
          section="switching"),
    Field("wake_backoff_sec", "arbiter", "Wake retry backoff (s)", "float", 1.0, "WAKE_BACKOFF_SEC", min=0,
          help="Initial delay between wake attempts; doubles after each failure (1s, 2s, 4s, …).",
          section="switching"),
    Field("comfy_free_wait_sec", "arbiter", "Wait for ComfyUI unload (s)", "float", 30.0,
          "COMFY_FREE_WAIT_SEC", min=0, section="switching"),
    Field("loading_max_sec", "arbiter", "Engine startup claim (s)", "int", 900, "LOADING_MAX_SEC", min=60,
          help="How long a starting engine may hold the GPU before its claim expires.",
          section="switching"),
    Field("comfy_url", "arbiter", "ComfyUI URL", "str", "http://127.0.0.1:8188", "COMFY_URL",
          help="Queue, model unload, and last-resort VRAM probe via /system_stats.",
          section="switching"),
    Field("health_interval_sec", "arbiter", "Health check interval (s)", "float", 15.0,
          "HEALTH_INTERVAL_SEC", min=0.1,
          help="How often to probe vLLM /health and ComfyUI /system_stats. Probes do not hold the arbiter lock.",
          section="health"),
    Field("health_timeout_sec", "arbiter", "Health check timeout (s)", "float", 5.0,
          "HEALTH_TIMEOUT_SEC", min=0.05,
          help="HTTP timeout for each health probe. Also used to verify the engine after a wake.",
          section="health"),
    Field("gpu_probe", "arbiter", "VRAM probe", "choice", "auto", "GPU_PROBE",
          choices=("auto", "nvml", "nvidia-smi", "comfy", "off"),
          help="auto tries NVML, then nvidia-smi, then ComfyUI. off disables probing.",
          section="vram"),
    Field("gpu_index", "arbiter", "GPU index", "int", 0, "GPU_INDEX", min=0,
          help="Which NVIDIA GPU to read. Single-card hosts leave this at 0.",
          section="vram"),
    Field("shared_vram", "arbiter", "Shared VRAM", "bool", False, "SHARED_VRAM",
          help="Off (default): one occupant at a time. On: keep several engines awake while "
               "their VRAM costs fit the budget; they serve in parallel. A wake that would not fit "
               "evicts least-recently used (after that victim's in-flight requests drain). "
               "ComfyUI still takes the whole GPU.",
          section="budget"),
    Field("vram_budget_mb", "arbiter", "VRAM budget (MiB)", "int", 0, "VRAM_BUDGET_MB", min=0,
          help="Card size used for shared placement. 0 = use the total from the VRAM probe.",
          section="budget"),
    Field("vram_safety_mb", "arbiter", "VRAM safety margin (MiB)", "int", 1024, "VRAM_SAFETY_MB", min=0,
          help="Headroom left unused so a wake or a ComfyUI prompt does not OOM. "
               "Usable budget is vram_budget_mb minus this.",
          section="budget"),
    Field("vram_reserve_mb", "arbiter", "Keep VRAM free (MiB)", "int", 0, "VRAM_RESERVE_MB", min=0,
          help="VRAM left for the desktop and other GPU apps. A wake asks ComfyUI to unload until "
               "this much is free on top of the engine's need, shared placement packs engines into "
               "the budget minus this, and a warning is logged when a woken engine leaves less. "
               "vLLM still takes its --gpu-memory-utilization share, so size that to match.",
          section="reserve"),
    Field("ram_reserve_mb", "arbiter", "Keep RAM free (MiB)", "int", 4096, "RAM_RESERVE_MB", min=0,
          help="Host RAM kept available for the OS. Sleep level 1 falls back to level 2 when "
               "backing the weights up to RAM would eat into it, and when available RAM drops "
               "below it an idle ComfyUI is told to drop its cached models. 0 disables.",
          section="reserve"),
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
    # Hub -----------------------------------------------------------------------
    Field("hub_enabled", "hub", "Enabled", "bool", False, "HUB_ENABLED", live=False,
          help="Model guides for ComfyUI, shared skills and an MCP gateway on one port."),
    Field("hub_host", "hub", "Listen host", "str", "", "HUB_HOST", live=False,
          help="Empty = LISTEN_HOST. The hub proxies MCP servers, so bind it narrowly "
               "(e.g. 127.0.0.1, or the docker bridge 172.17.0.1) or set a token."),
    Field("hub_port", "hub", "Listen port", "int", 8120, "HUB_PORT", live=False),
    Field("hub_dir", "hub", "Library folder", "str", "/data/hub", "HUB_DIR",
          help="Holds models/*.md (guides), skills/ and mcp.json. Re-read every few seconds."),
    Field("hub_skills_dirs", "hub", "Extra skills folders", "str", "", "HUB_SKILLS_DIRS",
          help="Comma-separated folders searched for */SKILL.md after <library>/skills."),
    Field("hub_token", "hub", "API token", "str", "", "HUB_TOKEN",
          help="When set, clients need Authorization: Bearer <token> (MCP included). "
               "Write-only: the current value is never shown here."),
    # Dashboard -----------------------------------------------------------------
    Field("dashboard_host", "dashboard", "Listen host", "str", "127.0.0.1", "DASHBOARD_HOST", live=False,
          help="127.0.0.1 keeps the dashboard local. 0.0.0.0 exposes it; set SWITCHYARD_TOKEN if you do."),
    Field("dashboard_port", "dashboard", "Listen port", "int", 8100, "DASHBOARD_PORT", live=False),
    Field("dashboard_token", "dashboard", "API token", "str", "", "SWITCHYARD_TOKEN",
          help="When set, every dashboard API call needs Authorization: Bearer <token>. "
               "Required if the dashboard is reachable from other machines. Write-only: "
               "the current value is never shown here."),
]
BY_KEY = {f.key: f for f in FIELDS}


# ------------------------------------------------------------------ engines spec
# :boot / :off (and :park / :on) after free_mb. Extra integers are ignored so a
# later vram_mb field can land without breaking this parser.
_ENGINE_FLAGS = {"boot": ("boot", True), "park": ("boot", False),
                 "off": ("enabled", False), "on": ("enabled", True)}


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_engine(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"bad engine entry: {raw}")
    e = dict(raw)  # keep unknown keys (vram_mb, quant, …) for a later budget PR
    e["name"] = str(e.get("name", "")).strip()
    e["backend"] = str(e.get("backend", "")).strip()
    e["listen"] = int(e.get("listen") or 0)
    e["free_mb"] = int(e.get("free_mb") or 0)
    e["enabled"] = _as_bool(e.get("enabled", True), True)
    e["boot"] = _as_bool(e.get("boot", False), False)
    return e


def parse_engines(spec: str | list) -> list[dict]:
    """'name=host:port@listen:free_mb[:vram_mb][:flag...],...' -> engine dicts."""
    if isinstance(spec, list):
        return [_normalize_engine(e) for e in spec]
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, rest = item.split("=", 1)
        backend, rest = rest.split("@", 1)
        parts = rest.split(":")
        listen = int(parts[0])
        free_mb, flags = 0, []
        if len(parts) > 1:
            try:
                free_mb = int(parts[1] or 0)
                flags = parts[2:]
            except ValueError:
                flags = parts[1:]
        entry = {"name": name, "backend": backend, "listen": listen, "free_mb": free_mb,
                 "enabled": True, "boot": False}
        for i, tok in enumerate(flags):
            t = tok.strip().lower()
            if not t:
                continue
            if t in _ENGINE_FLAGS:
                key, val = _ENGINE_FLAGS[t]
                entry[key] = val
                continue
            try:
                n = int(t)
            except ValueError:
                raise ValueError(f"unknown engine flag {tok!r}") from None
            if i == 0 and parts[1:2] and parts[1].strip().isdigit():
                entry["vram_mb"] = n  # the field right after free_mb is the VRAM cost
            # other numeric fields are reserved and ignored
        out.append(entry)
    return out


def _coerce(f: Field, value):
    if f.type == "engines":
        engines = parse_engines(value)
        seen, boots = set(), []
        for e in engines:
            name = e["name"]
            host, _, port = str(e.get("backend", "")).rpartition(":")
            if not name or name in seen or not host or not port.isdigit():
                raise ValueError(f"bad engine entry: {e}")
            if e["free_mb"] < 0:
                raise ValueError(f"{name}: free_mb must be >= 0")
            seen.add(name)
            if e["enabled"] and e["boot"]:
                boots.append(name)
            for k in ("vram_mb", "n_params", "n_layers", "n_kv_heads", "n_heads", "head_dim",
                      "hidden_size", "context_length"):
                if e.get(k) not in (None, ""):
                    e[k] = int(float(e[k]))
            for k in ("quant", "kv_quant"):
                if e.get(k):
                    e[k] = str(e[k])
                    quant_bytes(e[k])  # unknown quant -> ValueError at save time, not at wake
        if len(boots) > 1:
            raise ValueError(f"at most one enabled engine may load on boot, got {boots}")
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
            "sections": SECTIONS,
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
            self._persist()
        changed = {k: v for k, v in clean.items() if old[k] != v}
        for fn in self._listeners:
            try:
                fn(changed)
            except Exception:  # a listener must not break saving
                pass
        return {"changed": sorted(changed), "restart_pending": self.restart_pending()}

    def _persist(self) -> None:
        """Write saved values: create the parent dir and replace via a temp file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.saved, indent=2))
        tmp.replace(self.path)

    def reset(self, key: str) -> dict:
        """Forget a saved override (falls back to env/default after restart)."""
        with self._lock:
            self.saved.pop(key, None)
            self._persist()
        return {"reset": key}

"""Hub: what clients need to use the stack well, on one port.

  * Model guides: Markdown files in <hub_dir>/models/ (front matter lists the ComfyUI
    model files each guide covers). They are matched against ComfyUI's live inventory
    (GET /models, /models/<folder>), so a client sees which guides apply to what is
    installed and which installed files have no guide.
    Files under hidden path segments (".cache/…") are dropped, and <hub_dir>/ignore.txt can
    hide more (one glob per line: "folder/pattern", or a bare pattern matched on the file name).
  * Skills: folders holding a SKILL.md (Claude/Agent Skills layout) under <hub_dir>/skills/
    and any extra HUB_SKILLS_DIRS.
  * MCP: POST /mcp is a small MCP server (streamable HTTP, JSON responses, stateless) with
    tools over the guides, skills and server catalog. Servers listed in <hub_dir>/mcp.json
    with a "url" are proxied at /mcp/<name>, so a client configures one host. Servers with a
    "command" (stdio) are listed in the catalog for clients to launch themselves.

REST (all GET, JSON unless noted):
  /                     endpoint index
  /models               ComfyUI inventory grouped by guide (+ undocumented files)
  /guides               every guide (slug, title, summary, tags, installed files)
  /guides/<slug>        one guide; ?format=md returns the Markdown
  /skills               every skill (name, description, source)
  /skills/<name>        SKILL.md; /skills/<name>/<path> a supporting file
  /mcp-servers          shared MCP server catalog
  /search?q=…           keyword search over guides and skills

When HUB_TOKEN is set, every request needs Authorization: Bearer <token>.
Stdlib only.
"""
from __future__ import annotations

import fnmatch
import hmac
import http.client
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from . import __version__
from .config import Settings
from .events import Counters, log
from .httputil import HOP_BY_HOP, Handler, Server

C = "hub"
MCP_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INVENTORY_TTL = 60.0
SCAN_TTL = 5.0
MAX_FILE_BYTES = 512 * 1024
SKILL_DEPTH = 3
SKIP_FOLDERS = {"configs", "custom_nodes"}  # ComfyUI folders that hold no model weights


# ------------------------------------------------------------------ front matter
def _scalar(raw: str):
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        return [_scalar(x) for x in _split_inline(s[1:-1]) if x.strip()]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    return s


def _split_inline(s: str) -> list[str]:
    out, cur, quote_ch = [], "", ""
    for ch in s:
        if quote_ch:
            cur += ch
            if ch == quote_ch:
                quote_ch = ""
        elif ch in "\"'":
            quote_ch = ch
            cur += ch
        elif ch == ",":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def parse_front_matter(text: str) -> tuple[dict, str]:
    """A YAML subset: `key: value`, inline `[a, b]`, block lists, and `>` / `|` block scalars.
    Nested mappings are skipped. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    i = 1
    while i < end:
        line = lines[i]
        m = re.match(r"^([A-Za-z0-9_-]+):(.*)$", line)
        if not m:
            i += 1
            continue
        key, rest = m.group(1), m.group(2).strip()
        i += 1
        block: list[str] = []
        while i < end and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
            block.append(lines[i])
            i += 1
        if rest in (">", "|", ">-", "|-", ">+", "|+"):
            parts = [b.strip() for b in block]
            meta[key] = (" " if rest[0] == ">" else "\n").join(p for p in parts if p).strip()
        elif rest:
            meta[key] = _scalar(rest + (" " + " ".join(b.strip() for b in block if b.strip())
                                        if block and not rest.startswith(("[", "\"", "'")) else ""))
        else:
            items = [b.strip()[2:].strip() for b in block if b.strip().startswith("- ")]
            if items:
                meta[key] = [_scalar(x) for x in items]
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return meta, body


def _as_list(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _first_para(body: str) -> str:
    para: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            if para:
                break
            continue
        if s.startswith(("|", "-", "*", "```")) and not para:
            continue
        para.append(s)
    return " ".join(para)[:400]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ------------------------------------------------------------------ records
@dataclass
class Guide:
    slug: str
    title: str
    summary: str
    tags: list[str]
    files: list[str]          # glob patterns, "folder/name"
    body: str
    path: str
    matched: list[str] = field(default_factory=list)

    def card(self) -> dict:
        return {"slug": self.slug, "title": self.title, "summary": self.summary, "tags": self.tags,
                "patterns": self.files, "installed": self.matched}


@dataclass
class Skill:
    name: str
    description: str
    root: Path
    source: str

    def card(self) -> dict:
        return {"name": self.name, "description": self.description, "source": self.source}


def _matches(pattern: str, item: str) -> bool:
    p, it = pattern.lower(), item.lower()
    if fnmatch.fnmatchcase(it, p):
        return True
    # "folder/x*" should also match files in subfolders of that folder ("loras/a/b.safetensors").
    return p.endswith("/*") and it.startswith(p[:-1])


def _read_text(path: Path) -> str:
    with open(path, "rb") as f:
        return f.read(MAX_FILE_BYTES).decode("utf-8", "replace")


# ------------------------------------------------------------------ hub
class Hub:
    def __init__(self, settings: Settings):
        self.s = settings
        self.stats = Counters()
        self.server: Server | None = None
        self._lock = threading.Lock()
        self._guides: list[Guide] = []
        self._skills: dict[str, Skill] = {}
        self._servers: dict[str, dict] = {}
        self._scanned = 0.0
        self._inv: dict[str, list[str]] | None = None
        self._inv_at = 0.0
        self._inv_error: str | None = None

    # -- sources --------------------------------------------------------------
    def root(self) -> Path:
        return Path(self.s.get("hub_dir") or "/data/hub")

    def skill_dirs(self) -> list[Path]:
        dirs = [self.root() / "skills"]
        dirs += [Path(p) for p in _as_list(self.s.get("hub_skills_dirs"))]
        return dirs

    def ignored(self) -> list[str]:
        path = self.root() / "ignore.txt"
        try:
            lines = _read_text(path).splitlines()
        except OSError:
            return []
        return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    def _keep(self, folder: str, name: str, patterns: list[str]) -> bool:
        if any(part.startswith(".") for part in name.split("/")):
            return False
        item = f"{folder}/{name}"
        base = name.rsplit("/", 1)[-1].lower()
        for p in patterns:
            if "/" in p:
                if _matches(p, item):
                    return False
            elif fnmatch.fnmatchcase(base, p.lower()):
                return False
        return True

    def scan(self, force: bool = False) -> None:
        with self._lock:
            if not force and time.monotonic() - self._scanned < SCAN_TTL:
                return
            self._guides = self._load_guides()
            self._skills = self._load_skills()
            self._servers = self._load_servers()
            self._scanned = time.monotonic()

    def _load_guides(self) -> list[Guide]:
        out: list[Guide] = []
        base = self.root() / "models"
        if not base.is_dir():
            return out
        seen: set[str] = set()
        for path in sorted(base.rglob("*.md")):
            try:
                meta, body = parse_front_matter(_read_text(path))
            except OSError as e:
                log(C, f"cannot read {path}: {e}", "warn")
                continue
            slug = slugify(str(meta.get("slug") or path.stem))
            if not slug or slug in seen:
                continue
            seen.add(slug)
            title = str(meta.get("title") or _first_heading(body) or path.stem)
            out.append(Guide(slug=slug, title=title,
                             summary=str(meta.get("summary") or meta.get("description") or _first_para(body)),
                             tags=_as_list(meta.get("tags")), files=_as_list(meta.get("files")),
                             body=body, path=str(path)))
        return out

    def _load_skills(self) -> dict[str, Skill]:
        found: dict[str, Skill] = {}
        for base in self.skill_dirs():
            if not base.is_dir():
                continue
            for md in sorted(self._find_skill_files(base)):
                try:
                    meta, body = parse_front_matter(_read_text(md))
                except OSError:
                    continue
                name = str(meta.get("name") or md.parent.name).strip()
                if not name or name in found:  # first directory wins
                    continue
                desc = str(meta.get("description") or _first_para(body))
                found[name] = Skill(name=name, description=desc, root=md.parent, source=str(base))
        return found

    @staticmethod
    def _find_skill_files(base: Path) -> list[Path]:
        out: list[Path] = []

        def walk(d: Path, depth: int) -> None:
            md = d / "SKILL.md"
            if md.is_file():
                out.append(md)
                return  # a skill's own subfolders are its resources, not more skills
            if depth >= SKILL_DEPTH:
                return
            try:
                children = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                return
            for child in children:
                walk(child, depth + 1)

        walk(base, 0)
        return out

    def _load_servers(self) -> dict[str, dict]:
        path = self.root() / "mcp.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(_read_text(path))
        except (OSError, ValueError) as e:
            log(C, f"cannot read {path}: {e}", "warn")
            return {}
        raw = data.get("mcpServers", data.get("servers", {})) if isinstance(data, dict) else {}
        out = {}
        for name, cfg in (raw or {}).items():
            if isinstance(cfg, dict) and re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                out[name] = cfg
        return out

    # -- ComfyUI inventory -----------------------------------------------------
    def _comfy_get(self, path: str, timeout: float = 10):
        u = urlparse(self.s.get("comfy_url"))
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"ComfyUI {path} returned HTTP {resp.status}")
            return json.loads(data or b"null")
        finally:
            conn.close()

    def inventory(self, refresh: bool = False) -> dict[str, list[str]] | None:
        """{folder: [file, ...]} from ComfyUI, cached for INVENTORY_TTL seconds."""
        with self._lock:
            fresh = self._inv is not None and time.monotonic() - self._inv_at < INVENTORY_TTL
            if fresh and not refresh:
                return self._inv
        try:
            folders = self._comfy_get("/models")
            skip = self.ignored()
            inv: dict[str, list[str]] = {}
            for folder in folders or []:
                if not isinstance(folder, str) or folder in SKIP_FOLDERS:
                    continue
                files = self._comfy_get("/models/" + quote(folder))
                kept = sorted(str(f) for f in files or [] if self._keep(folder, str(f), skip))
                if kept:
                    inv[folder] = kept
            with self._lock:
                self._inv, self._inv_at, self._inv_error = inv, time.monotonic(), None
        except Exception as e:
            with self._lock:
                self._inv_error = str(e)
                if self._inv is None:
                    return None
        return self._inv

    def guides(self, refresh: bool = False) -> list[Guide]:
        self.scan()
        inv = self.inventory(refresh) or {}
        items = [f"{folder}/{name}" for folder, names in inv.items() for name in names]
        for g in self._guides:
            g.matched = [it for it in items if any(_matches(p, it) for p in g.files)]
        return list(self._guides)

    def models(self, folder: str | None = None, refresh: bool = False) -> dict:
        guides = self.guides(refresh)
        inv = self._inv or {}
        documented: set[str] = set()
        groups = []
        for g in guides:
            if not g.files:
                continue
            files = [f for f in g.matched if not folder or f.startswith(folder.rstrip("/") + "/")]
            documented.update(g.matched)
            if files or not folder:
                groups.append({"guide": g.slug, "title": g.title, "summary": g.summary,
                               "installed": bool(g.matched), "files": files})
        undocumented = [f"{fo}/{n}" for fo, names in inv.items() for n in names
                        if f"{fo}/{n}" not in documented and (not folder or fo == folder.rstrip("/"))]
        return {
            "comfy_url": self.s.get("comfy_url"),
            "comfy_reachable": self._inv_error is None and self._inv is not None,
            "comfy_error": self._inv_error,
            "folders": {k: len(v) for k, v in inv.items()},
            "models": groups,
            "undocumented": undocumented,
            "general_guides": [g.card() for g in guides if not g.files],
        }

    # -- lookups ---------------------------------------------------------------
    def guide(self, key: str) -> Guide | None:
        """By slug, title, or an installed/pattern file name."""
        guides = self.guides()
        k = (key or "").strip()
        if not k:
            return None
        ks = slugify(k)
        for g in guides:
            if g.slug == ks or g.title.lower() == k.lower():
                return g
        kl = k.lower()
        for g in guides:  # an installed file, by full path or file name
            if any(kl == m.lower() or m.lower().endswith("/" + kl) for m in g.matched):
                return g
        for g in guides:  # a pattern; a bare file name only matches a pattern's own name part
            for p in g.files:
                name = p.rsplit("/", 1)[-1]
                if _matches(p, kl) or (name.strip("*?") and fnmatch.fnmatchcase(kl.rsplit("/", 1)[-1], name.lower())):
                    return g
        for g in guides:
            if ks and (ks in g.slug or kl in g.title.lower()):
                return g
        return None

    def skills(self) -> list[Skill]:
        self.scan()
        return sorted(self._skills.values(), key=lambda s: s.name)

    def skill(self, name: str) -> Skill | None:
        self.scan()
        return self._skills.get((name or "").strip())

    def skill_file(self, sk: Skill, rel: str | None) -> tuple[str, list[str]]:
        """Text of SKILL.md (or a supporting file) plus the skill's file listing."""
        root = sk.root.resolve()
        listing = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
                listing.append(str(p.relative_to(root)))
            if len(listing) >= 200:
                break
        target = (root / (rel or "SKILL.md")).resolve()
        if target != root and root not in target.parents:
            raise PermissionError("path escapes the skill folder")
        if not target.is_file():
            raise FileNotFoundError(rel or "SKILL.md")
        return _read_text(target), listing

    def servers(self, base_url: str = "") -> list[dict]:
        self.scan()
        out = []
        for name, cfg in sorted(self._servers.items()):
            item = {"name": name, "description": cfg.get("description", "")}
            if cfg.get("url"):
                item["transport"] = "streamable-http"
                item["url"] = f"{base_url}/mcp/{name}" if base_url else f"/mcp/{name}"
            elif cfg.get("command"):
                item["transport"] = "stdio"
                item["command"] = cfg.get("command")
                item["args"] = cfg.get("args", [])
                if cfg.get("env"):
                    item["env_keys"] = sorted(cfg["env"])  # values may be secrets
            out.append(item)
        return out

    def search(self, query: str, limit: int = 8) -> list[dict]:
        stop = {"the", "and", "for", "with", "how", "what", "which", "use", "can", "you", "of", "to", "in", "on", "an"}
        # Crude stemming: long words match on their first 6 letters ("upscale" ~ "upscaling").
        terms = [t[:6] if len(t) > 6 else t for t in re.findall(r"[a-z0-9.]+", (query or "").lower())
                 if len(t) > 1 and t not in stop]
        if not terms:
            return []
        scored = []

        def score(title: str, head: str, body: str) -> int:
            t, h, b = title.lower(), head.lower(), body.lower()
            return sum(8 * (term in t) + 4 * (term in h) + min(b.count(term), 6) for term in terms)

        for g in self.guides():
            s = score(g.title, g.summary + " " + " ".join(g.tags + g.matched), g.body)
            if s:
                scored.append((s, {"kind": "guide", "id": g.slug, "title": g.title,
                                   "snippet": _snippet(g.body, terms) or g.summary}))
        for sk in self.skills():
            s = score(sk.name, sk.description, "")
            if s:
                scored.append((s, {"kind": "skill", "id": sk.name, "title": sk.name,
                                   "snippet": sk.description[:300]}))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:limit]]

    def status(self) -> dict:
        self.scan()
        with self._lock:
            inv = self._inv
            err = self._inv_error
            guides, skills, servers = self._guides, self._skills, self._servers
        documented = sum(1 for g in guides if g.matched)
        return {"port": self.s.boot["hub_port"], "dir": str(self.root()),
                "guides": len(guides), "guides_installed": documented, "skills": len(skills),
                "mcp_servers": len(servers), "models": sum(len(v) for v in (inv or {}).values()),
                "comfy_error": err, "auth": bool(self.token()), **self.stats.snapshot()}

    def token(self) -> str:
        return (self.s.get("hub_token") or "").strip()

    # -- MCP --------------------------------------------------------------------
    TOOLS = [
        {"name": "comfy_models",
         "description": "List the models installed in ComfyUI, grouped by the guide that documents them, "
                        "plus installed files no guide covers. Call this first to see what can be used.",
         "inputSchema": {"type": "object", "properties": {
             "folder": {"type": "string", "description": "Only this ComfyUI folder, e.g. loras or diffusion_models."},
             "refresh": {"type": "boolean", "description": "Re-read the inventory from ComfyUI now."}}}},
        {"name": "model_guide",
         "description": "Full usage guide for one model family: files, settings that work, how to prompt it. "
                        "Accepts a guide slug, title, or model file name.",
         "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        {"name": "search_guides",
         "description": "Keyword search over the model guides and shared skills. Returns the best matches "
                        "with a snippet; then read one with model_guide or get_skill.",
         "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                          "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                         "required": ["query"]}},
        {"name": "list_skills",
         "description": "List the shared skills (name and when to use each).",
         "inputSchema": {"type": "object", "properties": {"filter": {"type": "string",
                                                                     "description": "Optional substring filter."}}}},
        {"name": "get_skill",
         "description": "Read a skill's SKILL.md instructions, or one of its supporting files.",
         "inputSchema": {"type": "object", "properties": {
             "name": {"type": "string"},
             "file": {"type": "string", "description": "Path inside the skill folder; default SKILL.md."}},
             "required": ["name"]}},
        {"name": "list_mcp_servers",
         "description": "Shared MCP servers available through this hub, with the URL to connect to each.",
         "inputSchema": {"type": "object", "properties": {}}},
    ]

    def call_tool(self, name: str, args: dict, base_url: str) -> str:
        args = args or {}
        if name == "comfy_models":
            return json.dumps(self.models(args.get("folder") or None, bool(args.get("refresh"))), indent=1)
        if name == "model_guide":
            g = self.guide(str(args.get("name", "")))
            if not g:
                slugs = ", ".join(x.slug for x in self.guides())
                raise LookupError(f"no guide for {args.get('name')!r}. Guides: {slugs}")
            files = "\n".join(f"- {f}" for f in g.matched) or "- (none installed)"
            return f"{g.body}\n\n## Installed files covered by this guide\n{files}\n"
        if name == "search_guides":
            limit = max(1, min(int(args.get("limit") or 8), 20))
            hits = self.search(str(args.get("query", "")), limit)
            return json.dumps(hits, indent=1) if hits else "no matches"
        if name == "list_skills":
            flt = str(args.get("filter") or "").lower()
            rows = [f"- {s.name}: {s.description[:240]}" for s in self.skills()
                    if not flt or flt in s.name.lower() or flt in s.description.lower()]
            return "\n".join(rows) or "no skills"
        if name == "get_skill":
            sk = self.skill(str(args.get("name", "")))
            if not sk:
                raise LookupError(f"no skill named {args.get('name')!r}")
            text, listing = self.skill_file(sk, args.get("file"))
            extra = [f for f in listing if f != "SKILL.md"]
            if not args.get("file") and extra:
                text += "\n\n---\nSupporting files (read with get_skill file=…):\n" + "\n".join(
                    f"- {f}" for f in extra)
            return text
        if name == "list_mcp_servers":
            return json.dumps(self.servers(base_url), indent=1)
        raise LookupError(f"unknown tool {name}")

    def resources(self) -> list[dict]:
        out = [{"uri": f"switchyard://guides/{g.slug}", "name": g.slug, "title": g.title,
                "description": g.summary, "mimeType": "text/markdown"} for g in self.guides()]
        out += [{"uri": f"switchyard://skills/{s.name}", "name": s.name, "description": s.description[:300],
                 "mimeType": "text/markdown"} for s in self.skills()]
        return out

    def read_resource(self, uri: str) -> str:
        m = re.fullmatch(r"switchyard://(guides|skills)/(.+)", uri or "")
        if not m:
            raise LookupError(f"unknown resource {uri}")
        if m.group(1) == "guides":
            g = self.guide(m.group(2))
            if not g:
                raise LookupError(f"unknown resource {uri}")
            return g.body
        sk = self.skill(m.group(2))
        if not sk:
            raise LookupError(f"unknown resource {uri}")
        return self.skill_file(sk, None)[0]

    def rpc(self, msg: dict, base_url: str) -> dict | None:
        """One JSON-RPC message -> response (None for notifications)."""
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:
            return None

        def ok(result):
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        def err(code, message):
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                return ok({"protocolVersion": want if want in MCP_PROTOCOLS else MCP_PROTOCOLS[0],
                           "capabilities": {"tools": {"listChanged": False},
                                            "resources": {"listChanged": False}},
                           "serverInfo": {"name": "switchyard-hub", "version": __version__},
                           "instructions": (
                               "Model guides for the local ComfyUI, shared skills, and shared MCP servers. "
                               "Before writing a ComfyUI prompt or workflow, call comfy_models to see what is "
                               "installed, then model_guide for the model you pick, and follow its settings and "
                               "prompting rules. Use search_guides for task questions (upscaling, video, editing).")})
            if method == "ping":
                return ok({})
            if method == "tools/list":
                return ok({"tools": self.TOOLS})
            if method == "tools/call":
                self.stats.inc("tool_calls")
                try:
                    text = self.call_tool(params.get("name", ""), params.get("arguments") or {}, base_url)
                    return ok({"content": [{"type": "text", "text": text}], "isError": False})
                except (LookupError, PermissionError, FileNotFoundError, ValueError) as e:
                    return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
            if method == "resources/list":
                return ok({"resources": self.resources()})
            if method == "resources/read":
                uri = params.get("uri", "")
                return ok({"contents": [{"uri": uri, "mimeType": "text/markdown",
                                         "text": self.read_resource(uri)}]})
            if method in ("prompts/list",):
                return ok({"prompts": []})
            if method == "resources/templates/list":
                return ok({"resourceTemplates": []})
            return err(-32601, f"method not found: {method}")
        except LookupError as e:
            return err(-32002, str(e))
        except Exception as e:  # never kill the handler thread
            self.stats.inc("errors")
            return err(-32603, f"internal error: {e}")

    # -- HTTP --------------------------------------------------------------------
    def make_handler(self):
        hub = self

        class HubHandler(Handler):
            def base_url(self) -> str:
                host = self.headers.get("Host") or f"127.0.0.1:{hub.s.boot['hub_port']}"
                return f"http://{host}"

            def _auth_ok(self) -> bool:
                token = hub.token()
                if not token:
                    return True
                kind, _, got = (self.headers.get("Authorization") or "").partition(" ")
                if kind.lower() == "bearer" and hmac.compare_digest(got.strip().encode(), token.encode()):
                    return True
                self.send_json(401, {"error": "unauthorized"}, extra={"WWW-Authenticate": "Bearer"})
                return False

            def send_text(self, code: int, text: str, ctype="text/markdown; charset=utf-8") -> None:
                self.send_bytes(code, text.encode("utf-8"), ctype)

            def do_GET(self):  # noqa: N802
                if not self._auth_ok():
                    return
                u = urlparse(self.path)
                path = unquote(u.path).rstrip("/") or "/"
                q = parse_qs(u.query)
                hub.stats.inc("requests")
                try:
                    if path.startswith("/mcp"):
                        return self.mcp()
                    if path == "/":
                        return self.send_json(200, {
                            "name": "switchyard-hub", "version": __version__,
                            "mcp": f"{self.base_url()}/mcp",
                            "endpoints": ["/models", "/guides", "/guides/<slug>", "/skills",
                                          "/skills/<name>[/<file>]", "/mcp-servers", "/search?q="]})
                    if path == "/models":
                        return self.send_json(200, hub.models((q.get("folder") or [None])[0],
                                                              "refresh" in q))
                    if path == "/guides":
                        return self.send_json(200, [g.card() for g in hub.guides()])
                    if path.startswith("/guides/"):
                        g = hub.guide(path[len("/guides/"):])
                        if not g:
                            return self.send_json(404, {"error": "no such guide"})
                        if (q.get("format") or [""])[0] == "md":
                            return self.send_text(200, g.body)
                        return self.send_json(200, {**g.card(), "markdown": g.body})
                    if path == "/skills":
                        return self.send_json(200, [s.card() for s in hub.skills()])
                    if path.startswith("/skills/"):
                        name, _, rel = path[len("/skills/"):].partition("/")
                        sk = hub.skill(name)
                        if not sk:
                            return self.send_json(404, {"error": "no such skill"})
                        try:
                            text, listing = hub.skill_file(sk, rel or None)
                        except PermissionError:
                            return self.send_json(403, {"error": "path escapes the skill folder"})
                        except FileNotFoundError:
                            return self.send_json(404, {"error": "no such file"})
                        if rel:
                            return self.send_text(200, text, "text/plain; charset=utf-8")
                        return self.send_json(200, {**sk.card(), "files": listing, "markdown": text})
                    if path == "/mcp-servers":
                        return self.send_json(200, hub.servers(self.base_url()))
                    if path == "/search":
                        return self.send_json(200, hub.search((q.get("q") or [""])[0]))
                    self.send_json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):  # noqa: N802
                if not self._auth_ok():
                    return
                if urlparse(self.path).path.startswith("/mcp"):
                    return self.mcp()
                self.send_json(405, {"error": "method not allowed"})

            def do_DELETE(self):  # noqa: N802
                if not self._auth_ok():
                    return
                if urlparse(self.path).path.startswith("/mcp"):
                    return self.mcp()
                self.send_json(405, {"error": "method not allowed"})

            def mcp(self) -> None:
                path = urlparse(self.path).path.rstrip("/")
                if path == "/mcp":
                    return self.own_mcp()
                name, _, rest = path[len("/mcp/"):].partition("/")
                return self.proxy_mcp(name, rest)

            def own_mcp(self) -> None:
                if self.command != "POST":
                    # Stateless server: no server-initiated stream, no sessions to delete.
                    self.send_json(405, {"error": "use POST"}, extra={"Allow": "POST"})
                    return
                try:
                    msg = json.loads(self.body() or b"null")
                except ValueError:
                    self.send_json(400, {"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}})
                    return
                base = self.base_url()
                if isinstance(msg, list):
                    replies = [r for r in (hub.rpc(m, base) for m in msg if isinstance(m, dict)) if r]
                    if not replies:
                        self.send_response(202)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_json(200, replies)
                    return
                if not isinstance(msg, dict):
                    self.send_json(400, {"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32600, "message": "invalid request"}})
                    return
                reply = hub.rpc(msg, base)
                if reply is None:
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_json(200, reply)

            def proxy_mcp(self, name: str, rest: str) -> None:
                hub.scan()
                cfg = hub._servers.get(name)
                if not cfg or not cfg.get("url"):
                    self.send_json(404, {"error": f"no proxied MCP server named {name!r}"})
                    return
                target = urlparse(cfg["url"])
                upath = target.path or "/"
                if rest:
                    upath = upath.rstrip("/") + "/" + rest
                query = urlparse(self.path).query
                if query:
                    upath += "?" + query
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in HOP_BY_HOP and k.lower() != "authorization"}
                headers.update({str(k): str(v) for k, v in (cfg.get("headers") or {}).items()})
                body = self.body() or None
                cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
                hub.stats.inc("proxied")
                try:
                    conn = cls(target.hostname, target.port or (443 if target.scheme == "https" else 80),
                               timeout=float(cfg.get("timeout", 1200)))
                    conn.request(self.command, upath, body=body, headers=headers)
                    resp = conn.getresponse()
                except Exception as e:
                    hub.stats.inc("errors")
                    self.send_json(502, {"error": f"MCP server {name} unavailable: {e}"})
                    return
                try:
                    self.relay(resp)
                finally:
                    conn.close()

        return HubHandler

    def start(self) -> None:
        host = (self.s.get("hub_host") or "").strip() or self.s.get("listen_host")
        port = self.s.get("hub_port")
        self.scan(force=True)
        log(C, f"{host}:{port} serving {self.root()} ({len(self._guides)} guides, {len(self._skills)} skills, "
               f"{len(self._servers)} MCP servers); MCP at /mcp")
        if host in ("", "0.0.0.0", "::") and not self.token():
            log(C, f"listening on all interfaces ({host or '*'}) without HUB_TOKEN: anyone on the network "
                   "can read the guides and skills and use the proxied MCP servers. Set HUB_HOST or "
                   "HUB_TOKEN.", "warn")
        threading.Thread(target=self.inventory, daemon=True).start()
        self.server = Server((host, port), self.make_handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _snippet(body: str, terms: list[str], width: int = 240) -> str:
    low = body.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
    if pos < 0:
        return ""
    start = max(0, pos - width // 3)
    text = body[start:start + width].replace("\n", " ").strip()
    return ("…" if start else "") + text + ("…" if start + width < len(body) else "")

# Hub: model guides, shared skills and an MCP gateway

An LLM client that can generate images through ComfyUI still needs to know which models are installed and how to use them. Distilled models want few steps at CFG 1, others need a real negative prompt, prompt styles differ, and LoRAs only work with the base model they were trained for. The hub serves that knowledge to every client from one place:

- **Model guides.** Markdown files you write, one per model family. Each guide lists the ComfyUI model files it covers, and the hub matches those against ComfyUI's live model list (`GET /models`, `/models/<folder>`). Clients see which guides apply to what is installed. Installed files that no guide covers are listed separately, so you can tell when a guide is missing.
- **Skills.** Folders that contain a `SKILL.md`, in the Claude / Agent Skills layout.
- **MCP.** `POST /mcp` is an MCP server (streamable HTTP, JSON responses, no sessions) with tools over the guides and skills. Shared HTTP MCP servers listed in `mcp.json` are proxied at `/mcp/<name>`, so a client configures one host instead of several.

Turn it on with `HUB_ENABLED=true`. It is off by default.

## Library layout

```
$HUB_DIR/                  (default /data/hub)
├── models/                guides, *.md, searched recursively
│   ├── 00-overview.md     a guide without `files:` is a general (task) guide
│   └── example-model.md
├── skills/                <name>/SKILL.md (+ any supporting files)
├── mcp.json               shared MCP servers
└── ignore.txt             inventory entries to hide (optional)
```

The folder is re-read every few seconds, so edits show up without a restart. The ComfyUI inventory is cached for 60 s. Use `?refresh=1`, the `refresh` tool argument, or the dashboard's **Rescan** button to re-read it sooner.

### Guides

```markdown
---
title: Example Model 9B
summary: Fast text-to-image (4 steps). Prose prompts, CFG 1, no negatives.
tags: [image, fast]
files:
  - diffusion_models/example-model-9b*.safetensors
  - text_encoders/example_encoder.safetensors
  - vae/example-vae.safetensors
  - loras/example-model/*
---
# Example Model 9B

...settings that work, how to prompt it, LoRAs...
```

- `files` entries are globs over `folder/name`, using the names ComfyUI reports. `folder/*` also matches files in subfolders.
- One file can be covered by several guides. A shared VAE is fine.
- `slug` (optional) defaults to the file name. `summary` defaults to the first paragraph.
- The front matter is a YAML subset: scalars, `[inline, lists]`, `- block` lists, and `>` / `|` block text.

Write what a model needs to use the model well:
- the files and which loader or type each needs
- settings that are known to work (steps, CFG, sampler, scheduler, resolution)
- prompt style and length, with a weak and a strong example
- what the LoRAs do and their trigger words
- known failure modes


### ignore.txt

Some custom nodes register a folder that mirrors another folder, and model repos ship configs, tokenizers and audio samples next to their weights. `ignore.txt` hides those entries, one glob per line. A pattern that contains `/` matches `folder/name`; a bare pattern matches the file name. Paths with a hidden segment (`.cache/…`) are always dropped.

```
mirror_loras/*
*.json
*.lock
```

### mcp.json

The same shape as Claude Desktop and Cursor configs:

```json
{
  "mcpServers": {
    "comfyui": {"url": "http://127.0.0.1:8095/mcp", "description": "Generate with ComfyUI",
                "headers": {"Authorization": "Bearer upstream-token"}, "timeout": 1200},
    "audio":   {"command": "node", "args": ["/opt/audio-mcp.js"], "description": "TTS/STT (stdio)"}
  }
}
```

- **`url` servers** are proxied at `/mcp/<name>`. Every method, header (including `Mcp-Session-Id`), query string and streamed SSE response is passed through. The client's `Authorization` header is **not** forwarded; the server's own `headers` are added instead. That way the hub token and the upstream credential stay separate.
- **`command` servers** (stdio) are listed in the catalog so clients can launch them locally. Their `env` values are never returned, only the variable names.

## MCP tools

| Tool | What it returns |
|---|---|
| `comfy_models(folder?, refresh?)` | installed files grouped by guide, undocumented files, general guides |
| `model_guide(name)` | the full guide, by slug, title or model file name, with the installed files it covers |
| `search_guides(query, limit?)` | best-matching guides and skills with a snippet |
| `list_skills(filter?)` | skill names and descriptions |
| `get_skill(name, file?)` | `SKILL.md` (with a list of supporting files), or one supporting file |
| `list_mcp_servers()` | the shared server catalog with proxy URLs |

Guides and skills are also MCP resources (`switchyard://guides/<slug>`, `switchyard://skills/<name>`). The server's `instructions` tell the model to check `comfy_models` and the matching `model_guide` before it writes a ComfyUI prompt.

## Connecting clients

**Open WebUI**: Admin → Settings → External Tools → add a connection of type **MCP (Streamable HTTP)** with URL `http://host.docker.internal:8120/mcp` (bearer auth if you set `HUB_TOKEN`). Then enable the tool on the models that should use it. To reach a proxied server through the hub, add another connection with `…/mcp/<name>`.

**Claude Code / opencode / other MCP clients**: add a streamable-HTTP server with the same URL.

**Plain HTTP**: `GET /guides/<slug>?format=md` returns the Markdown, which is useful for RAG ingestion or a system prompt.

## Security

- The hub reads files and proxies MCP servers. Anyone who can reach its port can use whatever those servers allow (with a ComfyUI MCP server, that means running jobs on your GPU).
- Bind it narrowly with `HUB_HOST` (`127.0.0.1`, or the docker bridge `172.17.0.1` so containers can reach it but the LAN can't), or set `HUB_TOKEN`. On a wildcard bind with no token, Switchyard logs a warning at startup.
- Skill files are served only from inside each skill's folder; paths that escape it are refused. File reads are capped at 512 KiB.

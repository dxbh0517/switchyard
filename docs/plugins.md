# Engine adapter plugins

Switchyard can load third-party GPU occupants without editing `arbiter.py`. This
release ships the contract, the loader, and a dummy adapter. **The arbiter does
not use plugins yet** (that is a follow-up). You can list whatever was discovered
at `GET /api/plugins` on the dashboard.

Core stays stdlib-only (Python 3.10+). Plugins may use whatever they need; a
failing plugin is skipped and never takes down the process.

## Interface

Subclass `switchyard.plugins.api.EngineAdapter` and set `name`, `kind`, and
`API_VERSION`:

| Method | Role |
|---|---|
| `load()` / `unload()` | Connect or tear down. Do not occupy VRAM in `load()`. |
| `wake(level=2)` / `sleep(level=2)` | Occupy / release the GPU. Same names as PR #8's `VllmBackend`. |
| `swap_in()` / `swap_out()` | Aliases of `wake` / `sleep`. |
| `health()` → `HealthResult` | `ok`, `sleeping` (`None` if down), optional `detail`. |
| `memory_usage()` → `MemoryUsage` | `(vram_mb, ram_mb)`; either may be `None`. |

Optional: `is_sleeping()` (defaults to `health().sleeping`) and `estimate_vram()`
(defaults to `None`; PR #8 keeps the budget estimator separate).

`kind` should be `vllm` or `comfy` when wrapping those occupants, or a token of
your own (`llama.cpp`, `dummy`, …).

Copy [`examples/plugins/dummy.py`](../examples/plugins/dummy.py) as a starting
point. It fakes sleep/wake and reports canned VRAM/RAM numbers.

## Versioning

Core publishes three integers in `switchyard.plugins.api`:

- `PLUGIN_API_VERSION` — ABI this Switchyard implements (currently `1`)
- `PLUGIN_API_MIN` / `PLUGIN_API_MAX` — inclusive range the loader accepts

Adapters declare the version they were written against (`API_VERSION = 1`). If
that number is outside the supported range, the loader records status
`incompatible` with a message like:

```
plugin 'old' targets API_VERSION 0; this Switchyard supports 1-1
```

and moves on. Compatible additions bump `PLUGIN_API_VERSION` and widen `MAX`.
Required-method removals or return-shape changes raise `MIN`.

## Discovery

Scanned in this order. `PluginRegistry.by_name()` keeps the **last successful**
adapter per `name`; failed attempts stay in `records` so `/api/plugins` can show
them.

1. **Entry points**, group `switchyard.engines` (`importlib.metadata`). A
   third-party package registers:

   ```toml
   [project.entry-points."switchyard.engines"]
   myengine = "mypkg.adapter:MyAdapter"
   ```

   The object may be a class (constructed with no arguments) or an instance.

2. **Plugin directory**, `SWITCHYARD_PLUGIN_DIR`. `os.pathsep`-separated so
   several dirs work on Linux, macOS, and Windows (`:` vs `;`). Each `*.py` file
   and each package (subdirectory with `__init__.py`) is imported. Names starting
   with `_` are skipped. Export `ADAPTER` / `ADAPTERS`, or define `EngineAdapter`
   subclasses in that module.

   ```sh
   # Unix
   export SWITCHYARD_PLUGIN_DIR=/opt/switchyard/plugins
   # Windows
   set SWITCHYARD_PLUGIN_DIR=C:\switchyard\plugins
   ```

3. **Config list**, `SWITCHYARD_PLUGINS`. Comma-separated `module:Class` (the
   module must already be importable):

   ```sh
   export SWITCHYARD_PLUGINS=mypkg.adapter:MyAdapter,other.pkg:OtherAdapter
   ```

Errors are isolated per plugin and written to the activity log (`[plugins] …`).

## Trying the dummy adapter

```sh
python3 -m pip install -e .
export SWITCHYARD_PLUGIN_DIR="$PWD/examples/plugins"
python3 -m switchyard
# GET http://127.0.0.1:8100/api/plugins
```

You should see `"name": "dummy"`, `"kind": "dummy"`, `"status": "ok"`. The arbiter
will not route traffic to it until the follow-up that wires `discover()` into
occupancy.

## Convergence with PR #8 (`backends.py`)

PR #8 extracts `VllmBackend` and `ComfyBackend` with `wake` / `sleep` / `health`
/ `estimate_vram`. This plugin API uses those occupancy names on purpose. The
planned follow-up:

1. Wrap `VllmBackend` and `ComfyBackend` as built-in `EngineAdapter`s (`kind`
   `vllm` / `comfy`).
2. Map `health() -> bool | None` onto `HealthResult` (`None` ⇒ `ok=False`,
   `sleeping=None`).
3. Point the arbiter at `discover().by_name()` instead of inlined HTTP, keeping
   today's one-owner default.
4. Keep `estimate_vram` on the budget layer; live numbers come from
   `memory_usage()`.

Until that lands, installing a plugin only affects `/api/plugins`. It cannot
steal the GPU.

# Two vLLM engines on one GPU with Docker Compose

This walks through [`docker-compose.example.yml`](../docker-compose.example.yml):
Switchyard plus two vLLM engines (`qwen` and `vision`) that take turns on a
single NVIDIA GPU. The model names in the file are placeholders; swap in your own.

## The port map

Every engine has **two** ports. Clients use the *listen* port; Switchyard
forwards to the *backend* port where vLLM actually runs.

```
ENGINES=qwen=127.0.0.1:8002@8000:21400,vision=127.0.0.1:8003@8005:17500
        ──┬─ ─────┬──────────  ─┬── ──┬──
        name   backend        listen  free_MiB
```

| Engine | Clients call | Switchyard binds | vLLM runs on (`--port`) |
|---|---|---|---|
| `qwen` | `http://<host>:8000/v1/...` | `:8000` | `:8002` |
| `vision` | `http://<host>:8005/v1/...` | `:8005` | `:8003` |

Rules of thumb:

- The `--port` you pass to `vllm serve` must equal the **backend** port in `ENGINES`.
- Point clients (Open WebUI, scripts, …) at the **listen** port. Calling the
  backend port directly bypasses the arbiter, so nothing wakes the engine and
  nothing puts the other one to sleep.
- Listen and backend ports must not collide, because both are bound on the same host.
- `free_MiB` is how much free VRAM Switchyard waits for before waking that engine.
  Set it a little above what the engine needs once awake.

## What each engine needs

- `--enable-sleep-mode` on `vllm serve`.
- `VLLM_SERVER_DEV_MODE=1` in the environment. Together these expose `/sleep`,
  `/wake_up` and `/collective_rpc`, which Switchyard calls.
- A GPU claim before loading. The entrypoint runs a tiny Python one-liner that
  `POST`s `/gpu/loading?engine=NAME` to Switchyard and then `exec`s `vllm serve`
  with the service's `command:` as arguments:

  ```yaml
  entrypoint:
    - /bin/sh
    - -c
    - |
      python3 -c "import urllib.request as u
      try: u.urlopen(u.Request('http://127.0.0.1:8000/gpu/loading?engine=qwen', data=b'{}', method='POST'), timeout=3600).read()
      except Exception as e: print('gpu claim skipped:', e)"
      exec vllm serve "$$@"
    - vllm-serve
  ```

  - `engine=` is the engine **name** from `ENGINES`, not a port.
  - The claim can go to any of Switchyard's listen ports; the example uses `:8000` for both.
  - `$$@` is `$@` escaped for Compose. `vllm-serve` is just `$0`.
  - The request blocks until it is this engine's turn, so two engines never load
    at once. If Switchyard is unreachable, the engine starts anyway.
  - A claim expires after `LOADING_MAX_SEC` (default 900 s) if the engine
    never comes up.

## Networking

The example puts every service on `network_mode: host`, so `127.0.0.1` means the
host for all of them. That is the simplest setup and is why the claim URL uses
`127.0.0.1:8000`.

If you run the engines on a bridge network instead, use
`http://host.docker.internal:8000/...` in the claim (as in the README) and add
`extra_hosts: ["host.docker.internal:host-gateway"]` on Linux. The backend host
in `ENGINES` must then be an address Switchyard can reach (for example a
published port on `127.0.0.1`).

## Startup, step by step

1. `switchyard` starts and binds `:8000` and `:8005`.
2. Both engine containers start; each blocks in its `/gpu/loading` claim.
3. One claim is granted and that engine loads its weights. When Switchyard sees
   it answer `/health`, the claim is cleared and the engine is parked (put to
   sleep), unless it is marked **boot** (`:boot` in `ENGINES`, or the dashboard
   checkbox), in which case it is woken. Then the other claim is granted.
4. Parked engines stay asleep until a request arrives on their listen port, or
   until you press **Wake** in the dashboard.

Then:

```sh
curl http://127.0.0.1:8000/v1/models     # wakes qwen (sleeps vision if needed)
curl http://127.0.0.1:8005/v1/models     # wakes vision
curl http://127.0.0.1:8000/gpu/status    # who owns the GPU right now
```

The dashboard is on <http://127.0.0.1:8100/>.

## Adding ComfyUI

Copy `comfyui_node/gpu_arbiter` into `ComfyUI/custom_nodes/` and keep `COMFY_URL`
pointing at ComfyUI. With host networking the hook reaches Switchyard at
`127.0.0.1:8000`; set `GPU_ARBITER_URL=http://127.0.0.1:8000` if it does not find
it at its default `http://host.docker.internal:8000`.

## Troubleshooting

- **Clients hang or get 503:** check `GET /gpu/status` and the dashboard. An
  engine that failed to wake is marked unhealthy and returns HTTP 503 with
  `Retry-After` until a health check passes.
- **`404` on `/sleep` or `/wake_up`:** `VLLM_SERVER_DEV_MODE=1` or
  `--enable-sleep-mode` is missing on that engine.
- **Out of memory while loading:** lower `--gpu-memory-utilization`, or make
  sure the other engine's claim was honored (look for `... is loading` in the
  Switchyard log).
- **Engine never woken on request:** the client is calling the backend port.
  Use the listen port.

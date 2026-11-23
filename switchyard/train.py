"""`switchyard train -- CMD...`: run a GPU trainer with the whole card reserved.

1. POST /gpu/training/begin on an engine port. Switchyard waits for in-flight chat
   requests and ComfyUI prompts, sleeps every vLLM engine, unloads ComfyUI, and
   returns a lease. Chat requests and ComfyUI prompts queue until the lease ends.
2. Run CMD, renewing the lease every ttl/3 seconds.
3. POST /gpu/training/end when CMD exits, is interrupted, or crashes. If this
   wrapper is killed outright, the lease expires after `ttl` seconds.

Stdlib only, so it runs from any Python environment (including a trainer's venv).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ARBITER = "http://127.0.0.1:8000"
BEGIN_TIMEOUT = 3600  # waiting for a long ComfyUI prompt or chat generation is fine


def _say(msg: str) -> None:
    print(f"switchyard train: {msg}", file=sys.stderr, flush=True)


class Lease:
    def __init__(self, base: str, ttl: float, reason: str):
        self.base = base.rstrip("/")
        self.ttl = ttl
        self.reason = reason
        self.id: str | None = None
        self.lost = False
        self._stop = threading.Event()

    def _post(self, op: str, timeout: float, **params) -> dict:
        url = f"{self.base}/gpu/training/{op}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def begin(self) -> dict:
        res = self._post("begin", BEGIN_TIMEOUT, reason=self.reason, ttl=int(self.ttl))
        self.id = res["lease"]
        return res

    def renew_loop(self) -> None:
        every = max(5.0, self.ttl / 3)
        while not self._stop.wait(every):
            try:
                self._post("renew", 15, lease=self.id)
            except urllib.error.HTTPError as e:
                if e.code == 410:
                    self.lost = True
                    _say("lease was released by someone else; other GPU users may start now")
                    return
                _say(f"renew failed (HTTP {e.code}); retrying")
            except Exception as e:  # Switchyard restarting; keep trying until ttl runs out
                _say(f"renew failed ({e}); retrying")

    def end(self) -> None:
        self._stop.set()
        if not self.id or self.lost:
            return
        try:
            self._post("end", 30, lease=self.id)
            _say("GPU released")
        except Exception as e:
            _say(f"release failed ({e}); the lease expires in {int(self.ttl)}s")


def wait_for_free(base: str, min_free_mb: int, timeout: float = 60) -> int | None:
    """Poll /gpu/status until free VRAM >= min_free_mb. Returns the last reading."""
    deadline, free = time.time() + timeout, None
    while True:
        try:
            with urllib.request.urlopen(f"{base}/gpu/status", timeout=5) as r:
                free = (json.load(r).get("gpu") or {}).get("free_mb")
        except Exception:
            free = None
        if free is not None and free >= min_free_mb:
            return free
        if time.time() > deadline:
            return free
        time.sleep(2)


def parser(prog: str = "switchyard train") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Reserve the whole GPU through Switchyard, run a trainer, then release it.",
        epilog="Example: switchyard train --reason 'tina lora' -- python run.py train.yml",
    )
    p.add_argument("--arbiter", default=os.environ.get("GPU_ARBITER_URL") or DEFAULT_ARBITER,
                   help=f"any Switchyard engine port (env GPU_ARBITER_URL, default {DEFAULT_ARBITER})")
    p.add_argument("--reason", default="", help="shown in the dashboard and logs")
    p.add_argument("--ttl", type=float, default=300,
                   help="lease lifetime without a renew, seconds (default 300)")
    p.add_argument("--min-free-mb", type=int, default=0,
                   help="after reserving, wait up to 60s for this much free VRAM; abort if it never appears")
    p.add_argument("--fail-open", action="store_true",
                   help="run the command anyway if Switchyard cannot be reached")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    return p


def main(argv: list[str] | None = None, prog: str = "switchyard train") -> int:
    args = parser(prog).parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser(prog).error("no command given (put it after --)")
    base = args.arbiter.rstrip("/")
    lease = Lease(base, args.ttl, args.reason or os.path.basename(cmd[0]))

    _say(f"reserving the GPU at {base} (waits for running chat/ComfyUI work)")
    try:
        res = lease.begin()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _say(f"could not reserve the GPU: HTTP {e.code} {body}")
        return 75  # EX_TEMPFAIL
    except Exception as e:
        if not args.fail_open:
            _say(f"Switchyard unreachable ({e}); refusing to start (use --fail-open to run anyway)")
            return 75
        _say(f"Switchyard unreachable ({e}); running anyway (--fail-open)")
        lease = None
        res = {}

    if lease:
        slept = ", ".join(res.get("slept") or []) or "none awake"
        _say(f"GPU reserved (slept: {slept}; free {res.get('free_mb')} / {res.get('total_mb')} MiB)")
        if res.get("still_awake"):
            _say(f"warning: still awake: {', '.join(res['still_awake'])}")
        threading.Thread(target=lease.renew_loop, daemon=True).start()
        if args.min_free_mb:
            free = wait_for_free(base, args.min_free_mb)
            if free is None or free < args.min_free_mb:
                _say(f"only {free} MiB free, need {args.min_free_mb}; releasing and aborting")
                lease.end()
                return 75

    child: subprocess.Popen | None = None

    def forward(signum, _frame):
        if child and child.poll() is None:
            child.send_signal(signum)

    old = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        child = subprocess.Popen(cmd)
        code = child.wait()
    except FileNotFoundError as e:
        _say(str(e))
        code = 127
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                child.kill()
        if lease:
            lease.end()
        for s, h in old.items():
            signal.signal(s, h)
    _say(f"command exited with {code}")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())

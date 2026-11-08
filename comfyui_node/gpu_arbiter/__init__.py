"""Switchyard GPU arbiter hook for ComfyUI.

Before a prompt executes, POST /gpu/acquire puts every vLLM engine to sleep (waiting
for in-flight requests to finish). After it finishes, POST /gpu/release lets the
engines take the GPU back; Switchyard then calls ComfyUI's /free to unload models.
If Switchyard is unreachable, ComfyUI runs normally (fail-open).

GPU_ARBITER_URL: any engine port of Switchyard (default http://host.docker.internal:8000).
"""
import logging
import os
import urllib.request

import execution

ARBITER = os.environ.get("GPU_ARBITER_URL", "http://host.docker.internal:8000").rstrip("/")
log = logging.getLogger("gpu_arbiter")


def _post(path, timeout):
    req = urllib.request.Request(f"{ARBITER}{path}", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).read()


_orig_execute = execution.PromptExecutor.execute

if not getattr(_orig_execute, "_gpu_arbiter", False):
    def execute(self, *args, **kwargs):
        acquired = False
        try:
            # Long timeout: may wait for an LLM generation to finish.
            _post("/gpu/acquire", timeout=1800)
            acquired = True
        except Exception as e:
            log.warning("gpu_arbiter: acquire skipped (%s)", e)
        try:
            return _orig_execute(self, *args, **kwargs)
        finally:
            if acquired:
                try:
                    _post("/gpu/release", timeout=10)
                except Exception as e:
                    log.warning("gpu_arbiter: release failed (%s)", e)

    execute._gpu_arbiter = True
    execution.PromptExecutor.execute = execute
    log.info("gpu_arbiter: hooked PromptExecutor.execute -> %s", ARBITER)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

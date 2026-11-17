"""Minimal engine adapter: fakes occupancy, health, and memory.

Drop this file in a directory and point SWITCHYARD_PLUGIN_DIR at it, or copy the
class into your own package and register it (see docs/plugins.md). It never talks
to a GPU. Wake/sleep toggle an in-memory flag so the contract can be exercised
without vLLM or ComfyUI.
"""
from __future__ import annotations

from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage


class DummyAdapter(EngineAdapter):
    name = "dummy"
    kind = "dummy"
    API_VERSION = 1

    def __init__(self, vram_awake: int = 4096, vram_asleep: int = 256,
                 ram_awake: int = 2048, ram_asleep: int = 512):
        self._loaded = False
        self._awake = False
        self._vram_awake, self._vram_asleep = vram_awake, vram_asleep
        self._ram_awake, self._ram_asleep = ram_awake, ram_asleep

    def load(self) -> None:
        self._loaded = True
        self._awake = False

    def unload(self) -> None:
        self._loaded = False
        self._awake = False

    def health(self) -> HealthResult:
        if not self._loaded:
            return HealthResult(ok=False, sleeping=None, detail="not loaded")
        return HealthResult(ok=True, sleeping=not self._awake, detail="echo")

    def memory_usage(self) -> MemoryUsage:
        if not self._loaded:
            return MemoryUsage(vram_mb=None, ram_mb=None)
        if self._awake:
            return MemoryUsage(vram_mb=self._vram_awake, ram_mb=self._ram_awake)
        return MemoryUsage(vram_mb=self._vram_asleep, ram_mb=self._ram_asleep)

    def wake(self, level: int = 2) -> None:  # noqa: ARG002
        if not self._loaded:
            self.load()
        self._awake = True

    def sleep(self, level: int = 2) -> None:  # noqa: ARG002
        self._awake = False


ADAPTER = DummyAdapter

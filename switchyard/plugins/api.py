"""Engine adapter contract for third-party GPU occupants.

This module is the stable surface a drop-in plugin implements. Discovery lives in
`.loader`; the arbiter still talks to vLLM/ComfyUI directly (see `arbiter.py`).

Open PR #8 introduces `switchyard/backends.py` with `VllmBackend` / `ComfyBackend`.
Method names here match that draft where they overlap (`wake`, `sleep`, `health`,
`unload`). A follow-up will wrap those backends as `EngineAdapter`s and point the
arbiter at the plugin loader. Until then:

  Plugin API              PR #8 backends.py              Meaning
  ----------------------  -----------------------------  --------------------------------
  load() / unload()       (not present)                  connect / tear down the occupant
  wake(level) / sleep()   VllmBackend.wake / .sleep      occupy / release the GPU
  swap_in() / swap_out()  aliases of wake / sleep        same occupancy pair
  health() -> HealthResult  health() -> bool | None      up/down plus sleep bit
  memory_usage()          estimate_vram / .vram()        live VRAM (+ optional RAM)
  kind                    BACKENDS keys (vllm, comfy)    occupant type
  API_VERSION             (not present)                  plugin ABI the loader checks

`estimate_vram` stays on the budget layer from PR #8; adapters may override the
optional `estimate_vram()` below, which defaults to unknown.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Integer ABI. Bump PLUGIN_API_VERSION when adding required methods or changing
# return shapes. Widen PLUGIN_API_MAX for backward-compatible additions; raise
# PLUGIN_API_MIN only when old adapters cannot run. The loader never instantiates
# an adapter whose declared version sits outside [MIN, MAX].
PLUGIN_API_VERSION = 1
PLUGIN_API_MIN = 1
PLUGIN_API_MAX = 1

ENTRY_POINT_GROUP = "switchyard.engines"


def api_supported(version: int) -> bool:
    return PLUGIN_API_MIN <= version <= PLUGIN_API_MAX


def unsupported_version_message(name: str, version: int) -> str:
    return (f"plugin {name!r} targets API_VERSION {version}; "
            f"this Switchyard supports {PLUGIN_API_MIN}-{PLUGIN_API_MAX}")


@dataclass(frozen=True)
class HealthResult:
    """Liveness of an occupant. `ok=False` and `sleeping=None` means unreachable."""
    ok: bool
    sleeping: bool | None = None
    detail: str = ""


@dataclass(frozen=True)
class MemoryUsage:
    """Footprint in MiB. Either field may be None when the adapter cannot measure it."""
    vram_mb: int | None
    ram_mb: int | None = None


class EngineAdapter(ABC):
    """One GPU occupant: a vLLM-style engine, ComfyUI, or a third-party adapter.

    Class attributes (`name`, `kind`, `API_VERSION`) are metadata the loader reads.
    `kind` should use the same tokens PR #8 will register in `BACKENDS` (`vllm`,
    `comfy`) when wrapping those backends; custom adapters pick their own token.
    """

    name: str = ""
    kind: str = ""
    API_VERSION: int = PLUGIN_API_VERSION

    @abstractmethod
    def load(self) -> None:
        """Start or connect to the occupant. Must not occupy VRAM by itself."""

    @abstractmethod
    def unload(self) -> None:
        """Disconnect or stop the occupant. Releases any claim this adapter holds."""

    @abstractmethod
    def health(self) -> HealthResult:
        """Reachability and sleep state. Must not raise; return ok=False on failure."""

    @abstractmethod
    def memory_usage(self) -> MemoryUsage:
        """Current VRAM (and optional system RAM) in MiB."""

    @abstractmethod
    def wake(self, level: int = 2) -> None:
        """Occupy the GPU (PR #8 `VllmBackend.wake`). `level` matches vLLM sleep levels."""

    @abstractmethod
    def sleep(self, level: int = 2) -> None:
        """Release the GPU (PR #8 `VllmBackend.sleep`)."""

    def swap_in(self, level: int = 2) -> None:
        """Alias of `wake` for callers that think in swap/occupancy terms."""
        self.wake(level)

    def swap_out(self, level: int = 2) -> None:
        """Alias of `sleep` for callers that think in swap/occupancy terms."""
        self.sleep(level)

    def is_sleeping(self) -> bool | None:
        """PR #8 `VllmBackend.is_sleeping`. Derived from `health()` unless overridden."""
        return self.health().sleeping

    def estimate_vram(self) -> int | None:
        """Optional predicted footprint (PR #8 `estimate_vram`). Default: unknown."""
        return None

"""Engine adapter plugins: contract, versioning, and discovery.

The arbiter does not load plugins yet. Call `discover()` yourself, or GET
`/api/plugins` on the dashboard for a read-only listing.
"""
from .api import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_MAX,
    PLUGIN_API_MIN,
    PLUGIN_API_VERSION,
    EngineAdapter,
    HealthResult,
    MemoryUsage,
    api_supported,
)
from .loader import PluginInfo, PluginRegistry, discover

__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_MAX",
    "PLUGIN_API_MIN",
    "PLUGIN_API_VERSION",
    "EngineAdapter",
    "HealthResult",
    "MemoryUsage",
    "PluginInfo",
    "PluginRegistry",
    "api_supported",
    "discover",
]

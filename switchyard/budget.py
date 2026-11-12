"""VRAM budget: usable capacity, cost fallbacks, LRU eviction.

Costs are MiB. A missing budget (None/0) means "unknown" — the arbiter then
falls back to exclusive one-owner eviction.
"""
from __future__ import annotations


def usable_mb(budget_mb: int | None, safety_mb: int) -> int | None:
    if not budget_mb:
        return None
    return max(0, int(budget_mb) - int(safety_mb or 0))


def resolve_cost(measured_mb=None, configured_mb=None, estimated_mb=None,
                 free_mb: int = 0) -> tuple[int, str]:
    """Pick a cost and a label. First positive value wins."""
    for value, src in (
        (measured_mb, "measured"),
        (configured_mb, "configured"),
        (estimated_mb, "estimate"),
        (free_mb, "free_mb"),
    ):
        if value:
            return int(value), src
    return 0, "free_mb"


def lru_victims(resident_at: dict, costs: dict, keep: str, keep_cost: int,
                usable: int) -> list[str]:
    """Names to sleep (oldest last_infer first) so `keep` fits in `usable` MiB.

    `resident_at` maps engine name -> last-used timestamp. `keep` itself is never
    returned. If `keep` alone is larger than `usable`, every other resident is
    still evicted (same outcome as exclusive placement).
    """
    others = sorted((n for n in resident_at if n != keep), key=lambda n: resident_at[n])
    used = sum(costs.get(n, 0) for n in others) + int(keep_cost)
    out = []
    for n in others:
        if used <= usable:
            break
        out.append(n)
        used -= costs.get(n, 0)
    return out

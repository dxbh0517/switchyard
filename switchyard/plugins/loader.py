"""Discover and load engine adapters without crashing the core process.

Sources, in order (a later successful adapter of the same `name` replaces an
earlier one in `PluginRegistry.by_name()`; every attempt is kept in `records`):

  1. Python entry points in group `switchyard.engines` (`importlib.metadata`)
  2. Files and packages in `SWITCHYARD_PLUGIN_DIR` (`os.pathsep`-separated)
  3. Config list `SWITCHYARD_PLUGINS` of `module:Class` (comma-separated)

Each plugin is isolated: import errors, version mismatches, and constructor
failures are logged and recorded, then the scan continues.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from hashlib import sha1
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType

from ..events import log
from .api import (
    ENTRY_POINT_GROUP,
    EngineAdapter,
    PLUGIN_API_MAX,
    PLUGIN_API_MIN,
    PLUGIN_API_VERSION,
    api_supported,
    unsupported_version_message,
)

C = "plugins"


@dataclass
class PluginInfo:
    name: str
    kind: str
    api_version: int | None
    source: str            # entry_point | directory | config
    origin: str
    status: str            # ok | incompatible | error
    error: str | None = None
    adapter: EngineAdapter | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "api_version": self.api_version,
            "source": self.source, "origin": self.origin, "status": self.status,
            "error": self.error,
        }


@dataclass
class PluginRegistry:
    records: list[PluginInfo]

    def by_name(self) -> dict[str, EngineAdapter]:
        """Last successfully loaded adapter per `name`."""
        out: dict[str, EngineAdapter] = {}
        for rec in self.records:
            if rec.status == "ok" and rec.adapter is not None:
                out[rec.name] = rec.adapter
        return out

    def as_dict(self) -> dict:
        return {
            "api_version": PLUGIN_API_VERSION,
            "api_min": PLUGIN_API_MIN,
            "api_max": PLUGIN_API_MAX,
            "plugins": [r.as_dict() for r in self.records],
        }


def discover(plugin_dir: str | None = None, plugins: str | None = None,
             group: str | None = None, entry_points_iter=None) -> PluginRegistry:
    """Scan every source. Keyword args override the matching environment variable.

    `entry_points_iter` is a test seam: an iterable of objects with `.name`,
    `.value`, and `.load()`. When omitted, `importlib.metadata` is queried.
    """
    records: list[PluginInfo] = []
    if group is None:
        group = ENTRY_POINT_GROUP
    if plugin_dir is None:
        plugin_dir = os.environ.get("SWITCHYARD_PLUGIN_DIR", "")
    if plugins is None:
        plugins = os.environ.get("SWITCHYARD_PLUGINS", "")

    _absorb(records, lambda: _from_entry_points(group, entry_points_iter),
            source="entry_point", origin=group)
    for directory in _split_pathsep(plugin_dir):
        _absorb(records, lambda d=directory: _from_directory(d),
                source="directory", origin=directory)
    for spec in _split_comma(plugins):
        _absorb(records, lambda s=spec: _from_spec(s),
                source="config", origin=spec)
    return PluginRegistry(records)


def _absorb(records: list[PluginInfo], fn, source: str, origin: str) -> None:
    try:
        records.extend(fn())
    except Exception as e:
        msg = f"{source} scan failed for {origin}: {e}"
        log(C, msg, "error")
        records.append(PluginInfo(name="", kind="", api_version=None, source=source,
                                  origin=origin, status="error", error=msg))


def _split_pathsep(value: str) -> list[str]:
    return [p for p in (s.strip() for s in value.split(os.pathsep)) if p]


def _split_comma(value: str) -> list[str]:
    return [p for p in (s.strip() for s in value.split(",")) if p]


def _iter_entry_points(group: str):
    eps = entry_points()
    select = getattr(eps, "select", None)
    if select is not None:
        return list(select(group=group))
    return list(eps.get(group, []))  # pragma: no cover  (pre-3.10 dict interface)


def _from_entry_points(group: str, given) -> list[PluginInfo]:
    found = list(given) if given is not None else _iter_entry_points(group)
    out = []
    for ep in found:
        origin = getattr(ep, "value", None) or f"{group}:{getattr(ep, 'name', '?')}"
        name = getattr(ep, "name", "") or origin
        try:
            obj = ep.load()
        except Exception as e:
            out.append(_fail(name, "entry_point", origin, f"entry point load failed: {e}"))
            continue
        out.extend(_realize(obj, source="entry_point", origin=origin, fallback_name=name))
    return out


def _from_directory(directory: str) -> list[PluginInfo]:
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"plugin directory not found: {directory}")
    out = []
    for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        if child.is_file() and child.suffix == ".py" and not child.name.startswith("_"):
            out.extend(_from_path(child, package=False))
        elif child.is_dir() and (child / "__init__.py").is_file() and not child.name.startswith("_"):
            out.extend(_from_path(child, package=True))
    return out


def _from_path(path: Path, package: bool) -> list[PluginInfo]:
    origin = str(path)
    fallback = path.stem
    try:
        mod = _load_file(path, package=package)
    except Exception as e:
        return [_fail(fallback, "directory", origin, f"import failed: {e}")]
    return _from_module(mod, source="directory", origin=origin, fallback_name=fallback)


def _load_file(path: Path, package: bool) -> ModuleType:
    target = path / "__init__.py" if package else path
    name = _module_name(path)
    spec = importlib.util.spec_from_file_location(
        name, target, submodule_search_locations=[str(path)] if package else None)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _module_name(path: Path) -> str:
    ident = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", path.stem) or "plugin"
    return f"_switchyard_plugin_{stem}_{ident}"


def _from_spec(spec: str) -> list[PluginInfo]:
    if ":" not in spec:
        return [_fail(spec, "config", spec, "expected module:Class")]
    mod_name, _, attr = spec.partition(":")
    mod_name, attr = mod_name.strip(), attr.strip()
    if not mod_name or not attr:
        return [_fail(spec, "config", spec, "expected module:Class")]
    try:
        mod = importlib.import_module(mod_name)
        obj = mod
        for part in attr.split("."):
            obj = getattr(obj, part)
    except Exception as e:
        return [_fail(attr or spec, "config", spec, f"import failed: {e}")]
    return _realize(obj, source="config", origin=spec, fallback_name=attr)


def _from_module(mod: ModuleType, source: str, origin: str, fallback_name: str) -> list[PluginInfo]:
    try:
        objects = _exported(mod)
    except Exception as e:
        return [_fail(fallback_name, source, origin, f"scan failed: {e}")]
    if not objects:
        return [_fail(fallback_name, source, origin, "no EngineAdapter subclass or ADAPTER export")]
    out = []
    for obj in objects:
        out.extend(_realize(obj, source=source, origin=origin, fallback_name=fallback_name))
    return out


def _exported(mod: ModuleType) -> list:
    if hasattr(mod, "ADAPTERS"):
        return list(mod.ADAPTERS)
    if hasattr(mod, "ADAPTER"):
        return [mod.ADAPTER]
    found = []
    for obj in vars(mod).values():
        if inspect.isclass(obj) and issubclass(obj, EngineAdapter) and obj is not EngineAdapter:
            if obj.__module__ == mod.__name__:
                found.append(obj)
    return found


def _realize(obj, source: str, origin: str, fallback_name: str) -> list[PluginInfo]:
    name = getattr(obj, "name", None) or fallback_name
    version = _declared_version(obj)
    if inspect.isclass(obj):
        if not _is_adapter_class(obj):
            return [_fail(name, source, origin, f"{obj!r} is not an EngineAdapter subclass")]
        if version is None:
            return [_fail(name, source, origin, "does not declare API_VERSION")]
        if not api_supported(version):
            return [_reject(name, source, origin, version)]
        try:
            obj = obj()
        except Exception as e:
            return [_fail(name, source, origin, f"constructor failed: {e}", version)]
        name = getattr(obj, "name", None) or name
        version = _declared_version(obj) if _declared_version(obj) is not None else version
    else:
        if version is None:
            return [_fail(name, source, origin, "does not declare API_VERSION")]
        if not api_supported(version):
            return [_reject(name, source, origin, version)]
        if not isinstance(obj, EngineAdapter):
            return [_fail(name, source, origin, f"{type(obj).__name__} is not an EngineAdapter")]
    kind = getattr(obj, "kind", "") or ""
    rec = PluginInfo(name=str(name), kind=str(kind), api_version=version, source=source,
                     origin=origin, status="ok", adapter=obj)
    log(C, f"loaded {rec.name} ({rec.kind or 'adapter'}) from {source} {origin}")
    return [rec]


def _is_adapter_class(cls) -> bool:
    try:
        return issubclass(cls, EngineAdapter) and cls is not EngineAdapter
    except TypeError:
        return False


def _declared_version(obj) -> int | None:
    for attr in ("API_VERSION", "PLUGIN_API_VERSION"):
        if hasattr(obj, attr):
            try:
                return int(getattr(obj, attr))
            except (TypeError, ValueError):
                return None
    return None


def _reject(name: str, source: str, origin: str, version: int) -> PluginInfo:
    msg = unsupported_version_message(name, version)
    log(C, msg, "error")
    return PluginInfo(name=name, kind="", api_version=version, source=source, origin=origin,
                      status="incompatible", error=msg)


def _fail(name: str, source: str, origin: str, error: str, version: int | None = None) -> PluginInfo:
    log(C, f"{origin}: {error}", "error")
    return PluginInfo(name=name, kind="", api_version=version, source=source, origin=origin,
                      status="error", error=error)

"""Plugin contract, discovery sources, versioning, and isolation."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path

from switchyard import config
from switchyard.httputil import Server
from switchyard.plugins.api import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_MAX,
    PLUGIN_API_MIN,
    PLUGIN_API_VERSION,
    EngineAdapter,
    HealthResult,
    MemoryUsage,
    unsupported_version_message,
)
from switchyard.plugins.loader import discover

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"
DUMMY_PATH = EXAMPLES / "dummy.py"


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())


def _env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    for k, v in kwargs.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore(old: dict) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _EP:
    def __init__(self, name, obj=None, error=None, value="mod:Cls"):
        self.name, self.value, self._obj, self._error = name, value, obj, error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._obj


class DummyContractTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("example_dummy_adapter", DUMMY_PATH)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.ad = self.mod.DummyAdapter()

    def test_metadata_and_version(self):
        self.assertTrue(issubclass(self.mod.DummyAdapter, EngineAdapter))
        self.assertEqual(self.ad.name, "dummy")
        self.assertEqual(self.ad.kind, "dummy")
        self.assertEqual(self.ad.API_VERSION, PLUGIN_API_VERSION)
        self.assertEqual(self.mod.ADAPTER, self.mod.DummyAdapter)
        self.assertGreaterEqual(self.ad.API_VERSION, PLUGIN_API_MIN)
        self.assertLessEqual(self.ad.API_VERSION, PLUGIN_API_MAX)

    def test_lifecycle_health_memory_and_aliases(self):
        h = self.ad.health()
        self.assertIsInstance(h, HealthResult)
        self.assertFalse(h.ok)
        self.assertIsNone(h.sleeping)
        self.assertEqual(self.ad.memory_usage(), MemoryUsage(None, None))

        self.ad.load()
        h = self.ad.health()
        self.assertTrue(h.ok)
        self.assertTrue(h.sleeping)
        asleep = self.ad.memory_usage()
        self.assertEqual((asleep.vram_mb, asleep.ram_mb), (256, 512))

        self.ad.swap_in()
        self.assertFalse(self.ad.health().sleeping)
        self.assertFalse(self.ad.is_sleeping())
        awake = self.ad.memory_usage()
        self.assertEqual((awake.vram_mb, awake.ram_mb), (4096, 2048))
        self.assertIsNone(self.ad.estimate_vram())

        self.ad.swap_out()
        self.assertTrue(self.ad.is_sleeping())
        self.ad.wake()
        self.assertFalse(self.ad.is_sleeping())
        self.ad.sleep()
        self.assertTrue(self.ad.is_sleeping())

        self.ad.unload()
        self.assertFalse(self.ad.health().ok)


class DiscoveryTests(unittest.TestCase):
    def test_directory_file_and_package(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root / "echo.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class EchoAdapter(EngineAdapter):
                    name, kind, API_VERSION = "echo", "dummy", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True, True, "ok")
                    def memory_usage(self): return MemoryUsage(1, 2)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            _write(root / "pkg" / "__init__.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class PkgAdapter(EngineAdapter):
                    name, kind, API_VERSION = "pkg", "dummy", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True, True)
                    def memory_usage(self): return MemoryUsage(0, 0)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
                ADAPTER = PkgAdapter
            """)
            reg = discover(plugin_dir=str(root), plugins="", entry_points_iter=())
            names = {r.name: r for r in reg.records if r.status == "ok"}
            self.assertIn("echo", names)
            self.assertIn("pkg", names)
            self.assertEqual(names["echo"].source, "directory")
            self.assertEqual(names["pkg"].kind, "dummy")
            self.assertEqual(set(reg.by_name()), {"echo", "pkg"})

    def test_directory_from_env(self):
        old = _env(SWITCHYARD_PLUGIN_DIR=str(EXAMPLES), SWITCHYARD_PLUGINS="")
        try:
            reg = discover(entry_points_iter=())
            self.assertIn("dummy", reg.by_name())
            self.assertEqual(reg.by_name()["dummy"].kind, "dummy")
            self.assertTrue(all(r.source == "directory" for r in reg.records))
        finally:
            _restore(old)

    def test_config_module_class(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d) / "myplug.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class Listed(EngineAdapter):
                    name, kind, API_VERSION = "listed", "dummy", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True, False)
                    def memory_usage(self): return MemoryUsage(8, None)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            sys.path.insert(0, d)
            old = _env(SWITCHYARD_PLUGIN_DIR="", SWITCHYARD_PLUGINS="myplug:Listed")
            try:
                reg = discover(entry_points_iter=())
                listed = next(r for r in reg.records if r.name == "listed")
                self.assertEqual(listed.status, "ok")
                self.assertEqual(listed.source, "config")
                self.assertIn("listed", reg.by_name())
            finally:
                _restore(old)
                sys.path.remove(d)
                sys.modules.pop("myplug", None)

    def test_entry_points(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("example_dummy_adapter_ep", DUMMY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ep = _EP("dummy_ep", obj=mod.DummyAdapter, value="examples.plugins.dummy:DummyAdapter")
        reg = discover(plugin_dir="", plugins="", entry_points_iter=[ep])
        self.assertEqual(len(reg.records), 1)
        self.assertEqual(reg.records[0].status, "ok")
        self.assertEqual(reg.records[0].source, "entry_point")
        self.assertEqual(reg.records[0].name, "dummy")
        self.assertIsInstance(reg.by_name()["dummy"], EngineAdapter)

    def test_real_entry_point_group_is_iterable(self):
        from switchyard.plugins.loader import _iter_entry_points
        found = _iter_entry_points(ENTRY_POINT_GROUP)
        self.assertIsInstance(found, list)

    def test_version_rejection_does_not_raise(self):
        class TooNew(EngineAdapter):
            name, kind, API_VERSION = "future", "dummy", PLUGIN_API_MAX + 5
            def load(self): pass
            def unload(self): pass
            def health(self): return HealthResult(True)
            def memory_usage(self): return MemoryUsage(0)
            def wake(self, level=2): pass
            def sleep(self, level=2): pass

        reg = discover(plugin_dir="", plugins="", entry_points_iter=[_EP("future", TooNew)])
        rec = reg.records[0]
        self.assertEqual(rec.status, "incompatible")
        self.assertIn("future", rec.error)
        self.assertIn(str(PLUGIN_API_MAX + 5), rec.error)
        self.assertEqual(rec.error, unsupported_version_message("future", PLUGIN_API_MAX + 5))
        self.assertEqual(reg.by_name(), {})

    def test_broken_plugin_does_not_break_others(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write(root / "good.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class Good(EngineAdapter):
                    name, kind, API_VERSION = "good", "dummy", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True, True)
                    def memory_usage(self): return MemoryUsage(1, 1)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            _write(root / "boom.py", "raise RuntimeError('boom from plugin')\n")
            _write(root / "stale.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class Stale(EngineAdapter):
                    name, kind, API_VERSION = "stale", "dummy", 0
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True)
                    def memory_usage(self): return MemoryUsage(0)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            broken_ep = _EP("ep_broken", error=ImportError("nope"))
            reg = discover(plugin_dir=str(root), plugins="no.such.mod:Nope",
                           entry_points_iter=[broken_ep])
            statuses = {(r.name, r.status) for r in reg.records}
            self.assertIn(("good", "ok"), statuses)
            self.assertTrue(any(r.status == "error" and "boom" in (r.error or "") for r in reg.records))
            self.assertTrue(any(r.name == "stale" and r.status == "incompatible" for r in reg.records))
            self.assertTrue(any(r.source == "entry_point" and r.status == "error" for r in reg.records))
            self.assertTrue(any(r.source == "config" and r.status == "error" for r in reg.records))
            self.assertEqual(set(reg.by_name()), {"good"})

    def test_later_source_overrides_name(self):
        with tempfile.TemporaryDirectory() as d:
            plug, listed = Path(d) / "plug", Path(d) / "listed"
            _write(plug / "a.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class A(EngineAdapter):
                    name, kind, API_VERSION = "dup", "dir", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True)
                    def memory_usage(self): return MemoryUsage(1)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            _write(listed / "override.py", """
                from switchyard.plugins.api import EngineAdapter, HealthResult, MemoryUsage
                class Override(EngineAdapter):
                    name, kind, API_VERSION = "dup", "config", 1
                    def load(self): pass
                    def unload(self): pass
                    def health(self): return HealthResult(True)
                    def memory_usage(self): return MemoryUsage(2)
                    def wake(self, level=2): pass
                    def sleep(self, level=2): pass
            """)
            sys.path.insert(0, str(listed))
            try:
                reg = discover(plugin_dir=str(plug), plugins="override:Override",
                               entry_points_iter=())
                self.assertEqual(reg.by_name()["dup"].kind, "config")
                sources = [r.source for r in reg.records if r.name == "dup" and r.status == "ok"]
                self.assertEqual(sources, ["directory", "config"])
            finally:
                sys.path.remove(str(listed))
                sys.modules.pop("override", None)

    def test_missing_plugin_dir_is_isolated(self):
        reg = discover(plugin_dir="/no/such/switchyard-plugins-dir", plugins="",
                       entry_points_iter=())
        self.assertTrue(reg.records)
        self.assertEqual(reg.records[0].status, "error")
        self.assertEqual(reg.by_name(), {})

    def test_entry_point_scan_failure_does_not_raise(self):
        class Boom:
            def __iter__(self):
                raise RuntimeError("scan boom")

        reg = discover(plugin_dir="", plugins="", entry_points_iter=Boom())
        self.assertEqual(reg.records[0].status, "error")
        self.assertIn("scan boom", reg.records[0].error)
        self.assertEqual(reg.by_name(), {})


class PluginsApiTests(unittest.TestCase):
    def test_dashboard_lists_plugins(self):
        from switchyard.dashboard import Dashboard
        with tempfile.TemporaryDirectory() as d:
            s = config.Settings(os.path.join(d, "s.json"))
            port = _free_port()
            s.update({"dashboard_host": "127.0.0.1", "dashboard_port": port})
            old = _env(SWITCHYARD_PLUGIN_DIR=str(EXAMPLES), SWITCHYARD_PLUGINS="")
            try:
                dash = Dashboard(s, {"arbiter": None, "ovms": None, "stt": None})
                srv = Server(("127.0.0.1", port), dash.make_handler())
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                try:
                    import urllib.request
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/plugins", timeout=5) as r:
                        body = json.loads(r.read())
                finally:
                    srv.shutdown()
                    srv.server_close()
            finally:
                _restore(old)
        self.assertEqual(body["api_version"], PLUGIN_API_VERSION)
        names = [p["name"] for p in body["plugins"] if p["status"] == "ok"]
        self.assertIn("dummy", names)
        dummy = next(p for p in body["plugins"] if p["name"] == "dummy")
        self.assertEqual(dummy["kind"], "dummy")
        self.assertEqual(dummy["status"], "ok")
        self.assertIsNone(dummy["error"])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()

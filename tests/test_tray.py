"""Tray menu model and actions: pystray is mocked; GUI skipped if extras missing."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from switchyard import __version__
from switchyard.client import Client
from switchyard.tray import (
    TrayApp,
    View,
    config_path,
    load_config,
    menu_model,
    save_config,
    view_from,
)

try:
    from PIL import Image
    from switchyard.tray import make_icon
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

try:
    import psutil  # noqa: F401
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import pystray  # noqa: F401
    HAVE_PYSTRAY = True
except Exception:
    # Headless Linux: pystray may import-fail on missing Gtk/AppIndicator.
    HAVE_PYSTRAY = False


STATUS = {
    "version": "0.1.0",
    "arbiter": {
        "owner": "qwen",
        "engines": {
            "qwen": {"listen": 8000, "state": "awake"},
            "vision": {"listen": 8005, "state": "asleep"},
        },
        "gpu": {"used_mb": 4096, "total_mb": 24576, "free_mb": 20480, "source": "nvml"},
    },
}
SETTINGS_SHARED = {
    "fields": [{"key": "shared_vram", "type": "bool"}],
    "values": {"shared_vram": False},
}
RAM = {"used_mb": 8192, "total_mb": 16384, "percent": 50.0}


class MenuModelTests(unittest.TestCase):
    def test_online_menu_has_modes_engines_clear_open_quit(self):
        v = view_from(STATUS, SETTINGS_SHARED, RAM, "http://127.0.0.1:8100")
        texts = [i.text for i in menu_model(v)]
        self.assertTrue(any(t.startswith("Switchyard") for t in texts))
        self.assertTrue(any("VRAM" in t and "nvml" in t for t in texts))
        self.assertTrue(any(t.startswith("RAM") for t in texts))
        self.assertIn("Owner: qwen", texts)
        self.assertIn("Clear all…", texts)
        self.assertIn("Open dashboard", texts)
        self.assertIn("Quit", texts)
        mode = next(i for i in menu_model(v) if i.text == "Mode")
        names = [c.text for c in mode.children]
        self.assertEqual(names, ["Exclusive", "Shared VRAM"])
        self.assertTrue(next(c for c in mode.children if c.text == "Exclusive").checked)
        give = next(i for i in menu_model(v) if i.text == "Give GPU to")
        self.assertEqual([c.text for c in give.children], ["qwen", "vision", "ComfyUI"])
        self.assertTrue(next(c for c in give.children if c.text == "qwen").checked)

    def test_old_server_hides_shared_vram(self):
        v = view_from(STATUS, {"fields": [], "values": {}}, RAM, "http://x")
        mode = next(i for i in menu_model(v) if i.text == "Mode")
        shared = next(c for c in mode.children if c.text == "Shared VRAM")
        self.assertFalse(shared.visible)

    def test_offline_menu(self):
        v = View(online=False, url="http://127.0.0.1:8100", error="unreachable (down)")
        texts = [i.text for i in menu_model(v)]
        self.assertIn("Retry", texts)
        self.assertIn("Quit", texts)
        self.assertNotIn("Clear all…", texts)
        self.assertTrue(any("unreachable" in t for t in texts))
        self.assertTrue(v.tooltip.startswith("Switchyard · server unreachable"))

    def test_tooltip_includes_vram_and_ram(self):
        v = view_from(STATUS, None, RAM, "http://x")
        self.assertIn("GPU", v.tooltip)
        self.assertIn("RAM", v.tooltip)
        self.assertIn("qwen", v.tooltip)


class ConfigTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tray.json")
            with patch.dict(os.environ, {"SWITCHYARD_TRAY_CONFIG": path}):
                self.assertEqual(config_path(), __import__("pathlib").Path(path))
                save_config({"url": None, "check_updates": True, "check_interval_hours": 24})
                self.assertTrue(load_config()["check_updates"])
                self.assertFalse(os.path.exists(path + ".tmp") or os.path.exists(os.path.join(d, "tray.tmp")))


class TrayAppTests(unittest.TestCase):
    def _app(self, **kw):
        client = kw.pop("client", MagicMock(spec=Client))
        client.base = "http://127.0.0.1:8100"
        client.status.return_value = STATUS
        client.settings.return_value = SETTINGS_SHARED
        client.clear_all.return_value = []
        cfg = {"check_updates": False, "check_interval_hours": 24}
        ram = kw.pop("ram_fn", lambda: RAM)
        confirm = kw.pop("confirm_fn", lambda *a, **k: True)
        opener = kw.pop("open_fn", MagicMock())
        app = TrayApp(client, cfg, ram_fn=ram, confirm_fn=confirm, open_fn=opener, **kw)
        app._bg = lambda fn: fn()
        app.status = STATUS
        app.view = view_from(STATUS, SETTINGS_SHARED, RAM, client.base)
        return app, client

    def test_dispatch_give_and_mode_and_clear(self):
        app, client = self._app()
        app.dispatch("give", "vision")
        app.dispatch("mode", True)
        app.dispatch("clear")
        # background threads; join by calling the bodies directly as well
        app._do_clear()
        client.clear_all.assert_called()
        client.give_gpu.assert_called_with("vision", STATUS)
        client.set_shared.assert_called_with(True)

    def test_clear_aborts_when_confirm_false(self):
        app, client = self._app(confirm_fn=lambda *a, **k: False)
        app.dispatch("clear")
        client.clear_all.assert_not_called()

    def test_open_dashboard_and_quit(self):
        opened = []
        app, client = self._app(open_fn=lambda url: opened.append(url))
        icon = MagicMock()
        app.icon = icon
        app.dispatch("open")
        app.dispatch("quit")
        self.assertEqual(opened, ["http://127.0.0.1:8100/"])
        icon.stop.assert_called_once()

    def test_collect_offline(self):
        from switchyard.client import ApiError
        client = MagicMock(spec=Client)
        client.base = "http://127.0.0.1:8100"
        client.status.side_effect = ApiError("unreachable (down)")
        app = TrayApp(client, {"check_updates": False}, ram_fn=lambda: RAM)
        v = app.collect()
        self.assertFalse(v.online)
        self.assertIn("unreachable", v.error)

    def test_periodic_update_off_by_default(self):
        called = []
        app, _ = self._app(check_fn=lambda: called.append(1) or Simple("no"))
        app.cfg["check_updates"] = False
        app.collect()
        self.assertEqual(called, [])

    def test_toggle_updates_persists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tray.json")
            with patch.dict(os.environ, {"SWITCHYARD_TRAY_CONFIG": path}):
                app, _ = self._app()
                app.dispatch("toggle_updates")
                self.assertTrue(app.cfg["check_updates"])
                with open(path) as f:
                    self.assertTrue(json.loads(f.read())["check_updates"])


class Simple:
    def __init__(self, msg):
        self.available = False
        self.message = msg


class CliTests(unittest.TestCase):
    def test_help_lists_tray_and_update(self):
        from switchyard.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        out = buf.getvalue()
        self.assertIn("tray", out)
        self.assertIn("update", out)
        self.assertIn("SWITCHYARD_URL", out)
        self.assertIn(__version__, out)

    def test_version(self):
        from switchyard.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())

    def test_no_args_starts_server(self):
        from switchyard.__main__ import main
        with patch("switchyard.__main__.serve") as serve:
            main([])
            serve.assert_called_once()

    def test_update_help(self):
        from switchyard.__main__ import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["update", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("yes", buf.getvalue())

    def test_tray_module_help(self):
        from switchyard.tray import main as tray_main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            tray_main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("--url", buf.getvalue())

    def test_parser_tray_flags(self):
        from switchyard.__main__ import _parser
        ns = _parser().parse_args(["tray", "--url", "http://x:1", "--check-updates"])
        self.assertEqual(ns.command, "tray")
        self.assertEqual(ns.url, "http://x:1")
        self.assertTrue(ns.check_updates)

    def test_tray_command_calls_run_not_serve(self):
        from switchyard.__main__ import main
        with patch("switchyard.tray.run") as run, patch("switchyard.__main__.serve") as serve:
            main(["tray", "--url", "http://x"])
        run.assert_called_once()
        serve.assert_not_called()

    def test_update_command_calls_update_not_serve(self):
        from switchyard.__main__ import main
        with patch("switchyard.update.main", return_value=0) as upd, \
             patch("switchyard.__main__.serve") as serve, \
             self.assertRaises(SystemExit) as cm:
            main(["update", "--yes"])
        self.assertEqual(cm.exception.code, 0)
        upd.assert_called_once()
        serve.assert_not_called()


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class IconTests(unittest.TestCase):
    def test_make_icon_headless(self):
        img = make_icon(0.5, True)
        self.assertEqual(img.size, (64, 64))
        self.assertIsInstance(img, Image.Image)
        offline = make_icon(None, False)
        self.assertEqual(offline.size, (64, 64))


@unittest.skipUnless(HAVE_PIL and HAVE_PSUTIL and HAVE_PYSTRAY,
                     "pystray backend not importable (needs Gtk/AppIndicator)")
class ExtraImportTests(unittest.TestCase):
    def test_pystray_menu_builds_without_running_gui(self):
        import pystray
        app, _ = TrayAppTests()._app()
        menu = app.pystray_menu()
        self.assertIsInstance(menu, pystray.Menu)


if __name__ == "__main__":
    unittest.main()

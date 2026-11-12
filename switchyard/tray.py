"""System tray / menu bar client for a running Switchyard server.

Optional extra: pip install 'switchyard[tray]'  (pystray, Pillow, psutil)

  switchyard tray
  python -m switchyard.tray
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .client import (
    ApiError,
    Client,
    default_url,
    engines_map,
    fmt_mib,
    owner_of,
    shared_enabled,
    shared_supported,
    vram_used_total,
)

POLL_SEC = 4
UPDATE_INTERVAL_SEC = 24 * 3600


def config_path() -> Path:
    override = os.environ.get("SWITCHYARD_TRAY_CONFIG")
    if override:
        return Path(override)
    if os.name == "nt":
        root = os.environ.get("APPDATA") or str(Path.home())
        return Path(root) / "switchyard" / "tray.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "switchyard" / "tray.json"
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "switchyard" / "tray.json"


def load_config(path: Path | None = None) -> dict:
    path = path or config_path()
    defaults = {"url": None, "check_updates": False, "check_interval_hours": 24}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            defaults.update({k: data[k] for k in defaults if k in data})
    except (OSError, ValueError):
        pass
    return defaults


def save_config(cfg: dict, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp.replace(path)


def ram_usage() -> dict | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    return {
        "used_mb": int((vm.total - vm.available) // (1024 * 1024)),
        "total_mb": int(vm.total // (1024 * 1024)),
        "percent": float(vm.percent),
    }


def confirm(message: str, title: str = "Switchyard") -> bool:
    """Ask before Clear all when a dialog toolkit is available."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        ok = bool(messagebox.askyesno(title, message, parent=root))
        root.destroy()
        return ok
    except Exception:
        return True  # ponytail: no dialog; the menu click is the confirmation


@dataclass
class Item:
    text: str
    action: str | None = None
    arg: object = None
    checked: bool | None = None
    radio: bool = False
    enabled: bool = True
    visible: bool = True
    default: bool = False
    children: list | None = None


@dataclass
class View:
    online: bool = False
    url: str = ""
    version: str | None = None
    error: str | None = None
    owner: str | None = None
    engines: list[str] = field(default_factory=list)
    engine_states: dict = field(default_factory=dict)
    vram_used: int | None = None
    vram_total: int | None = None
    vram_source: str | None = None
    ram_used: int | None = None
    ram_total: int | None = None
    ram_percent: float | None = None
    shared_ok: bool = False
    shared_on: bool = False
    update_label: str | None = None
    check_updates: bool = False

    @property
    def vram_text(self) -> str:
        text = fmt_mib(self.vram_used, self.vram_total)
        if text == "—":
            return "VRAM unknown"
        if self.vram_source:
            return f"VRAM {text} ({self.vram_source})"
        return f"VRAM {text}"

    @property
    def ram_text(self) -> str:
        if self.ram_total is None:
            return "RAM —"
        pct = f"{self.ram_percent:.0f}%" if self.ram_percent is not None else fmt_mib(self.ram_used, self.ram_total)
        return f"RAM {fmt_mib(self.ram_used, self.ram_total)} ({pct})"

    @property
    def tooltip(self) -> str:
        if not self.online:
            return "Switchyard · server unreachable"
        owner = self.owner or "idle"
        return f"Switchyard · GPU {fmt_mib(self.vram_used, self.vram_total)} · {self.ram_text} · {owner}"

    @property
    def vram_frac(self) -> float | None:
        if self.vram_used is None or not self.vram_total:
            return None
        return max(0.0, min(1.0, self.vram_used / self.vram_total))


def view_from(status: dict | None, settings: dict | None, ram: dict | None,
              url: str, error: str | None = None, check_updates: bool = False,
              update_label: str | None = None) -> View:
    v = View(url=url, check_updates=check_updates, update_label=update_label, error=error)
    if ram:
        v.ram_used, v.ram_total = ram.get("used_mb"), ram.get("total_mb")
        v.ram_percent = ram.get("percent")
    if not isinstance(status, dict):
        return v
    v.online = True
    v.version = status.get("version")
    v.owner = owner_of(status)
    engines = engines_map(status)
    v.engines = list(engines)
    v.engine_states = {n: (e.get("state") if isinstance(e, dict) else None) for n, e in engines.items()}
    v.vram_used, v.vram_total, v.vram_source = vram_used_total(status)
    v.shared_ok = shared_supported(settings, status)
    v.shared_on = shared_enabled(settings, status) if v.shared_ok else False
    return v


def menu_model(view: View) -> list[Item]:
    title = f"Switchyard v{view.version or __version__}"
    items = [Item(title, enabled=False)]
    if not view.online:
        items += [
            Item(view.error or "Server unreachable", enabled=False),
            Item(view.url or "", enabled=False, visible=bool(view.url)),
            Item("—"),
            Item("Retry", "retry"),
        ]
    else:
        owner = "ComfyUI" if view.owner == "comfy" else (view.owner or "idle")
        items += [
            Item(view.vram_text, enabled=False),
            Item(view.ram_text, enabled=False),
            Item(f"Owner: {owner}", enabled=False),
            Item("—"),
            Item("Mode", children=[
                Item("Exclusive", "mode", False, checked=not view.shared_on, radio=True),
                Item("Shared VRAM", "mode", True, checked=view.shared_on, radio=True,
                     visible=view.shared_ok, enabled=view.shared_ok),
            ]),
            Item("Give GPU to", children=[
                *[Item(name, "give", name, checked=view.owner == name, radio=True)
                  for name in view.engines],
                Item("ComfyUI", "give", "comfy", checked=view.owner == "comfy", radio=True),
            ]),
            Item("Clear all…", "clear"),
        ]
    items += [
        Item("—"),
        Item("Open dashboard", "open", default=True),
        Item(view.update_label or "Check for updates", "update"),
        Item("Check periodically", "toggle_updates", checked=view.check_updates),
        Item("Quit", "quit"),
    ]
    return [it for it in items if it.visible]


def make_icon(frac: float | None = None, online: bool = True):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = (21, 128, 61, 255) if online else (185, 28, 28, 255)
    try:
        d.rounded_rectangle((4, 4, 60, 60), radius=14, fill=fill)
    except Exception:
        d.rectangle((4, 4, 60, 60), fill=fill)
    d.polygon([(20, 46), (32, 18), (44, 46)], outline=(255, 255, 255, 255))
    if frac is not None:
        d.rectangle((8, 54, 56, 58), fill=(255, 255, 255, 70))
        d.rectangle((8, 54, 8 + max(1, int(48 * frac)), 58), fill=(255, 255, 255, 255))
    return img


def _require_extra() -> None:
    missing = []
    for mod, pip in (("PIL", "Pillow"), ("psutil", "psutil")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    try:
        import pystray  # noqa: F401
    except ImportError:
        missing.append("pystray")
    except Exception as e:
        raise SystemExit(
            "switchyard tray could not load a system tray backend.\n"
            f"  {type(e).__name__}: {e}\n"
            "Linux: install AppIndicator or GTK (GNOME needs the AppIndicator extension).\n"
            "The Xorg fallback needs an X11 session and has no menu — only a click action."
        ) from e
    if missing:
        raise SystemExit(
            "switchyard tray needs the optional extra:\n"
            "  python3 -m pip install 'switchyard[tray]'\n"
            f"missing: {', '.join(missing)}"
        )


class TrayApp:
    def __init__(self, client: Client, cfg: dict, *, check_fn=None, ram_fn=ram_usage,
                 confirm_fn=confirm, open_fn=webbrowser.open, apply_fn=None):
        self.client = client
        self.cfg = cfg
        self.check_fn = check_fn
        self.ram_fn = ram_fn
        self.confirm_fn = confirm_fn
        self.open_fn = open_fn
        self.apply_fn = apply_fn
        self.icon = None
        self._lock = threading.Lock()
        self._last_update_check = 0.0
        self._update_label = None
        self._pending_info = None
        self.view = View(url=client.base, check_updates=bool(cfg.get("check_updates")))
        self.status: dict | None = None
        self.settings: dict | None = None

    def collect(self) -> View:
        error = None
        status = None
        settings = None
        try:
            status = self.client.status()
            settings = self.client.settings()
        except ApiError as e:
            error = str(e)
        ram = None
        try:
            ram = self.ram_fn()
        except Exception:
            ram = None
        self.status, self.settings = status, settings
        now = time.time()
        interval = float(self.cfg.get("check_interval_hours") or 24) * 3600
        if self.cfg.get("check_updates") and self.check_fn and (now - self._last_update_check) >= interval:
            self._poll_update()
        return view_from(
            status, settings, ram, self.client.base, error=error,
            check_updates=bool(self.cfg.get("check_updates")),
            update_label=self._update_label,
        )

    def _poll_update(self) -> None:
        self._last_update_check = time.time()
        try:
            info = self.check_fn()
        except Exception:
            return
        self._pending_info = info
        if getattr(info, "available", False):
            self._update_label = f"Update to {info.latest}…"
        else:
            self._update_label = None

    def refresh(self) -> None:
        view = self.collect()
        with self._lock:
            self.view = view
        icon = self.icon
        if icon is None:
            return
        try:
            icon.title = view.tooltip
            icon.icon = make_icon(view.vram_frac, view.online)
            try:
                icon.menu = self.pystray_menu()
            except Exception:
                pass
            icon.update_menu()
        except Exception:
            pass

    def dispatch(self, action: str, arg=None) -> None:
        if action == "quit":
            if self.icon:
                self.icon.stop()
            return
        if action == "open":
            self.open_fn(self.client.base + "/")
            return
        if action == "retry":
            self._bg(self.refresh)
            return
        if action == "toggle_updates":
            self.cfg["check_updates"] = not self.cfg.get("check_updates")
            try:
                save_config(self.cfg)
            except OSError:
                pass
            if self.cfg["check_updates"]:
                self._last_update_check = 0.0
            else:
                self._update_label = None
            self._bg(self.refresh)
            return
        if action == "update":
            self._bg(self._do_update)
            return
        if action == "clear":
            if not self.confirm_fn("Sleep every engine and unload ComfyUI from VRAM?"):
                return
            self._bg(self._do_clear)
            return
        if action == "mode":
            self._bg(lambda: self._call(lambda: self.client.set_shared(bool(arg))))
            return
        if action == "give":
            self._bg(lambda: self._call(lambda: self.client.give_gpu(str(arg), self.status)))
            return

    def _do_clear(self) -> None:
        self._call(lambda: self.client.clear_all(self.status))

    def _do_update(self) -> None:
        if not self.check_fn:
            return
        try:
            info = self.check_fn()
        except Exception as e:
            self._update_label = f"Update check failed: {e}"
            self.refresh()
            return
        self._pending_info = info
        if not info.available:
            self._update_label = info.message or "Up to date"
            self.refresh()
            return
        notes = info.notes or "(no notes)"
        prompt = f"Update Switchyard {info.current} → {info.latest}?\n\n{notes[:800]}"
        if info.install != "pip":
            prompt = info.lines() + "\n\nOpen the releases page?"
            if self.confirm_fn(prompt):
                self.open_fn(info.url)
            return
        if not self.confirm_fn(prompt):
            return
        if self.apply_fn:
            self.apply_fn(info, confirm=True)
        else:
            from .update import apply
            apply(info, confirm=True)

    def _call(self, fn) -> None:
        try:
            fn()
        except ApiError as e:
            with self._lock:
                self.view.error = str(e)
        self.refresh()

    def _bg(self, fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def pystray_menu(self):
        import pystray
        with self._lock:
            model = menu_model(self.view)

        def handler(act, arg):
            return lambda *_: self.dispatch(act, arg)

        def convert(it: Item):
            if it.text == "—":
                return pystray.Menu.SEPARATOR
            if it.children is not None:
                kids = [convert(c) for c in it.children if c.visible]
                submenu = pystray.Menu(*kids) if kids else pystray.Menu(pystray.MenuItem("(none)", None, enabled=False))
                return pystray.MenuItem(it.text, submenu)
            checked = (lambda _item, flag=it.checked: flag) if it.checked is not None else None
            action = handler(it.action, it.arg) if it.action else (lambda *_: None)
            return pystray.MenuItem(
                it.text, action, checked=checked, radio=it.radio,
                enabled=it.enabled, default=it.default,
            )

        return pystray.Menu(*[convert(it) for it in model])


def run(url: str | None = None, check_updates: bool | None = None, *, client: Client | None = None,
        icon_cls=None) -> None:
    _require_extra()
    import pystray
    cfg = load_config()
    if check_updates is True:
        cfg["check_updates"] = True
    base = url or cfg.get("url") or default_url()
    app = TrayApp(client or Client(base), cfg, check_fn=lambda: _check())
    app.view = app.collect()
    Icon = icon_cls or pystray.Icon
    icon = Icon("switchyard", make_icon(app.view.vram_frac, app.view.online),
                app.view.tooltip, app.pystray_menu())
    app.icon = icon

    def setup(icon):
        icon.visible = True

        def loop():
            while getattr(icon, "visible", True):
                time.sleep(POLL_SEC)
                app.refresh()
        threading.Thread(target=loop, daemon=True).start()

    icon.run(setup=setup)


def _check():
    from .update import check
    return check()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="switchyard tray",
        description="System tray / menu bar icon for a running Switchyard dashboard.",
    )
    parser.add_argument("--url", default=None,
                        help="dashboard URL (default: SWITCHYARD_URL or http://127.0.0.1:8100)")
    parser.add_argument("--check-updates", action="store_true",
                        help="enable periodic GitHub update checks (off by default)")
    args = parser.parse_args(argv)
    run(url=args.url, check_updates=True if args.check_updates else None)


if __name__ == "__main__":
    main()

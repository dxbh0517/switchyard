"""python -m switchyard"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from . import __version__
from .config import Settings
from .dashboard import Dashboard
from .events import log

_HELP_EPILOG = """\
Commands:
  (none)   Start the arbiter, shim, STT and dashboard
  ctl      Control a running Switchyard (see `switchyard ctl --help`)
  tray     System tray / menu bar icon (pip install -e ".[tray]")
  update   Check GitHub for a newer version

Environment:
  SWITCHYARD_SETTINGS  Path to the settings file (default: /data/settings.json)
  ARBITER_ENABLED      Enable the GPU arbiter
  OVMS_SHIM_ENABLED    Enable the OVMS shim
  STT_ENABLED          Enable speech to text
  DASHBOARD_HOST       Dashboard bind address (default: 127.0.0.1)
  LISTEN_HOST          Bind address for arbiter, OVMS shim, and STT
  SWITCHYARD_URL       Dashboard URL for ctl and tray (default: http://127.0.0.1:8100)
  SWITCHYARD_TOKEN     Bearer token the dashboard API requires (empty = no auth)

See the README for the full list.

Warning: the dashboard can change settings. Keep DASHBOARD_HOST on localhost,
or set SWITCHYARD_TOKEN before exposing it.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="switchyard",
        description=f"Switchyard {__version__}: share one GPU between vLLM engines and ComfyUI, "
                    "with an OVMS shim, CPU speech to text, and a dashboard.",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"switchyard {__version__}")
    sub = parser.add_subparsers(dest="command")
    # `ctl` is dispatched before parsing (it has its own parser); listed here for --help.
    sub.add_parser("ctl", help="control a running Switchyard over its dashboard API", add_help=False)
    tray = sub.add_parser("tray", help="system tray / menu bar (needs the [tray] extra)")
    tray.add_argument("--url", default=None,
                      help="dashboard URL (default: SWITCHYARD_URL or http://127.0.0.1:8100)")
    tray.add_argument("--check-updates", action="store_true",
                      help="enable periodic GitHub update checks (off by default)")
    upd = sub.add_parser("update", help="check GitHub for a newer Switchyard and optionally install")
    upd.add_argument("-y", "--yes", action="store_true",
                     help="install after printing release notes (pip installs only; never silent)")
    return parser


def serve() -> None:
    settings = Settings()
    log("switchyard", f"v{__version__}, settings file {settings.path}")
    components: dict = {"arbiter": None, "ovms": None, "stt": None}
    if settings.get("arbiter_enabled"):
        from .arbiter import Arbiter
        components["arbiter"] = Arbiter(settings)
    if settings.get("ovms_enabled"):
        from .ovms_shim import OvmsShim
        components["ovms"] = OvmsShim(settings)
    if settings.get("stt_enabled"):
        from .stt import Stt
        components["stt"] = Stt(settings)
    for comp in components.values():
        if comp:
            comp.start()
    Dashboard(settings, components).start()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "ctl":
        from .ctl import main as ctl_main
        raise SystemExit(ctl_main(argv[1:], prog="switchyard ctl"))
    args = _parser().parse_args(argv)
    if args.command == "tray":
        from .tray import run
        run(url=args.url, check_updates=True if args.check_updates else None)
        return
    if args.command == "update":
        from .update import main as update_main
        raise SystemExit(update_main(yes=args.yes))
    serve()


if __name__ == "__main__":
    main()

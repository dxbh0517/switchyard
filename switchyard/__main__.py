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
Environment:
  SWITCHYARD_SETTINGS  Path to the settings file (default: /data/settings.json)
  ARBITER_ENABLED      Enable the GPU arbiter
  OVMS_SHIM_ENABLED    Enable the OVMS shim
  STT_ENABLED          Enable speech to text
  DASHBOARD_HOST       Dashboard bind address (default: 127.0.0.1)
  LISTEN_HOST          Bind address for arbiter, OVMS shim, and STT

See the README for the full list.

Warning: the dashboard has no auth and can change settings.
Keep DASHBOARD_HOST on localhost.
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
    return parser


def main(argv: list[str] | None = None) -> None:
    _parser().parse_args(argv)
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


if __name__ == "__main__":
    main()

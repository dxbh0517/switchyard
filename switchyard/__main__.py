"""python -m switchyard"""
from __future__ import annotations

import signal
import sys
import time

from . import __version__
from .config import Settings
from .dashboard import Dashboard
from .events import log


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ctl":
        from .ctl import main as ctl_main
        raise SystemExit(ctl_main(sys.argv[2:], prog="switchyard ctl"))
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

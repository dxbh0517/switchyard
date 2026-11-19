"""Port picking for tests that bind a listener after choosing its port.

``bind(("127.0.0.1", 0))`` hands out a port from the kernel's ephemeral range,
the same pool outgoing client sockets draw from. Between picking the port and
the server binding it, a client connection (health probe, proxied request) can
take it and the bind fails with EADDRINUSE. Picking below the ephemeral range
(32768+ on Linux, 49152+ elsewhere) and never handing out a port twice closes
that race.
"""
from __future__ import annotations

import random
import socket
import threading

_LOW, _HIGH = 20000, 32000
_lock = threading.Lock()
_handed_out: set[int] = set()


def free_port() -> int:
    with _lock:
        for _ in range(1000):
            port = random.randint(_LOW, _HIGH)
            if port in _handed_out:
                continue
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _handed_out.add(port)
            return port
    raise RuntimeError("no free test port")

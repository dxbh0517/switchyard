"""System reserve: RAM/VRAM kept back for the desktop and OS."""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest import mock

from test_switchyard import fake_comfy, fake_engine, free_port, post, serve

from switchyard import analytics, config, reserve
from switchyard.arbiter import Arbiter


class HelperTests(unittest.TestCase):
    def test_meminfo_and_pressure_parse(self):
        with tempfile.TemporaryDirectory() as d:
            mi, psi = os.path.join(d, "meminfo"), os.path.join(d, "psi")
            with open(mi, "w") as f:
                f.write("MemTotal:       65536000 kB\nMemFree: 1 kB\nMemAvailable:    8192000 kB\n")
            with open(psi, "w") as f:
                f.write("some avg10=12.50 avg60=1.00 avg300=0.00 total=1\n"
                        "full avg10=7.25 avg60=0.00 avg300=0.00 total=1\n")
            self.assertEqual(reserve.meminfo(mi), {"total_mb": 64000, "available_mb": 8000})
            self.assertEqual(reserve.memory_pressure(psi), {"some_avg10": 12.5, "full_avg10": 7.25})
            self.assertIsNone(reserve.meminfo(os.path.join(d, "missing")))

    def test_level1_fits_and_ram_low(self):
        self.assertTrue(reserve.level1_fits(30000, 20000, 4096))
        self.assertFalse(reserve.level1_fits(20000, 18000, 4096))
        self.assertTrue(reserve.level1_fits(None, 18000, 4096))
        self.assertTrue(reserve.level1_fits(1000, 18000, 0))
        self.assertTrue(reserve.ram_low(1000, 4096))
        self.assertFalse(reserve.ram_low(None, 4096))
        self.assertFalse(reserve.ram_low(1000, 0))


class ArbiterReserveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.e_state, he = fake_engine()
        self.c_state, hc = fake_comfy()
        self.srv = [serve(he), serve(hc)]
        (_, pe), (_, pc) = self.srv
        self.listen = free_port()
        self.s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        self.s.update({"engines": f"a=127.0.0.1:{pe}@{self.listen}:1000:20000",
                       "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                       "gpu_probe": "comfy", "sleep_level": 1, "ram_reserve_mb": 4096})
        self.arb = Arbiter(self.s)
        self.arb.start()
        time.sleep(0.3)

    def tearDown(self):
        analytics.reset()
        for srv in [s for s, _ in self.srv] + self.arb.servers:
            srv.shutdown()
            srv.server_close()
        self.tmp.cleanup()

    def _sleep_after_chat(self, available_mb):
        post(f"http://127.0.0.1:{self.listen}/v1/chat/completions",
             {"messages": [{"role": "user", "content": "hi"}]})
        mem = {"total_mb": 64000, "available_mb": available_mb}
        with mock.patch.object(reserve, "meminfo", return_value=mem):
            self.arb.sleep("a", "test")
        return [c for c in self.e_state["calls"] if c.startswith("/sleep")][-1]

    def test_level1_kept_when_ram_fits(self):
        self.assertEqual(self._sleep_after_chat(40000), "/sleep?level=1")
        self.assertEqual(self.arb.state["sleep_level"]["a"], 1)

    def test_level1_falls_back_to_level2_when_reserve_would_break(self):
        self.assertEqual(self._sleep_after_chat(16000), "/sleep?level=2")
        self.assertEqual(self.arb.state["sleep_level"]["a"], 2)

    def test_vram_reserve_shrinks_budget_and_raises_wake_target(self):
        self.s.update({"vram_reserve_mb": 2048, "vram_budget_mb": 24000, "vram_safety_mb": 1000})
        self.assertEqual(self.arb.usable(), 24000 - 1000 - 2048)
        self.assertEqual(self.arb.wake_target_mb("a"), 1000 + 2048)

    def test_low_ram_unloads_idle_comfy_once_per_cooldown(self):
        freed = self.c_state["freed"]
        low = {"total_mb": 64000, "available_mb": 1000}
        with mock.patch.object(reserve, "meminfo", return_value=low):
            t = self.arb.check_ram(grace=0, cooldown=3600, last_action=0.0)
            self.assertEqual(self.c_state["freed"], freed + 1)
            self.assertEqual(self.arb.check_ram(grace=0, cooldown=3600, last_action=t), t)
            self.assertEqual(self.c_state["freed"], freed + 1)
            st = self.arb.status()["reserve"]
        self.assertTrue(st["ram_low"])
        self.assertEqual(st["ram_available_mb"], 1000)

    def test_low_ram_leaves_busy_comfy_alone(self):
        self.arb.state["comfy_active"] = 1
        freed = self.c_state["freed"]
        with mock.patch.object(reserve, "meminfo", return_value={"total_mb": 64000, "available_mb": 1000}):
            self.arb.check_ram(grace=0, cooldown=0, last_action=0.0)
        self.assertEqual(self.c_state["freed"], freed)


if __name__ == "__main__":
    unittest.main()

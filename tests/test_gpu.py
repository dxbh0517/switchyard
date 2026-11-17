"""GPU VRAM probe: mocked NVML / nvidia-smi, plus the fallback chain."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from switchyard import config
from switchyard.arbiter import Arbiter
from switchyard.gpu import GpuProbe, GpuReading, _read_nvml, _read_smi


def _settings(**kw) -> tuple[config.Settings, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    s = config.Settings(os.path.join(tmp.name, "s.json"))
    if kw:
        s.update(kw)
    return s, tmp


class FakeNvml:
    def __init__(self, *, total=24 * 2**30, used=4 * 2**30, free=20 * 2**30,
                 name="NVIDIA GeForce RTX 4090", index=0, fail=None):
        self.total, self.used, self.free = total, used, free
        self.name, self.index, self.fail = name, index, fail
        self.inits = 0

    def nvmlInit(self):
        self.inits += 1
        if self.fail == "init":
            raise RuntimeError("nvmlInit failed")

    def nvmlDeviceGetHandleByIndex(self, i):
        if self.fail == "index" or i != self.index:
            raise RuntimeError("bad index")
        return object()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=self.total, used=self.used, free=self.free)

    def nvmlDeviceGetName(self, handle):
        return self.name


class NvmlSmiTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("pynvml", None)

    def test_nvml_bytes_and_name(self):
        sys.modules["pynvml"] = FakeNvml(name=b"NVIDIA GeForce RTX 3090")
        got = _read_nvml(0)
        self.assertEqual(got.source, "nvml")
        self.assertEqual(got.name, "NVIDIA GeForce RTX 3090")
        self.assertEqual(got.total_mb, round(24 * 2**30 / 2**20))
        self.assertEqual(got.used_mb, round(4 * 2**30 / 2**20))
        self.assertEqual(got.free_mb, round(20 * 2**30 / 2**20))

    def test_nvml_missing_and_init_failure(self):
        sys.modules["pynvml"] = None
        self.assertIsNone(_read_nvml(0))
        sys.modules["pynvml"] = FakeNvml(fail="init")
        self.assertIsNone(_read_nvml(0))

    def test_smi_parses_csv_and_passes_index(self):
        with patch("switchyard.gpu.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0, stdout=" 24576, 4096.0, 20480, NVIDIA GeForce RTX 4090 \n")
            got = _read_smi(1)
        self.assertEqual(got.source, "nvidia-smi")
        self.assertEqual((got.total_mb, got.used_mb, got.free_mb), (24576, 4096, 20480))
        self.assertEqual(got.name, "NVIDIA GeForce RTX 4090")
        self.assertEqual(got.index, 1)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:4], ["nvidia-smi", "-i", "1", "--query-gpu=memory.total,memory.used,memory.free,name"])
        self.assertIn("csv,noheader,nounits", cmd[-1])

    def test_smi_errors(self):
        with patch("switchyard.gpu.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(_read_smi(0))
        with patch("switchyard.gpu.subprocess.run",
                   return_value=SimpleNamespace(returncode=1, stdout="")):
            self.assertIsNone(_read_smi(0))
        with patch("switchyard.gpu.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout="not-csv\n")):
            self.assertIsNone(_read_smi(0))


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.s, self.tmp = _settings(gpu_probe="auto", gpu_index=0)
        self.addCleanup(self.tmp.cleanup)

    def _probe(self, **kw) -> GpuProbe:
        for k, v in kw.items():
            self.s.update({k: v})
        return GpuProbe(self.s, ttl_sec=0)

    def test_auto_prefers_nvml(self):
        nv = GpuReading("nvml", 0, 24000, 4000, 20000, "nv")
        smi = GpuReading("nvidia-smi", 0, 1, 1, 1, "smi")
        with patch("switchyard.gpu._read_nvml", return_value=nv) as n, \
             patch("switchyard.gpu._read_smi", return_value=smi) as s, \
             patch("switchyard.gpu._read_comfy", return_value=smi) as c:
            got = self._probe().read()
        self.assertEqual(got.source, "nvml")
        n.assert_called_once_with(0)
        s.assert_not_called()
        c.assert_not_called()

    def test_auto_falls_back_to_smi_then_comfy(self):
        smi = GpuReading("nvidia-smi", 0, 24576, 1000, 23576, "smi")
        comfy = GpuReading("comfy", 0, 100, 50, 50, "c")
        with patch("switchyard.gpu._read_nvml", return_value=None), \
             patch("switchyard.gpu._read_smi", return_value=smi) as s, \
             patch("switchyard.gpu._read_comfy", return_value=comfy):
            self.assertEqual(self._probe().read().source, "nvidia-smi")
            s.assert_called_once()
        with patch("switchyard.gpu._read_nvml", return_value=None), \
             patch("switchyard.gpu._read_smi", return_value=None), \
             patch("switchyard.gpu._read_comfy", return_value=comfy) as c:
            got = self._probe().read()
        self.assertEqual(got.source, "comfy")
        c.assert_called_once()

    def test_auto_unknown_when_all_fail(self):
        with patch("switchyard.gpu._read_nvml", return_value=None), \
             patch("switchyard.gpu._read_smi", return_value=None), \
             patch("switchyard.gpu._read_comfy", return_value=None):
            got = self._probe().read()
        self.assertEqual(got.source, "unknown")
        self.assertIsNone(got.free_mb)

    def test_forced_source_does_not_fall_through(self):
        smi = GpuReading("nvidia-smi", 0, 8, 1, 7, "smi")
        with patch("switchyard.gpu._read_nvml", return_value=None) as n, \
             patch("switchyard.gpu._read_smi", return_value=smi) as s, \
             patch("switchyard.gpu._read_comfy", return_value=smi) as c:
            got = self._probe(gpu_probe="nvml").read()
        self.assertEqual(got.source, "unknown")
        n.assert_called_once()
        s.assert_not_called()
        c.assert_not_called()

    def test_off_skips_backends(self):
        with patch("switchyard.gpu._read_nvml") as n, \
             patch("switchyard.gpu._read_smi") as s, \
             patch("switchyard.gpu._read_comfy") as c:
            got = self._probe(gpu_probe="off").read()
        self.assertEqual(got.source, "off")
        self.assertIsNone(got.free_mb)
        n.assert_not_called()
        s.assert_not_called()
        c.assert_not_called()

    def test_gpu_index_forwarded(self):
        with patch("switchyard.gpu._read_nvml", return_value=None) as n, \
             patch("switchyard.gpu._read_smi", return_value=None) as s, \
             patch("switchyard.gpu._read_comfy", return_value=None) as c:
            self._probe(gpu_index=2).read()
        n.assert_called_once_with(2)
        s.assert_called_once_with(2)
        c.assert_called_once()
        self.assertEqual(c.call_args.args[1], 2)

    def test_cache_ttl(self):
        s, tmp = _settings(gpu_probe="nvidia-smi")
        self.addCleanup(tmp.cleanup)
        probe = GpuProbe(s, ttl_sec=60)
        reading = GpuReading("nvidia-smi", 0, 8, 1, 7, "x")
        with patch("switchyard.gpu._read_smi", return_value=reading) as smi, \
             patch("switchyard.gpu._read_nvml", return_value=None), \
             patch("switchyard.gpu._read_comfy", return_value=None):
            self.assertIs(probe.read(), probe.read())
            smi.assert_called_once()
        probe = GpuProbe(s, ttl_sec=0)
        with patch("switchyard.gpu._read_smi", return_value=reading) as smi, \
             patch("switchyard.gpu._read_nvml", return_value=None), \
             patch("switchyard.gpu._read_comfy", return_value=None):
            probe.read()
            probe.read()
            self.assertEqual(smi.call_count, 2)


class ComfyParseTests(unittest.TestCase):
    def test_comfy_system_stats(self):
        payload = {"devices": [{"type": "cuda", "name": "cuda:0 Fake GPU : x",
                                "vram_free": 30 * 2**30, "vram_total": 32 * 2**30}]}

        from io import BytesIO

        s, tmp = _settings(gpu_probe="comfy", comfy_url="http://127.0.0.1:8188")
        self.addCleanup(tmp.cleanup)
        with patch("switchyard.gpu.urllib.request.urlopen",
                   return_value=BytesIO(json.dumps(payload).encode())):
            got = GpuProbe(s, ttl_sec=0).read()
        self.assertEqual(got.source, "comfy")
        self.assertEqual(got.name, "Fake GPU")
        self.assertEqual(got.free_mb, round(30 * 2**30 / 2**20))
        self.assertEqual(got.used_mb, got.total_mb - got.free_mb)


class UnloadWarnTests(unittest.TestCase):
    def test_unload_warns_when_probe_off(self):
        from switchyard import events
        s, tmp = _settings(engines="a=127.0.0.1:9@8:1000", gpu_probe="off",
                           comfy_url="http://127.0.0.1:1")
        self.addCleanup(tmp.cleanup)
        arb = Arbiter(s)
        before = max((e["id"] for e in events.since(0)), default=0)
        arb.comfy_unload(1000)
        self.assertTrue(any("VRAM unknown" in e["msg"] for e in events.since(before)))
        self.assertEqual(arb.gpu["source"], "off")


if __name__ == "__main__":
    unittest.main()

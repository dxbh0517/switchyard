"""Training lease: a trainer reserves the whole GPU; chat/ComfyUI wait until it ends."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error

from test_switchyard import fake_comfy, fake_engine, free_port, get, post, serve, wait_until

from switchyard import analytics, config
from switchyard import train as train_mod
from switchyard.arbiter import Arbiter, LeaseLost, TrainingConflict


class TrainingLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a_state, ha = fake_engine()
        self.b_state, hb = fake_engine()
        self.c_state, hc = fake_comfy()
        self.srv = [serve(ha), serve(hb), serve(hc)]
        (_, pa), (_, pb), (_, pc) = self.srv
        self.la, self.lb = free_port(), free_port()
        s = config.Settings(os.path.join(self.tmp.name, "s.json"))
        s.update({"engines": f"a=127.0.0.1:{pa}@{self.la}:1000,b=127.0.0.1:{pb}@{self.lb}:1000",
                  "comfy_url": f"http://127.0.0.1:{pc}", "listen_host": "127.0.0.1",
                  "gpu_probe": "comfy"})
        self.arb = Arbiter(s)
        self.arb.start()
        self.base = f"http://127.0.0.1:{self.la}"
        time.sleep(0.3)

    def tearDown(self):
        analytics.reset()
        for srv in [s for s, _ in self.srv] + self.arb.servers:
            srv.shutdown()
            srv.server_close()
        self.tmp.cleanup()

    def chat(self, port):
        return post(f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "hi"}]})

    def test_begin_sleeps_engines_and_unloads_comfy(self):
        self.chat(self.la)
        self.assertFalse(self.a_state["sleeping"])
        freed = self.c_state["freed"]
        res = post(f"{self.base}/gpu/training/begin?reason=lora&ttl=60")
        self.assertTrue(res["lease"])
        self.assertEqual(res["slept"], ["a"])
        self.assertTrue(self.a_state["sleeping"])
        self.assertGreater(self.c_state["freed"], freed)
        st = get(f"{self.base}/gpu/status")
        self.assertEqual(st["owner"], "training")
        self.assertTrue(st["training"]["ready"])
        self.assertEqual(st["training"]["reason"], "lora")

    def test_chat_waits_for_the_lease_then_serves(self):
        lease = post(f"{self.base}/gpu/training/begin")["lease"]
        done = threading.Event()
        threading.Thread(target=lambda: (self.chat(self.lb), done.set()), daemon=True).start()
        time.sleep(0.5)
        self.assertFalse(done.is_set(), "chat must not wake an engine during training")
        self.assertTrue(self.b_state["sleeping"])
        post(f"{self.base}/gpu/training/end?lease={lease}")
        self.assertTrue(done.wait(10))
        self.assertIsNone(self.arb.state["training"])

    def test_comfy_waits_for_the_lease(self):
        lease = post(f"{self.base}/gpu/training/begin")["lease"]
        done = threading.Event()
        threading.Thread(target=lambda: (post(f"{self.base}/gpu/acquire"), done.set()),
                         daemon=True).start()
        time.sleep(0.5)
        self.assertFalse(done.is_set())
        self.assertEqual(self.arb.state["comfy_active"], 0, "a waiting prompt must not block begin")
        post(f"{self.base}/gpu/training/end?lease={lease}")
        self.assertTrue(done.wait(10))
        post(f"{self.base}/gpu/release")

    def test_begin_drains_inflight_and_blocks_new_requests(self):
        hold = threading.Event()
        self.a_state["hold"] = hold
        threading.Thread(target=lambda: self.chat(self.la), daemon=True).start()
        self.assertTrue(wait_until(lambda: self.a_state["serving"] == 1))
        got = {}
        t = threading.Thread(target=lambda: got.update(self.arb.training_begin("x", 60)), daemon=True)
        t.start()
        self.assertTrue(wait_until(lambda: self.arb.state["training"] is not None))
        late = threading.Event()
        threading.Thread(target=lambda: (self.chat(self.lb), late.set()), daemon=True).start()
        time.sleep(0.4)
        self.assertFalse(got, "begin must wait for the in-flight request")
        self.assertFalse(late.is_set(), "a new request must queue behind the pending trainer")
        self.a_state["hold"] = None
        hold.set()
        t.join(10)
        self.assertTrue(got.get("lease"))
        self.assertTrue(self.a_state["sleeping"])
        self.assertFalse(late.is_set())
        self.arb.training_end(got["lease"])
        self.assertTrue(late.wait(10))

    def test_second_trainer_conflicts_and_bad_lease_is_gone(self):
        lease = self.arb.training_begin("one", 60)["lease"]
        with self.assertRaises(TrainingConflict):
            self.arb.training_begin("two", 60)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            post(f"{self.base}/gpu/training/begin")
        self.assertEqual(cm.exception.code, 409)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            post(f"{self.base}/gpu/training/renew?lease=nope")
        self.assertEqual(cm.exception.code, 410)
        self.assertEqual(post(f"{self.base}/gpu/training/renew?lease={lease}")["lease"], lease)
        self.arb.training_end(lease)
        with self.assertRaises(LeaseLost):
            self.arb.training_end(lease)

    def test_unrenewed_lease_expires(self):
        lease = self.arb.training_begin("crashy", 60)["lease"]
        self.arb.state["training"]["expires"] = time.time() - 1
        self.arb._expire_training()
        self.assertIsNone(self.arb.state["training"])
        with self.assertRaises(LeaseLost):
            self.arb.training_renew(lease)
        self.chat(self.la)  # the GPU is usable again

    def test_dashboard_force_release(self):
        self.arb.training_begin("stuck", 60)
        self.arb.action("training_end", None)
        self.assertTrue(wait_until(lambda: self.arb.state["training"] is None))

    def test_boot_engine_parked_while_training_then_warmed(self):
        self.arb.s.update({"engines": f"a={self.arb.engines['a']['backend']}@{self.la}:1000:boot,"
                                       f"b={self.arb.engines['b']['backend']}@{self.lb}:1000"})
        lease = self.arb.training_begin("t", 60)["lease"]
        self.a_state["sleeping"] = False  # the engine restarted awake during training
        self.arb.sleeping["a"] = False
        self.arb.on_backend_ready("a")
        self.assertTrue(self.a_state["sleeping"], "boot must not wake an engine during training")
        self.arb.training_end(lease)
        self.assertTrue(wait_until(lambda: self.a_state["sleeping"] is False, 10),
                        "the boot engine comes back after training")

    def test_train_wrapper_runs_command_and_releases(self):
        code = train_mod.main(["--arbiter", self.base, "--reason", "unit", "--",
                               sys.executable, "-c",
                               "import json,urllib.request,sys;"
                               f"s=json.load(urllib.request.urlopen('{self.base}/gpu/status'));"
                               "sys.exit(7 if s['owner']=='training' else 1)"])
        self.assertEqual(code, 7, "the command runs while the lease is held")
        self.assertIsNone(self.arb.state["training"])

    def test_train_wrapper_refuses_without_switchyard(self):
        code = train_mod.main(["--arbiter", f"http://127.0.0.1:{free_port()}", "--",
                               sys.executable, "-c", "pass"])
        self.assertEqual(code, 75)


if __name__ == "__main__":
    unittest.main()

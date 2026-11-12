"""GitHub update check: mocked network, no pip install actually run."""
from __future__ import annotations

import contextlib
import io
import json
import urllib.error
import unittest
from email.message import Message
from types import SimpleNamespace
from urllib.parse import urlparse

from switchyard import __version__
from switchyard.update import (
    UpdateInfo,
    apply,
    check,
    detect_install,
    newer_than,
    parse_version,
    pip_spec,
)


def _headers(d=None):
    m = Message()
    for k, v in (d or {}).items():
        m[k] = str(v)
    return m


class FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._raw = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.status = status
        self.headers = _headers(headers)

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def gh(routes):
    """Map path -> (status, payload, headers?)."""
    def open_url(req, timeout=None):
        path = urlparse(req.full_url).path
        spec = routes.get(path)
        if spec is None:
            raise urllib.error.URLError("offline")
        if isinstance(spec, Exception):
            raise spec
        status, payload, *rest = spec
        headers = rest[0] if rest else {}
        raw = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "err", _headers(headers), io.BytesIO(raw))
        return FakeResp(payload, status, headers)
    return open_url


class VersionTests(unittest.TestCase):
    def test_parse_and_compare(self):
        self.assertEqual(parse_version("v1.2.3"), (1, 2, 3))
        self.assertTrue(newer_than("0.2.0", "0.1.0"))
        self.assertFalse(newer_than("0.1.0", "0.1.0"))
        self.assertFalse(newer_than("0.1.0", "0.2.0"))
        self.assertTrue(newer_than("v0.1.1", "0.1.0"))


class DetectTests(unittest.TestCase):
    def test_env_override(self):
        self.assertEqual(detect_install({"SWITCHYARD_INSTALL": "pip"}), "pip")
        self.assertEqual(detect_install({"SWITCHYARD_INSTALL": "docker"}), "docker")

    def test_dockerenv_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            self.assertEqual(detect_install({}, dockerenv=f.name), "docker")


class CheckTests(unittest.TestCase):
    def test_release_newer_and_pypi(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (
                200, {"tag_name": "v0.9.0", "body": "fixed the GPU", "html_url": "https://example/r"}),
            "/pypi/switchyard/json": (200, {"releases": {"0.9.0": [{}]}, "info": {"version": "0.9.0"}}),
        })
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertTrue(info.available)
        self.assertEqual(info.latest, "0.9.0")
        self.assertIn("fixed the GPU", info.notes)
        self.assertTrue(info.via_pypi)
        self.assertEqual(pip_spec(info), "switchyard==0.9.0")

    def test_no_releases_uses_tags(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (404, {"message": "Not Found"}),
            "/repos/dxbh0517/switchyard/tags": (200, [
                {"name": "v0.0.1"}, {"name": "v0.2.0"}, {"name": "not-a-version"},
            ]),
            "/pypi/switchyard/json": (404, {}),
        })
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertTrue(info.available)
        self.assertEqual(info.latest, "0.2.0")
        self.assertIn("No release notes", info.notes)
        self.assertFalse(info.via_pypi)
        self.assertIn("@v0.2.0", pip_spec(info))

    def test_no_releases_or_tags(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (404, {"message": "Not Found"}),
            "/repos/dxbh0517/switchyard/tags": (200, []),
        })
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertFalse(info.available)
        self.assertEqual(info.kind, "none")
        self.assertIn("No GitHub releases", info.message)

    def test_already_current(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (
                200, {"tag_name": f"v{__version__}", "body": ""}),
        })
        info = check(current=__version__, opener=o, install="pip")
        self.assertFalse(info.available)
        self.assertIn("up to date", info.message.lower())

    def test_offline(self):
        o = gh({})
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertEqual(info.kind, "error")
        self.assertIn("offline", info.message)

    def test_rate_limit(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (
                403, {"message": "API rate limit exceeded"},
                {"X-RateLimit-Remaining": "0"}),
        })
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertEqual(info.kind, "error")
        self.assertIn("rate limit", info.message.lower())

    def test_docker_does_not_get_a_pip_spec_install(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (
                200, {"tag_name": "v9.0.0", "body": "big"}),
            "/pypi/switchyard/json": (404, {}),
        })
        info = check(current="0.1.0", opener=o, install="docker")
        self.assertTrue(info.available)
        self.assertIn("Docker", info.message)


class ApplyTests(unittest.TestCase):
    def test_pip_install_only_after_confirm(self):
        info = UpdateInfo(current="0.1.0", latest="0.2.0", tag="v0.2.0",
                          available=True, install="pip", via_pypi=True)
        ran = []

        def fake_run(cmd, check=False):
            ran.append(cmd)
            return SimpleNamespace(returncode=0)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(apply(info, confirm=False, run=fake_run), 0)
            self.assertEqual(ran, [])
            self.assertEqual(apply(info, confirm=True, run=fake_run), 0)
        self.assertEqual(ran[0][1:4], ["-m", "pip", "install"])
        self.assertIn("--upgrade", ran[0])
        self.assertIn("switchyard==0.2.0", ran[0])

    def test_docker_never_runs_pip(self):
        info = UpdateInfo(current="0.1.0", latest="0.2.0", tag="v0.2.0",
                          available=True, install="docker")
        ran = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(apply(info, confirm=True, run=lambda *a, **k: ran.append(a) or SimpleNamespace(returncode=0)), 0)
        self.assertEqual(ran, [])

    def test_cli_yes_still_prints_and_confirms_explicitly(self):
        o = gh({
            "/repos/dxbh0517/switchyard/releases/latest": (
                200, {"tag_name": "v9.9.9", "body": "notes here"}),
            "/pypi/switchyard/json": (404, {}),
        })
        ran = []
        info = check(current="0.1.0", opener=o, install="pip")
        self.assertIn("notes here", info.notes)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = apply(info, confirm=True,
                       run=lambda cmd, check=False: ran.append(cmd) or SimpleNamespace(returncode=0))
        self.assertEqual(rc, 0)
        self.assertTrue(any("v9.9.9" in str(c) for c in ran))


if __name__ == "__main__":
    unittest.main()

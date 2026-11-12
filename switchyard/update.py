"""Check GitHub for a newer Switchyard and optionally install it. Stdlib only.

Never installs unless the caller confirms. Docker / source checkouts are notify-only.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__

GITHUB_REPO = "dxbh0517/switchyard"
RELEASES_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags?per_page=30"
PYPI_URL = "https://pypi.org/pypi/switchyard/json"
GIT_SPEC = f"git+https://github.com/{GITHUB_REPO}.git@{{tag}}"


class CheckError(Exception):
    """Lookup failed (offline, rate limit, bad payload)."""


@dataclass
class UpdateInfo:
    current: str
    latest: str | None = None
    tag: str | None = None
    notes: str = ""
    available: bool = False
    kind: str = "ok"          # ok | none | error
    message: str = ""
    install: str = "pip"      # pip | docker | source
    via_pypi: bool = False
    url: str = f"https://github.com/{GITHUB_REPO}/releases"

    def lines(self) -> str:
        bits = [f"Installed: {self.current} ({self.install})"]
        if self.latest:
            bits.append(f"Latest:    {self.latest}")
        if self.available and self.notes:
            bits.append("")
            bits.append("What's new:")
            bits.append(self.notes)
        if self.message:
            bits.append("")
            bits.append(self.message)
        return "\n".join(bits)


def parse_version(text: str) -> tuple[int, ...]:
    text = str(text or "").strip()
    if text.lower().startswith("v") and text[1:2].isdigit():
        text = text[1:]
    parts = []
    for chunk in text.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts) or (0,)


def newer_than(latest: str, current: str) -> bool:
    return parse_version(latest) > parse_version(current)


def detect_install(environ: dict | None = None, dockerenv: str = "/.dockerenv") -> str:
    env = environ if environ is not None else os.environ
    forced = (env.get("SWITCHYARD_INSTALL") or "").strip().lower()
    if forced in ("pip", "docker", "source"):
        return forced
    if os.path.exists(dockerenv):
        return "docker"
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="replace")
        if "docker" in cgroup or "containerd" in cgroup or "kubepods" in cgroup:
            return "docker"
    except OSError:
        pass
    try:
        importlib.metadata.distribution("switchyard")
        return "pip"
    except importlib.metadata.PackageNotFoundError:
        return "source"


def _headers() -> dict:
    return {
        "User-Agent": f"switchyard/{__version__}",
        "Accept": "application/vnd.github+json",
    }


def _http_json(url: str, timeout: float = 10, opener=None) -> tuple[int, object, dict]:
    req = urllib.request.Request(url, headers=_headers())
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(req, timeout=timeout) as resp:
            raw = resp.read()
            headers = {k.lower(): v for k, v in getattr(resp, "headers", {}).items()}
            status = getattr(resp, "status", 200)
            return status, json.loads(raw) if raw else None, headers
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp else b""
        headers = {k.lower(): v for k, v in (e.headers.items() if e.headers else [])}
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return e.code, payload, headers
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise CheckError(f"offline ({e})") from e
    except ValueError as e:
        raise CheckError(f"non-JSON response from {url}") from e


def _rate_limited(status: int, headers: dict, payload) -> str | None:
    if status not in (403, 429):
        return None
    remaining = headers.get("x-ratelimit-remaining")
    msg = ""
    if isinstance(payload, dict):
        msg = str(payload.get("message") or "")
    if remaining == "0" or "rate limit" in msg.lower() or status == 429:
        return "GitHub API rate limit reached; try again later."
    if status == 403:
        return msg or "GitHub refused the request (HTTP 403)."
    return None


def _clip(text: str, n: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) > n:
        return text[:n].rstrip() + "\n…"
    return text


def _from_release(payload: dict) -> tuple[str, str, str, str]:
    tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
    ver = tag[1:] if tag.lower().startswith("v") and tag[1:2].isdigit() else tag
    notes = _clip(str(payload.get("body") or ""))
    url = str(payload.get("html_url") or f"https://github.com/{GITHUB_REPO}/releases")
    return ver, tag or ver, notes, url


def _from_tags(payload) -> tuple[str, str] | None:
    if not isinstance(payload, list) or not payload:
        return None
    best = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        ver = name[1:] if name.lower().startswith("v") and name[1:2].isdigit() else name
        key = parse_version(ver)
        if best is None or key > best[0]:
            best = (key, ver, name)
    if best is None:
        return None
    return best[1], best[2]


def pypi_has(version: str, opener=None) -> bool:
    try:
        status, payload, _ = _http_json(PYPI_URL, opener=opener)
    except CheckError:
        return False
    if status != 200 or not isinstance(payload, dict):
        return False
    releases = payload.get("releases")
    if isinstance(releases, dict) and version in releases:
        return True
    info = payload.get("info")
    return isinstance(info, dict) and str(info.get("version")) == version


def check(current: str | None = None, opener=None, install: str | None = None) -> UpdateInfo:
    current = current or __version__
    info = UpdateInfo(current=current, install=install or detect_install())
    try:
        status, payload, headers = _http_json(RELEASES_LATEST, opener=opener)
    except CheckError as e:
        info.kind = "error"
        info.message = str(e)
        return info
    limited = _rate_limited(status, headers, payload)
    if limited:
        info.kind = "error"
        info.message = limited
        return info
    if status == 200 and isinstance(payload, dict) and payload.get("tag_name"):
        ver, tag, notes, url = _from_release(payload)
        info.latest, info.tag, info.notes, info.url = ver, tag, notes, url
    elif status in (404, 200):
        try:
            tstatus, tpayload, theaders = _http_json(TAGS_URL, opener=opener)
        except CheckError as e:
            info.kind = "error"
            info.message = str(e)
            return info
        limited = _rate_limited(tstatus, theaders, tpayload)
        if limited:
            info.kind = "error"
            info.message = limited
            return info
        picked = _from_tags(tpayload) if tstatus == 200 else None
        if not picked:
            info.kind = "none"
            info.message = "No GitHub releases or tags yet."
            return info
        ver, tag = picked
        info.latest, info.tag = ver, tag
        info.notes = f"No release notes (tag {tag})."
        info.url = f"https://github.com/{GITHUB_REPO}/releases/tag/{tag}"
    else:
        info.kind = "error"
        info.message = f"GitHub returned HTTP {status}."
        return info
    info.available = newer_than(info.latest or "0", current)
    if not info.available:
        info.message = f"Already up to date ({current})."
    else:
        info.via_pypi = pypi_has(info.latest or "", opener=opener)
        if info.install == "pip":
            src = "PyPI" if info.via_pypi else f"git tag {info.tag}"
            info.message = f"A newer version is available. pip will install from {src}."
        elif info.install == "docker":
            info.message = (
                "This looks like a Docker install. Rebuild the image or pull a new tag; "
                "Switchyard will not auto-update Docker."
            )
        else:
            info.message = (
                "This looks like a source checkout. git pull / pip install -e . "
                "(or install a published tag); Switchyard will not overwrite the tree."
            )
    return info


def pip_spec(info: UpdateInfo) -> str:
    if info.via_pypi and info.latest:
        return f"switchyard=={info.latest}"
    tag = info.tag or info.latest
    if not tag:
        raise CheckError("no tag to install")
    return GIT_SPEC.format(tag=tag)


def apply(info: UpdateInfo, *, confirm: bool, run=subprocess.run) -> int:
    """Install only after confirm=True, and only for pip installs."""
    if not info.available:
        return 0
    if info.install != "pip":
        print("\nNot installing automatically.")
        return 0
    if not confirm:
        print("\nCancelled.")
        return 0
    spec = pip_spec(info)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    print("\nRunning:", " ".join(cmd))
    proc = run(cmd, check=False)
    return int(getattr(proc, "returncode", 1) or 0)


def _confirm_tty(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print("Re-run with --yes to install (pip installs only), or update manually.")
        return False
    try:
        answer = input("Install this update? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def main(argv: list[str] | None = None, *, yes: bool | None = None, opener=None) -> int:
    if yes is None:
        parser = argparse.ArgumentParser(
            prog="switchyard update",
            description="Check GitHub releases/tags of dxbh0517/switchyard and optionally upgrade.",
        )
        parser.add_argument(
            "-y", "--yes", action="store_true",
            help="install after printing release notes (pip installs only; never silent)",
        )
        args = parser.parse_args(argv)
        yes = args.yes
    try:
        info = check(opener=opener)
    except CheckError as e:
        print(e, file=sys.stderr)
        return 1
    print(info.lines())
    if info.kind == "error":
        return 1
    if not info.available:
        return 0
    return apply(info, confirm=_confirm_tty(yes))


if __name__ == "__main__":
    raise SystemExit(main())

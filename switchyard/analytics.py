"""Durable GPU/RAM telemetry for the arbiter (stdlib sqlite3, WAL).

Events are typed (`KINDS`) and queued to a background writer so `emit()` never
blocks the arbiter or touches the database under `Arbiter.cond`. Disk I/O,
retention pruning, and peak tracking happen on the writer thread.

Hook points for in-flight PRs (call these; do not invent extra kinds):
  record_health_check  — error-recovery health polls
  record_wake_retry    — already wired from the wake loop; reuse there
  record_wake_failure  — already wired from the wake loop
  record_oom           — VRAM / process OOM
  record_evict         — VRAM-budgeting victim sleep

Environment:
  SWITCHYARD_ANALYTICS=off     disable recording and return empty history
  SWITCHYARD_ANALYTICS_DB      sqlite path (default: <settings-dir>/analytics.sqlite)
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .events import log

KINDS = (
    "load", "unload", "wake", "sleep", "swap", "evict",
    "health_check", "wake_retry", "wake_failure", "oom", "error",
)
ERROR_KINDS = ("error", "oom", "wake_failure")

_DEFAULT_MAX_AGE_DAYS = 14
_DEFAULT_MAX_ROWS = 100_000
_QUEUE_MAX = 8192
_LIMIT_MAX = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    engine TEXT,
    duration_ms INTEGER,
    vram_before_mb INTEGER,
    vram_after_mb INTEGER,
    ram_before_mb INTEGER,
    ram_after_mb INTEGER,
    details TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_engine ON events(engine);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value INTEGER
);
"""

_INSERT = """INSERT INTO events
    (ts, kind, engine, duration_ms, vram_before_mb, vram_after_mb,
     ram_before_mb, ram_after_mb, details)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_recorder = None  # Recorder | NullRecorder | None
_lock = threading.Lock()


def ram_used_mb() -> int | None:
    """Host RAM in use (MiB). Linux /proc/meminfo; None on other platforms."""
    try:
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        if total is None or avail is None:
            return None
        return max(0, (total - avail) // 1024)
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_since(since: str | float | int) -> str:
    """Normalize `since` to the same UTC-ms ISO form stored in `events.ts`."""
    if isinstance(since, (int, float)) or (
            isinstance(since, str) and since.strip().replace(".", "", 1).isdigit()):
        dt = datetime.fromtimestamp(float(since), timezone.utc)
    else:
        s = str(since).strip()
        if not s:
            raise ValueError("empty since")
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _disabled() -> bool:
    return os.environ.get("SWITCHYARD_ANALYTICS", "").strip().lower() in ("0", "off", "false", "no")


def db_path(settings=None) -> Path:
    override = os.environ.get("SWITCHYARD_ANALYTICS_DB")
    if override:
        return Path(override)
    base = Path(settings.path if settings is not None
                else os.environ.get("SWITCHYARD_SETTINGS", "/data/settings.json"))
    return base.parent / "analytics.sqlite"


def _empty_history() -> dict:
    return {"events": [], "summary": {"counts": {}, "peak_vram_mb": None, "peak_ram_mb": None, "errors": 0}}


def _row(kind, engine, duration_ms, vram_before_mb, vram_after_mb,
         ram_before_mb, ram_after_mb, details, ts) -> dict:
    return {
        "ts": ts or utc_now(),
        "kind": kind,
        "engine": engine,
        "duration_ms": duration_ms,
        "vram_before_mb": vram_before_mb,
        "vram_after_mb": vram_after_mb,
        "ram_before_mb": ram_before_mb,
        "ram_after_mb": ram_after_mb,
        "details": details or {},
    }


class Recorder:
    """Thread-safe event recorder. `emit()` only enqueues; a daemon writer owns sqlite."""

    enabled = True

    def __init__(self, path: str | os.PathLike, *, max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
                 max_rows: int = _DEFAULT_MAX_ROWS, start_writer: bool = True):
        self.path = Path(path)
        self.max_age_days = max_age_days
        self.max_rows = max_rows
        self._q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._peak_vram: int | None = None
        self._peak_ram: int | None = None
        self._writes = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        if start_writer:
            self.start_writer()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
            rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            self._peak_vram = rows.get("peak_vram_mb")
            self._peak_ram = rows.get("peak_ram_mb")
        finally:
            conn.close()

    def start_writer(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, name="analytics-writer", daemon=True)
        self._thread.start()

    def emit(self, kind: str, engine: str | None = None, duration_ms: int | None = None,
             vram_before_mb: int | None = None, vram_after_mb: int | None = None,
             ram_before_mb: int | None = None, ram_after_mb: int | None = None,
             details: dict | None = None, ts: str | None = None) -> None:
        """Queue one event. Never raises, never writes sqlite."""
        try:
            if ram_after_mb is None:
                ram_after_mb = ram_used_mb()
            self._q.put_nowait(_row(kind, engine, duration_ms, vram_before_mb, vram_after_mb,
                                    ram_before_mb, ram_after_mb, details, ts))
        except queue.Full:
            pass
        except Exception:
            pass

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until the writer has drained work queued before this call."""
        if not self._thread or not self._thread.is_alive():
            self._drain_inline()
            return
        done = threading.Event()
        self._q.put(("__ctl__", "flush", done))
        if not done.wait(timeout):
            raise TimeoutError("analytics writer did not flush")

    def close(self) -> None:
        self._stopped.set()
        try:
            self._q.put(("__ctl__", "stop", None), timeout=1)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def history(self, limit: int = 100, since: str | float | None = None,
                kind: str | None = None, engine: str | None = None) -> dict:
        limit = max(1, min(int(limit or 100), _LIMIT_MAX))
        where, args = [], []
        if since not in (None, ""):
            where.append("ts >= ?")
            args.append(_parse_since(since))
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if engine:
            where.append("engine = ?")
            args.append(engine)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        conn = self._connect()
        try:
            events = [self._event(r) for r in conn.execute(
                f"SELECT * FROM events {clause} ORDER BY id DESC LIMIT ?", [*args, limit])]
            counts = {k: n for k, n in conn.execute(
                f"SELECT kind, COUNT(*) FROM events {clause} GROUP BY kind", args)}
            peaks = dict(conn.execute(
                "SELECT key, value FROM meta WHERE key IN ('peak_vram_mb', 'peak_ram_mb')"))
            errors = sum(counts.get(k, 0) for k in ERROR_KINDS)
            return {"events": events, "summary": {
                "counts": counts,
                "peak_vram_mb": peaks.get("peak_vram_mb"),
                "peak_ram_mb": peaks.get("peak_ram_mb"),
                "errors": errors,
            }}
        finally:
            conn.close()

    def prune(self) -> None:
        conn = self._connect()
        try:
            self._prune(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _event(r: sqlite3.Row) -> dict:
        details = r["details"]
        try:
            details = json.loads(details) if details else {}
        except ValueError:
            details = {}
        return {
            "id": r["id"], "ts": r["ts"], "kind": r["kind"], "engine": r["engine"],
            "duration_ms": r["duration_ms"],
            "vram_before_mb": r["vram_before_mb"], "vram_after_mb": r["vram_after_mb"],
            "ram_before_mb": r["ram_before_mb"], "ram_after_mb": r["ram_after_mb"],
            "details": details,
        }

    def _run(self) -> None:
        conn = None
        while not self._stopped.is_set():
            try:
                if conn is None:
                    conn = self._connect()
                    self._load_peaks(conn)
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if isinstance(item, tuple) and item and item[0] == "__ctl__":
                    conn.commit()
                    _, cmd, ev = item
                    if cmd == "flush" and ev is not None:
                        ev.set()
                    if cmd == "stop":
                        break
                    continue
                self._insert(conn, item)
                conn.commit()
                self._writes += 1
                if self._writes % 32 == 0:
                    self._prune(conn)
                    conn.commit()
            except Exception as e:
                log("analytics", f"writer error: {e}", "error")
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                time.sleep(0.2)
        if conn is not None:
            try:
                conn.commit()
                conn.close()
            except Exception:
                pass

    def _drain_inline(self) -> None:
        """Used when the writer is not running (tests)."""
        conn = self._connect()
        try:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, tuple) and item and item[0] == "__ctl__":
                    if item[2] is not None:
                        item[2].set()
                    continue
                self._insert(conn, item)
            self._prune(conn)
            conn.commit()
        finally:
            conn.close()

    def _load_peaks(self, conn: sqlite3.Connection) -> None:
        rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        self._peak_vram = rows.get("peak_vram_mb")
        self._peak_ram = rows.get("peak_ram_mb")

    def _insert(self, conn: sqlite3.Connection, e: dict) -> None:
        conn.execute(_INSERT, (
            e["ts"], e["kind"], e["engine"], e["duration_ms"],
            e["vram_before_mb"], e["vram_after_mb"],
            e["ram_before_mb"], e["ram_after_mb"],
            json.dumps(e["details"]) if e["details"] else None,
        ))
        changed = False
        for v in (e["vram_before_mb"], e["vram_after_mb"]):
            if v is not None:
                if self._peak_vram is None or v > self._peak_vram:
                    self._peak_vram = int(v)
                    changed = True
        for v in (e["ram_before_mb"], e["ram_after_mb"]):
            if v is not None:
                if self._peak_ram is None or v > self._peak_ram:
                    self._peak_ram = int(v)
                    changed = True
        if changed:
            if self._peak_vram is not None:
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('peak_vram_mb', ?)",
                             (self._peak_vram,))
            if self._peak_ram is not None:
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('peak_ram_mb', ?)",
                             (self._peak_ram,))

    def _prune(self, conn: sqlite3.Connection) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.max_age_days)).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
        conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        extra = n - self.max_rows
        if extra > 0:
            conn.execute(
                "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id ASC LIMIT ?)",
                (extra,),
            )


class NullRecorder:
    """No-op recorder when SWITCHYARD_ANALYTICS=off."""

    enabled = False
    path = None

    def emit(self, *a, **k) -> None:
        pass

    def history(self, **k) -> dict:
        return _empty_history()

    def flush(self, timeout: float = 5.0) -> None:
        pass

    def close(self) -> None:
        pass

    def prune(self) -> None:
        pass

    def start_writer(self) -> None:
        pass


def configure(settings=None, path: str | os.PathLike | None = None, **kwargs) -> Recorder | NullRecorder:
    """Install the process-wide recorder. Safe to call more than once (tests)."""
    global _recorder
    with _lock:
        if _recorder is not None:
            _recorder.close()
            _recorder = None
        if _disabled():
            _recorder = NullRecorder()
        else:
            _recorder = Recorder(path or db_path(settings), **kwargs)
        return _recorder


def configure_if_needed(settings=None, **kwargs) -> Recorder | NullRecorder:
    if _recorder is None:
        return configure(settings, **kwargs)
    return _recorder


def reset() -> None:
    global _recorder
    with _lock:
        if _recorder is not None:
            _recorder.close()
            _recorder = None


def get() -> Recorder | NullRecorder:
    rec = _recorder
    return rec if rec is not None else NullRecorder()


def emit(kind: str, engine: str | None = None, **fields) -> None:
    try:
        get().emit(kind, engine=engine, **fields)
    except Exception:
        pass


def history(limit: int = 100, since=None, kind: str | None = None, engine: str | None = None) -> dict:
    try:
        return get().history(limit=limit, since=since, kind=kind, engine=engine)
    except Exception as e:
        log("analytics", f"history failed: {e}", "error")
        return _empty_history()


def record_health_check(engine: str | None = None, ok: bool = True,
                        duration_ms: int | None = None, details: dict | None = None, **fields) -> None:
    extra = {"ok": ok, **(details or {})}
    emit("health_check", engine=engine, duration_ms=duration_ms, details=extra, **fields)


def record_wake_retry(engine: str | None = None, attempt: int | None = None,
                      error: str | None = None, details: dict | None = None, **fields) -> None:
    extra = {**(details or {})}
    if attempt is not None:
        extra["attempt"] = attempt
    if error is not None:
        extra["error"] = error
    emit("wake_retry", engine=engine, details=extra, **fields)


def record_wake_failure(engine: str | None = None, error: str | None = None,
                        details: dict | None = None, **fields) -> None:
    extra = {**(details or {})}
    if error is not None:
        extra["error"] = error
    emit("wake_failure", engine=engine, details=extra, **fields)


def record_oom(engine: str | None = None, details: dict | None = None, **fields) -> None:
    emit("oom", engine=engine, details=details, **fields)


def record_evict(engine: str | None = None, reason: str | None = None,
                 details: dict | None = None, **fields) -> None:
    extra = {**(details or {})}
    if reason is not None:
        extra["reason"] = reason
    emit("evict", engine=engine, details=extra, **fields)

"""OpenAI-compatibility shim in front of OpenVINO Model Server (OVMS).

  * Chat/completions: OVMS replies without an "id" field, which some clients (e.g. the
    vLLM Semantic Router) reject. The shim adds one to JSON replies and to every SSE chunk.
  * Rerank (/v1/rerank, /rerank): Qwen3-Reranker only scores correctly when the query and
    each document are wrapped in its yes/no judging prompt; clients send plain text.
  * Everything else is passed through; /v1/* is mapped to OVMS /v3/*.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import uuid
from urllib.parse import urlparse

from .config import Settings
from .events import Counters, log
from .httputil import Handler, Server

C = "ovms"
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINK_END = "</think>"


def split_think(message: dict) -> None:
    """Some small models still write a short thought ending in </think> when thinking is
    off, and OVMS leaves it in `content`. Move it to `reasoning_content`."""
    content = message.get("content")
    if isinstance(content, str) and THINK_END in content:
        thought, _, answer = content.rpartition(THINK_END)
        message["content"] = answer.lstrip()
        if not message.get("reasoning_content"):
            message["reasoning_content"] = thought.replace("<think>", "").strip()


class StreamThinkFilter:
    """Streaming version of split_think: holds back the first HOLD characters of content;
    if </think> shows up in them, only the text after it is sent. Later chunks pass through."""

    HOLD = 300

    def __init__(self):
        self.buf = ""
        self.done = False

    def feed(self, obj: dict) -> dict | None:
        choices = obj.get("choices") or []
        if self.done or not choices:
            return obj
        choice = choices[0]
        delta = choice.setdefault("delta", {})
        text = delta.pop("content", None)
        if text:
            self.buf += text
        if THINK_END in self.buf:
            self.buf = self.buf.rpartition(THINK_END)[2].lstrip()
            self.done = True
        elif len(self.buf) > self.HOLD or choice.get("finish_reason"):
            self.done = True
        if self.done:
            if self.buf:
                delta["content"] = self.buf
            self.buf = ""
            return obj
        keep = any(v for v in delta.values()) or choice.get("finish_reason") or obj.get("usage")
        return obj if keep else None


class OvmsShim:
    def __init__(self, settings: Settings):
        self.s = settings
        self.stats = Counters()
        self.models: list = []
        self.up: bool | None = None

    def upstream(self, method: str, path: str, body: bytes | None, headers: dict, timeout: float = 900):
        u = urlparse(self.s.get("ovms_upstream"))
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        conn.request(method, path, body=body, headers=headers)
        return conn, conn.getresponse()

    def poller(self) -> None:
        """Model states for the dashboard (OVMS /v1/config lists every servable)."""
        while True:
            try:
                conn, resp = self.upstream("GET", "/v1/config", None, {}, timeout=5)
                data = json.loads(resp.read())
                conn.close()
                models = []
                for name, info in sorted(data.items()):
                    versions = info.get("model_version_status") or [{}]
                    models.append({"name": name, "state": versions[0].get("state", "?")})
                if not self.up:
                    log(C, f"OVMS reachable ({len(models)} models)")
                self.models, self.up = models, True
            except Exception as e:
                if self.up is not False:
                    log(C, f"OVMS unreachable: {e}", "warn")
                self.up = False
            time.sleep(10)

    def status(self) -> dict:
        return {"upstream": self.s.get("ovms_upstream"), "up": self.up, "models": self.models,
                "port": self.s.boot["ovms_port"], **self.stats.snapshot()}

    def make_handler(self):
        shim = self

        class OvmsHandler(Handler):
            def rerank(self, body: bytes) -> None:
                shim.stats.inc("reranks")
                try:
                    req = json.loads(body or b"{}")
                    docs = [d.get("text", "") if isinstance(d, dict) else str(d) for d in req.get("documents", [])]
                    instruction = req.get("instruction") or shim.s.get("rerank_instruction")
                    out = {
                        "model": shim.s.get("rerank_model"),
                        "query": f"{PREFIX}<Instruct>: {instruction}\n<Query>: {req['query']}\n",
                        "documents": [f"<Document>: {d}{SUFFIX}" for d in docs],
                    }
                    if req.get("top_n"):
                        out["top_n"] = req["top_n"]
                    conn, resp = shim.upstream("POST", "/v3/rerank", json.dumps(out).encode(),
                                               {"Content-Type": "application/json"})
                    try:
                        data = json.loads(resp.read())
                    finally:
                        conn.close()
                    if resp.status != 200:
                        shim.stats.inc("errors")
                        self.send_json(resp.status, data)
                        return
                    if req.get("return_documents"):
                        for item in data.get("results", []):
                            item["document"] = {"text": docs[item["index"]]}
                    self.send_json(200, data)
                except Exception as e:
                    shim.stats.inc("errors")
                    self.send_json(502, {"error": {"message": str(e), "type": "upstream_error"}})

            def proxy(self) -> None:
                body = self.body() or None
                bare = urlparse(self.path).path.rstrip("/")
                if self.command == "POST" and bare in ("/v1/rerank", "/rerank", "/v3/rerank"):
                    self.rerank(body or b"")
                    return
                path = "/v3/" + self.path[4:] if self.path.startswith("/v1/") else self.path
                bare = urlparse(path).path
                add_id = self.command == "POST" and bare in ("/v3/chat/completions", "/v3/completions")
                if self.command == "POST":
                    shim.stats.inc("embeddings" if bare.endswith("/embeddings") else "requests")
                try:
                    conn, resp = shim.upstream(self.command, path, body, self.forward_headers())
                except Exception as e:
                    shim.stats.inc("errors")
                    self.send_json(502, {"error": {"message": f"OVMS unavailable: {e}", "type": "upstream_error"}})
                    return
                try:
                    if resp.status >= 400:
                        shim.stats.inc("errors")
                    if not add_id or resp.status != 200:
                        data = resp.read()
                        self.send_response(resp.status)
                        for k, v in resp.getheaders():
                            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                                self.send_header(k, v)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    rid = ("chatcmpl-" if "chat" in path else "cmpl-") + uuid.uuid4().hex
                    if "text/event-stream" not in resp.getheader("Content-Type", ""):
                        obj = json.loads(resp.read())
                        obj.setdefault("id", rid)
                        for choice in obj.get("choices") or []:
                            split_think(choice.get("message") or {})
                        self.send_json(200, obj)
                        return

                    think = StreamThinkFilter()

                    def add(line: bytes) -> bytes:
                        if line.startswith(b"data: {"):
                            try:
                                obj = think.feed(json.loads(line[6:]))
                                if obj is None:
                                    return b""
                                obj.setdefault("id", rid)
                                return b"data: " + json.dumps(obj).encode() + b"\n"
                            except ValueError:
                                pass
                        return line

                    self.relay(resp, transform_line=add)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    conn.close()

            do_GET = do_POST = do_DELETE = proxy  # noqa: N815

        return OvmsHandler

    def start(self) -> None:
        host, port = self.s.get("listen_host"), self.s.get("ovms_port")
        log(C, f"{host}:{port} -> {self.s.get('ovms_upstream')} (rerank model {self.s.get('rerank_model')})")
        threading.Thread(target=self.poller, daemon=True).start()
        self.server = Server((host, port), self.make_handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

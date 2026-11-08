"""OpenAI-compatible speech to text on the CPU (Granite Speech CTC models via transformers).

POST /v1/audio/transcriptions  (multipart: file, model, language, response_format)
GET  /v1/models, /health

torch/transformers are imported when the model loads, so the rest of Switchyard runs
without them when STT is disabled.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP

from .config import Settings
from .events import Counters, log
from .httputil import Handler, Server

C = "stt"
SR = 16000


def parse_multipart(ctype: str, body: bytes) -> dict:
    """multipart/form-data -> {name: bytes}. Uses the stdlib email parser."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            out[name] = part.get_payload(decode=True) or b""
    return out


class Stt:
    def __init__(self, settings: Settings):
        self.s = settings
        self.stats = Counters()
        self.lock = threading.Lock()   # one transcription at a time; each uses all threads
        self.model = self.processor = None
        self.loaded_path: str | None = None
        self.state = "not loaded"
        self.error: str | None = None
        settings.on_change(self._on_change)

    # ------------------------------------------------------------ model
    def load(self) -> None:
        path = self.s.get("stt_model_path")
        self.state, self.error = "loading", None
        log(C, f"loading {path}")
        t = time.time()
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor

            processor = AutoProcessor.from_pretrained(path)
            # bf16 checkpoints run faster in fp32 on CPUs without AVX-512 BF16.
            model = AutoModelForCTC.from_pretrained(path, dtype=torch.float32).eval()
            with self.lock:
                self.processor, self.model, self.loaded_path = processor, model, path
            self.state = "ready"
            log(C, f"model ready in {time.time() - t:.1f}s")
        except Exception as e:
            self.state, self.error = "error", str(e)
            log(C, f"model load failed: {e}", "error")

    def reload(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()

    def _on_change(self, changed: dict) -> None:
        if "stt_model_path" in changed:
            self.reload()

    # ------------------------------------------------------------ audio
    @staticmethod
    def decode(raw: bytes):
        import numpy as np

        p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
                            "-ar", str(SR), "-f", "f32le", "pipe:1"], input=raw, capture_output=True)
        if p.returncode != 0:
            raise ValueError(f"could not decode audio: {p.stderr.decode()[:300]}")
        return np.frombuffer(p.stdout, dtype=np.float32)

    def split(self, audio) -> list:
        """Cut long audio into <= chunk_sec pieces at the quietest point of each piece's last 3 s."""
        import numpy as np

        n = int(self.s.get("stt_chunk_sec") * SR)
        out, start = [], 0
        while len(audio) - start > n:
            win = audio[start + n - 3 * SR: start + n]
            frame = SR // 50  # 20 ms
            energy = np.square(win[: len(win) // frame * frame]).reshape(-1, frame).mean(axis=1)
            cut = start + n - 3 * SR + int(np.argmin(energy)) * frame + frame // 2
            out.append(audio[start:cut])
            start = cut
        out.append(audio[start:])
        return [c for c in out if len(c) > SR // 10]

    def transcribe(self, audio) -> str:
        import torch

        texts = []
        with self.lock, torch.inference_mode():
            torch.set_num_threads(self.s.get("stt_threads"))
            for chunk in self.split(audio):
                inputs = self.processor([chunk], sampling_rate=SR, return_tensors="pt")
                ids = self.model.generate(**inputs)
                texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return " ".join(t for t in texts if t)

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "model": self.loaded_path,
                "port": self.s.boot["stt_port"], **self.stats.snapshot()}

    # ------------------------------------------------------------ HTTP
    def make_handler(self):
        stt = self

        class SttHandler(Handler):
            def do_GET(self):  # noqa: N802
                if self.path.startswith("/health"):
                    code = 200 if stt.state == "ready" else 503
                    self.send_json(code, {"status": "ok" if code == 200 else stt.state})
                elif self.path.startswith("/v1/models"):
                    self.send_json(200, {"object": "list", "data": [
                        {"id": stt.s.get("stt_model_id"), "object": "model", "owned_by": "switchyard"}]})
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                body = self.body()
                if not self.path.startswith("/v1/audio/transcriptions"):
                    self.send_json(404, {"error": "not found"})
                    return
                if stt.state != "ready":
                    self.send_json(503, {"error": {"message": f"STT model {stt.state}"}})
                    return
                try:
                    form = parse_multipart(self.headers.get("Content-Type", ""), body)
                    if "file" not in form:
                        raise ValueError("missing 'file'")
                    fmt = form.get("response_format", b"json").decode() or "json"
                    language = form.get("language", b"").decode() or None
                    audio = stt.decode(form["file"])
                except ValueError as e:
                    stt.stats.inc("errors")
                    self.send_json(400, {"error": {"message": str(e)}})
                    return
                t = time.time()
                try:
                    text = stt.transcribe(audio)
                except Exception as e:
                    stt.stats.inc("errors")
                    log(C, f"transcription failed: {e}", "error")
                    self.send_json(500, {"error": {"message": str(e)}})
                    return
                dur, took = len(audio) / SR, time.time() - t
                stt.stats.inc("requests")
                stt.stats.inc("audio_sec", dur)
                stt.stats.set("last_rtf", round(dur / max(took, 1e-6), 1))
                log(C, f"{dur:.1f}s audio in {took:.2f}s ({dur / max(took, 1e-6):.1f}x realtime)")
                if fmt == "text":
                    self.send_bytes(200, text.encode(), "text/plain; charset=utf-8")
                elif fmt == "verbose_json":
                    self.send_json(200, {"task": "transcribe", "language": language or "auto",
                                         "duration": dur, "text": text, "segments": []})
                elif fmt in ("srt", "vtt"):
                    self.send_json(400, {"error": {"message": "srt/vtt need timestamps, which this model does not produce"}})
                else:
                    self.send_json(200, {"text": text})

        return SttHandler

    def start(self) -> None:
        port = self.s.get("stt_port")
        log(C, f"listening on :{port}")
        self.reload()
        srv = Server(("0.0.0.0", port), self.make_handler())
        threading.Thread(target=srv.serve_forever, daemon=True).start()

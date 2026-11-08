# Switchyard: GPU arbiter + OVMS shim + speech to text + dashboard, one process.
#   docker build -t switchyard .                       # with STT (CPU torch, ~1 GB)
#   docker build --build-arg WITH_STT=0 -t switchyard . # arbiter + shim only (stdlib)
ARG BASE=python:3.12-slim
FROM ${BASE}
ARG WITH_STT=1
RUN if [ "$WITH_STT" = "1" ] && ! command -v ffmpeg >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*; fi
RUN if [ "$WITH_STT" = "1" ]; then \
      pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
   && pip install --no-cache-dir "transformers>=5.16.0" numpy; fi
COPY switchyard /app/switchyard
WORKDIR /app
ENV PYTHONUNBUFFERED=1 SWITCHYARD_SETTINGS=/data/settings.json
VOLUME /data
CMD ["python3", "-m", "switchyard"]

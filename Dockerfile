# syntax=docker/dockerfile:1
# Build:  docker build -t ai-service .            (Docling + RapidOCR)
# Not built in the environment where this was written (no Docker daemon): see docs/table_reliability/REPORT_FASE_7.md.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt requirements-docling.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements-docling.txt

FROM python:3.11-slim
# libglib: OpenCV; libgomp: OpenMP used by LightGBM and onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
WORKDIR /app
COPY app ./app
COPY config ./config
COPY models ./models
# Page PNGs (evidence page_image.path is relative to /app) and the Docling/Hugging Face model cache must be writable.
RUN mkdir -p /app/out/pages && chown -R appuser:appuser /app/out /home/appuser
USER appuser
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/healthz', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]

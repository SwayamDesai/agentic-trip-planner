# Single-stage: the dependency tree is mostly pure Python, so a builder stage
# would save little and cost clarity.
FROM python:3.13-slim

# Faster, quieter, and no .pyc clutter in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements first, so a code change does not invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root. The volume is chowned so the app can write its cache and
# checkpoints without running as root to do it.
RUN useradd --create-home --uid 10001 atlas \
    && mkdir -p /data \
    && chown -R atlas:atlas /app /data
USER atlas

# All mutable state on the volume. Without this the cache and checkpoints live
# on the container filesystem and vanish on every deploy — which on this app
# means re-spending LLM quota to rebuild what it already knew.
# GATEWAY_LIMITER: the SQLite-backed bucket, not the `limits` default.
# `limits` has no file-backed store, so without Redis it runs in memory and a
# restart resets every counter — a volume cannot persist what was never on
# disk. Switch to limits + GATEWAY_STORAGE_URI=redis://... once a Redis exists.
#
# (A `#` comment cannot appear inside a line continuation: Docker swallows the
# rest of the ENV silently, which is how this was set to nothing the first time.)
ENV TRIP_CACHE_DIR=/data/cache \
    TRIP_DB=/data/trips.sqlite \
    GATEWAY_DB=/data/gateway.sqlite \
    GATEWAY=1 \
    PORT=8000 \
    GATEWAY_LIMITER=bucket

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",8000)}/health', timeout=4).status==200 else 1)"

# ONE worker, deliberately. Per-run metrics and the Langfuse trace id are
# module-level, and the concurrency limiter is in-process — several workers
# would interleave them and each keep its own limiter. Concurrency is bounded
# by the gateway instead, which is what the quota wants anyway.
CMD ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]

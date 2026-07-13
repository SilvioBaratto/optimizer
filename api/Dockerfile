# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — copies cleanly to the runtime stage without PYTHONPATH hacks
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install dependencies before copying app code (layer cache)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and regenerate the BAML client from baml_src/
COPY api/ api/
WORKDIR /build/api
RUN baml-cli generate

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

# Virtualenv (includes all packages + scripts: alembic, baml-cli)
COPY --from=builder /venv /venv

# Application code with the generated baml_client already in place
COPY --from=builder /build/api/ ./api/

COPY api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app /entrypoint.sh

ENV PATH=/venv/bin:$PATH

USER appuser
WORKDIR /app/api

# Prometheus metrics — also the healthcheck target
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/metrics').read()"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "app.worker"]

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# uv binary (pinned to match the workspace lockfile tooling)
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_DOWNLOADS=0

# Workspace metadata + only the members the daemon needs. optimizer/ (portopt-core,
# the skfolio stack) is deliberately NOT copied — `--package portopt` installs the
# ingestion daemon + its workspace dep portopt-db, and nothing from the optimizer lib.
COPY pyproject.toml uv.lock ./
COPY packages/portopt-db/ packages/portopt-db/
COPY ingestion/ ingestion/

# Real (non-editable) install into an isolated venv so the runtime stage needs no
# build-tree source paths. --frozen: respect the committed lock. --no-dev: no test deps.
RUN uv venv /venv
ENV VIRTUAL_ENV=/venv PATH=/venv/bin:$PATH
RUN uv sync --frozen --no-dev --no-editable --package portopt --active

# Regenerate the BAML client from baml_src/ (baml-cli ships with baml-py)
RUN cd ingestion && baml-cli generate

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

# Virtualenv (includes all runtime packages + scripts: alembic, baml-cli)
COPY --from=builder /venv /venv

# Application source (generated baml_client + entrypoint live outside the
# installed `app` package, so the source tree is still needed).
COPY --from=builder /build/ingestion/ ./ingestion/

# The Alembic tree lives in portopt-db (single migration owner) and sits outside
# the installed package, so ship it explicitly; entrypoint migrates from here.
COPY --from=builder /build/packages/portopt-db/alembic /app/portopt-db/alembic
COPY --from=builder /build/packages/portopt-db/alembic.ini /app/portopt-db/alembic.ini

COPY ingestion/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app /entrypoint.sh

ENV PATH=/venv/bin:$PATH

USER appuser
WORKDIR /app/ingestion

# Prometheus metrics — also the healthcheck target
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/metrics').read()"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "app.worker"]

# Fund bridge image — the deepagents/langgraph decision engine + drift-monitor
# daemon. Unlike ingestion/Dockerfile (which installs ONLY `--package portopt`
# and deliberately omits the optimizer/skfolio stack), fund is the one member
# that may import both `optimizer` and `portopt_db`, so its closure pulls
# portopt-core (skfolio) + portopt-db + deepagents/langgraph.
#
# IMPORTANT — editable install, not `--no-editable`:
#   fund.agents.backend resolves its runtime knowledge relative to the SOURCE file
#   (``Path(__file__).parents[3]`` → the `fund/` dir; ``.parent`` → repo root):
#     * theory  = <repo>/optimizerwiki
#     * skills  = fund/src/fund/skills
#   A `--no-editable` wheel would relocate `fund` into site-packages and those
#   paths would resolve into the venv (no optimizerwiki/, no skills/). So we keep
#   the members editable and copy the source tree to the SAME `/app` path in the
#   runtime stage, which preserves the editable finder's absolute paths.
#
# The markdown the agent reads (optimizerwiki/**.md, skills SKILL.md/reference.md)
# is kept in the build context by fund/Dockerfile.dockerignore, which overrides the
# root .dockerignore (that one strips *.md).

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# uv binary (same pin as ingestion/Dockerfile — matches the workspace lockfile).
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_DOWNLOADS=0

# Workspace metadata + only the members in portopt-fund's closure. `ingestion/`
# is NOT copied (not a fund dependency; uv tolerates the absent workspace member,
# exactly as ingestion/Dockerfile omits fund/). README.md is required by the root
# pyproject's `readme = "README.md"` when portopt-core builds.
COPY pyproject.toml uv.lock README.md ./
COPY packages/portopt-core/ packages/portopt-core/
COPY packages/portopt-db/ packages/portopt-db/
COPY fund/ fund/
# Runtime knowledge the agent stages on every call (NOT a Python package).
COPY optimizerwiki/ optimizerwiki/

# Fail the build early (before the slow sync) if the agent's runtime resources
# didn't make it into the context — otherwise the miss only surfaces at first
# run as a FileNotFoundError from stage_theory.
RUN test -d /app/optimizerwiki/docs \
    && test -d /app/optimizerwiki/openwiki \
    && test -d /app/fund/src/fund/skills/optimization-objective-map \
    || (echo "FATAL: fund runtime resources missing from build context" && exit 1)

# Editable install into an isolated venv. --frozen: respect the committed lock.
# --no-dev: no test deps. NO --no-editable (see header): members stay editable and
# resolve to /app/... , which the runtime stage reproduces at the same path.
RUN uv venv /venv
ENV VIRTUAL_ENV=/venv PATH=/venv/bin:$PATH
RUN uv sync --frozen --no-dev --package portopt-fund --active

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

# The venv (all runtime packages + console scripts: fund / fund-tui / fund-worker,
# alembic) and the source tree at the SAME /app path the editable finder recorded.
COPY --from=builder /venv /venv
COPY --from=builder /app /app

COPY fund/entrypoint.sh /entrypoint.sh
# appuser must own /app because the agent regenerates fund/.runtime/agent-fs there
# on every run (stage_theory wipes + recopies the staged theory/skills workdir).
RUN chmod +x /entrypoint.sh \
    && chown -R appuser:appuser /app /entrypoint.sh

ENV VIRTUAL_ENV=/venv PATH=/venv/bin:$PATH

USER appuser
WORKDIR /app

# No HTTP API, no Prometheus (out of scope for the fund daemon — see worker.py).
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "fund.worker"]

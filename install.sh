#!/usr/bin/env bash
# portopt bootstrap (mac/linux): clone the repo (needs git; PORTOPT_HOME, default
# ~/optimizer), install uv if missing, install the portopt CLI, then launch the
# interactive setup wizard.
#
#   curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.sh | bash
#   curl -LsSf .../install.sh | bash -s -- --non-interactive --llm-provider openrouter …
set -euo pipefail

# The CLI drives `docker compose` against a checkout of this repo (compose file,
# Dockerfiles, scripts/), and installing it into a uv tool venv does not provide
# one. Leave a clone behind, reuse an existing one, and point the CLI at it.
PORTOPT_REPO="${PORTOPT_REPO:-https://github.com/SilvioBaratto/optimizer.git}"
PORTOPT_REF="${PORTOPT_REF:-development}"
PORTOPT_HOME="${PORTOPT_HOME:-$HOME/optimizer}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch the optimizer checkout; install it and re-run." >&2
  exit 1
fi

if [ -f "${PORTOPT_HOME}/docker-compose.yml" ]; then
  echo "Using the existing checkout at ${PORTOPT_HOME}"
else
  echo "Cloning ${PORTOPT_REPO} into ${PORTOPT_HOME}..."
  git clone "${PORTOPT_REPO}" "${PORTOPT_HOME}"
  # A separate checkout (not `clone --branch`) so PORTOPT_REF may be a branch, tag
  # or commit.
  git -C "${PORTOPT_HOME}" checkout --quiet "${PORTOPT_REF}"
fi
export OPTIMIZER_REPO="${PORTOPT_HOME}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv drops tool executables in ~/.local/bin, so put it on PATH unconditionally.
# That keeps the portopt shim callable in this session even when uv already existed.
export PATH="$HOME/.local/bin:$PATH"

# release.yml only builds the portopt-core library, never the CLI member, so
# install the CLI straight from git. The install tracks a branch instead of a release.
echo "Installing the portopt CLI from ${PORTOPT_REPO}@${PORTOPT_REF}..."
uv tool install --force "git+${PORTOPT_REPO}@${PORTOPT_REF}#subdirectory=ingestion"

echo "Launching the setup wizard..."
# `curl | bash` binds this shell's stdin to the piped script, not the terminal, so
# an interactive `portopt setup` would read EOF and abort. When a controlling
# terminal is readable, reconnect stdin to it; otherwise (headless / CI, no
# /dev/tty) run as-is so a flag-driven `--non-interactive` install still works.
if [ -r /dev/tty ]; then
  portopt setup "$@" < /dev/tty
else
  portopt setup "$@"
fi

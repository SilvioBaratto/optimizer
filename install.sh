#!/usr/bin/env bash
# portopt bootstrap (mac/linux): install uv if missing, install the portopt CLI,
# then launch the interactive setup wizard.
#
#   curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.sh | bash
#   curl -LsSf .../install.sh | bash -s -- --non-interactive --llm-provider openrouter …
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv drops tool executables in ~/.local/bin, so put it on PATH unconditionally.
# That keeps the portopt shim callable in this session even when uv already existed.
export PATH="$HOME/.local/bin:$PATH"

# release.yml only builds the portopt-core library, never the CLI member, so
# install the CLI straight from git. The install tracks a branch instead of a release.
PORTOPT_REPO="${PORTOPT_REPO:-https://github.com/SilvioBaratto/optimizer.git}"
PORTOPT_REF="${PORTOPT_REF:-development}"
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

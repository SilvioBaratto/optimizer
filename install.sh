#!/usr/bin/env bash
# portopt bootstrap (mac/linux): install uv if missing, install the portopt CLI,
# then launch the interactive setup wizard.
#
#   curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.sh | bash
#   curl -LsSf .../install.sh | bash -s -- --non-interactive --llm-provider openrouter …
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing the portopt CLI..."
uv tool install portopt

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

#!/usr/bin/env bash
# portopt bootstrap (mac/linux): install uv if missing, install the portopt CLI,
# then launch the interactive setup wizard.
#
#   curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/main/install.sh | bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing the portopt CLI..."
uv tool install portopt

echo "Launching the setup wizard..."
portopt setup

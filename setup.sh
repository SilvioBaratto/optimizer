#!/usr/bin/env bash
# optimizer setup (macOS / Linux / Git Bash): the from-clone front door.
#
#   git clone https://github.com/SilvioBaratto/optimizer && cd optimizer
#   ./setup.sh
#
# Razor-thin funnel — no logic lives here. It ensures uv, installs the `portopt`
# CLI from THIS checkout (the `ingestion` member provides the `portopt` dist; it
# pulls `portopt-db` via a workspace source that isn't on PyPI), exports
# OPTIMIZER_REPO so the out-of-repo tool venv can still locate scripts/optimizer,
# then hands off to the tested core `portopt setup` with the caller's args.
# Keep LF-only (.gitattributes).
set -euo pipefail

here="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPTIMIZER_REPO="$here"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# uv drops tool executables in ~/.local/bin, so put it on PATH unconditionally.
# A uv that was already installed (Homebrew, a distro package) does not imply
# ~/.local/bin is on PATH, and the `exec portopt` below needs the fresh shim.
export PATH="$HOME/.local/bin:$PATH"

cd "$here"
echo "Installing the portopt CLI from this checkout..."
uv tool install --from ./ingestion portopt

exec portopt setup "$@"

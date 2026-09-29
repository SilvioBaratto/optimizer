#!/usr/bin/env bash
# optimizer setup (macOS / Linux / Git Bash): the from-clone front door.
#
#   git clone https://github.com/SilvioBaratto/optimizer && cd optimizer
#   ./setup.sh
#
# Razor-thin funnel — no logic lives here. It ensures uv, installs the `portopt`
# CLI from THIS checkout (the workspace, not PyPI: `portopt` pulls `portopt-db`
# via a workspace source that isn't published), exports OPTIMIZER_REPO so the
# out-of-repo tool venv can still locate scripts/optimizer, then hands off to the
# tested core `portopt setup` with the caller's args. Keep LF-only (.gitattributes).
set -euo pipefail

here="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPTIMIZER_REPO="$here"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Astral)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$here"
echo "Installing the portopt CLI from this checkout..."
uv tool install --from . portopt

exec portopt setup "$@"

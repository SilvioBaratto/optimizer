#!/usr/bin/env bash
# portopt bootstrap (mac/linux): clone the repo (needs git), install uv if missing,
# install the portopt CLI, then launch the interactive setup wizard from inside the
# checkout.
#
#   curl -LsSf https://raw.githubusercontent.com/SilvioBaratto/optimizer/development/install.sh | bash
#   curl -LsSf .../install.sh | bash -s -- --non-interactive --llm-provider openrouter …
#
# The checkout lands in ~/optimizer (override with PORTOPT_HOME, or its alias
# PORTOPT_DIR); a re-run updates it. PORTOPT_REF may be a branch, a tag or a commit
# SHA.
set -euo pipefail

# The CLI drives `docker compose` against a checkout of this repo (compose file,
# Dockerfiles, scripts/), and installing it into a uv tool venv does not provide
# one. `portopt setup` also renders ./secrets against the working directory, so keep
# a checkout at a stable path, point the CLI at it and run the wizard from inside it;
# otherwise setup dies on "no configuration file provided" after it has already
# persisted the secrets.
PORTOPT_REPO="${PORTOPT_REPO:-https://github.com/SilvioBaratto/optimizer.git}"
PORTOPT_REF="${PORTOPT_REF:-development}"
# install.ps1 and the README name this PORTOPT_HOME; PORTOPT_DIR is the same setting.
PORTOPT_DIR="${PORTOPT_DIR:-${PORTOPT_HOME:-$HOME/optimizer}}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch the optimizer checkout; install it and re-run." >&2
  exit 1
fi

# Reduce a clone URL to host/owner/repo so https, ssh and scp-style spellings of the
# same repo compare equal.
repo_id() {
  printf '%s' "$1" | sed -E 's#^[A-Za-z0-9+.-]+://##; s#^[^@/]+@##; s#:#/#; s#/?\.git/?$##; s#/$##'
}

if [ -d "$PORTOPT_DIR/.git" ]; then
  # fetch + checkout + merge rewrite the working copy, so only do it to a checkout of
  # this repo. A developer's own clone, or an unrelated repo, would otherwise be
  # switched to PORTOPT_REF without a word.
  origin_url="$(git -C "$PORTOPT_DIR" remote get-url origin 2>/dev/null || true)"
  if [ -z "$origin_url" ] || [ "$(repo_id "$origin_url")" != "$(repo_id "$PORTOPT_REPO")" ]; then
    echo "error: ${PORTOPT_DIR} is a git checkout of '${origin_url:-no origin remote}', not ${PORTOPT_REPO}." >&2
    echo "Point PORTOPT_DIR (or PORTOPT_HOME) at another path, or at a clone of ${PORTOPT_REPO}, and re-run." >&2
    exit 1
  fi
  echo "Updating the checkout in ${PORTOPT_DIR}..."
  git -C "$PORTOPT_DIR" fetch --quiet origin "$PORTOPT_REF" \
    && git -C "$PORTOPT_DIR" checkout --quiet "$PORTOPT_REF" \
    && git -C "$PORTOPT_DIR" merge --quiet --ff-only FETCH_HEAD \
    || echo "warning: could not update ${PORTOPT_DIR}; using it as it is." >&2
elif [ -f "${PORTOPT_DIR}/docker-compose.yml" ]; then
  # A tree that carries the compose file but no git metadata (an unpacked archive)
  # cannot be updated, and `git clone` would refuse to write into it.
  echo "Using the existing checkout at ${PORTOPT_DIR}"
else
  echo "Cloning ${PORTOPT_REPO}@${PORTOPT_REF} into ${PORTOPT_DIR}..."
  # `--branch` accepts a branch or tag name only. A pinned commit SHA takes a plain
  # clone followed by a checkout, which also keeps this path in step with the update
  # path above.
  if ! git clone --quiet --branch "$PORTOPT_REF" "$PORTOPT_REPO" "$PORTOPT_DIR" 2>/dev/null; then
    git clone --quiet "$PORTOPT_REPO" "$PORTOPT_DIR"
    git -C "$PORTOPT_DIR" checkout --quiet "$PORTOPT_REF"
  fi
fi
# The tool venv lives outside the repo, so tell the CLI where the checkout is.
export OPTIMIZER_REPO="$PORTOPT_DIR"
cd "$PORTOPT_DIR"

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
# terminal can be opened, reconnect stdin to it; otherwise (headless / CI, no
# controlling terminal) run as-is so a flag-driven `--non-interactive` install
# still works. The probe really opens the device: /dev/tty is rw-rw-rw- even with
# no controlling terminal, so `-r` would pass and the redirect would then abort
# the script under `set -e`.
if { : < /dev/tty; } 2>/dev/null; then
  portopt setup "$@" < /dev/tty
else
  portopt setup "$@"
fi

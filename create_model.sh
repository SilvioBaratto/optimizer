#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELFILE="$SCRIPT_DIR/Modelfile"

usage() {
  echo "Usage: $0 --original_model <name> --new_model <name>"
  echo ""
  echo "Creates an Ollama model from the Modelfile, replacing the FROM base model."
  echo ""
  echo "  --original_model   Base model to use in FROM (e.g. qwen3.5:4b)"
  echo "  --new_model        Name for the new model (e.g. finance-qwen)"
  exit 1
}

ORIGINAL_MODEL=""
NEW_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --original_model) ORIGINAL_MODEL="$2"; shift 2 ;;
    --new_model)      NEW_MODEL="$2";      shift 2 ;;
    *)                usage ;;
  esac
done

if [[ -z "$ORIGINAL_MODEL" || -z "$NEW_MODEL" ]]; then
  usage
fi

if [[ ! -f "$MODELFILE" ]]; then
  echo "Error: Modelfile not found at $MODELFILE"
  exit 1
fi

TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

sed "s|^FROM .*|FROM $ORIGINAL_MODEL|" "$MODELFILE" > "$TMPFILE"

echo "Creating model '$NEW_MODEL' from base '$ORIGINAL_MODEL'..."
ollama create "$NEW_MODEL" -f "$TMPFILE"
echo "Done. Run with: ollama run $NEW_MODEL"

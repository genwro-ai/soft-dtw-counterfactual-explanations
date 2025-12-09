#!/usr/bin/env bash
# Create and sync a uv-managed virtual environment for this project.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/ for installation steps."
  exit 1
fi

uv venv
uv sync

echo ""
echo "Environment created. Activate with:"
echo "  source .venv/bin/activate"

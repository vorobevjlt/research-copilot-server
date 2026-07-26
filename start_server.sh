#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting the API server (Uvicorn)..."

if command -v uv >/dev/null 2>&1; then
  exec uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
elif command -v poetry >/dev/null 2>&1; then
  exec poetry run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
elif command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
else
  echo "Error: install uv (recommended) or Poetry, then try again." >&2
  exit 1
fi

#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure Homebrew binaries (like tesseract) are on PATH for "macOS"
export PATH="/opt/homebrew/bin:$PATH"

echo "Starting Celery Worker (foreground)..."

run_celery () {
  "$1" -A src.services.celery:celery_app worker --loglevel=info --pool=threads
}

# Use the uv-managed project environment first
if command -v uv >/dev/null 2>&1; then
  exec uv run celery -A src.services.celery:celery_app worker --loglevel=info --pool=threads
fi

# Fall back to the local virtual environment
if [ -x ".venv/bin/celery" ]; then
  run_celery ".venv/bin/celery"
  exit $?
fi

# Fall back to whichever celery is on PATH
if command -v celery >/dev/null 2>&1; then
  run_celery "$(command -v celery)"
  exit $?
fi

echo "Celery command not found. Activate your virtual environment or install dependencies."
exit 1

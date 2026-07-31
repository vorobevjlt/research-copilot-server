#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDIS_COMPOSE_FILE="$SCRIPT_DIR/redis/docker-compose.yaml"

cd "$SCRIPT_DIR"

# Make Homebrew-installed document-processing tools available on macOS.
export PATH="/opt/homebrew/bin:$PATH"

SERVICE_NAMES=(
  "Redis"
  "API server"
  "Celery worker"
)
SERVICE_PIDS=()
API_COMMAND=()
CELERY_COMMAND=()
SHUTTING_DOWN=false
REDIS_LAUNCHED=false

cleanup() {
  exit_code=$?

  if [ "$SHUTTING_DOWN" = true ]; then
    return
  fi

  SHUTTING_DOWN=true
  trap - EXIT INT TERM HUP

  echo
  echo "Stopping all services..."

  for pid in "${SERVICE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  for pid in "${SERVICE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  if [ "$REDIS_LAUNCHED" = true ]; then
    docker compose \
      -f "$REDIS_COMPOSE_FILE" \
      down >/dev/null 2>&1 || true
  fi

  echo "All services stopped."
  exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

select_runtime_commands() {
  if command -v uv >/dev/null 2>&1; then
    API_COMMAND=(
      uv run uvicorn src.server:app
      --reload
      --host 0.0.0.0
      --port 8000
    )
    CELERY_COMMAND=(
      uv run celery
      -A src.services.celery:celery_app
      worker
      --loglevel=info
      --pool=threads
    )
    return
  fi

  if command -v poetry >/dev/null 2>&1; then
    API_COMMAND=(
      poetry run uvicorn src.server:app
      --reload
      --host 0.0.0.0
      --port 8000
    )
    CELERY_COMMAND=(
      poetry run celery
      -A src.services.celery:celery_app
      worker
      --loglevel=info
      --pool=threads
    )
    return
  fi

  if [ -x "$SCRIPT_DIR/.venv/bin/uvicorn" ] &&
     [ -x "$SCRIPT_DIR/.venv/bin/celery" ]; then
    API_COMMAND=(
      "$SCRIPT_DIR/.venv/bin/uvicorn" src.server:app
      --reload
      --host 0.0.0.0
      --port 8000
    )
    CELERY_COMMAND=(
      "$SCRIPT_DIR/.venv/bin/celery"
      -A src.services.celery:celery_app
      worker
      --loglevel=info
      --pool=threads
    )
    return
  fi

  if command -v uvicorn >/dev/null 2>&1 &&
     command -v celery >/dev/null 2>&1; then
    API_COMMAND=(
      uvicorn src.server:app
      --reload
      --host 0.0.0.0
      --port 8000
    )
    CELERY_COMMAND=(
      celery
      -A src.services.celery:celery_app
      worker
      --loglevel=info
      --pool=threads
    )
    return
  fi

  echo "Error: could not find both Uvicorn and Celery." >&2
  echo "Install uv (recommended), Poetry, or the local project environment." >&2
  exit 1
}

configure_guardrails() {
  local guardrails_cli="$SCRIPT_DIR/.venv/bin/guardrails"
  local guardrails_python="$SCRIPT_DIR/.venv/bin/python"

  if [ ! -x "$guardrails_cli" ] || [ ! -x "$guardrails_python" ]; then
    return
  fi

  if ! "$guardrails_python" -c \
    'from guardrails.settings import settings; raise SystemExit(settings.rc.enable_metrics is not False)'; then
    echo "Disabling Guardrails telemetry..."
    "$guardrails_cli" configure --disable-metrics
  fi
}

start_redis() {
  echo "Starting Redis..."
  exec docker compose -f "$REDIS_COMPOSE_FILE" up
}

start_api() {
  echo "Starting API server..."
  exec "${API_COMMAND[@]}"
}

start_celery() {
  echo "Starting Celery worker..."
  exec "${CELERY_COMMAND[@]}"
}

if [ ! -f "$REDIS_COMPOSE_FILE" ]; then
  echo "Error: Redis configuration not found at $REDIS_COMPOSE_FILE." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is required to start Redis." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: the Docker Compose plugin is required to start Redis." >&2
  exit 1
fi

select_runtime_commands
configure_guardrails

echo "Starting all server services..."

start_redis &
SERVICE_PIDS+=("$!")
REDIS_LAUNCHED=true

start_api &
SERVICE_PIDS+=("$!")

start_celery &
SERVICE_PIDS+=("$!")

echo
echo "All services started."
echo "API: http://localhost:8000"
echo "Press Ctrl+C to stop everything."

while true; do
  for index in "${!SERVICE_PIDS[@]}"; do
    pid="${SERVICE_PIDS[$index]}"

    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        service_exit_code=0
      else
        service_exit_code=$?
      fi

      echo
      echo "${SERVICE_NAMES[$index]} stopped unexpectedly (exit $service_exit_code)." >&2

      if [ "$service_exit_code" -eq 0 ]; then
        exit 1
      fi

      exit "$service_exit_code"
    fi
  done

  sleep 1
done

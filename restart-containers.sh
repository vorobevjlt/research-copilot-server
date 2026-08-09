#!/usr/bin/env bash

set -euo pipefail

REDIS_CONTAINER="redis_server"

wait_for_redis() {
  local timeout_seconds="${1:-30}"
  local elapsed_seconds=0

  while (( elapsed_seconds < timeout_seconds )); do
    if [ "$(docker exec "$REDIS_CONTAINER" redis-cli ping 2>/dev/null || true)" = "PONG" ]; then
      return 0
    fi

    sleep 1
    ((elapsed_seconds += 1))
  done

  echo "Error: Redis did not respond with PONG within ${timeout_seconds}s." >&2
  return 1
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or is not available on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running." >&2
  exit 1
fi

if ! docker inspect "$REDIS_CONTAINER" >/dev/null 2>&1; then
  echo "Error: required container '$REDIS_CONTAINER' does not exist." >&2
  exit 1
fi

echo "Restarting Redis..."
docker restart "$REDIS_CONTAINER" >/dev/null
wait_for_redis

echo
echo "Redis restarted successfully."
echo "Redis: redis://127.0.0.1:6379"

#!/bin/bash
set -e

PROJECT_DIR="/home/ubuntu/swissenergyforecast"
SERVICES=("backend" "frontend")

echo "🚀 Starting deployment in $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "📥 Pulling latest changes..."
git pull origin main

echo "🏗️  Building and starting containers..."
docker compose -f compose.yaml -f compose.prod.yaml up -d --build

# --- Reusable Healthcheck Function ---
check_health() {
  local SERVICE=$1
  local CONTAINER_ID=$(docker compose ps -q "$SERVICE")

  echo "🩺 Checking health for $SERVICE..."

  for i in {1..12}; do
    local STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "no-healthcheck")

    if [ "$STATUS" = "healthy" ]; then
      echo "✅ $SERVICE is healthy!"
      return 0
    elif [ "$STATUS" = "unhealthy" ]; then
      echo "❌ $SERVICE is unhealthy!"
      docker compose logs "$SERVICE"
      return 1
    elif [ "$STATUS" = "no-healthcheck" ]; then
      # If no healthcheck is defined, check if the container is simply running
      local RUNNING=$(docker inspect --format='{{.State.Running}}' "$CONTAINER_ID")
      if [ "$RUNNING" = "true" ]; then
        echo "⚠️  $SERVICE has no healthcheck, but is running."
        return 0
      fi
    fi

    echo "⏳ $SERVICE status: $STATUS... retrying (try $i/12)"
    sleep 5
  done

  echo "🛑 Timeout: $SERVICE failed to become healthy."
  return 1
}

# --- Execute checks for all services ---
for SERVICE in "${SERVICES[@]}"; do
  if ! check_health "$SERVICE"; then
    exit 1
  fi
done

echo "🎉 All services are up and healthy!"
docker image prune -f
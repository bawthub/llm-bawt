#!/usr/bin/env bash
set -euo pipefail

COMPOSE_BASE="docker-compose.yml"
COMPOSE_DEV="docker-compose.dev.yml"

# Dev compose. COMPOSE_FILE (shell or .env) is the single source of truth for
# the deployed file set, including overlays like docker-compose.ops.yml; -f
# flags would override it and drop them (TASK-1145). Name base+dev only when
# COMPOSE_FILE is defined nowhere.
dev_compose() {
  if [[ -n "${COMPOSE_FILE:-}" ]] || grep -qE '^COMPOSE_FILE=' .env 2>/dev/null; then
    docker compose "$@"
  else
    docker compose -f "$COMPOSE_BASE" -f "$COMPOSE_DEV" "$@"
  fi
}

usage() {
  cat <<EOF
Docker management script for llm-bawt

USAGE:
    ./start.sh [COMMAND] [OPTIONS]

COMMANDS:
    up              Start containers (production mode)
    dev             Start containers in dev mode (live source mounting)
    down            Stop and remove containers
    restart         Restart containers
    rebuild         Rebuild and restart containers
    logs            Follow container logs
    status          Show container status
    exec            Execute command in container
    shell           Open bash shell in container

OPTIONS:
    -f, --force     Force rebuild (ignores cache)
    -d, --detach    Run in background (default for up/dev)

EXAMPLES:
    # Start production mode
    ./start.sh up

    # Start dev mode with live code mounting
    ./start.sh dev

    # Rebuild and restart
    ./start.sh rebuild

    # View logs
    ./start.sh logs

    # Stop everything
    ./start.sh down

    # Open shell in container
    ./start.sh shell

    # Run a command in container
    ./start.sh exec llm --status

DEVELOPMENT:
    Dev mode mounts ./src for live code changes.
    After changing code: ./start.sh restart
    No rebuild needed - just restart!

PRODUCTION:
    Production mode has source baked into image.
    After changing code: ./start.sh rebuild
EOF
}

case "${1:-help}" in
  up)
    echo "Starting containers (production mode)..."
    docker compose up -d
    echo "✓ Services started on ports 8001 (MCP) and 8642 (LLM)"
    echo "Run './start.sh logs' to view output"
    ;;

  dev)
    echo "Starting containers (dev mode with live mounting)..."
    dev_compose up -d
    echo "✓ Dev mode started - source code mounted from ./src"
    echo "✓ After code changes: ./start.sh restart"
    echo "Run './start.sh logs' to view output"
    ;;

  down)
    echo "Stopping containers..."
    docker compose down
    echo "✓ Containers stopped and removed"
    ;;

  restart)
    echo "Restarting containers..."
    if docker compose ps | grep -q llm-bawt-app; then
      # Check if dev mode (has source mount)
      if docker inspect llm-bawt-app 2>/dev/null | grep -q "/app/src"; then
        dev_compose restart
        echo "✓ Dev containers restarted"
      else
        docker compose restart
        echo "✓ Production containers restarted"
      fi
    else
      echo "No containers running. Use './start.sh up' or './start.sh dev'"
      exit 1
    fi
    ;;

  rebuild)
    echo "Rebuilding and restarting containers..."
    docker compose down
    docker compose up -d --build
    echo "✓ Containers rebuilt and started"
    echo "Run './start.sh logs' to view output"
    ;;

  logs)
    echo "Following container logs (Ctrl+C to exit)..."
    docker compose logs -f --tail=50
    ;;

  status|ps)
    echo "Container status:"
    docker compose ps
    echo ""
    echo "To check service health: llm --status"
    ;;

  exec)
    shift
    if [[ $# -eq 0 ]]; then
      echo "Usage: ./start.sh exec <command>"
      echo "Example: ./start.sh exec llm --status"
      exit 1
    fi
    docker compose exec app "$@"
    ;;

  shell|bash)
    echo "Opening shell in container..."
    docker compose exec app bash
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    echo "Unknown command: $1"
    echo ""
    usage
    exit 1
    ;;
esac

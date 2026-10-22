#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$ROOT_DIR/.logs"

MCP_HOST="0.0.0.0"
MCP_PORT="8001"
SERVICE_HOST="0.0.0.0"
SERVICE_PORT="8642"

# Use 127.0.0.1 for inter-service calls (0.0.0.0 causes HTTP 421 from FastMCP)
MCP_URL="http://127.0.0.1:${MCP_PORT}"

MCP_PID_FILE="$RUN_DIR/mcp-server.pid"
SERVICE_PID_FILE="$RUN_DIR/llm-service.pid"

LOG_MODE="file"  # file | stdout
VERBOSE_FLAG=""
DEBUG_FLAG=""
DEV_MODE=false

mkdir -p "$RUN_DIR" "$LOG_DIR"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file")"
  kill -0 "$pid" >/dev/null 2>&1
}

start_mcp_server() {
  if is_running "$MCP_PID_FILE"; then
    echo "MCP server already running (pid $(cat "$MCP_PID_FILE"))"
    return 0
  fi

  echo "[mcp] Starting on ${MCP_HOST}:${MCP_PORT}..."

  # --quiet suppresses uv's progress/build output
  # In dev mode, use --no-sync to avoid rebuilding
  UV_FLAGS="--quiet"
  if [[ "$DEV_MODE" == true ]]; then
    UV_FLAGS="--quiet --no-sync"
  fi

  if [[ "$LOG_MODE" == "stdout" ]]; then
    LLM_BAWT_LOG_PREFIX="mcp" \
    LLM_BAWT_MCP_SERVER_VERBOSE=${VERBOSE_FLAG:+1} \
    LLM_BAWT_MCP_SERVER_DEBUG=${DEBUG_FLAG:+1} \
    uv run $UV_FLAGS --extra mcp llm-mcp-server --transport http --host "$MCP_HOST" --port "$MCP_PORT" &
    echo $! > "$MCP_PID_FILE"
  else
    LLM_BAWT_LOG_PREFIX="mcp" \
    LLM_BAWT_MCP_SERVER_VERBOSE=${VERBOSE_FLAG:+1} \
    LLM_BAWT_MCP_SERVER_DEBUG=${DEBUG_FLAG:+1} \
    nohup uv run $UV_FLAGS --extra mcp llm-mcp-server --transport http --host "$MCP_HOST" --port "$MCP_PORT" \
      > "$LOG_DIR/mcp-server.log" 2>&1 &
    echo $! > "$MCP_PID_FILE"
  fi
}

start_llm_service() {
  if is_running "$SERVICE_PID_FILE"; then
    echo "llm-service already running (pid $(cat "$SERVICE_PID_FILE"))"
    return 0
  fi

  # Set up dev mode flags early (needed for health check and service start)
  RELOAD_FLAG=""
  UV_FLAGS="--quiet"
  if [[ "$DEV_MODE" == true ]]; then
    RELOAD_FLAG="--reload"
    UV_FLAGS="--quiet --no-sync"
  fi

  # Wait for MCP server to be ready before starting llm-service
  echo "[llm] Waiting for MCP server..."
  for i in {1..20}; do
    if uv run $UV_FLAGS python -c "import socket; s=socket.socket(); s.settimeout(0.5); exit(0 if s.connect_ex(('127.0.0.1', $MCP_PORT))==0 else 1)" 2>/dev/null; then
      echo "[llm] MCP server ready"
      break
    fi
    sleep 0.5
  done

  echo "[llm] Starting on ${SERVICE_HOST}:${SERVICE_PORT}..."

  # Embeddings are served by local-model-bridge; the CPU app image does not
  # include the memory extra (sentence-transformers/torch).
  if [[ "$DEV_MODE" == true ]]; then
    echo "[llm] Dev mode enabled - auto-reload on code changes"
  fi

  if [[ "$LOG_MODE" == "stdout" ]]; then
    LLM_BAWT_LOG_PREFIX="" \
    LLM_BAWT_MCP_SERVER_URL="$MCP_URL" \
    uv run $UV_FLAGS --extra service --extra search llm-service --host "$SERVICE_HOST" --port "$SERVICE_PORT" $RELOAD_FLAG $VERBOSE_FLAG $DEBUG_FLAG &
    echo $! > "$SERVICE_PID_FILE"
  else
    LLM_BAWT_LOG_PREFIX="" \
    LLM_BAWT_MCP_SERVER_URL="$MCP_URL" \
    nohup uv run $UV_FLAGS --extra service --extra search llm-service --host "$SERVICE_HOST" --port "$SERVICE_PORT" $RELOAD_FLAG $VERBOSE_FLAG $DEBUG_FLAG \
      > "$LOG_DIR/llm-service.log" 2>&1 &
    echo $! > "$SERVICE_PID_FILE"
  fi
}

stop_service() {
  local name="$1"
  local pid_file="$2"
  local pattern="$3"
  
  local pid=""
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
  fi

  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping $name (pid $pid)..."
    kill "$pid" || true
    # Wait for process to actually exit (up to 10 seconds)
    for i in {1..20}; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
    # Force kill if still running
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Force killing $name..."
      kill -9 "$pid" 2>/dev/null || true
      sleep 0.5
    fi
  elif [[ -n "$pattern" ]]; then
    # Fallback: check for processes matching the pattern
    # Use pgrep to see if any exist first
    if pgrep -f "$pattern" >/dev/null; then
      echo "$name pidfile missing/stale, but process found. Killing by pattern '$pattern'..."
      pkill -f "$pattern" || true
      # Wait for processes to exit
      for i in {1..20}; do
        if ! pgrep -f "$pattern" >/dev/null; then
          break
        fi
        sleep 0.5
      done
      # Force kill if still running
      if pgrep -f "$pattern" >/dev/null; then
        echo "Force killing $name..."
        pkill -9 -f "$pattern" 2>/dev/null || true
        sleep 0.5
      fi
    else
      echo "$name not running (pidfile missing)"
    fi
  else
    echo "$name not running (pidfile missing)"
  fi
  rm -f "$pid_file"
}

# Docker foreground mode (TASK-1286): this script is PID 1, and the kernel
# drops SIGTERM to PID 1 unless the process handles it. Without the trap below,
# `docker stop`/`restart` waited out the 10s stop timeout and then SIGKILLed
# every process, so the app's lifespan shutdown never ran. Forward the signal
# and give the children SHUTDOWN_WAIT_SECONDS to exit, inside Docker's window.
# uvicorn's own graceful timeout is 3s (api.py), so 8s leaves the lifespan
# teardown room to finish. Exit 0 when every child stopped on its own, 1 when
# one had to be SIGKILLed (visible in `docker inspect` as a degraded stop).
SHUTDOWN_WAIT_SECONDS="${LLM_BAWT_SHUTDOWN_WAIT_SECONDS:-8}"
MONITOR_SLEEP_PID=""

shutdown_foreground() {
  trap '' TERM INT
  echo "[server] $1 received; stopping llm-service and MCP server..."
  if [[ -n "$MONITOR_SLEEP_PID" ]]; then
    kill "$MONITOR_SLEEP_PID" 2>/dev/null || true
  fi
  local pids=() alive=() pid_file pid
  for pid_file in "$SERVICE_PID_FILE" "$MCP_PID_FILE"; do
    if is_running "$pid_file"; then
      pids+=("$(cat "$pid_file")")
    fi
  done
  if (( ${#pids[@]} )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
  fi
  # Count 0.2s polls rather than reading $SECONDS: that is whole seconds, so a
  # deadline taken just before a tick boundary would cut the wait short.
  local ticks=0 rc=0 max_ticks=$((SHUTDOWN_WAIT_SECONDS * 5))
  while (( ${#pids[@]} )); do
    alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive+=("$pid")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      break
    fi
    if (( ticks >= max_ticks )); then
      echo "[server] Still running after ${SHUTDOWN_WAIT_SECONDS}s, force killing: ${alive[*]}"
      kill -KILL "${alive[@]}" 2>/dev/null || true
      rc=1
      break
    fi
    sleep 0.2
    ticks=$((ticks + 1))
  done
  rm -f "$SERVICE_PID_FILE" "$MCP_PID_FILE"
  echo "[server] Shutdown complete"
  exit "$rc"
}

status() {
  if is_running "$MCP_PID_FILE"; then
    echo "MCP server: running (pid $(cat "$MCP_PID_FILE")) on port ${MCP_PORT}"
  else
    echo "MCP server: stopped"
  fi

  if is_running "$SERVICE_PID_FILE"; then
    echo "llm-service: running (pid $(cat "$SERVICE_PID_FILE")) on port ${SERVICE_PORT}"
  else
    echo "llm-service: stopped"
  fi

  echo "Logs: $LOG_DIR"
  echo "Mode: $LOG_MODE"
  echo "MCP URL: $MCP_URL"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|restart|status] [--stdout|--logfile] [--verbose|--debug] [--dev]

Commands:
  start     Start MCP server + llm-service (background, logs in .logs/)
  stop      Stop both services
  restart   Stop then start
  status    Show status + log dir

Options:
  --stdout   Run in background but keep logs on stdout (no log files)
  --logfile  Write logs to .logs/ (default)
  --verbose  Pass verbose logging to servers
  --debug    Pass debug logging to servers
  --dev      Enable development mode with auto-reload
EOF
}

cmd="${1:-start}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stdout) LOG_MODE="stdout" ;;
    --logfile) LOG_MODE="file" ;;
    --verbose) VERBOSE_FLAG="--verbose" ;;
    --debug) DEBUG_FLAG="--debug" ;;
    --dev) DEV_MODE=true ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done
case "$cmd" in
  start)
    if [[ "$LOG_MODE" == "stdout" ]]; then
      trap 'shutdown_foreground SIGTERM' TERM
      trap 'shutdown_foreground SIGINT' INT
    fi
    start_mcp_server
    start_llm_service
    status

    # If running in stdout mode (Docker), keep the script alive by tailing logs
    if [[ "$LOG_MODE" == "stdout" ]]; then
      echo ""
      echo "Services started. Monitoring processes..."
      # Keep the script running and monitor child processes
      while true; do
        # Check if both services are still running
        if ! is_running "$MCP_PID_FILE" || ! is_running "$SERVICE_PID_FILE"; then
          echo "[ERROR] One or more services stopped unexpectedly"
          status
          exit 1
        fi
        # Sleep in the background and wait on it: bash runs a trap only after
        # a foreground command returns, but `wait` returns on a signal at once.
        sleep 10 &
        MONITOR_SLEEP_PID=$!
        wait "$MONITOR_SLEEP_PID" || true
      done
    fi
    ;;
  stop)
    stop_service "llm-service" "$SERVICE_PID_FILE" "llm-service"
    stop_service "mcp server" "$MCP_PID_FILE" "llm-mcp-server"
    ;;
  restart)
    stop_service "llm-service" "$SERVICE_PID_FILE" "llm-service"
    stop_service "mcp server" "$MCP_PID_FILE" "llm-mcp-server"
    start_mcp_server
    start_llm_service
    status
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 1
    ;;
esac

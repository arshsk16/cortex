#!/usr/bin/env bash
# =============================================================================
# Cortex container entrypoint
# =============================================================================
# Responsibilities:
#   1. Wait for PostgreSQL to accept connections.
#   2. Run Alembic migrations (idempotent — safe on every restart).
#   3. Start the Uvicorn ASGI server.
#
# Environment variables used directly by this script:
#   DATABASE_URL      — checked for reachability (asyncpg URL format)
#   WORKERS           — number of Uvicorn worker processes (default: 1)
#   HOST              — bind address (default: 0.0.0.0)
#   PORT              — bind port    (default: 8000)
# =============================================================================

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
MAX_WAIT_SECONDS="${DB_WAIT_SECONDS:-30}"

# ---------------------------------------------------------------------------
# Helper: wait for PostgreSQL to be ready
# ---------------------------------------------------------------------------
wait_for_db() {
    # Extract host and port from DATABASE_URL
    # URL format: postgresql+asyncpg://user:pass@host:port/db
    local db_url="${DATABASE_URL:-}"
    if [[ -z "$db_url" ]]; then
        echo "[entrypoint] ERROR: DATABASE_URL is not set" >&2
        exit 1
    fi

    # Strip the dialect prefix and extract host:port
    local hostport
    hostport=$(echo "$db_url" | sed -E 's|.*@([^/]+)/.*|\1|')
    local dbhost dbport
    dbhost=$(echo "$hostport" | cut -d: -f1)
    dbport=$(echo "$hostport" | cut -d: -f2)
    dbport="${dbport:-5432}"

    echo "[entrypoint] Waiting for PostgreSQL at $dbhost:$dbport (max ${MAX_WAIT_SECONDS}s)..."
    local elapsed=0
    until python -c "
import socket, sys
try:
    s = socket.create_connection(('$dbhost', $dbport), timeout=1)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        if [[ $elapsed -ge $MAX_WAIT_SECONDS ]]; then
            echo "[entrypoint] ERROR: PostgreSQL not ready after ${MAX_WAIT_SECONDS}s — aborting" >&2
            exit 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "[entrypoint] PostgreSQL is ready (waited ${elapsed}s)"
}

# ---------------------------------------------------------------------------
# Helper: run Alembic migrations
# ---------------------------------------------------------------------------
run_migrations() {
    echo "[entrypoint] Running Alembic migrations..."
    alembic -c /app/alembic.ini upgrade head
    echo "[entrypoint] Migrations complete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
wait_for_db
run_migrations

echo "[entrypoint] Starting Cortex on ${HOST}:${PORT} (workers=${WORKERS})..."
exec uvicorn cortex.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --no-access-log

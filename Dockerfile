# syntax=docker/dockerfile:1.7
# =============================================================================
# Cortex — Production Dockerfile (multi-stage, Python 3.12-slim)
# =============================================================================
# Stage 1  builder  — install uv and build the project wheel
# Stage 2  runtime  — lean image that only installs the pre-built wheel
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Install uv (pinned release for reproducibility).
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

WORKDIR /build

# uv environment variables required for correct Docker operation:
#   UV_LINK_MODE=copy   — use file copy instead of hardlinks (hardlinks fail
#                         across Docker filesystem layers)
#   UV_PYTHON_DOWNLOADS=never — use the system Python from the base image;
#                         do NOT attempt to download a managed Python
#   UV_COMPILE_BYTECODE=1 — pre-compile .pyc files at build time so the
#                         runtime image starts faster
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

# Copy dependency manifests first to maximise layer cache.
COPY pyproject.toml uv.lock README.md ./

# Install all production dependencies into .venv (skip the project itself).
# --no-dev is omitted: dev packages are optional extras (extra == 'dev'),
# not a uv dev group, so they are already excluded from the default sync.
RUN uv sync --frozen --no-install-project

# Copy source and install the project itself.
COPY src/ ./src/
RUN uv sync --frozen


# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Non-root user for security.
RUN groupadd --gid 1001 cortex \
 && useradd  --uid 1001 --gid cortex --no-create-home --shell /usr/sbin/nologin cortex

# System packages:
#   curl   — used by the Docker HEALTHCHECK command
#   libpq5 — PostgreSQL client library required by asyncpg
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends curl libpq5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder --chown=cortex:cortex /build/.venv /app/.venv

# Copy application source, Alembic migrations, and config.
COPY --chown=cortex:cortex src/        /app/src/
COPY --chown=cortex:cortex alembic/    /app/alembic/
COPY --chown=cortex:cortex alembic.ini /app/alembic.ini

# Copy the container entrypoint script.
COPY --chown=cortex:cortex docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Persistent-volume mount points (mapped by docker-compose).
RUN mkdir -p /app/storage/documents /app/storage/chroma \
 && chown -R cortex:cortex /app/storage

# Activate the virtual environment for every subsequent RUN / CMD.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER cortex

EXPOSE 8000

# Liveness probe via the Phase 15 /health/live endpoint.
# start_period allows migrations to run before checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

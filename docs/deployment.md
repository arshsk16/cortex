# Cortex — Deployment Guide

## Overview

Cortex runs as a single Docker container backed by PostgreSQL 16 and Redis 7.
All three services are orchestrated by the included `docker-compose.yml`.

```
┌─────────────────────────────────────────────┐
│             docker-compose stack             │
│                                             │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  cortex  │──▶│ postgres │   │  redis  │ │
│  │  :8000   │   │  :5432   │   │  :6379  │ │
│  └──────────┘   └──────────┘   └─────────┘ │
│       │              │               │      │
│  cortex_documents  postgres_data  redis_data│
│  cortex_chroma     (named volumes)          │
└─────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24.x |
| Docker Compose | v2.x (`docker compose`) |

---

## Quick Start

### 1. Configure secrets

```bash
cp .env.docker .env.docker.local
```

Edit `.env.docker.local` and fill in every value marked `<CHANGE_ME>`:

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `REDIS_PASSWORD` | Redis AUTH password |
| `JWT_SECRET_KEY` | 32-byte hex secret — generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GEMINI_API_KEY` | Google AI Studio API key (optional; leave blank to disable LLM calls) |

### 2. Build and start the stack

```bash
docker compose --env-file .env.docker.local up -d --build
```

On first boot the `cortex` container will:
1. Wait for PostgreSQL to be ready.
2. Run Alembic migrations automatically (`alembic upgrade head`).
3. Start the Uvicorn ASGI server.

### 3. Verify

```bash
# Check all containers are healthy
docker compose ps

# Tail logs
docker compose logs -f cortex

# Liveness probe (should return {"status": "ok"})
curl http://localhost:8000/health/live

# Readiness probe (includes DB check)
curl http://localhost:8000/health/ready

# API docs
open http://localhost:8000/docs
```

---

## Stopping and restarting

```bash
# Stop without removing volumes
docker compose down

# Restart after a config change (rebuild image)
docker compose --env-file .env.docker.local up -d --build

# Full teardown including volumes (DESTRUCTIVE — all data is lost)
docker compose down -v
```

---

## Running Alembic Migrations Manually

Migrations run automatically on every startup, but you can also run them
manually against the running `db` container:

```bash
# Apply all pending migrations
docker compose exec cortex alembic -c /app/alembic.ini upgrade head

# Show current migration state
docker compose exec cortex alembic -c /app/alembic.ini current

# Show migration history
docker compose exec cortex alembic -c /app/alembic.ini history --verbose

# Downgrade one revision
docker compose exec cortex alembic -c /app/alembic.ini downgrade -1
```

---

## Health Checks

Cortex exposes two health endpoints for use with container orchestrators:

| Endpoint | Purpose | HTTP status |
|----------|---------|-------------|
| `GET /health/live` | **Liveness**: process alive, event loop responsive | `200` always |
| `GET /health/ready` | **Readiness**: all dependencies (DB) healthy | `200` healthy / `503` degraded |
| `GET /health` | Full status with component details | `200` / `503` |

The `Dockerfile` and `docker-compose.yml` both configure Docker health checks
pointing at `/health/live`.

---

## Scaling

To run more Uvicorn workers (same container):

```bash
# In .env.docker.local
WORKERS=4
docker compose --env-file .env.docker.local up -d --build
```

To run multiple container replicas (requires an external load balancer):

```bash
docker compose --env-file .env.docker.local up -d --scale cortex=3
```

> Note: rate limiting is in-process (Phase 15). Multiple replicas have
> independent rate-limit counters. A Redis-backed limiter (Phase 17+) is
> required for accurate per-IP limiting across replicas.

---

## Environment Variables Reference

All variables from `.env.docker` are supported. The most important ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *(built by compose)* | asyncpg connection URL |
| `REDIS_URL` | *(built by compose)* | Redis connection URL |
| `JWT_SECRET_KEY` | **required** | JWT signing secret (≥32 chars) |
| `GEMINI_API_KEY` | `""` | Google AI API key |
| `WORKERS` | `2` | Uvicorn worker process count |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LOG_JSON` | `true` | Emit JSON logs (recommended in prod) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Global per-IP rate limit |
| `RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE` | `10` | Auth-route rate limit |
| `GEMINI_REQUEST_TIMEOUT_SECONDS` | `30` | Max seconds to wait for Gemini |

---

## Persistent Volumes

| Volume | Mount | Contents |
|--------|-------|----------|
| `postgres_data` | PostgreSQL internal | All database data |
| `redis_data` | Redis internal | AOF persistence file |
| `cortex_documents` | `/app/storage/documents` | Uploaded PDF files |
| `cortex_chroma` | `/app/storage/chroma` | ChromaDB vector index |

Back up all four volumes before major upgrades.

---

## Troubleshooting

### Container exits immediately

```bash
docker compose logs cortex
```

Common causes:
- `DATABASE_URL` not reachable — check `db` container is healthy.
- `JWT_SECRET_KEY` shorter than 32 characters — Pydantic validation fails.

### Migration fails on startup

```bash
docker compose exec cortex alembic -c /app/alembic.ini current
docker compose exec cortex alembic -c /app/alembic.ini upgrade head
```

### Port already in use

Change `CORTEX_PORT`, `POSTGRES_PORT`, or `REDIS_PORT` in your env file.

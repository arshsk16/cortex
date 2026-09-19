# Cortex

Enterprise-grade AI platform backend built with FastAPI, PostgreSQL, SQLAlchemy, and Alembic.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL 14+

## Quick start

```bash
# Install dependencies (creates .venv)
uv sync --all-extras

# Copy environment file and edit DATABASE_URL if needed
cp .env.example .env

# Configure DATABASE_URL and JWT_SECRET_KEY in .env

# Apply database migrations (once models exist)
uv run alembic upgrade head

# Start the API server
uv run cortex
# or
uv run uvicorn cortex.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Health check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

## Authentication (Phase 1)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/register` | Create account; returns JWT + user |
| `POST` | `/api/v1/login` | Email/password login; returns JWT + user |
| `GET` | `/api/v1/me` | Current user (`Authorization: Bearer <token>`) |

Set a strong `JWT_SECRET_KEY` (min 32 characters) in `.env` before deploying.

## Document management (Phase 2)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload PDF (max 25 MB); returns metadata |
| `GET` | `/api/v1/documents` | List owned documents (paginated) |
| `GET` | `/api/v1/documents/{id}` | Retrieve owned document metadata |
| `DELETE` | `/api/v1/documents/{id}` | Delete owned document and stored file |

Uploaded files are stored under `storage/documents/` (configurable via `DOCUMENT_STORAGE_PATH`).

## Document ingestion (Phase 3)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/{id}/process` | Extract, clean, chunk PDF; store chunks |

Status flow: `uploaded` → `processing` → `ready` (or `failed` on error).

## Embeddings & vector store (Phase 4)

During processing, each chunk is embedded with `sentence-transformers` (default: `BAAI/bge-small-en-v1.5`) and stored in persistent ChromaDB. Chunk text remains in PostgreSQL; Chroma stores embeddings plus `chunk_id`, `document_id`, and `chunk_index` metadata only.

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding batch size |
| `CHROMA_PERSIST_DIRECTORY` | `storage/chroma` | ChromaDB data directory |
| `CHROMA_COLLECTION_NAME` | `document_chunks` | Chroma collection name |

## Project layout

```
cortex/
├── alembic/                 # Database migration environment
├── src/cortex/
│   ├── api/                 # HTTP routers, endpoints, DI wiring
│   ├── core/                # Settings, logging, exceptions
│   ├── db/                  # Engine, sessions, ORM models
│   ├── schemas/             # Pydantic v2 request/response models
│   ├── services/            # Business logic (SOLID, injectable)
│   ├── utils/               # Shared pure helpers
│   └── main.py              # App factory and ASGI entrypoint
├── tests/
├── .env.example
├── alembic.ini
└── pyproject.toml
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check src tests
uv run mypy src
```

## Architecture notes

- **Routers** only handle HTTP concerns and delegate to services.
- **Services** contain business logic and receive collaborators via constructors.
- **Dependency injection** is wired in `api/deps.py` using FastAPI `Depends`.
- **Settings** are loaded once via `pydantic-settings` from environment / `.env`.
- **Database** access is async (`asyncpg` + SQLAlchemy 2.0 asyncio).

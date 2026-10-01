# Cortex

Cortex is an enterprise-grade AI platform backend. It provides a robust foundation for building AI-powered applications, offering document ingestion, Retrieval-Augmented Generation (RAG), agentic workflows with tool calling, and long-term conversational memory. It is built with FastAPI, PostgreSQL, Redis, and ChromaDB.

## Architecture & Components

- **Web Framework**: FastAPI (async ASGI, Pydantic v2 validation).
- **Relational Database**: PostgreSQL 14+ via SQLAlchemy 2.0 (asyncpg) and Alembic for migrations.
- **Vector Database**: ChromaDB for storing document embeddings and extracted memory fragments.
- **State & Cache**: Redis for agent state persistence, JWT revocation blocklists, and sliding-window rate limiting.
- **LLM & Embeddings**: Abstracted provider interfaces (supporting models like `BAAI/bge-small-en-v1.5`).

## Core Features

### 1. Document & RAG Pipeline
- **Upload & Streaming Validation**: Secure PDF upload with streaming file-size enforcement (max 25 MB).
- **Ingestion**: Asynchronous extraction, cleaning, chunking, and embedding.
- **Retrieval**: Semantic search against ChromaDB using cosine similarity.

### 2. Agent & Tool Calling
- **Agent Orchestration**: ReAct-style agent orchestration powered by the LLM.
- **Tools**: Includes `RAGSearchTool`, `DocumentListTool`, and a `CalculatorTool`.
- **Streaming**: Supports Server-Sent Events (SSE) for streaming agent thought processes and responses in real time (`/api/v1/agent/stream`).

### 3. Conversation & Long-Term Memory
- **Conversations**: Full chat history persisted to PostgreSQL.
- **Memory Extraction**: A background `MemoryExtractorService` analyzes conversations to extract persistent user facts and preferences, embedding them into a dedicated Chroma memory collection.
- **State Store**: Redis caches agent state with TTLs to support multi-turn conversational agents.

### 4. Authentication & Security Hardening
- **JWT Authentication**: Secure Bearer tokens with `jti` (JWT ID) claims.
- **Revocation Blocklist**: Redis-backed token blocklist for immediate, fail-open logout revocation.
- **Rate Limiting**: Sliding-window rate limiter with per-IP isolation and Trusted Proxy CIDR support to prevent `X-Forwarded-For` spoofing.
- **API Gating**: Interactive OpenAPI docs (`/docs`, `/redoc`) are automatically disabled in production environments.

## Deployment & DevOps

### Docker & Docker Compose
A complete local environment is provided via `docker-compose.yml`, spinning up PostgreSQL, Redis, and the Cortex API (via the included `Dockerfile`).

### Kubernetes
Production-ready Kubernetes manifests are provided in the `k8s/` directory (deployable via Kustomize). Features include:
- Deployments, Services, and Ingress configuration.
- ConfigMaps and Secrets for environment variables.
- Horizontal Pod Autoscaler (HPA) and resource limits.
- Liveness and readiness probes pointing to `/api/v1/health/live` and `/api/v1/health/ready`.

### CI/CD
GitHub Actions (`.github/workflows/ci.yml`) enforces quality by running:
- Linting (`ruff`) and Type Checking (`mypy`).
- Full test suite execution (pytest).
- Docker image build and containerized smoke tests against live backing services.

### Performance Benchmarking
A suite of performance and load tests is located in `benchmarks/`. The application is capable of handling high concurrency, with metrics tracked for PostgreSQL loads, RAG retrieval times, and API latencies.

## Quality & Testing

The project is rigorously tested, currently passing **786 automated tests** (pytest) covering unit, integration, and security edge cases. Code formatting and linting are strictly enforced via `ruff`, ensuring zero violations.

## Quick Start

### Option 1: Local Development (uv)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync --all-extras

# 2. Configure environment
cp .env.example .env
# Edit .env to set DATABASE_URL, REDIS_URL, and a strong JWT_SECRET_KEY (min 32 chars)

# 3. Apply database migrations
uv run alembic upgrade head

# 4. Start the server
uv run cortex
# Or use uvicorn directly: uv run uvicorn cortex.main:app --reload
```

### Option 2: Docker Compose

Requires Docker and Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

### Accessing the API

- **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Development mode only)
- **Liveness Probe**: [http://localhost:8000/api/v1/health/live](http://localhost:8000/api/v1/health/live)
- **Readiness Probe**: [http://localhost:8000/api/v1/health/ready](http://localhost:8000/api/v1/health/ready)

## Project Structure

```text
cortex/
├── .github/                 # CI/CD workflows
├── alembic/                 # Database migrations
├── benchmarks/              # Performance and load testing scripts
├── docker/                  # Docker-related entrypoints
├── k8s/                     # Kubernetes manifests & kustomize config
├── src/cortex/
│   ├── agent/               # Agent orchestration, tools, and streaming
│   ├── api/                 # HTTP routers, dependencies, and endpoints
│   ├── core/                # Settings, rate-limiting, security, blocklist
│   ├── db/                  # SQLAlchemy models, sessions, engine
│   ├── embeddings/          # Embedding providers (SentenceTransformers)
│   ├── llm/                 # LLM provider interfaces
│   ├── retrieval/           # RAG retrieval logic
│   ├── schemas/             # Pydantic v2 schemas
│   ├── services/            # Business logic (Auth, Document, Memory, etc.)
│   ├── state_store/         # Redis / Null state store
│   ├── vectorstore/         # ChromaDB interfaces
│   └── main.py              # Application factory and ASGI entry
├── tests/                   # 780+ pytest suite
├── .env.example             # Template environment variables
├── docker-compose.yml       # Local multi-container stack
├── Dockerfile               # Production container definition
├── pyproject.toml           # Project metadata and dependencies
└── uv.lock                  # Pinned dependency lockfile
```

## Known Production Limitations

- **ChromaDB**: Currently utilizes the persistent file-based Chroma client (`PersistentClient`). For highly available, multi-replica deployments (e.g., scaled via Kubernetes HPA), this should be migrated to a remote ChromaDB server (`HttpClient`) to avoid split-brain vector state.
- **HMAC JWT**: Access tokens use symmetric signing (HS256). The `JWT_SECRET_KEY` must be securely managed (e.g., via Kubernetes Secrets) and kept strictly confidential.

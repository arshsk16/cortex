# Cortex — Kubernetes Deployment Guide

## Overview

Cortex ships with production-ready Kubernetes manifests in `k8s/`.

```
k8s/
├── namespace.yaml           Cortex namespace
├── configmap.yaml           Non-sensitive configuration
├── secret.yaml              Secret template (fill before applying)
├── kustomization.yaml       Kustomize entry point
├── postgres/
│   ├── statefulset.yaml     PostgreSQL 16 StatefulSet with PVC
│   └── service.yaml         Headless ClusterIP service
├── redis/
│   ├── deployment.yaml      Redis 7 Deployment
│   ├── pvc.yaml             Redis PersistentVolumeClaim
│   └── service.yaml         ClusterIP service
└── cortex/
    ├── deployment.yaml      Cortex API Deployment (2 replicas)
    ├── service.yaml         ClusterIP service
    ├── pvc.yaml             Document + ChromaDB PVCs
    ├── hpa.yaml             HorizontalPodAutoscaler (2–10 replicas)
    └── ingress.yaml         NGINX Ingress (replace domain)
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| kubectl | ≥ 1.28 |
| Kubernetes cluster | ≥ 1.28 |
| NGINX Ingress Controller | ≥ 1.10 (optional) |
| Metrics Server | ≥ 0.7 (required for HPA) |

---

## Quick Deploy

### 1. Fill in secrets

```bash
# Generate a strong JWT secret
python -c "import secrets; print(secrets.token_hex(32))"

# Base64-encode each secret value
echo -n 'MY_POSTGRES_PASSWORD' | base64
echo -n 'MY_REDIS_PASSWORD'    | base64
echo -n 'MY_JWT_SECRET'        | base64
echo -n 'MY_GEMINI_API_KEY'    | base64
```

Edit `k8s/secret.yaml` and replace every `<base64:CHANGE_ME>` placeholder.

### 2. Set your image

Edit `k8s/cortex/deployment.yaml` and replace `cortex:latest` with your
image reference (e.g. `ghcr.io/your-org/cortex:v0.1.0`).

### 3. Apply with Kustomize

```bash
# Apply everything in the correct order
kubectl apply -k k8s/

# Or step-by-step:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/
kubectl apply -f k8s/cortex/
```

### 4. Run migrations

Migrations run automatically via the entrypoint on pod startup.
To run them manually:

```bash
kubectl exec -n cortex deploy/cortex -- \
  alembic -c /app/alembic.ini upgrade head
```

### 5. Verify

```bash
# Check pod status
kubectl get pods -n cortex

# Wait for all pods to be ready
kubectl rollout status deployment/cortex -n cortex

# Liveness probe
kubectl exec -n cortex deploy/cortex -- \
  curl -fsS http://localhost:8000/health/live

# Readiness probe
kubectl exec -n cortex deploy/cortex -- \
  curl -fsS http://localhost:8000/health/ready

# Tail logs
kubectl logs -n cortex -l app=cortex -f
```

---

## Health Probes

| Probe | Endpoint | Behaviour |
|-------|----------|-----------|
| Liveness | `/health/live` | Always 200 — restarts on event-loop freeze |
| Readiness | `/health/ready` | 200 only when DB is healthy — removes pod from LB on DB down |

The Deployment sets `initialDelaySeconds: 60` for liveness (allows migrations to
complete on first pod start).

---

## Scaling

### Manual scaling

```bash
kubectl scale deployment cortex --replicas=4 -n cortex
```

### Automatic scaling (HPA)

The `cortex/hpa.yaml` autoscaler scales between **2–10** replicas based on
CPU (70%) and memory (80%) utilisation.

```bash
kubectl get hpa -n cortex
```

> **Note**: The in-process rate limiter (Phase 15) is per-pod. Deploy a
> Redis-backed rate limiter for accurate per-IP limiting across replicas.

---

## Updating the Application

```bash
# 1. Build and push a new image
docker build -t ghcr.io/your-org/cortex:v0.2.0 .
docker push ghcr.io/your-org/cortex:v0.2.0

# 2. Update the deployment (rolling update — zero downtime)
kubectl set image deployment/cortex \
  cortex=ghcr.io/your-org/cortex:v0.2.0 -n cortex

# 3. Monitor the rollout
kubectl rollout status deployment/cortex -n cortex

# 4. Roll back if needed
kubectl rollout undo deployment/cortex -n cortex
```

---

## Persistent Storage

| PVC | Default size | Contents |
|-----|-------------|----------|
| `postgres-data` (StatefulSet) | 10 Gi | PostgreSQL data |
| `redis-pvc` | 2 Gi | Redis AOF journal |
| `cortex-documents-pvc` | 20 Gi | Uploaded PDFs |
| `cortex-chroma-pvc` | 10 Gi | ChromaDB vector index |

Change `storage:` values in the manifest files before first apply.
PVC sizes cannot be reduced after creation.

---

## Deleting the Stack

```bash
# Delete all resources in the namespace (PRESERVES PVCs)
kubectl delete namespace cortex

# Also delete persistent volumes (DESTRUCTIVE — all data lost)
kubectl delete pvc -n cortex --all
```

---

## CI/CD — GitHub Actions

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | push / PR to `main` | lint → pytest → docker build |
| `docker-publish.yml` | push tag `v*` | build + push to GHCR |

### Setting up CI secrets

In your GitHub repository go to **Settings → Secrets and variables → Actions**
and add:

| Secret | Value |
|--------|-------|
| *(none required for GHCR push)* | `GITHUB_TOKEN` is automatic |

To push to another registry (DockerHub, ECR, etc.) add the appropriate
`DOCKER_USERNAME` / `DOCKER_PASSWORD` secrets and update `docker-publish.yml`.

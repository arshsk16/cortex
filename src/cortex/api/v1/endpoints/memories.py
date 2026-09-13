"""Memory management HTTP endpoints -- Phase 13A."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from cortex.api.deps import CurrentActiveUserDep, MemoryServiceDep
from cortex.schemas.memory import (
    MemoryCreate,
    MemoryList,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)

router = APIRouter(prefix="/memories", tags=["Memories"])


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a memory",
    description=(
        "Persist a new memory entry and index its embedding for semantic search."
    ),
    responses={
        201: {"description": "Memory created"},
        401: {"description": "Not authenticated"},
        422: {"description": "Validation error"},
    },
)
async def create_memory(
    payload: MemoryCreate,
    current_user: CurrentActiveUserDep,
    memory_service: MemoryServiceDep,
) -> MemoryResponse:
    """Create a new memory for the authenticated user."""
    memory = await memory_service.create(user=current_user, payload=payload)
    return MemoryResponse(memory=memory)


@router.get(
    "",
    response_model=MemoryList,
    status_code=status.HTTP_200_OK,
    summary="List memories",
    description="Return a paginated list of memories owned by the authenticated user.",
    responses={
        200: {"description": "Paginated memory list"},
        401: {"description": "Not authenticated"},
    },
)
async def list_memories(
    current_user: CurrentActiveUserDep,
    memory_service: MemoryServiceDep,
    skip: Annotated[int, Query(ge=0, description="Records to skip")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=100, description="Max records to return")
    ] = 50,
) -> MemoryList:
    """List memories for the authenticated user."""
    return await memory_service.list(user=current_user, skip=skip, limit=limit)


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic memory search",
    description=(
        "Embed ``query`` and return the most similar memories owned by the "
        "authenticated user. Results are ordered by descending cosine similarity."
    ),
    responses={
        200: {"description": "Semantic search results"},
        401: {"description": "Not authenticated"},
        422: {"description": "Validation error"},
    },
)
async def search_memories(
    payload: MemorySearchRequest,
    current_user: CurrentActiveUserDep,
    memory_service: MemoryServiceDep,
) -> MemorySearchResponse:
    """Semantic search over the authenticated user's memories."""
    results = await memory_service.search(
        user=current_user,
        query=payload.query,
        limit=payload.limit,
    )
    return MemorySearchResponse(results=results, query=payload.query)


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a memory",
    description="Retrieve a single memory owned by the authenticated user.",
    responses={
        200: {"description": "Memory record"},
        401: {"description": "Not authenticated"},
        404: {"description": "Memory not found"},
    },
)
async def get_memory(
    memory_id: str,
    current_user: CurrentActiveUserDep,
    memory_service: MemoryServiceDep,
) -> MemoryResponse:
    """Return a single owned memory."""
    memory = await memory_service.get(memory_id=memory_id, user=current_user)
    return MemoryResponse(memory=memory)


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a memory",
    description=(
        "Delete an owned memory and remove its embedding from the semantic index."
    ),
    responses={
        204: {"description": "Memory deleted"},
        401: {"description": "Not authenticated"},
        404: {"description": "Memory not found"},
    },
)
async def delete_memory(
    memory_id: str,
    current_user: CurrentActiveUserDep,
    memory_service: MemoryServiceDep,
) -> None:
    """Delete an owned memory."""
    await memory_service.delete(memory_id=memory_id, user=current_user)

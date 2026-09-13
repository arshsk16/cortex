"""Semantic retrieval HTTP endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

from cortex.api.deps import CurrentActiveUserDep, RetrieverDep
from cortex.schemas.retrieval import (
    RetrievalChunkResult,
    RetrievalRequest,
    RetrievalResponse,
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic chunk search",
    description=(
        "Embed the query with the configured embedding model, search the "
        "vector store for the most similar chunks owned by the authenticated "
        "user, and return ranked results with their chunk text hydrated from "
        "PostgreSQL.  Optionally restrict the search to a single document by "
        "supplying ``document_id`` (the document must have status ``ready``)."
    ),
    responses={
        200: {"description": "Ranked list of matching chunks"},
        400: {"description": "Empty query or document not ready"},
        401: {"description": "Not authenticated"},
        404: {"description": "Document not found"},
        422: {"description": "Validation error"},
    },
)
async def semantic_search(
    payload: RetrievalRequest,
    current_user: CurrentActiveUserDep,
    retriever: RetrieverDep,
) -> RetrievalResponse:
    """Return the top-*k* chunks most semantically similar to the query.

    The endpoint is intentionally thin: all business logic (ownership
    enforcement, embedding generation, vector search, text hydration) lives
    in :class:`~cortex.retrieval.semantic.SemanticRetriever`.
    """
    results = await retriever.retrieve(
        query=payload.query,
        user=current_user,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )

    return RetrievalResponse(
        query=payload.query,
        top_k=payload.top_k,
        results=[
            RetrievalChunkResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                text=r.text,
                score=r.score,
            )
            for r in results
        ],
        result_count=len(results),
    )

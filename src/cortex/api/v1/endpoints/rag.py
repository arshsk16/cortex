"""RAG query HTTP endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

from cortex.api.deps import CurrentActiveUserDep, RAGServiceDep
from cortex.schemas.rag import CitationResponse, RAGQueryRequest, RAGResponse

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG query — answer a question from document context",
    description=(
        "Retrieve the most relevant document chunks owned by the authenticated user, "
        "build a structured prompt, and generate a grounded answer using Gemini. "
        "Each request is stateless; no conversation history is maintained. "
        "Optionally restrict retrieval to a single document by supplying "
        "``document_id`` (the document must have status ``ready``)."
    ),
    responses={
        200: {"description": "Grounded answer with source citations"},
        400: {"description": "Empty question or document not ready"},
        401: {"description": "Not authenticated"},
        404: {"description": "Document not found"},
        503: {"description": "LLM generation failed"},
        422: {"description": "Validation error"},
    },
)
async def rag_query(
    payload: RAGQueryRequest,
    current_user: CurrentActiveUserDep,
    rag_service: RAGServiceDep,
) -> RAGResponse:
    """Answer the user's question using retrieved document context.

    The endpoint is intentionally thin:  all business logic lives in
    :class:`~cortex.services.rag.RAGService`.
    """
    result = await rag_service.answer(
        question=payload.question,
        user=current_user,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )

    citations = [
        CitationResponse(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
        )
        for chunk in result.retrieved_chunks
    ]

    return RAGResponse(
        answer=result.answer,
        citations=citations,
    )

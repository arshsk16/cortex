"""Pydantic schemas for the RAG query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    """Request payload for POST /api/v1/rag/query."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question to answer from document context",
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document UUID to restrict retrieval to a single owned "
            "document.  The document must have status 'ready'."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve as context",
    )


class CitationResponse(BaseModel):
    """A single source citation in the RAG response."""

    document_id: str = Field(..., description="UUID of the source document")
    chunk_id: str = Field(..., description="UUID of the source chunk")
    chunk_index: int = Field(
        ..., ge=0, description="Zero-based position of the chunk within the document"
    )


class RAGResponse(BaseModel):
    """Response envelope for POST /api/v1/rag/query."""

    answer: str = Field(
        ...,
        description="The model's answer grounded in the retrieved document context",
    )
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description=(
            "Source citations corresponding to the chunks used to generate the answer"
        ),
    )

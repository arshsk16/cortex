"""Document management HTTP endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from cortex.api.deps import (
    CurrentActiveUserDep,
    DocumentServiceDep,
    IngestionServiceDep,
)
from cortex.schemas.document import DocumentList, DocumentResponse
from cortex.schemas.ingestion import DocumentProcessResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
    description=(
        "Upload a PDF file (max 25 MB). The authenticated user becomes the "
        "owner of the stored document."
    ),
    responses={
        201: {"description": "Document uploaded successfully"},
        400: {"description": "Invalid file type or size"},
        401: {"description": "Not authenticated"},
        422: {"description": "Validation error"},
    },
)
async def upload_document(
    current_user: CurrentActiveUserDep,
    document_service: DocumentServiceDep,
    file: Annotated[UploadFile, File(description="PDF file to upload")],
    title: Annotated[str | None, Form(description="Optional document title")] = None,
) -> DocumentResponse:
    """Accept a PDF upload and persist document metadata."""
    content = await file.read()
    document = await document_service.upload(
        user=current_user,
        content=content,
        original_filename=file.filename or "document.pdf",
        mime_type=file.content_type,
        title=title,
    )
    return DocumentResponse(document=document)


@router.get(
    "",
    response_model=DocumentList,
    status_code=status.HTTP_200_OK,
    summary="List documents",
    description="Return a paginated list of documents owned by the authenticated user.",
    responses={
        200: {"description": "Paginated document list"},
        401: {"description": "Not authenticated"},
    },
)
async def list_documents(
    current_user: CurrentActiveUserDep,
    document_service: DocumentServiceDep,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Maximum records to return"),
    ] = 50,
) -> DocumentList:
    """List documents for the authenticated user."""
    return await document_service.list(user=current_user, skip=skip, limit=limit)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document",
    description="Retrieve metadata for a single owned document.",
    responses={
        200: {"description": "Document metadata"},
        401: {"description": "Not authenticated"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    current_user: CurrentActiveUserDep,
    document_service: DocumentServiceDep,
) -> DocumentResponse:
    """Return metadata for an owned document."""
    document = await document_service.get(document_id=document_id, user=current_user)
    return DocumentResponse(document=document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete an owned document and remove its stored file.",
    responses={
        204: {"description": "Document deleted"},
        401: {"description": "Not authenticated"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    current_user: CurrentActiveUserDep,
    document_service: DocumentServiceDep,
) -> None:
    """Delete an owned document."""
    await document_service.delete(document_id=document_id, user=current_user)


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process document",
    description=(
        "Extract text from an owned PDF, clean and chunk it, and persist chunks. "
        "Reprocessing replaces any existing chunks for the document."
    ),
    responses={
        200: {"description": "Document processed successfully"},
        401: {"description": "Not authenticated"},
        404: {"description": "Document not found"},
        422: {"description": "Document processing failed"},
    },
)
async def process_document(
    document_id: str,
    current_user: CurrentActiveUserDep,
    ingestion_service: IngestionServiceDep,
) -> DocumentProcessResponse:
    """Run the ingestion pipeline for an owned document."""
    return await ingestion_service.process(document_id=document_id, user=current_user)

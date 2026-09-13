"""Document list tool — exposes the user's document inventory to the agent.

This tool is intentionally **read-only**.  It returns document metadata
(title, status, ID) so the agent can inform the user which documents are
available or suggest using a specific document_id with rag_search.

It wraps :class:`~cortex.services.document.DocumentService` and enforces
user ownership through that service's existing ownership checks.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from cortex.agent.base import Tool

if TYPE_CHECKING:
    from cortex.db.models.user import User
    from cortex.services.document import DocumentService

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50


class DocumentListTool(Tool):
    """List the user's uploaded documents.

    Returns document titles, IDs, and ingestion status.  Useful when the
    agent needs to check which documents are available before searching or
    when the user asks about their document library.
    """

    def __init__(self, document_service: DocumentService) -> None:
        self._doc_service = document_service

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "document_list"

    @property
    def description(self) -> str:
        return (
            "List the user's uploaded documents with their titles, IDs, and "
            "processing status.  Use this tool when the user asks what documents "
            "are available, or before calling rag_search to find a relevant "
            "document_id to restrict retrieval to a specific document."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of documents to return "
                        f"(1–{_MAX_LIMIT}, default {_DEFAULT_LIMIT})"
                    ),
                    "minimum": 1,
                    "maximum": _MAX_LIMIT,
                },
            },
        }

    async def execute(self, args: dict[str, Any], user: User) -> str:
        """Return a JSON-formatted summary of the user's document library."""
        limit: int = min(
            max(int(args.get("limit", _DEFAULT_LIMIT)), 1),
            _MAX_LIMIT,
        )

        doc_list = await self._doc_service.list(user=user, skip=0, limit=limit)

        if not doc_list.items:
            return (
                "The user has no uploaded documents yet.  "
                "They must upload and process a document before searching."
            )

        docs_summary = [
            {
                "id": doc.id,
                "title": doc.title,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc in doc_list.items
        ]

        result = {
            "total": doc_list.total,
            "returned": len(docs_summary),
            "documents": docs_summary,
        }

        logger.info(
            "DocumentListTool: user_id=%s returned=%d total=%d",
            user.id,
            len(docs_summary),
            doc_list.total,
        )
        return json.dumps(result, indent=2)

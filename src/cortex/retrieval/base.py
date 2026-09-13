"""Retriever abstraction for semantic search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cortex.retrieval.models import RetrievalResult

if TYPE_CHECKING:
    from cortex.db.models.user import User


class Retriever(ABC):
    """Return ranked, user-scoped chunks relevant to a natural-language query.

    Implementations embed the query, search a vector store for nearest
    neighbours, and hydrate the text from PostgreSQL.  The abstraction
    keeps callers decoupled from any specific embedding model or vector
    backend.
    """

    @abstractmethod
    async def retrieve(
        self,
        *,
        query: str,
        user: User,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-*k* chunks most relevant to ``query``.

        Parameters
        ----------
        query:
            Natural-language search query.
        user:
            Authenticated owner; only chunks belonging to this user are
            returned (ownership enforced at the document level).
        top_k:
            Maximum number of results to return.
        document_id:
            Optional filter to restrict search to a single document owned
            by ``user``.
        """

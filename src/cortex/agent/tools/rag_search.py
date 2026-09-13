"""RAG search tool — wraps the existing SemanticRetriever.

Design note
-----------
This tool intentionally does **not** duplicate RAG prompting or citation
logic.  Its ``execute`` method calls :meth:`~cortex.retrieval.base.Retriever.retrieve`
directly and returns a formatted observation string for the agent's
reasoning context.

The :class:`~cortex.retrieval.models.RetrievalResult` objects are
*also* stored on the tool instance so that :class:`~cortex.agent.service.AgentService`
can accumulate all retrieved chunks across multiple calls.  The final
grounded answer is then produced by the existing
:class:`~cortex.services.prompt_builder.PromptBuilder` +
:class:`~cortex.llm.base.LLMProvider` pipeline — identical to
:class:`~cortex.services.rag.RAGService`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cortex.agent.base import Tool
from cortex.retrieval.models import RetrievalResult

if TYPE_CHECKING:
    from cortex.db.models.user import User
    from cortex.retrieval.base import Retriever

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_MAX_TOP_K = 20


class RAGSearchTool(Tool):
    """Search the user's documents using semantic retrieval.

    Wraps :class:`~cortex.retrieval.base.Retriever` (concretely
    :class:`~cortex.retrieval.semantic.SemanticRetriever`), which remains
    the single authoritative retrieval component.

    After calling ``execute``, callers should read ``last_results`` to
    obtain the structured :class:`~cortex.retrieval.models.RetrievalResult`
    list for citation accumulation.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._last_results: list[RetrievalResult] = []

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search the user's uploaded documents for information relevant to "
            "a query.  Use this tool when the question requires facts or details "
            "from documents.  Returns excerpts from the most semantically "
            "similar document chunks."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        f"Number of chunks to retrieve (1–{_MAX_TOP_K}, "
                        f"default {_DEFAULT_TOP_K})"
                    ),
                    "minimum": 1,
                    "maximum": _MAX_TOP_K,
                },
                "document_id": {
                    "type": "string",
                    "description": (
                        "Optional UUID to restrict search to a single document"
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, args: dict[str, Any], user: User) -> str:
        """Run semantic retrieval and return formatted chunk observations.

        Side-effect: stores :class:`~cortex.retrieval.models.RetrievalResult`
        objects in ``last_results`` for the agent service to accumulate.
        """
        query: str = args.get("query", "").strip()
        if not query:
            self._last_results = []
            return "Error: 'query' argument is required and must not be empty."

        top_k: int = min(
            max(int(args.get("top_k", _DEFAULT_TOP_K)), 1),
            _MAX_TOP_K,
        )
        document_id: str | None = args.get("document_id")

        results: list[RetrievalResult] = await self._retriever.retrieve(
            query=query,
            user=user,
            top_k=top_k,
            document_id=document_id,
        )
        self._last_results = results

        if not results:
            logger.info(
                "RAGSearchTool: no results for query=%r user_id=%s",
                query[:80],
                user.id,
            )
            return (
                f"No document excerpts found for the query: {query!r}. "
                "The user may not have uploaded relevant documents."
            )

        lines: list[str] = [
            f"Found {len(results)} relevant excerpt(s) for query: {query!r}\n"
        ]
        for i, chunk in enumerate(results, start=1):
            lines.append(
                f"[Excerpt {i} | document_id={chunk.document_id} "
                f"chunk_index={chunk.chunk_index} score={chunk.score:.3f}]\n"
                f"{chunk.text.strip()}"
            )
        observation = "\n\n".join(lines)

        logger.info(
            "RAGSearchTool: query=%r user_id=%s results=%d",
            query[:80],
            user.id,
            len(results),
        )
        return observation

    # ------------------------------------------------------------------
    # Result accessor (used by AgentService to accumulate citations)
    # ------------------------------------------------------------------

    @property
    def last_results(self) -> list[RetrievalResult]:
        """Most recently retrieved RetrievalResult list."""
        return list(self._last_results)

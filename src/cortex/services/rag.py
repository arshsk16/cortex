"""RAG (Retrieval-Augmented Generation) orchestration service."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.llm.base import LLMProvider
from cortex.retrieval.base import Retriever
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.db.models.conversation import Message
    from cortex.db.models.user import User
    from cortex.services.conversation import ConversationService

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrate retrieval and generation to answer user questions.

    Responsibilities (in order):

    1. Validate the question is non-empty.
    2. Optionally load conversation history from :class:`ConversationService`.
    3. Call :class:`~cortex.retrieval.base.Retriever` to fetch ranked chunks.
    4. Call :class:`~cortex.services.prompt_builder.PromptBuilder` to assemble
       the prompt (system + optional history + context + question).
    5. Call :class:`~cortex.llm.base.LLMProvider` to generate the answer.
    6. Optionally store user/assistant messages via :class:`ConversationService`.
    7. Return the answer plus citations derived from retrieved chunks.

    **Backward compatibility:** ``conversation_service`` and
    ``conversation_history_limit`` are optional.  When ``conversation_service``
    is ``None`` (Phase 6 RAG endpoint), the service operates exactly as before —
    stateless, no DB writes.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        conversation_history_limit: int = 10,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_provider
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._history_limit = conversation_history_limit

    async def answer(
        self,
        *,
        question: str,
        user: User,
        top_k: int = 5,
        document_id: str | None = None,
        conversation_id: str | None = None,
    ) -> RAGResult:
        """Answer ``question`` using retrieved document context.

        Parameters
        ----------
        question:
            Natural-language question from the user.
        user:
            Authenticated owner; retrieval is scoped to this user's documents.
        top_k:
            Maximum number of chunks to retrieve and include in the prompt.
        document_id:
            Optional filter to restrict retrieval to a single owned document.
        conversation_id:
            When set (and ``conversation_service`` is present), load history,
            store user + assistant messages, and record token usage.
        """
        question = question.strip()
        if not question:
            raise BadRequestError(
                "Question must not be empty",
                details={"field": "question"},
            )

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1: Load conversation history (optional)
        # ------------------------------------------------------------------
        history: list[Message] = []
        if conversation_id and self._conv_service:
            history = await self._conv_service.get_history(
                conversation_id=conversation_id,
                user_id=user.id,
                limit=self._history_limit,
            )

        # ------------------------------------------------------------------
        # Step 2: Persist user message (before retrieval so ordering is stable)
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            # Auto-title the conversation from the very first user message
            await self._conv_service.set_auto_title_if_needed(
                conversation_id=conversation_id,
                first_message=question,
            )

        # ------------------------------------------------------------------
        # Step 3: Semantic retrieval
        # ------------------------------------------------------------------
        t_retrieval = time.perf_counter()
        chunks: list[RetrievalResult] = await self._retriever.retrieve(
            query=question,
            user=user,
            top_k=top_k,
            document_id=document_id,
        )
        retrieval_latency_ms = (time.perf_counter() - t_retrieval) * 1000

        # ------------------------------------------------------------------
        # Step 4: Build structured prompt
        # ------------------------------------------------------------------
        prompt = self._prompt_builder.build(
            question=question,
            retrieved_chunks=chunks,
            history=history if history else None,
        )

        # ------------------------------------------------------------------
        # Step 5: Generate answer
        # ------------------------------------------------------------------
        t_llm = time.perf_counter()
        try:
            answer_text = await self._llm.generate(prompt)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError(
                "LLM generation failed unexpectedly",
                details={"reason": str(exc)},
            ) from exc
        llm_latency_ms = (time.perf_counter() - t_llm) * 1000

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        # ------------------------------------------------------------------
        # Step 6: Structured request logging
        # ------------------------------------------------------------------
        logger.info(
            "RAG complete: user_id=%s conversation_id=%s chunks=%d "
            "retrieval_ms=%.1f llm_ms=%.1f total_ms=%.1f answer_len=%d",
            user.id,
            conversation_id,
            len(chunks),
            retrieval_latency_ms,
            llm_latency_ms,
            total_latency_ms,
            len(answer_text),
        )

        # ------------------------------------------------------------------
        # Step 7: Persist assistant message + token usage (optional)
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            citation_dicts = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ]
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer_text,
                citations=citation_dicts or None,
            )

            # Estimate token usage: ~4 chars per token (rough heuristic)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(answer_text) // 4
            await self._conv_service.record_token_usage(
                conversation_id=conversation_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return RAGResult(
            answer=answer_text,
            retrieved_chunks=chunks,
        )


class RAGResult:
    """Result returned by :meth:`RAGService.answer`.

    Attributes
    ----------
    answer:
        The model's text response.
    retrieved_chunks:
        The ranked chunks used to build the prompt; used by the endpoint to
        build citation metadata.
    """

    __slots__ = ("answer", "retrieved_chunks")

    def __init__(
        self,
        *,
        answer: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> None:
        self.answer = answer
        self.retrieved_chunks = retrieved_chunks

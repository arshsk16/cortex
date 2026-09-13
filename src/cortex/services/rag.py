"""RAG (Retrieval-Augmented Generation) orchestration service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.llm.base import LLMProvider
from cortex.retrieval.base import Retriever
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.db.models.user import User

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrate retrieval and generation to answer user questions.

    Responsibilities (in order):

    1. Validate the question is non-empty.
    2. Call :class:`~cortex.retrieval.base.Retriever` to fetch ranked chunks.
    3. Call :class:`~cortex.services.prompt_builder.PromptBuilder` to assemble
       the prompt (system instructions + context + question).
    4. Call :class:`~cortex.llm.base.LLMProvider` to generate the answer.
    5. Return the answer plus citations derived from retrieved chunks.

    This service contains *no* LLM-specific code, *no* embedding logic, and
    *no* SQL — it delegates all of those concerns to its collaborators.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_provider
        self._prompt_builder = prompt_builder

    async def answer(
        self,
        *,
        question: str,
        user: User,
        top_k: int = 5,
        document_id: str | None = None,
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

        Returns
        -------
        RAGResult
            Dataclass containing the answer text and citation metadata.

        Raises
        ------
        BadRequestError
            If ``question`` is empty after stripping whitespace.
        ServiceUnavailableError
            If the LLM provider fails.
        NotFoundError / BadRequestError
            Propagated from the retriever when the document is not found or
            not yet processed.
        """
        question = question.strip()
        if not question:
            raise BadRequestError(
                "Question must not be empty",
                details={"field": "question"},
            )

        # Step 1: Retrieve relevant chunks (ownership enforced inside retriever)
        chunks: list[RetrievalResult] = await self._retriever.retrieve(
            query=question,
            user=user,
            top_k=top_k,
            document_id=document_id,
        )

        logger.info(
            "RAG retrieval complete: question=%r user_id=%s chunks=%d",
            question[:80],
            user.id,
            len(chunks),
        )

        # Step 2: Build structured prompt
        prompt = self._prompt_builder.build(
            question=question,
            retrieved_chunks=chunks,
        )

        # Step 3: Generate answer
        try:
            answer_text = await self._llm.generate(prompt)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError(
                "LLM generation failed unexpectedly",
                details={"reason": str(exc)},
            ) from exc

        logger.info(
            "RAG answer generated: user_id=%s answer_len=%d citations=%d",
            user.id,
            len(answer_text),
            len(chunks),
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

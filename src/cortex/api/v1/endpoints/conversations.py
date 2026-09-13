"""Conversation CRUD and chat endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from cortex.api.deps import (
    ConversationRAGServiceDep,
    ConversationServiceDep,
    CurrentActiveUserDep,
    SessionDep,
)
from cortex.core.exceptions import ServiceUnavailableError
from cortex.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    CitationRead,
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationRead,
    ConversationRename,
    MessageRead,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    payload: ConversationCreate,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    session: SessionDep,
) -> ConversationRead:
    """Create a new conversation thread owned by the authenticated user."""
    conv = await conv_service.create_conversation(
        user_id=current_user.id,
        title=payload.title,
    )
    await session.commit()
    return ConversationRead.model_validate(conv)


@router.get(
    "",
    response_model=ConversationList,
    status_code=status.HTTP_200_OK,
    summary="List all conversations for the current user",
)
async def list_conversations(
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    limit: int = 50,
    offset: int = 0,
) -> ConversationList:
    """Return conversations ordered newest-first."""
    convs = await conv_service.list_conversations(
        user_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    return ConversationList(
        conversations=[ConversationRead.model_validate(c) for c in convs],
        total=len(convs),
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    status_code=status.HTTP_200_OK,
    summary="Get a conversation with its full message history",
)
async def get_conversation(
    conversation_id: str,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
) -> ConversationDetail:
    """Return a conversation and all its messages (ownership enforced)."""
    conv = await conv_service.get_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    messages = await conv_service.get_history(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    return ConversationDetail(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[MessageRead.model_validate(m) for m in messages],
    )


@router.patch(
    "/{conversation_id}",
    response_model=ConversationRead,
    status_code=status.HTTP_200_OK,
    summary="Rename a conversation",
)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationRename,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    session: SessionDep,
) -> ConversationRead:
    """Update the title of an owned conversation."""
    conv = await conv_service.rename_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        title=payload.title,
    )
    await session.commit()
    return ConversationRead.model_validate(conv)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: str,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    session: SessionDep,
) -> None:
    """Delete an owned conversation and cascade all its messages."""
    await conv_service.delete_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    await session.commit()


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageRead],
    status_code=status.HTTP_200_OK,
    summary="List messages in a conversation",
)
async def list_messages(
    conversation_id: str,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    limit: int | None = None,
) -> list[MessageRead]:
    """Return all messages in chronological order."""
    messages = await conv_service.get_history(
        conversation_id=conversation_id,
        user_id=current_user.id,
        limit=limit,
    )
    return [MessageRead.model_validate(m) for m in messages]


# ---------------------------------------------------------------------------
# Chat — multi-turn RAG
# ---------------------------------------------------------------------------


@router.post(
    "/{conversation_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and get a grounded AI response",
    description=(
        "Loads conversation history, retrieves document context, builds a "
        "structured prompt, generates an answer with Gemini, and persists "
        "both the user message and assistant response.  Ownership is enforced."
    ),
    responses={
        200: {"description": "AI answer with citations"},
        400: {"description": "Empty message or document not ready"},
        401: {"description": "Not authenticated"},
        403: {"description": "Conversation not owned by user"},
        404: {"description": "Conversation or document not found"},
        503: {"description": "LLM generation failed"},
    },
)
async def chat(
    conversation_id: str,
    payload: ChatRequest,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    rag_service: ConversationRAGServiceDep,
    session: SessionDep,
) -> ChatResponse:
    """Multi-turn chat — runs the full conversation-aware RAG pipeline."""
    # Verify the conversation exists and is owned (raises 403/404 otherwise)
    await conv_service.get_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )

    result = await rag_service.answer(
        question=payload.message,
        user=current_user,
        top_k=payload.top_k,
        document_id=payload.document_id,
        conversation_id=conversation_id,
    )

    await session.commit()

    # Build richer citations (document_name populated where chunk has it)
    citations = [
        CitationRead(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            chunk_index=c.chunk_index,
            document_name=getattr(c, "document_name", None),
        )
        for c in result.retrieved_chunks
    ]

    # Fetch the just-created assistant message to get its id
    history = await conv_service.get_history(
        conversation_id=conversation_id,
        user_id=current_user.id,
    )
    last_msg = history[-1] if history else None

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=last_msg.id if last_msg else "",
        answer=result.answer,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Streaming chat — SSE
# ---------------------------------------------------------------------------


@router.post(
    "/{conversation_id}/stream",
    status_code=status.HTTP_200_OK,
    summary="Streaming chat via Server-Sent Events",
    description=(
        "Runs the same RAG pipeline as /chat but streams the LLM response "
        "token-by-token using Server-Sent Events.  Connect with EventSource. "
        "Each event has data: <chunk>.  A final event data: [DONE] is emitted."
    ),
    response_class=StreamingResponse,
)
async def stream_chat(
    conversation_id: str,
    payload: ChatRequest,
    current_user: CurrentActiveUserDep,
    conv_service: ConversationServiceDep,
    rag_service: ConversationRAGServiceDep,
    session: SessionDep,
) -> StreamingResponse:
    """Multi-turn streaming chat using Server-Sent Events."""

    async def _event_generator() -> AsyncIterator[str]:
        """Run retrieval + prompt building first, then stream generation."""
        import time

        from cortex.retrieval.models import RetrievalResult

        # Ownership check
        await conv_service.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.id,
        )

        question = payload.message.strip()
        if not question:
            yield "event: error\ndata: Question must not be empty\n\n"
            return

        t_start = time.perf_counter()

        # Load history
        history = await conv_service.get_history(
            conversation_id=conversation_id,
            user_id=current_user.id,
            limit=rag_service._history_limit,
        )

        # Persist user message + auto-title
        await conv_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        await conv_service.set_auto_title_if_needed(
            conversation_id=conversation_id,
            first_message=question,
        )

        # Retrieval
        chunks: list[RetrievalResult] = await rag_service._retriever.retrieve(
            query=question,
            user=current_user,
            top_k=payload.top_k,
            document_id=payload.document_id,
        )

        # Build prompt
        prompt = rag_service._prompt_builder.build(
            question=question,
            retrieved_chunks=chunks,
            history=history if history else None,
        )

        # Stream generation
        full_answer_parts: list[str] = []
        try:
            stream = await rag_service._llm.generate_stream(prompt)
            async for token in stream:
                if token:
                    full_answer_parts.append(token)
                    # SSE format: "data: <payload>\n\n"
                    yield f"data: {token}\n\n"
                    await asyncio.sleep(0)  # yield control to event loop
        except ServiceUnavailableError as exc:
            yield f"event: error\ndata: {exc}\n\n"
            return

        full_answer = "".join(full_answer_parts)

        # Persist assistant message
        citation_dicts = [
            {
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        await conv_service.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_answer,
            citations=citation_dicts or None,
        )

        # Record token usage
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(full_answer) // 4
        await conv_service.record_token_usage(
            conversation_id=conversation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        await session.commit()

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Streaming RAG complete: user_id=%s conversation_id=%s "
            "chunks=%d total_ms=%.1f answer_len=%d",
            current_user.id,
            conversation_id,
            len(chunks),
            total_ms,
            len(full_answer),
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

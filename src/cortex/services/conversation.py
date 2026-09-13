"""Conversation management service — CRUD + history."""

from __future__ import annotations

import logging
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.exceptions import ForbiddenError, NotFoundError
from cortex.db.models.conversation import Conversation, Message, TokenUsage

logger = logging.getLogger(__name__)

_MAX_TITLE_LEN = 60


def _auto_title(first_message: str) -> str:
    """Derive a conversation title from the first user message.

    Trims whitespace and truncates to 60 characters.  No LLM call required.
    """
    stripped = first_message.strip()
    if len(stripped) <= _MAX_TITLE_LEN:
        return stripped
    return stripped[:_MAX_TITLE_LEN].rstrip()


class ConversationService:
    """Business logic for conversation threads.

    All public methods accept an explicit ``user_id`` parameter and enforce
    ownership — no HTTP-layer logic is present.

    The session is injected per-request from the FastAPI DI graph following
    the same pattern used by ``DocumentService``.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Conversation CRUD
    # ------------------------------------------------------------------

    async def create_conversation(
        self,
        *,
        user_id: str,
        title: str | None = None,
    ) -> Conversation:
        """Create and persist a new conversation.

        Parameters
        ----------
        user_id:
            Owner of the conversation.
        title:
            Optional explicit title.  If omitted, defaults to
            ``"New Conversation"``; callers may update this after the first
            message via :meth:`rename_conversation` or rely on
            :meth:`_set_auto_title_if_needed`.
        """
        conv = Conversation(
            id=str(uuid4()),
            user_id=user_id,
            title=title or "New Conversation",
        )
        self._session.add(conv)
        await self._session.flush()
        await self._session.refresh(conv)
        logger.info("Created conversation id=%s user_id=%s", conv.id, user_id)
        return conv

    async def get_conversation(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> Conversation:
        """Return the conversation or raise NotFoundError / ForbiddenError."""
        conv = await self._session.get(Conversation, conversation_id)
        if conv is None:
            raise NotFoundError(
                "Conversation not found",
                details={"conversation_id": conversation_id},
            )
        if conv.user_id != user_id:
            raise ForbiddenError("You do not own this conversation")
        return conv

    async def list_conversations(
        self,
        *,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """Return the user's conversations ordered newest-first."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def rename_conversation(
        self,
        *,
        conversation_id: str,
        user_id: str,
        title: str,
    ) -> Conversation:
        """Set a new title on an owned conversation."""
        conv = await self.get_conversation(
            conversation_id=conversation_id, user_id=user_id
        )
        conv.title = title.strip()[:255]
        await self._session.flush()
        return conv

    async def delete_conversation(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> None:
        """Delete an owned conversation and cascade its messages/token usage."""
        conv = await self.get_conversation(
            conversation_id=conversation_id, user_id=user_id
        )
        await self._session.delete(conv)
        await self._session.flush()
        logger.info(
            "Deleted conversation id=%s user_id=%s", conversation_id, user_id
        )

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    async def get_history(
        self,
        *,
        conversation_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Return messages in chronological order.

        Parameters
        ----------
        conversation_id:
            Conversation to load history from.
        user_id:
            Enforces ownership.
        limit:
            If set, only the most recent *limit* messages are returned.
        """
        # Ownership check
        await self.get_conversation(
            conversation_id=conversation_id, user_id=user_id
        )

        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        if limit is not None:
            # Load the last N messages: subquery for the newest N, then re-order
            sub = (
                select(Message.id)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .subquery()
            )
            stmt = (
                select(Message)
                .where(Message.id.in_(select(sub)))
                .order_by(Message.created_at.asc())
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> Message:
        """Persist a new message and update the conversation's updated_at.

        Does NOT enforce ownership — caller is responsible for having verified
        ownership before reaching this method.
        """
        msg = Message(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
        )
        self._session.add(msg)

        # Touch updated_at on the conversation
        conv = await self._session.get(Conversation, conversation_id)
        if conv is not None:
            from datetime import UTC, datetime

            conv.updated_at = datetime.now(UTC)

        await self._session.flush()
        return msg

    # ------------------------------------------------------------------
    # Token usage
    # ------------------------------------------------------------------

    async def record_token_usage(
        self,
        *,
        conversation_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float = 0.0,
    ) -> TokenUsage:
        """Persist a token usage record for a single generation request."""
        usage = TokenUsage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
        )
        self._session.add(usage)
        await self._session.flush()
        return usage

    # ------------------------------------------------------------------
    # Auto-title helper
    # ------------------------------------------------------------------

    async def set_auto_title_if_needed(
        self,
        *,
        conversation_id: str,
        first_message: str,
    ) -> None:
        """Set an auto-generated title from the first message if still default."""
        conv = await self._session.get(Conversation, conversation_id)
        if conv is None:
            return
        if conv.title == "New Conversation":
            conv.title = _auto_title(first_message)
            await self._session.flush()
            logger.debug(
                "Auto-titled conversation id=%s title=%r",
                conversation_id,
                conv.title,
            )

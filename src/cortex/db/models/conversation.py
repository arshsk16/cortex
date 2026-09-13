"""Conversation, Message, and TokenUsage ORM models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cortex.db.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from cortex.db.models.user import User


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A named conversation thread owned by a single user."""

    __tablename__ = "conversations"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Conversation",
        server_default="New Conversation",
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    token_usages: Mapped[list[TokenUsage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation id={self.id!r} user_id={self.user_id!r} "
            f"title={self.title!r}>"
        )


class Message(UUIDPrimaryKeyMixin, Base):
    """A single immutable turn in a conversation (user or assistant)."""

    __tablename__ = "messages"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="'user' or 'assistant'",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSONB array of citation objects; null for user-role messages
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True, default=None
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id!r} conversation_id={self.conversation_id!r} "
            f"role={self.role!r}>"
        )


class TokenUsage(UUIDPrimaryKeyMixin, Base):
    """Per-request LLM token consumption — provider-independent."""

    __tablename__ = "token_usages"

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    conversation: Mapped[Conversation] = relationship(back_populates="token_usages")

    def __repr__(self) -> str:
        return (
            f"<TokenUsage id={self.id!r} "
            f"conversation_id={self.conversation_id!r} "
            f"total_tokens={self.total_tokens!r}>"
        )

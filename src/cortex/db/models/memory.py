"""Memory ORM model -- durable long-term semantic memory records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cortex.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from cortex.db.models.user import User


class Memory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A user-owned long-term memory entry.

    PostgreSQL is the durable source of truth for every memory record.
    Chroma holds only the embedding vector and a ``memory_id`` reference
    used to look up the full content after a similarity search.
    """

    __tablename__ = "memories"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Arbitrary caller-supplied JSON metadata (tags, source, etc.)
    memory_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "memory_metadata",
        JSONB,
        nullable=True,
        default=None,
    )

    user: Mapped[User] = relationship(back_populates="memories")

    def __repr__(self) -> str:
        return (
            f"<Memory id={self.id!r} user_id={self.user_id!r} "
            f"content={self.content[:40]!r}>"
        )

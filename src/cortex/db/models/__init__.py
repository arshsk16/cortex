"""ORM model registry.

Import every model module here so Alembic and ``Base.metadata`` discover tables.
"""

from cortex.db.base import Base
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.document_chunk import DocumentChunk
from cortex.db.models.user import User, UserRole

__all__ = [
    "Base",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "User",
    "UserRole",
]

"""NullStateStore — silent no-op for local dev and testing without Redis."""

from __future__ import annotations

import logging
from typing import Any

from cortex.state_store.base import StateStore

logger = logging.getLogger(__name__)


class NullStateStore(StateStore):
    """No-op ``StateStore`` used when Redis is not configured.

    Every method succeeds silently so the application behaves identically
    whether Redis is present or not.  A one-time info log is emitted at
    construction time so operators know state is not being persisted.
    """

    def __init__(self) -> None:
        logger.info(
            "StateStore: NullStateStore active — agent state will not be "
            "persisted to Redis. Set REDIS_URL to enable."
        )

    async def save(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """No-op save."""

    async def load(self, key: str) -> dict[str, Any] | None:
        """Always returns None — no state is stored."""
        return None

    async def delete(self, key: str) -> None:
        """No-op delete."""

    async def ping(self) -> bool:
        """Always reachable (nothing to reach)."""
        return True

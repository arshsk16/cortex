"""State-store abstraction — provider-neutral interface for ephemeral agent state.

The ``StateStore`` ABC decouples ``AgentService`` from any specific backend
(Redis, Memcached, in-process dict, …).  Only the interface is used inside
agent code; the concrete implementation is injected via FastAPI dependency
injection.

Implementations
---------------
``RedisStateStore``
    Production implementation backed by ``redis.asyncio``.
``NullStateStore``
    No-op implementation used when Redis is disabled (``REDIS_URL`` unset)
    or in local/test environments.  All operations succeed silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateStore(ABC):
    """Ephemeral key-value store for agent execution state snapshots.

    Design contract
    ---------------
    * All methods are async and must not block the event loop.
    * Values are plain ``dict`` objects (JSON-serialisable).
    * Implementations MUST NOT raise on backend failure; they should log
      the error and return a safe default (``None`` for :meth:`load`,
      ``False`` for :meth:`ping`).  The caller should not need to wrap
      state-store calls in try/except.
    * Keys are caller-constructed and already namespaced; the store does
      not add additional prefixes.
    """

    @abstractmethod
    async def save(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """Persist ``value`` under ``key`` with an expiry of ``ttl_seconds``.

        Overwrites any existing value.  The key is automatically removed by
        the backend when ``ttl_seconds`` elapses.

        Parameters
        ----------
        key:
            Namespaced state key (e.g. ``cortex:agent:run:{uid}:{cid}:{rid}``).
        value:
            JSON-serialisable dictionary to store.
        ttl_seconds:
            Time-to-live in seconds.  Must be > 0.
        """

    @abstractmethod
    async def load(self, key: str) -> dict[str, Any] | None:
        """Return the stored value, or ``None`` if the key does not exist.

        Parameters
        ----------
        key:
            The same key previously passed to :meth:`save`.

        Returns
        -------
        dict | None
            Deserialised dictionary, or ``None`` if not found / expired.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove ``key`` immediately regardless of its TTL.

        Used on successful run completion so the slot is freed without
        waiting for natural expiry.  Safe to call on a non-existent key.
        """

    @abstractmethod
    async def ping(self) -> bool:
        """Return ``True`` if the backend is reachable, ``False`` otherwise."""

"""Agent execution state-store package.

Provides a provider-neutral ``StateStore`` abstraction for ephemeral agent
execution state, with two concrete implementations:

* :class:`~cortex.state_store.null.NullStateStore` — silent no-op (default
  when ``REDIS_URL`` is not configured).
* :class:`~cortex.state_store.redis_store.RedisStateStore` — production
  implementation backed by ``redis.asyncio``.
"""

from cortex.state_store.base import StateStore
from cortex.state_store.null import NullStateStore
from cortex.state_store.redis_store import RedisStateStore

__all__ = [
    "NullStateStore",
    "RedisStateStore",
    "StateStore",
]

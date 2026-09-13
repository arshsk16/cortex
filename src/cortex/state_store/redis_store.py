"""RedisStateStore — production StateStore backed by redis.asyncio.

Key features
------------
* Async, non-blocking: uses ``redis.asyncio.Redis`` throughout.
* All backend errors are caught and logged at WARNING level; no exception
  is ever propagated to the caller.
* Serialisation: plain ``json`` (stdlib) — no pickle, no msgpack.
* Connection is managed externally (passed in at construction) so the
  application lifespan owns the lifecycle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis

from cortex.state_store.base import StateStore

logger = logging.getLogger(__name__)

# Maximum JSON payload size stored in Redis (bytes).
# Payloads exceeding this are logged and NOT stored (avoids Redis OOM).
_MAX_PAYLOAD_BYTES = 256 * 1024  # 256 KiB


class RedisStateStore(StateStore):
    """``StateStore`` implementation backed by a ``redis.asyncio.Redis`` client.

    Parameters
    ----------
    client:
        A connected (or lazy-connected) ``redis.asyncio.Redis`` instance.
        The caller is responsible for opening and closing the connection via
        the application lifespan.
    """

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def save(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """Serialise ``value`` to JSON and store under ``key`` with TTL."""
        try:
            payload = json.dumps(value, default=str)
            payload_bytes = len(payload.encode())
            if payload_bytes > _MAX_PAYLOAD_BYTES:
                logger.warning(
                    "StateStore: payload for key=%r exceeds %d bytes (%d bytes); "
                    "skipping save to Redis.",
                    key,
                    _MAX_PAYLOAD_BYTES,
                    payload_bytes,
                )
                return
            await self._client.setex(key, ttl_seconds, payload)
            logger.debug("StateStore: saved key=%r ttl=%ds", key, ttl_seconds)
        except Exception:
            logger.warning("StateStore: failed to save key=%r", key, exc_info=True)

    async def load(self, key: str) -> dict[str, Any] | None:
        """Fetch and deserialise the stored dict, or return ``None``."""
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning(
                    "StateStore: unexpected type %s for key=%r; discarding.",
                    type(data).__name__,
                    key,
                )
                return None
            logger.debug("StateStore: loaded key=%r", key)
            return data
        except json.JSONDecodeError:
            logger.warning(
                "StateStore: corrupt JSON for key=%r; discarding.", key, exc_info=True
            )
            return None
        except Exception:
            logger.warning("StateStore: failed to load key=%r", key, exc_info=True)
            return None

    async def delete(self, key: str) -> None:
        """Remove ``key`` from Redis immediately."""
        try:
            await self._client.delete(key)
            logger.debug("StateStore: deleted key=%r", key)
        except Exception:
            logger.warning("StateStore: failed to delete key=%r", key, exc_info=True)

    async def ping(self) -> bool:
        """Return ``True`` if Redis responds to PING."""
        try:
            return bool(await self._client.ping())
        except Exception:
            logger.warning("StateStore: ping failed", exc_info=True)
            return False

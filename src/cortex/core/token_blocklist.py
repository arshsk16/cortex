"""JWT token blocklist backed by Redis for revocation support.

Design
------
* When a user logs out, the token's ``jti`` is stored in Redis with a TTL
  equal to the token's remaining lifetime.  Once the token would have
  expired naturally, the blocklist entry also expires — no manual cleanup.
* ``get_current_user`` in ``deps.py`` checks this store after decoding the
  token to reject revoked tokens.
* If Redis is unavailable, :meth:`TokenBlocklist.revoke` logs a warning and
  silently fails (the token remains valid until its natural expiry).
  :meth:`TokenBlocklist.is_revoked` returns ``False`` (fail-open) so that a
  Redis outage does not lock out all authenticated users.

This matches the existing Redis fallback strategy used by ``RedisStateStore``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Redis key prefix for revoked JTIs.
_KEY_PREFIX = "cortex:blocklist:jti:"


class TokenBlocklist:
    """Per-jti revocation store backed by a ``redis.asyncio.Redis`` client.

    Parameters
    ----------
    client:
        A connected (or lazy-connected) ``redis.asyncio.Redis`` instance.
        Lifecycle management (connect / close) is the caller's responsibility.
    """

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def revoke(self, jti: str, ttl_seconds: int) -> None:
        """Mark *jti* as revoked; the entry expires after *ttl_seconds*.

        A ``ttl_seconds`` value of 0 or negative means the token is already
        expired — no action is taken (the token is harmless regardless).

        On Redis failure, logs at WARNING and returns without raising.
        """
        if ttl_seconds <= 0:
            logger.debug("TokenBlocklist: jti=%r already expired — skip revoke", jti)
            return
        key = _KEY_PREFIX + jti
        try:
            await self._client.setex(key, ttl_seconds, "1")
            logger.info("TokenBlocklist: revoked jti=%r ttl=%ds", jti, ttl_seconds)
        except Exception:
            logger.warning(
                "TokenBlocklist: failed to revoke jti=%r (Redis error)",
                jti,
                exc_info=True,
            )

    async def is_revoked(self, jti: str) -> bool:
        """Return ``True`` if *jti* has been revoked.

        Returns ``False`` on Redis failure (fail-open) to avoid locking out
        users during a Redis outage.
        """
        key = _KEY_PREFIX + jti
        try:
            result = await self._client.exists(key)
            return bool(result)
        except Exception:
            logger.warning(
                "TokenBlocklist: failed to check jti=%r"
                " (Redis error) — treating as not revoked",
                jti,
                exc_info=True,
            )
            return False

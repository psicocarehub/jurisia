"""
Redis cache service with JSON serialization and TTL support.

Usage:
    cache = CacheService()
    await cache.connect()

    await cache.set("key", {"data": 1}, ttl=3600)
    result = await cache.get("key")  # {"data": 1} or None
    await cache.delete("key")
    await cache.invalidate_pattern("case_analysis:*")
"""

import json
import logging
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_redis_client = None


async def get_redis():
    """Get or create a shared async Redis connection."""
    global _redis_client
    if _redis_client is None:
        try:
            from redis.asyncio import from_url
            _redis_client = from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            await _redis_client.ping()
            logger.info("Redis connected: %s", settings.REDIS_URL)
        except Exception as e:
            logger.warning("Redis unavailable, caching disabled: %s", e)
            _redis_client = None
    return _redis_client


async def close_redis():
    """Close the Redis connection (call on shutdown)."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


class CacheService:
    """Async Redis cache with namespace prefixes and JSON ser/de."""

    DEFAULT_TTL = 3600  # 1 hour

    # TTL presets for different data types
    TTL_ANALYSIS = 3600       # 1h — case analyses
    TTL_JUDGE_PROFILE = 86400 # 24h — judge profiles (slow-changing)
    TTL_RAG = 1800            # 30min — RAG results
    TTL_PREDICTION = 7200     # 2h — outcome predictions

    def __init__(self, prefix: str = "jurisai") -> None:
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None on miss or if Redis is unavailable."""
        client = await get_redis()
        if not client:
            return None
        try:
            raw = await client.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug("Cache get failed for %s: %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """Set a cached value with TTL. Returns False if Redis is unavailable."""
        client = await get_redis()
        if not client:
            return False
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            await client.setex(self._key(key), ttl, serialized)
            return True
        except Exception as e:
            logger.debug("Cache set failed for %s: %s", key, e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cached key."""
        client = await get_redis()
        if not client:
            return False
        try:
            await client.delete(self._key(key))
            return True
        except Exception as e:
            logger.debug("Cache delete failed for %s: %s", key, e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern (e.g. 'case_analysis:*')."""
        client = await get_redis()
        if not client:
            return 0
        try:
            full_pattern = self._key(pattern)
            keys = []
            async for key in client.scan_iter(match=full_pattern, count=100):
                keys.append(key)
            if keys:
                await client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.debug("Cache invalidate failed for %s: %s", pattern, e)
            return 0

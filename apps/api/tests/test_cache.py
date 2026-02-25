"""Tests for CacheService (mocked Redis)."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.cache import CacheService


@pytest.fixture
def cache():
    return CacheService(prefix="test")


@pytest.mark.asyncio
async def test_get_returns_none_when_redis_unavailable(cache):
    with patch("app.services.cache.get_redis", return_value=None):
        result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_set_returns_false_when_redis_unavailable(cache):
    with patch("app.services.cache.get_redis", return_value=None):
        result = await cache.set("key", {"v": 1})
    assert result is False


@pytest.mark.asyncio
async def test_get_and_set_with_mock_redis(cache):
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value='{"value": 42}')
    mock_redis.setex = AsyncMock()

    with patch("app.services.cache.get_redis", return_value=mock_redis):
        await cache.set("key", {"value": 42}, ttl=60)
        mock_redis.setex.assert_called_once()

        result = await cache.get("key")
        assert result == {"value": 42}


@pytest.mark.asyncio
async def test_delete_with_mock_redis(cache):
    mock_redis = AsyncMock()
    mock_redis.delete = AsyncMock()

    with patch("app.services.cache.get_redis", return_value=mock_redis):
        result = await cache.delete("key")
    assert result is True
    mock_redis.delete.assert_called_once_with("test:key")


@pytest.mark.asyncio
async def test_invalidate_pattern(cache):
    mock_redis = AsyncMock()

    async def mock_scan_iter(match=None, count=None):
        for key in ["test:a:1", "test:a:2"]:
            yield key

    mock_redis.scan_iter = mock_scan_iter
    mock_redis.delete = AsyncMock()

    with patch("app.services.cache.get_redis", return_value=mock_redis):
        count = await cache.invalidate_pattern("a:*")
    assert count == 2


def test_key_prefix(cache):
    assert cache._key("foo:bar") == "test:foo:bar"


def test_ttl_constants():
    assert CacheService.TTL_ANALYSIS == 3600
    assert CacheService.TTL_JUDGE_PROFILE == 86400
    assert CacheService.TTL_RAG == 1800

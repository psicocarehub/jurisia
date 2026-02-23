"""
Integration test script — validates complete flow without running the server.

Tests: Auth tokens, Supabase connectivity, ES/Qdrant search, data integrity.

Usage:
    python scripts/integration_test.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
from jose import jwt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("integration")

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "apps" / "api"))
from app.config import settings

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = {"pass": 0, "fail": 0}


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        results["pass"] += 1
        logger.info("%s %s %s", PASS, name, detail)
    else:
        results["fail"] += 1
        logger.error("%s %s %s", FAIL, name, detail)


async def test_supabase_connectivity():
    """Test Supabase REST API connectivity."""
    logger.info("\n=== Supabase ===")
    headers = {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{settings.SUPABASE_URL}/rest/v1/tenants?select=id,name&limit=1",
            headers=headers,
            timeout=10.0,
        )
        check("Supabase REST API", resp.status_code == 200, f"status={resp.status_code}")

        if resp.status_code == 200:
            tenants = resp.json()
            check("Tenant exists", len(tenants) > 0, f"count={len(tenants)}")

        resp = await client.get(
            f"{settings.SUPABASE_URL}/rest/v1/users?select=id,email&limit=5",
            headers=headers,
            timeout=10.0,
        )
        check("Users table", resp.status_code == 200, f"status={resp.status_code}")
        if resp.status_code == 200:
            users = resp.json()
            check("Users exist", len(users) > 0, f"count={len(users)}")

        resp = await client.get(
            f"{settings.SUPABASE_URL}/rest/v1/judge_profiles?select=id,name&limit=5",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            judges = resp.json()
            check("Judge profiles seeded", len(judges) > 0, f"count={len(judges)}")

        resp = await client.get(
            f"{settings.SUPABASE_URL}/rest/v1/alerts?select=id,title&limit=5",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            alerts = resp.json()
            check("Alerts seeded", len(alerts) > 0, f"count={len(alerts)}")

        resp = await client.get(
            f"{settings.SUPABASE_URL}/rest/v1/law_article_versions?select=id,law_name&limit=5",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            versions = resp.json()
            check("Law versions seeded", len(versions) > 0, f"count={len(versions)}")


async def test_elasticsearch():
    """Test Elasticsearch connectivity and data."""
    logger.info("\n=== Elasticsearch ===")
    from elasticsearch import AsyncElasticsearch

    kwargs = {"hosts": [settings.ELASTICSEARCH_URL]}
    if settings.ES_API_KEY:
        kwargs["api_key"] = settings.ES_API_KEY
    es = AsyncElasticsearch(**kwargs)

    try:
        info = await es.info()
        check("ES connectivity", True, f"cluster={info.get('cluster_name', 'N/A')}")

        index_name = f"{settings.ES_INDEX_PREFIX}_chunks"
        exists = await es.indices.exists(index=index_name)
        check(f"Index '{index_name}' exists", exists)

        if exists:
            count_resp = await es.count(index=index_name)
            doc_count = count_resp.get("count", 0)
            check("Documents indexed", doc_count > 0, f"count={doc_count}")

            search_resp = await es.search(
                index=index_name,
                body={"query": {"match": {"content": "ICMS PIS COFINS"}}, "size": 3},
            )
            hits = search_resp["hits"]["total"]["value"]
            check("Search works (ICMS PIS COFINS)", hits > 0, f"hits={hits}")

            search_resp = await es.search(
                index=index_name,
                body={"query": {"match": {"content": "LGPD proteção dados"}}, "size": 3},
            )
            hits = search_resp["hits"]["total"]["value"]
            check("Search works (LGPD)", hits > 0, f"hits={hits}")

            agg_resp = await es.search(
                index=index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "by_court": {"terms": {"field": "court", "size": 10}},
                        "by_area": {"terms": {"field": "area", "size": 10}},
                    },
                },
            )
            courts = [b["key"] for b in agg_resp["aggregations"]["by_court"]["buckets"]]
            areas = [b["key"] for b in agg_resp["aggregations"]["by_area"]["buckets"]]
            check("Multiple courts indexed", len(courts) >= 2, f"courts={courts}")
            check("Multiple areas indexed", len(areas) >= 3, f"areas={areas}")

    except Exception as e:
        check("ES connectivity", False, str(e))
    finally:
        await es.close()


async def test_qdrant():
    """Test Qdrant connectivity and data."""
    logger.info("\n=== Qdrant ===")
    from qdrant_client import AsyncQdrantClient

    kwargs = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    qdrant = AsyncQdrantClient(**kwargs)

    try:
        collections = await qdrant.get_collections()
        names = [c.name for c in collections.collections]
        check("Qdrant connectivity", True, f"collections={names}")
        check(
            f"Collection '{settings.QDRANT_COLLECTION}' exists",
            settings.QDRANT_COLLECTION in names,
        )

        if settings.QDRANT_COLLECTION in names:
            info = await qdrant.get_collection(settings.QDRANT_COLLECTION)
            count = info.points_count
            check("Vectors indexed", count > 0, f"count={count}")
    except Exception as e:
        check("Qdrant connectivity", False, str(e))
    finally:
        await qdrant.close()


async def test_auth_tokens():
    """Test JWT token generation and validation."""
    logger.info("\n=== Auth ===")
    from app.core.auth import create_access_token, ALGORITHM

    token_data = {
        "sub": "test-user-id",
        "tenant_id": "test-tenant-id",
        "role": "lawyer",
        "email": "test@jurisai.com",
    }

    token = create_access_token(token_data)
    check("Token generation", bool(token), f"len={len(token)}")

    decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    check("Token decoding", decoded["sub"] == "test-user-id")
    check("Token has tenant_id", decoded["tenant_id"] == "test-tenant-id")
    check("Token has expiry", "exp" in decoded)


async def test_voyage_embeddings():
    """Test Voyage AI embedding API."""
    logger.info("\n=== Voyage AI Embeddings ===")
    if not settings.VOYAGE_API_KEY:
        check("Voyage API key configured", False)
        return

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.VOYAGE_API_KEY}"},
            json={
                "model": settings.EMBEDDING_MODEL,
                "input": ["Teste de embedding jurídico"],
                "input_type": "document",
            },
            timeout=30.0,
        )
        check("Voyage API connectivity", resp.status_code == 200, f"status={resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            check("Embedding dimensions", len(embedding) == settings.EMBEDDING_DIM, f"dim={len(embedding)}")


async def test_redis():
    """Test Redis/Upstash connectivity."""
    logger.info("\n=== Redis ===")
    if not settings.REDIS_URL:
        check("Redis URL configured", False)
        return

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await r.ping()
        check("Redis connectivity", True)

        await r.set("jurisai:test", "ok", ex=10)
        val = await r.get("jurisai:test")
        check("Redis read/write", val == "ok")
        await r.close()
    except Exception as e:
        check("Redis connectivity", False, str(e))


async def test_config():
    """Test that all critical config values are set."""
    logger.info("\n=== Configuration ===")
    check("SUPABASE_URL set", bool(settings.SUPABASE_URL))
    check("SUPABASE_ANON_KEY set", bool(settings.SUPABASE_ANON_KEY))
    check("ELASTICSEARCH_URL set", bool(settings.ELASTICSEARCH_URL))
    check("ES_API_KEY set", bool(settings.ES_API_KEY))
    check("QDRANT_URL set", bool(settings.QDRANT_URL))
    check("QDRANT_API_KEY set", bool(settings.QDRANT_API_KEY))
    check("VOYAGE_API_KEY set", bool(settings.VOYAGE_API_KEY))
    check("SECRET_KEY set", bool(settings.SECRET_KEY))
    check("DEEPSEEK_API_KEY set", bool(settings.DEEPSEEK_API_KEY))


async def main():
    logger.info("=" * 60)
    logger.info("JurisAI Integration Test")
    logger.info("=" * 60)

    await test_config()
    await test_auth_tokens()
    await test_supabase_connectivity()
    await test_elasticsearch()
    await test_qdrant()
    await test_voyage_embeddings()
    await test_redis()

    logger.info("\n" + "=" * 60)
    total = results["pass"] + results["fail"]
    logger.info(
        "Results: %d/%d passed (%d failed)",
        results["pass"],
        total,
        results["fail"],
    )
    logger.info("=" * 60)

    return results["fail"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

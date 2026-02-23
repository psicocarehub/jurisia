"""
Centralized factory for authenticated Elasticsearch and Qdrant clients.
All services should use these instead of creating clients directly.
"""

from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient

from app.config import settings


def create_es_client() -> AsyncElasticsearch:
    kwargs: dict = {"hosts": [settings.ELASTICSEARCH_URL]}
    if settings.ES_API_KEY:
        kwargs["api_key"] = settings.ES_API_KEY
    return AsyncElasticsearch(**kwargs)


def create_qdrant_client() -> AsyncQdrantClient:
    kwargs: dict = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return AsyncQdrantClient(**kwargs)

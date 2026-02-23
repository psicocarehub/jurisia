from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Juris.AI"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-to-a-secure-random-string"
    API_KEY_HEADER: str = "X-API-Key"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://jurisai:jurisai_dev@localhost:5432/jurisai"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Elasticsearch
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ES_INDEX_PREFIX: str = "jurisai"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "legal_documents"

    # LLM Providers
    GAIA_BASE_URL: str = "http://localhost:8080/v1"
    GAIA_MODEL_NAME: str = "gaia-legal-reasoning-v1"
    DEEPSEEK_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    MARITACA_API_KEY: str = ""

    # Embeddings
    VOYAGE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "voyage-law-2"
    EMBEDDING_DIM: int = 1024

    # OCR
    PADDLE_OCR_ENDPOINT: str = ""
    SURIYA_OCR_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""

    # Ingestion
    DATAJUD_API_KEY: str = ""

    # Storage
    S3_BUCKET: str = "jurisai-documents"
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3001"

    # Routing thresholds
    ROUTER_THRESHOLD_TIER1: float = 0.6
    ROUTER_THRESHOLD_TIER2: float = 0.85

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

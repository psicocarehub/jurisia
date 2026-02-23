from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Juris.AI"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-to-a-secure-random-string"
    API_KEY_HEADER: str = "X-API-Key"

    # Database (Supabase)
    DATABASE_URL: str = "postgresql+asyncpg://jurisai:jurisai_dev@localhost:5432/jurisai"
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"

    # Elasticsearch
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ES_INDEX_PREFIX: str = "jurisai"
    ES_API_KEY: str = ""

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "legal_documents"
    QDRANT_API_KEY: str = ""

    # LLM Providers
    GAIA_BASE_URL: str = ""
    GAIA_MODEL_NAME: str = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    DEEPSEEK_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    QWEN_API_KEY: str = ""
    KIMI_API_KEY: str = ""
    MINIMAX_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    XAI_API_KEY: str = ""
    MARITACA_API_KEY: str = ""

    # Embeddings & Reranking (Voyage AI)
    VOYAGE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "voyage-3-large"
    EMBEDDING_DIM: int = 1024
    RERANK_MODEL: str = "voyage-rerank-2"

    # OCR
    PADDLE_OCR_ENDPOINT: str = ""
    FAL_API_KEY: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""

    # Ingestion
    DATAJUD_API_KEY: str = ""
    JUIT_API_KEY: str = ""
    JUIT_API_URL: str = "https://api.juit.dev"
    TRANSPARENCIA_API_KEY: str = ""
    CNPJ_DATA_DIR: str = "/tmp/jurisai_data/cnpj"

    # Storage (Supabase Storage)
    S3_BUCKET: str = "documents"
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Memory
    SUPERMEMORY_API_KEY: str = ""

    # Observability
    SENTRY_DSN: str = ""
    SENTRY_AUTH_TOKEN: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3001"

    # Routing thresholds
    ROUTER_THRESHOLD_TIER1: float = 0.6
    ROUTER_THRESHOLD_TIER2: float = 0.85

    # Auth
    JWT_SECRET: str = ""
    JWT_EXPIRES_IN: str = "24h"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

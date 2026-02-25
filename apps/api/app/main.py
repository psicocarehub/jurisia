import logging
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1.router import api_router
from app.config import settings
from app.core.exceptions import AppError
from app.core.middleware import AuditMiddleware, TenantMiddleware
from app.core.rate_limit import limiter
from app.db.session import engine

logger = logging.getLogger(__name__)

if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=0.2,
        profiles_sample_rate=0.1,
        environment="development" if settings.DEBUG else "production",
        send_default_pii=False,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger("jurisai").info("Juris.AI API starting — using Supabase REST API")
    from app.services.cache import get_redis, close_redis
    await get_redis()
    yield
    await close_redis()
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title="Juris.AI API",
    version="0.1.0",
    description=(
        "API do Juris.AI — Copiloto de IA para advogados brasileiros.\n\n"
        "Oferece busca semântica em jurisprudência, geração de petições, "
        "análise de processos (Raio-X), perfil de juízes, e chat jurídico com RAG."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "cases", "description": "Gerenciamento de processos judiciais"},
        {"name": "documents", "description": "Upload, OCR e classificação de documentos"},
        {"name": "petitions", "description": "Geração e edição de petições com IA"},
        {"name": "chat", "description": "Chat jurídico com RAG e streaming"},
        {"name": "search", "description": "Busca semântica híbrida (BM25 + vetorial)"},
        {"name": "jurimetrics", "description": "Perfil de juízes e predição de resultados"},
        {"name": "updates", "description": "Feed de atualizações legislativas e jurisprudenciais"},
        {"name": "alerts", "description": "Alertas de mudanças legislativas"},
        {"name": "feedback", "description": "Feedback do usuário sobre respostas da IA"},
        {"name": "memory", "description": "Memória de longo prazo (knowledge graph)"},
        {"name": "admin", "description": "Operações administrativas"},
        {"name": "compliance", "description": "Conformidade CNJ Res. 615/2025 e LGPD"},
    ],
)

app.state.limiter = limiter


# --------------- Exception Handlers ---------------

async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def _validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Dados de entrada inválidos",
                "details": exc.errors(),
            }
        },
    )


async def _generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "Erro interno do servidor"}},
    )


app.add_exception_handler(AppError, _app_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, _validation_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(Exception, _generic_error_handler)  # type: ignore[arg-type]


# --------------- Security Headers Middleware ---------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# --------------- Middleware Stack ---------------

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TenantMiddleware)
app.add_middleware(AuditMiddleware)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.APP_NAME}

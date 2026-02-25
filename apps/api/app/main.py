from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.router import api_router
from app.config import settings
from app.core.middleware import AuditMiddleware, TenantMiddleware
from app.db.session import engine

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

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
    import logging
    logging.getLogger("jurisai").info("Juris.AI API starting — using Supabase REST API")
    yield
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title="Juris.AI API",
    version="0.1.0",
    description="Copiloto de IA para advogados brasileiros",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

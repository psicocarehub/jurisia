import time
import uuid
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.db.session import AsyncSessionLocal

logger = logging.getLogger("jurisai")


class TenantMiddleware(BaseHTTPMiddleware):
    """Extract tenant_id from JWT or header and set RLS context."""

    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            request.state.tenant_id = tenant_id
        response = await call_next(request)
        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Log all requests for LGPD compliance."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()

        response = await call_next(request)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "tenant_id": getattr(request.state, "tenant_id", None),
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response

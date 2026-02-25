"""
Structured exception classes and FastAPI exception handlers.

Provides consistent JSON error responses across all endpoints:
  {"error": {"code": "NOT_FOUND", "message": "...", "details": ...}}
"""

from typing import Any, Optional


class AppError(Exception):
    """Base exception with structured fields."""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "Erro interno do servidor",
        details: Optional[Any] = None,
    ) -> None:
        self.message = message
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            body["details"] = self.details
        return {"error": body}


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"

    def __init__(self, resource: str = "Recurso", identifier: str = "") -> None:
        msg = f"{resource} não encontrado"
        if identifier:
            msg += f": {identifier}"
        super().__init__(message=msg)


class ValidationError(AppError):
    status_code = 422
    code = "VALIDATION_ERROR"

    def __init__(self, message: str = "Dados inválidos", details: Optional[Any] = None) -> None:
        super().__init__(message=message, details=details)


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "SERVICE_UNAVAILABLE"

    def __init__(self, service: str = "Serviço externo") -> None:
        super().__init__(message=f"{service} temporariamente indisponível")


class RateLimitError(AppError):
    status_code = 429
    code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, message: str = "Limite de requisições excedido") -> None:
        super().__init__(message=message)


class ForbiddenError(AppError):
    status_code = 403
    code = "FORBIDDEN"

    def __init__(self, message: str = "Acesso negado") -> None:
        super().__init__(message=message)

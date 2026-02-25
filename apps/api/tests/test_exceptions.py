"""Tests for structured exception classes."""

from app.core.exceptions import (
    AppError,
    NotFoundError,
    ValidationError,
    ServiceUnavailableError,
    RateLimitError,
    ForbiddenError,
)


def test_not_found_error():
    err = NotFoundError("Processo", "case-123")
    assert err.status_code == 404
    assert err.code == "NOT_FOUND"
    d = err.to_dict()
    assert d["error"]["code"] == "NOT_FOUND"
    assert "case-123" in d["error"]["message"]


def test_validation_error():
    err = ValidationError("Campo inválido", details={"field": "cnj"})
    assert err.status_code == 422
    d = err.to_dict()
    assert d["error"]["details"] == {"field": "cnj"}


def test_service_unavailable():
    err = ServiceUnavailableError("Elasticsearch")
    assert err.status_code == 503
    assert "Elasticsearch" in err.message


def test_rate_limit_error():
    err = RateLimitError()
    assert err.status_code == 429


def test_forbidden_error():
    err = ForbiddenError()
    assert err.status_code == 403


def test_base_app_error():
    err = AppError("test", details=[1, 2, 3])
    d = err.to_dict()
    assert d["error"]["code"] == "INTERNAL_ERROR"
    assert d["error"]["details"] == [1, 2, 3]

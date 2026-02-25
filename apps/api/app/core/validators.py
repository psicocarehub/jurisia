"""
Brazilian legal-document validators for Pydantic field_validator usage.

Covers CNJ case numbers, CPF, and CNPJ with digit-check algorithms.
"""

import re
from typing import Optional


_CNJ_RE = re.compile(r"^\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}$")


def validate_cnj(value: Optional[str]) -> Optional[str]:
    """Validate CNJ process number format: NNNNNNN-DD.AAAA.J.TR.OOOO"""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if not _CNJ_RE.match(v):
        raise ValueError(
            f"Número CNJ inválido: '{v}'. "
            "Formato esperado: NNNNNNN-DD.AAAA.J.TR.OOOO"
        )
    return v


def _only_digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def validate_cpf(value: Optional[str]) -> Optional[str]:
    """Validate CPF (11 digits + check digits). Accepts with or without dots/dash."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None

    digits = _only_digits(v)
    if len(digits) != 11:
        raise ValueError(f"CPF deve ter 11 dígitos, recebido: {len(digits)}")

    if digits == digits[0] * 11:
        raise ValueError("CPF inválido (todos dígitos iguais)")

    # First check digit
    total = sum(int(digits[i]) * (10 - i) for i in range(9))
    d1 = 11 - (total % 11)
    d1 = 0 if d1 >= 10 else d1
    if int(digits[9]) != d1:
        raise ValueError("CPF inválido (dígito verificador 1)")

    # Second check digit
    total = sum(int(digits[i]) * (11 - i) for i in range(10))
    d2 = 11 - (total % 11)
    d2 = 0 if d2 >= 10 else d2
    if int(digits[10]) != d2:
        raise ValueError("CPF inválido (dígito verificador 2)")

    return digits


def validate_cnpj(value: Optional[str]) -> Optional[str]:
    """Validate CNPJ (14 digits + check digits). Accepts with or without formatting."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None

    digits = _only_digits(v)
    if len(digits) != 14:
        raise ValueError(f"CNPJ deve ter 14 dígitos, recebido: {len(digits)}")

    if digits == digits[0] * 14:
        raise ValueError("CNPJ inválido (todos dígitos iguais)")

    # First check digit
    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights1[i] for i in range(12))
    d1 = 11 - (total % 11)
    d1 = 0 if d1 >= 10 else d1
    if int(digits[12]) != d1:
        raise ValueError("CNPJ inválido (dígito verificador 1)")

    # Second check digit
    weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(digits[i]) * weights2[i] for i in range(13))
    d2 = 11 - (total % 11)
    d2 = 0 if d2 >= 10 else d2
    if int(digits[13]) != d2:
        raise ValueError("CNPJ inválido (dígito verificador 2)")

    return digits


def validate_client_document(value: Optional[str]) -> Optional[str]:
    """Auto-detect and validate as CPF (11 digits) or CNPJ (14 digits)."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None

    digits = _only_digits(v)
    if len(digits) == 11:
        return validate_cpf(v)
    elif len(digits) == 14:
        return validate_cnpj(v)
    else:
        raise ValueError(
            f"Documento deve ser CPF (11 dígitos) ou CNPJ (14 dígitos), "
            f"recebido: {len(digits)} dígitos"
        )

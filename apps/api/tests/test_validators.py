"""Tests for Brazilian document validators."""

import pytest

from app.core.validators import (
    validate_cnj,
    validate_cpf,
    validate_cnpj,
    validate_client_document,
)


class TestCNJ:
    def test_valid_cnj(self):
        assert validate_cnj("0000001-23.2024.8.26.0100") == "0000001-23.2024.8.26.0100"

    def test_none_returns_none(self):
        assert validate_cnj(None) is None

    def test_empty_returns_none(self):
        assert validate_cnj("") is None
        assert validate_cnj("  ") is None

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Número CNJ inválido"):
            validate_cnj("1234567")

    def test_missing_dots(self):
        with pytest.raises(ValueError):
            validate_cnj("0000001-23-2024-8-26-0100")

    def test_whitespace_stripped(self):
        assert validate_cnj(" 0000001-23.2024.8.26.0100 ") == "0000001-23.2024.8.26.0100"


class TestCPF:
    def test_valid_cpf(self):
        result = validate_cpf("529.982.247-25")
        assert result == "52998224725"

    def test_valid_cpf_digits_only(self):
        result = validate_cpf("52998224725")
        assert result == "52998224725"

    def test_none_returns_none(self):
        assert validate_cpf(None) is None

    def test_empty_returns_none(self):
        assert validate_cpf("") is None

    def test_all_same_digits(self):
        with pytest.raises(ValueError, match="todos dígitos iguais"):
            validate_cpf("111.111.111-11")

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="11 dígitos"):
            validate_cpf("1234")

    def test_invalid_check_digit(self):
        with pytest.raises(ValueError, match="dígito verificador"):
            validate_cpf("52998224720")


class TestCNPJ:
    def test_valid_cnpj(self):
        result = validate_cnpj("11.222.333/0001-81")
        assert result == "11222333000181"

    def test_valid_cnpj_digits_only(self):
        result = validate_cnpj("11222333000181")
        assert result == "11222333000181"

    def test_none_returns_none(self):
        assert validate_cnpj(None) is None

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="14 dígitos"):
            validate_cnpj("123456")

    def test_all_same_digits(self):
        with pytest.raises(ValueError, match="todos dígitos iguais"):
            validate_cnpj("11111111111111")

    def test_invalid_check_digit(self):
        with pytest.raises(ValueError, match="dígito verificador"):
            validate_cnpj("11222333000100")


class TestClientDocument:
    def test_auto_detect_cpf(self):
        result = validate_client_document("529.982.247-25")
        assert result == "52998224725"

    def test_auto_detect_cnpj(self):
        result = validate_client_document("11.222.333/0001-81")
        assert result == "11222333000181"

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="CPF.*ou CNPJ"):
            validate_client_document("123456789")

    def test_none_returns_none(self):
        assert validate_client_document(None) is None

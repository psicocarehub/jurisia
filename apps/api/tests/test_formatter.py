"""Tests for petition formatter."""

import pytest
from app.services.petition.formatter import PetitionFormatter, AI_DISCLAIMER


def test_format_abnt_wraps_in_div():
    formatter = PetitionFormatter()
    result = formatter.format_abnt("Test content paragraph")
    assert "<div" in result
    assert "Times New Roman" in result
    assert "Test content paragraph" in result


def test_format_abnt_detects_headings():
    formatter = PetitionFormatter()
    result = formatter.format_abnt("DOS FATOS\n\nConteudo aqui")
    assert "<h2" in result
    assert "DOS FATOS" in result


def test_format_oab_includes_header():
    formatter = PetitionFormatter()
    result = formatter.format_oab("Texto da peticao")
    assert "EXCELENTISSIMO" in result
    assert "OAB" in result


def test_add_ai_label_footer():
    formatter = PetitionFormatter()
    result = formatter.add_ai_label("Conteudo")
    assert AI_DISCLAIMER in result
    assert result.index("Conteudo") < result.index(AI_DISCLAIMER)


def test_add_ai_label_header():
    formatter = PetitionFormatter()
    result = formatter.add_ai_label("TEXTO_PRINCIPAL_AQUI", position="header")
    assert AI_DISCLAIMER in result
    assert result.index("Aviso") < result.index("TEXTO_PRINCIPAL_AQUI")

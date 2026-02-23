"""Tests for petition exporter (PDF/DOCX)."""

import pytest


def test_export_pdf_generates_bytes():
    from app.services.petition.exporter import export_pdf
    result = export_pdf("<p>Test content</p>", title="Test")
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:4] == b'%PDF'


def test_export_docx_generates_bytes():
    from app.services.petition.exporter import export_docx
    result = export_docx("<p>Test content</p>", title="Test")
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:2] == b'PK'


def test_export_pdf_with_ai_disclaimer():
    from app.services.petition.exporter import export_pdf
    result = export_pdf("<p>AI content</p>", title="AI Test", ai_generated=True)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_export_docx_with_ai_disclaimer():
    from app.services.petition.exporter import export_docx
    result = export_docx("<p>AI content</p>", title="AI Test", ai_generated=True)
    assert isinstance(result, bytes)
    assert len(result) > 0

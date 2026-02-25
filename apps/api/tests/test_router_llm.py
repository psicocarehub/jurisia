"""Tests for LLM Router tier selection."""

import pytest

from app.services.llm.router import LLMRouter


def test_router_instantiation():
    router = LLMRouter()
    assert router is not None


def test_router_model_tiers():
    router = LLMRouter()
    tiers = router.model_tiers
    assert isinstance(tiers, dict)
    assert "low" in tiers
    assert "medium" in tiers
    assert "high" in tiers


def test_router_fallback_order():
    router = LLMRouter()
    fallback = router.fallback_order
    assert isinstance(fallback, list)
    assert len(fallback) > 0


def test_router_selects_tier():
    router = LLMRouter()
    tiers = router.model_tiers
    for tier_name in ("low", "medium", "high"):
        model = tiers.get(tier_name)
        assert model is not None, f"Tier '{tier_name}' should have a model configured"

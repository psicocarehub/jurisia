"""
LLM provider configurations for LiteLLM routing.
"""

from typing import Any

from app.config import settings


def get_provider_config(provider: str = "gaia") -> dict[str, Any]:
    """Get provider config for direct LiteLLM calls."""
    configs: dict[str, dict[str, Any]] = {}

    if settings.GAIA_BASE_URL:
        configs["gaia"] = {
            "model": f"openai/{settings.GAIA_MODEL_NAME}",
            "api_base": settings.GAIA_BASE_URL,
            "api_key": "dummy",
        }

    if settings.OPENAI_API_KEY:
        configs["openai"] = {
            "model": "gpt-4o",
            "api_key": settings.OPENAI_API_KEY,
        }

    if settings.DEEPSEEK_API_KEY:
        configs["deepseek"] = {
            "model": "deepseek/deepseek-reasoner",
            "api_key": settings.DEEPSEEK_API_KEY,
        }

    if settings.XAI_API_KEY:
        configs["xai"] = {
            "model": "xai/grok-2-latest",
            "api_key": settings.XAI_API_KEY,
        }

    if settings.MARITACA_API_KEY:
        configs["maritaca"] = {
            "model": "maritaca/sabia-3",
            "api_key": settings.MARITACA_API_KEY,
        }

    if settings.ANTHROPIC_API_KEY:
        configs["anthropic"] = {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": settings.ANTHROPIC_API_KEY,
        }

    if provider in configs:
        return configs[provider]

    for name in ["openai", "anthropic", "xai", "deepseek", "gaia", "maritaca"]:
        if name in configs:
            return configs[name]

    raise RuntimeError("Nenhum provider LLM configurado.")

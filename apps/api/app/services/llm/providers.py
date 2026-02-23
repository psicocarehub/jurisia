"""
LLM provider configuration stubs.
"""

from typing import Any, Dict, Optional

from app.config import settings


def get_provider_config(
    provider: str = "gaia",
    tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Get provider config. Stub for LiteLLM/RouteLLM routing."""
    configs = {
        "gaia": {
            "model": f"openai/{settings.GAIA_MODEL_NAME}",
            "api_base": settings.GAIA_BASE_URL,
            "api_key": "dummy",
        },
        "deepseek": {
            "model": "deepseek/deepseek-reasoner",
            "api_key": settings.DEEPSEEK_API_KEY,
        },
        "maritaca": {
            "model": "maritaca/sabia-3",
            "api_key": settings.MARITACA_API_KEY,
        },
        "anthropic": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": settings.ANTHROPIC_API_KEY,
        },
    }
    return configs.get(provider, configs["gaia"])

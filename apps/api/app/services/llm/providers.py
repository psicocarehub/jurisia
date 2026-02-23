"""
LLM provider configurations for LiteLLM routing.
Updated Feb/2026: DeepSeek V3.2, Qwen 3.5, Kimi K2.5, Gemini 3 Pro.
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

    if settings.DEEPSEEK_API_KEY:
        configs["deepseek"] = {
            "model": "deepseek/deepseek-chat",
            "api_key": settings.DEEPSEEK_API_KEY,
        }

    if settings.QWEN_API_KEY:
        configs["qwen"] = {
            "model": "qwen/qwen3.5",
            "api_key": settings.QWEN_API_KEY,
        }

    if settings.KIMI_API_KEY:
        configs["kimi"] = {
            "model": "kimi/moonshot-v1-k2.5",
            "api_key": settings.KIMI_API_KEY,
        }

    if settings.MINIMAX_API_KEY:
        configs["minimax"] = {
            "model": "minimax/minimax-m2.5",
            "api_key": settings.MINIMAX_API_KEY,
        }

    if settings.OPENAI_API_KEY:
        configs["openai"] = {
            "model": "openai/gpt-5.2-pro",
            "api_key": settings.OPENAI_API_KEY,
        }

    if settings.XAI_API_KEY:
        configs["xai"] = {
            "model": "xai/grok-3",
            "api_key": settings.XAI_API_KEY,
        }

    if settings.MARITACA_API_KEY:
        configs["maritaca"] = {
            "model": "maritaca/sabia-3",
            "api_key": settings.MARITACA_API_KEY,
        }

    if settings.ANTHROPIC_API_KEY:
        configs["anthropic"] = {
            "model": "anthropic/claude-opus-4-20260120",
            "api_key": settings.ANTHROPIC_API_KEY,
        }

    if settings.GOOGLE_API_KEY:
        configs["gemini"] = {
            "model": "gemini/gemini-3-pro",
            "api_key": settings.GOOGLE_API_KEY,
        }

    if provider in configs:
        return configs[provider]

    for name in ["deepseek", "qwen", "anthropic", "gemini", "openai", "kimi", "xai", "gaia", "maritaca", "minimax"]:
        if name in configs:
            return configs[name]

    raise RuntimeError("Nenhum provider LLM configurado.")

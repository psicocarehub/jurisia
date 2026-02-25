import logging
import os
from pathlib import Path
from typing import Optional

import litellm
from litellm import acompletion

from app.config import settings

logger = logging.getLogger(__name__)

if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    logger.info("Langfuse observability enabled")

_INTENT_DIR = Path(__file__).resolve().parents[5] / "training" / "models" / "intent"
_complexity_model = None
_complexity_loaded = False


def _load_complexity_model():
    global _complexity_model, _complexity_loaded
    if _complexity_loaded:
        return _complexity_model
    _complexity_loaded = True
    path = _INTENT_DIR / "complexity_classifier.joblib"
    if path.exists():
        try:
            import joblib
            _complexity_model = joblib.load(path)
            logger.info("Loaded trained complexity classifier")
        except Exception as e:
            logger.warning("Failed to load complexity classifier: %s", e)
    return _complexity_model


class LLMRouter:
    """
    Multi-tier LLM router with ML-based complexity selection.

    Uses trained classifier when available, falls back to keyword heuristic.
    """

    @property
    def model_tiers(self) -> dict:
        tiers = {}

        if settings.GAIA_BASE_URL and settings.GAIA_BASE_URL.startswith("http"):
            tiers["low"] = {
                "model": f"openai/{settings.GAIA_MODEL_NAME}",
                "api_base": settings.GAIA_BASE_URL,
                "api_key": "dummy",
            }

        if settings.DEEPSEEK_API_KEY:
            tiers["medium"] = {
                "model": "deepseek/deepseek-chat",
                "api_key": settings.DEEPSEEK_API_KEY,
            }

        if settings.QWEN_API_KEY:
            tiers["medium_qwen"] = {
                "model": "qwen/qwen3.5",
                "api_key": settings.QWEN_API_KEY,
            }

        if settings.MARITACA_API_KEY:
            tiers["medium_pt"] = {
                "model": "maritaca/sabia-3",
                "api_key": settings.MARITACA_API_KEY,
            }

        if settings.KIMI_API_KEY:
            tiers["medium_agent"] = {
                "model": "kimi/moonshot-v1-k2.5",
                "api_key": settings.KIMI_API_KEY,
            }

        if settings.MINIMAX_API_KEY:
            tiers["medium_minimax"] = {
                "model": "minimax/minimax-m2.5",
                "api_key": settings.MINIMAX_API_KEY,
            }

        if settings.XAI_API_KEY:
            tiers["medium_x"] = {
                "model": "xai/grok-3",
                "api_key": settings.XAI_API_KEY,
            }

        if settings.GOOGLE_API_KEY:
            tiers["high"] = {
                "model": "gemini/gemini-3-pro",
                "api_key": settings.GOOGLE_API_KEY,
            }

        if settings.ANTHROPIC_API_KEY:
            tiers["high_opus"] = {
                "model": "anthropic/claude-opus-4-6",
                "api_key": settings.ANTHROPIC_API_KEY,
            }

        if settings.OPENAI_API_KEY:
            tiers["high_openai"] = {
                "model": "openai/gpt-5.2-pro",
                "api_key": settings.OPENAI_API_KEY,
            }

        return tiers

    COMPLEXITY_INDICATORS = [
        "analise", "análise", "compare", "redija", "elabore",
        "petição", "recurso", "fundamentação", "tese",
        "constitucional", "conflito de normas", "precedente",
        "jurisprudência", "súmula", "acórdão", "habeas corpus",
        "mandado de segurança", "ação civil pública",
    ]

    def _select_tier(self, query: str, requested_tier: Optional[str] = None) -> str:
        if requested_tier:
            return requested_tier

        model = _load_complexity_model()
        if model is not None:
            try:
                pred = model.predict([query])[0]
                config_path = _INTENT_DIR / "complexity_classifier_config.json"
                if config_path.exists():
                    import json
                    labels = json.loads(config_path.read_text())["labels"]
                    tier = labels[pred]
                else:
                    tier = ["high", "low", "medium"][pred]
                return tier
            except Exception as e:
                logger.debug("ML complexity classification failed: %s", e)

        query_lower = query.lower()
        complexity_score = sum(
            1 for indicator in self.COMPLEXITY_INDICATORS if indicator in query_lower
        )

        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"

    def _get_available_tier(self, preferred: str) -> str:
        """Fall back to an available tier if preferred has no API key."""
        tiers = self.model_tiers

        if preferred in tiers:
            api_key = tiers[preferred].get("api_key", "")
            if api_key:
                return preferred

        fallback_order = ["high", "high_opus", "high_openai", "medium", "medium_qwen", "medium_agent", "medium_minimax", "medium_x", "medium_pt", "low"]
        for tier in fallback_order:
            if tier in tiers:
                key = tiers[tier].get("api_key", "")
                if key:
                    return tier

        raise RuntimeError("Nenhum provider LLM configurado. Configure ao menos uma API key.")

    async def generate(
        self,
        messages: list[dict],
        stream: bool = False,
        tier: Optional[str] = None,
        **kwargs,
    ) -> dict:
        import logging
        logger = logging.getLogger(__name__)

        query = messages[-1]["content"] if messages else ""
        selected_tier = self._select_tier(query, tier)
        selected_tier = self._get_available_tier(selected_tier)

        tried_tiers: set[str] = set()
        fallback_order = ["medium", "high", "high_opus", "high_openai", "medium_qwen",
                          "medium_agent", "medium_minimax", "medium_x", "medium_pt", "low"]

        while True:
            model_config = self.model_tiers[selected_tier]
            tried_tiers.add(selected_tier)

            try:
                response = await acompletion(
                    model=model_config["model"],
                    messages=messages,
                    api_base=model_config.get("api_base"),
                    api_key=model_config.get("api_key"),
                    stream=stream,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                )

                if stream:
                    return response

                return {
                    "content": response.choices[0].message.content,
                    "model": model_config["model"],
                    "tier": selected_tier,
                    "tokens_input": response.usage.prompt_tokens if response.usage else 0,
                    "tokens_output": response.usage.completion_tokens if response.usage else 0,
                    "thinking": getattr(response.choices[0].message, "reasoning_content", ""),
                }
            except Exception as e:
                logger.warning("LLM tier '%s' (%s) failed: %s", selected_tier, model_config["model"], e)
                next_tier = None
                for t in fallback_order:
                    if t not in tried_tiers and t in self.model_tiers:
                        next_tier = t
                        break
                if next_tier is None:
                    raise
                logger.info("Falling back to tier '%s'", next_tier)
                selected_tier = next_tier

    async def quick_classify(self, prompt: str) -> str:
        result = await self.generate(
            messages=[{"role": "user", "content": prompt}],
            tier="medium",
            max_tokens=50,
            temperature=0.1,
        )
        return result["content"]

    def get_available_providers(self) -> list[str]:
        """Return list of configured provider names."""
        return list(self.model_tiers.keys())

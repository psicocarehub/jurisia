from typing import Optional

from litellm import acompletion

from app.config import settings


class LLMRouter:
    """
    Multi-tier LLM router with intelligent complexity-based selection.

    Tier hierarchy (atualizado Fev/2026):
      Tier 0 — Cache/RAG (custo zero)
      Tier 1 — GAIA 4B fine-tuned (self-hosted Modal, 70-80% das queries)
      Tier 2 — DeepSeek V3.2 ($0.28/$1.10/M) ou Qwen 3.5 ($0.50/$2.00/M)
      Tier 3 — Gemini 3 Pro ($1.25/$5.00/M) ou Claude Opus 4.6 ($$)
      Fallback — GPT-5.2 Pro, Claude, Kimi K2.5, MiniMax M2.5
    """

    @property
    def model_tiers(self) -> dict:
        tiers = {}

        if settings.GAIA_BASE_URL:
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
                "model": "anthropic/claude-opus-4-20260120",
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
        query = messages[-1]["content"] if messages else ""
        selected_tier = self._select_tier(query, tier)
        selected_tier = self._get_available_tier(selected_tier)
        model_config = self.model_tiers[selected_tier]

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

from typing import Optional

from litellm import acompletion

from app.config import settings


class LLMRouter:
    MODEL_TIERS = {
        "low": {
            "model": f"openai/{settings.GAIA_MODEL_NAME}",
            "api_base": settings.GAIA_BASE_URL,
            "api_key": "dummy",
        },
        "medium": {
            "model": "deepseek/deepseek-reasoner",
            "api_key": settings.DEEPSEEK_API_KEY,
        },
        "medium_pt": {
            "model": "maritaca/sabia-3",
            "api_key": settings.MARITACA_API_KEY,
        },
        "high": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": settings.ANTHROPIC_API_KEY,
        },
    }

    COMPLEXITY_INDICATORS = [
        "analise", "análise", "compare", "redija", "elabore",
        "petição", "recurso", "fundamentação", "tese",
        "constitucional", "conflito de normas", "precedente",
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
        tier_config = self.MODEL_TIERS[preferred]
        api_key = tier_config.get("api_key", "")

        if api_key and api_key != "dummy":
            return preferred

        fallback_order = ["high", "medium", "medium_pt", "low"]
        for tier in fallback_order:
            key = self.MODEL_TIERS[tier].get("api_key", "")
            if key and key != "dummy":
                return tier

        return "low"

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
        model_config = self.MODEL_TIERS[selected_tier]

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
            tier="low",
            max_tokens=50,
            temperature=0.1,
        )
        return result["content"]

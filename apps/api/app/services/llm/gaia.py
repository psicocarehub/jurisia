"""
GAIA Client — cliente direto para inferência vLLM/Modal.

Cliente para GAIA fine-tuned (Gemma-3-Gaia-PT-BR) via
endpoint vLLM ou Modal, formato compatível com OpenAI.
"""

import json
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import settings


class GAIAClient:
    """
    Cliente direto para GAIA (vLLM/Modal).

    Formato de requisição compatível com OpenAI API.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or settings.GAIA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.GAIA_MODEL_NAME

    def _build_messages_payload(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Monta payload no formato OpenAI chat completions."""
        return {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Gera resposta (sem streaming).

        Args:
            messages: Lista de mensagens no formato OpenAI
            max_tokens: Máximo de tokens na resposta
            temperature: Temperatura de amostragem

        Returns:
            Dict com content, usage, finish_reason
        """
        payload = self._build_messages_payload(
            messages, max_tokens, temperature, stream=False
        )
        url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "content": message.get("content", ""),
            "usage": usage,
            "finish_reason": choice.get("finish_reason", "stop"),
            "model": data.get("model", self.model_name),
        }

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Gera resposta em streaming (SSE).

        Args:
            messages: Lista de mensagens
            max_tokens: Máximo de tokens
            temperature: Temperatura

        Yields:
            Chunks de texto conforme gerados
        """
        payload = self._build_messages_payload(
            messages, max_tokens, temperature, stream=True
        )
        url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            return
                        # Parse SSE JSON e extrair delta content
                        try:
                            data = json.loads(chunk)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            pass

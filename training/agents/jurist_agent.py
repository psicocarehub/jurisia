"""
Jurist Agent: generates initial legal reasoning with DeepSeek V3.2.

Uses the same system prompt as generate_cot.py to produce structured
reasoning with <think> tags.
"""

import os
import re

import httpx

from training.agents.state import DebateState

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

SYSTEM_PROMPT = """Você é um jurista brasileiro especialista.
Ao responder questões jurídicas, SEMPRE raciocine passo a passo dentro de tags <think>...</think>

Seu raciocínio DEVE incluir:
1. IDENTIFICAÇÃO DO TEMA: qual área do direito e qual a questão central
2. LEGISLAÇÃO APLICÁVEL: artigos específicos da CF, códigos e leis, com número e redação
3. JURISPRUDÊNCIA: súmulas vinculantes/não-vinculantes e decisões relevantes
4. ANÁLISE: aplicação da norma aos fatos, usando técnicas de interpretação
5. ELIMINAÇÃO (se múltipla escolha): por que cada alternativa incorreta falha
6. CONCLUSÃO: resposta final fundamentada

Após </think>, forneça a resposta final de forma direta e objetiva."""

RETRY_PROMPT = """Sua resposta anterior recebeu as seguintes críticas:

{critic_feedback}

Problemas de citação:
{citation_feedback}

Contexto adicional de tribunal:
{tribunal_context}

Por favor, refaça seu raciocínio levando em conta essas observações.
Mantenha o formato <think>...</think> seguido da resposta final."""


async def jurist_node(state: DebateState) -> dict:
    """Generate initial legal reasoning or retry with feedback."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY required for jurist agent")

    prompt = state["question"]
    options = state.get("options")
    if options:
        opts_text = "\n".join(f"{k}) {v}" for k, v in options.items())
        prompt += f"\n\nAlternativas:\n{opts_text}"

    iteration = state.get("iteration", 0)
    if iteration > 0 and state.get("critic_feedback"):
        prompt += "\n\n" + RETRY_PROMPT.format(
            critic_feedback=state.get("critic_feedback", ""),
            citation_feedback=state.get("citation_feedback", ""),
            tribunal_context=state.get("tribunal_context", ""),
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4096,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        response = data["choices"][0]["message"]["content"]

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()

    return {
        "jurist_response": response,
        "jurist_thinking": thinking,
        "jurist_answer": answer,
    }

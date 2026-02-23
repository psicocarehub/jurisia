"""
Refiner Agent: incorporates all feedback and produces the final refined trace.

Receives the jurist's response, citation verification, tribunal patterns,
and critic feedback, then produces the definitive CoT trace.
Decides whether another iteration is needed based on critic severity.
"""

import os
import re

import httpx

from training.agents.state import DebateState

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

REFINER_SYSTEM = """Você é um jurista sênior responsável por REFINAR e APRIMORAR respostas jurídicas.

Você receberá:
1. A questão original
2. A resposta inicial do jurista
3. Resultado da verificação de citações
4. Contexto real de tribunais (súmulas mais citadas, teses)
5. Crítica detalhada com problemas identificados

Sua tarefa é produzir a VERSÃO FINAL da resposta, que deve:
- Corrigir todas as citações incorretas
- Incorporar súmulas e teses relevantes dos tribunais
- Resolver as lacunas e inconsistências apontadas pelo crítico
- Manter o formato <think>...</think> seguido da resposta final
- Ser completa, bem fundamentada e em português brasileiro

IMPORTANTE: A versão final deve ser MELHOR que a original em todos os aspectos."""


async def refiner_node(state: DebateState) -> dict:
    """Produce the final refined trace."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    critic_severity = state.get("critic_severity", "none")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    should_retry = critic_severity == "high" and iteration < max_iter

    if not api_key:
        return {
            "refined_response": state.get("jurist_response", ""),
            "refined_thinking": state.get("jurist_thinking", ""),
            "refined_answer": state.get("jurist_answer", ""),
            "iteration": iteration + 1,
            "should_retry": False,
            "final_trace": _build_final_trace(state, state.get("jurist_response", "")),
        }

    if critic_severity in ("none", "low") and state.get("citation_score", 1.0) >= 0.8:
        return {
            "refined_response": state.get("jurist_response", ""),
            "refined_thinking": state.get("jurist_thinking", ""),
            "refined_answer": state.get("jurist_answer", ""),
            "iteration": iteration + 1,
            "should_retry": False,
            "final_trace": _build_final_trace(state, state.get("jurist_response", "")),
        }

    user_prompt = f"""## Questão Original
{state.get('question', '')}

## Resposta do Jurista (versão {iteration + 1})
{state.get('jurist_response', '')}

## Verificação de Citações
{state.get('citation_feedback', '')}

## Contexto de Tribunais
{state.get('tribunal_context', '')}

## Crítica Recebida
{state.get('critic_feedback', '')}

---
Produza a versão refinada e definitiva. Use formato <think>...</think> + resposta final."""

    try:
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
                        {"role": "system", "content": REFINER_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.4,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            response = data["choices"][0]["message"]["content"]
    except Exception as e:
        response = state.get("jurist_response", "")
        should_retry = False

    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()

    return {
        "refined_response": response,
        "refined_thinking": thinking,
        "refined_answer": answer,
        "iteration": iteration + 1,
        "should_retry": should_retry,
        "final_trace": _build_final_trace(state, response),
    }


def _build_final_trace(state: DebateState, final_response: str) -> dict:
    """Build the final trace dict for output."""
    think_match = re.search(r"<think>(.*?)</think>", final_response, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = final_response.split("</think>", 1)[1].strip() if "</think>" in final_response else final_response.strip()

    return {
        "question": state.get("question", ""),
        "options": state.get("options"),
        "correct_answer": state.get("correct_answer"),
        "area": state.get("area", "geral"),
        "difficulty": state.get("difficulty", "medium"),
        "thinking": thinking,
        "answer": answer,
        "full_response": final_response,
        "citation_score": state.get("citation_score", 0.0),
        "citation_checks": len(state.get("citation_checks", [])),
        "critic_severity": state.get("critic_severity", "none"),
        "iterations": state.get("iteration", 0) + 1,
        "pipeline": "debate",
    }

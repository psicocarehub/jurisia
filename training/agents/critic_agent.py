"""
Critic Agent: challenges weak arguments and finds gaps.

Reviews the jurist's response along with verification and tribunal data,
then identifies:
- Arguments without legal basis
- Conclusions contradicting dominant jurisprudence
- Missing analysis points
- Logical inconsistencies
"""

import os
import re

import httpx

from training.agents.state import DebateState

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

CRITIC_SYSTEM = """Você é um revisor jurídico rigoroso e experiente.
Sua função é CRITICAR e DESAFIAR a resposta jurídica fornecida.

Analise a resposta e identifique:
1. ARGUMENTOS SEM FUNDAMENTAÇÃO: afirmações sem citação de lei/jurisprudência
2. CITAÇÕES INCORRETAS: artigos inexistentes ou aplicados incorretamente
3. JURISPRUDÊNCIA IGNORADA: súmulas ou teses que deveriam ter sido mencionadas
4. LACUNAS: aspectos importantes do caso que não foram analisados
5. INCONSISTÊNCIAS: contradições lógicas ou jurídicas na argumentação
6. QUALIDADE: se a resposta é superficial ou aprofundada o suficiente

Forneça sua crítica de forma estruturada.
Ao final, classifique a SEVERIDADE dos problemas: NENHUMA, BAIXA, MÉDIA ou ALTA.

Se a severidade for ALTA, a resposta precisa ser refeita.
Se for MÉDIA, pode ser refinada.
Se for BAIXA ou NENHUMA, a resposta está aceitável."""


async def critic_node(state: DebateState) -> dict:
    """Critique the jurist's response."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {
            "critic_feedback": "Critic agent não disponível (sem API key).",
            "critic_issues": [],
            "critic_severity": "none",
        }

    jurist_response = state.get("jurist_response", "")
    citation_feedback = state.get("citation_feedback", "")
    tribunal_context = state.get("tribunal_context", "")
    question = state.get("question", "")

    user_prompt = f"""## Questão Original
{question}

## Resposta do Jurista
{jurist_response}

## Verificação de Citações
{citation_feedback}

## Contexto do Tribunal (patterns reais)
{tribunal_context}

---
Analise criticamente a resposta acima. Identifique todos os problemas e classifique a severidade."""

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
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            feedback = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "critic_feedback": f"Erro no agente crítico: {e}",
            "critic_issues": [],
            "critic_severity": "none",
        }

    issues: list[str] = []
    for pattern in [
        r"(?:\d+[\.\)]\s*)(.*?(?:sem fundamentação|incorret[oa]|inconsistên|lacuna|ausência|falt[oa]u|ignor).*?)(?:\n|$)",
    ]:
        for match in re.finditer(pattern, feedback, re.IGNORECASE):
            issues.append(match.group(1).strip()[:200])

    if not issues:
        for line in feedback.split("\n"):
            line = line.strip()
            if line.startswith(("-", "•", "*")) and len(line) > 20:
                issues.append(line.lstrip("-•* ").strip()[:200])

    feedback_lower = feedback.lower()
    if "alta" in feedback_lower and "severidade" in feedback_lower:
        severity = "high"
    elif "média" in feedback_lower and "severidade" in feedback_lower:
        severity = "medium"
    elif "baixa" in feedback_lower and "severidade" in feedback_lower:
        severity = "low"
    elif "nenhuma" in feedback_lower and "severidade" in feedback_lower:
        severity = "none"
    else:
        severity = "low" if len(issues) <= 2 else "medium" if len(issues) <= 5 else "high"

    return {
        "critic_feedback": feedback,
        "critic_issues": issues[:10],
        "critic_severity": severity,
    }

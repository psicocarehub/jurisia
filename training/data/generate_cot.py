"""
Gerar reasoning traces jurídicos usando teacher model (DeepSeek V3.2 / Qwen 3.5).

Meta: 50K-100K traces de alta qualidade para SFT do GAIA.

Teacher models disponíveis (Fev/2026):
  - deepseek-chat (DeepSeek V3.2): $0.28/$1.10/M — melhor custo-benefício, MIT
  - qwen3.5 (Qwen 3.5): $0.50/$2.00/M — Apache 2.0, ótimo jurídico
  - moonshot-v1-k2.5 (Kimi K2.5): $0.45/$2.20/M — líder agentic tasks

Pipeline:
1. Carregar questões OAB + cenários jurídicos
2. Para cada questão: gerar 5-10 traces com teacher
3. Rejection sampling: manter apenas onde resposta = gabarito
4. Self-consistency: selecionar trace mais comum
5. Salvar em JSONL

Usage:
    python -m training.data.generate_cot --questions questions.jsonl --output traces.jsonl
"""

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

# ============================================
# SYSTEM PROMPT
# ============================================

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


# ============================================
# API CALL
# ============================================

TEACHER_ENDPOINTS: dict[str, tuple[str, str]] = {
    "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "QWEN_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1/chat/completions", "KIMI_API_KEY"),
}


async def call_teacher(
    model: str,
    system: str,
    user: str,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """
    Call teacher model API (DeepSeek V3.2, Qwen 3.5, or Kimi K2.5).

    Args:
        model: Model identifier (e.g. deepseek-chat, qwen3.5, moonshot-v1-k2.5)
        system: System prompt
        user: User message
        api_key: API key (default: from env based on model prefix)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Raw response content from model
    """
    provider = next((p for p in TEACHER_ENDPOINTS if p in model), "deepseek")
    endpoint, env_key = TEACHER_ENDPOINTS[provider]

    api_key = api_key or os.environ.get(env_key)
    if not api_key:
        raise ValueError(f"{env_key} must be set or passed as api_key")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ============================================
# PARSING & HELPERS
# ============================================

def parse_response(response: str) -> tuple[str, str]:
    """
    Extract thinking and answer from response with <think> tags.

    Args:
        response: Full model response

    Returns:
        Tuple of (thinking_content, answer)
    """
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""

    if "</think>" in response:
        answer = response.split("</think>", 1)[1].strip()
    else:
        answer = response.strip()

    return thinking, answer


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison (multiple choice or free text).

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (letter for MC, truncated for free text)
    """
    answer = answer.strip().upper()
    match = re.search(r"[A-E]", answer[:10])
    return match.group(0) if match else answer[:50]


def select_best_trace(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select best trace via self-consistency (most common answer).

    Args:
        traces: List of trace dicts with 'answer' key

    Returns:
        Best trace (most common normalized answer)
    """
    if not traces:
        raise ValueError("traces cannot be empty")
    answers = [normalize_answer(t["answer"]) for t in traces]
    most_common = Counter(answers).most_common(1)[0][0]
    for t in traces:
        if normalize_answer(t["answer"]) == most_common:
            return t
    return traces[0]


# ============================================
# MAIN GENERATION
# ============================================

async def generate_traces(
    questions: list[dict[str, Any]],
    teacher_model: str = "deepseek-chat",
    traces_per_question: int = 5,
    output_file: str = "training/data/raw_traces.jsonl",
    api_key: str | None = None,
    min_traces: int = 5,
    max_traces: int = 10,
) -> int:
    """
    Generate reasoning traces using teacher model.

    Args:
        questions: List of question dicts with keys: question, options?, correct_answer?, area?, difficulty?
        teacher_model: Teacher model identifier
        traces_per_question: Traces to generate per question (or random between min/max)
        output_file: Output JSONL path
        api_key: API key (or env)
        min_traces: Minimum traces per question
        max_traces: Maximum traces per question

    Returns:
        Number of successfully generated trace entries
    """
    results: list[dict[str, Any]] = []

    for i, q in enumerate(questions):
        prompt = q["question"]
        if q.get("options"):
            options = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
            prompt += f"\n\nAlternativas:\n{options}"

        # Allow variable traces per question
        n = min(max_traces, max(min_traces, traces_per_question))

        traces: list[dict[str, Any]] = []
        for _ in range(n):
            try:
                response = await call_teacher(
                    model=teacher_model,
                    system=SYSTEM_PROMPT,
                    user=prompt,
                    api_key=api_key,
                )
            except Exception as e:
                print(f"  Erro questão {i+1}: {e}")
                continue

            thinking, answer = parse_response(response)

            if q.get("correct_answer"):
                is_correct = normalize_answer(answer) == normalize_answer(q["correct_answer"])
            else:
                is_correct = True  # Open-ended question

            traces.append({
                "thinking": thinking,
                "answer": answer,
                "is_correct": is_correct,
                "full_response": response,
            })

        correct_traces = [t for t in traces if t["is_correct"]]

        if correct_traces:
            best_trace = select_best_trace(correct_traces)
            results.append({
                "question": q["question"],
                "options": q.get("options"),
                "correct_answer": q.get("correct_answer"),
                "area": q.get("area", "geral"),
                "difficulty": q.get("difficulty", "medium"),
                "thinking": best_trace["thinking"],
                "answer": best_trace["answer"],
                "full_response": best_trace["full_response"],
                "num_correct": len(correct_traces),
                "num_total": n,
            })
            print(f"  [{i+1}/{len(questions)}] OK ({len(correct_traces)}/{n} corretos)")
        else:
            print(f"  [{i+1}/{len(questions)}] SKIP (nenhum trace correto)")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success_rate = len(results) / len(questions) * 100 if questions else 0
    print(f"\nGerados {len(results)} traces ({success_rate:.1f}% sucesso)")
    return len(results)


def load_questions(path: str) -> list[dict[str, Any]]:
    """Load questions from JSONL file."""
    questions: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


async def generate_traces_debate(
    questions: list[dict[str, Any]],
    output_file: str = "training/data/raw_traces.jsonl",
    max_iterations: int = 2,
) -> int:
    """
    Generate reasoning traces using multi-agent debate pipeline.

    Each question goes through: Jurist -> Verifier -> Judge -> Critic -> Refiner
    with optional retry loop. Produces higher quality traces than single-model.

    Args:
        questions: List of question dicts
        output_file: Output JSONL path
        max_iterations: Max debate iterations per question

    Returns:
        Number of successfully generated trace entries
    """
    from training.agents.debate_graph import run_debate

    results: list[dict[str, Any]] = []

    for i, q in enumerate(questions):
        try:
            trace = await run_debate(
                question=q["question"],
                options=q.get("options"),
                correct_answer=q.get("correct_answer"),
                area=q.get("area", "geral"),
                difficulty=q.get("difficulty", "medium"),
                max_iterations=max_iterations,
            )

            if not trace or not trace.get("thinking"):
                print(f"  [{i+1}/{len(questions)}] SKIP (trace vazio)")
                continue

            if q.get("correct_answer"):
                pred = normalize_answer(trace.get("answer", ""))
                gold = normalize_answer(q["correct_answer"])
                if pred != gold:
                    print(f"  [{i+1}/{len(questions)}] SKIP (resposta incorreta: {pred} vs {gold})")
                    continue

            results.append(trace)
            iters = trace.get("iterations", 1)
            cit_score = trace.get("citation_score", 0)
            severity = trace.get("critic_severity", "?")
            print(f"  [{i+1}/{len(questions)}] OK (iters={iters}, citations={cit_score:.0%}, severity={severity})")

        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] ERRO: {e}")
            continue

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success_rate = len(results) / len(questions) * 100 if questions else 0
    print(f"\nGerados {len(results)} traces via debate ({success_rate:.1f}% sucesso)")
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT traces with teacher model")
    parser.add_argument("--questions", "-q", default="training/data/questions.jsonl", help="Input questions JSONL")
    parser.add_argument("--output", "-o", default="training/data/raw_traces.jsonl", help="Output traces JSONL")
    parser.add_argument("--model", "-m", default="deepseek-chat", help="Teacher model (deepseek-chat, qwen3.5, moonshot-v1-k2.5)")
    parser.add_argument("--mode", default="simple", choices=["simple", "debate"], help="Generation mode: simple (single model) or debate (multi-agent)")
    parser.add_argument("--traces-per-question", "-n", type=int, default=5, help="Traces per question (simple mode)")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max debate iterations (debate mode)")
    parser.add_argument("--min-traces", type=int, default=5)
    parser.add_argument("--max-traces", type=int, default=10)
    args = parser.parse_args()

    if not Path(args.questions).exists():
        print(f"Arquivo não encontrado: {args.questions}")
        print("Crie um JSONL com questões. Exemplo por linha:")
        print('  {"question": "...", "options": {"A": "...", "B": "..."}, "correct_answer": "A", "area": "penal"}')
        return

    questions = load_questions(args.questions)
    print(f"Carregadas {len(questions)} questões")
    print(f"Modo: {args.mode}")

    if args.mode == "debate":
        asyncio.run(
            generate_traces_debate(
                questions=questions,
                output_file=args.output,
                max_iterations=args.max_iterations,
            )
        )
    else:
        asyncio.run(
            generate_traces(
                questions=questions,
                teacher_model=args.model,
                traces_per_question=args.traces_per_question,
                output_file=args.output,
                min_traces=args.min_traces,
                max_traces=args.max_traces,
            )
        )


if __name__ == "__main__":
    main()

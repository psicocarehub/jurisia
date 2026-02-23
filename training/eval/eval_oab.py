"""
Avaliação no benchmark OAB (Exame da Ordem).

Carrega modelo, roda em questões OAB e calcula accuracy.

Usage:
    python -m training.eval.eval_oab --model training/grpo/gaia-legal-reasoning --questions oab_questions.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================
# ANSWER NORMALIZATION
# ============================================

def normalize_answer(answer: str) -> str:
    """Extract letter (A-E) from answer for multiple choice."""
    answer = answer.strip().upper()
    match = re.search(r"[A-E]", answer[:20])
    return match.group(0) if match else answer[:50]


def extract_answer_from_response(response: str) -> str:
    """Extract final answer from model response (after </think> if present)."""
    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()
    return response[:200].strip()


# ============================================
# EVALUATION
# ============================================

def load_questions(path: str) -> list[dict[str, Any]]:
    """Load OAB questions from JSONL."""
    questions: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_eval(
    model_path: str,
    questions_path: str,
    output_path: str | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    device: str | None = None,
) -> dict[str, float]:
    """
    Run OAB evaluation.

    Args:
        model_path: Path to model (HuggingFace or local)
        questions_path: JSONL with question, options, correct_answer
        output_path: Optional path to save per-question results
        max_new_tokens: Max generation length
        temperature: Sampling temperature (low for eval)
        device: Device to use (auto if None)

    Returns:
        Dict with accuracy and per-area metrics
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Carregando modelo: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    questions = load_questions(questions_path)
    print(f"Questões: {len(questions)}")

    correct = 0
    results: list[dict[str, Any]] = []
    area_correct: dict[str, int] = {}
    area_total: dict[str, int] = {}

    for i, q in enumerate(questions):
        prompt = q["question"]
        if q.get("options"):
            options = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
            prompt += f"\n\nAlternativas:\n{options}\n\nResposta:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = extract_answer_from_response(response)
        pred = normalize_answer(answer)
        gold = normalize_answer(q.get("correct_answer", ""))
        is_correct = pred == gold
        if is_correct:
            correct += 1

        area = q.get("area", "geral")
        area_total[area] = area_total.get(area, 0) + 1
        if is_correct:
            area_correct[area] = area_correct.get(area, 0) + 1

        results.append({
            "question_id": i,
            "pred": pred,
            "gold": gold,
            "correct": is_correct,
            "area": area,
        })
        print(f"  [{i+1}/{len(questions)}] {pred} vs {gold} {'✓' if is_correct else '✗'}")

    accuracy = correct / len(questions) if questions else 0.0
    metrics: dict[str, float] = {"accuracy": accuracy, "total": float(len(questions)), "correct": float(correct)}
    for area in area_total:
        metrics[f"accuracy_{area}"] = area_correct.get(area, 0) / area_total[area]

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(questions)})")
    for area in sorted(area_total.keys()):
        acc = area_correct.get(area, 0) / area_total[area]
        print(f"  {area}: {acc:.2%} ({area_correct.get(area, 0)}/{area_total[area]})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
        print(f"Resultados salvos em {output_path}")

    return metrics


# ============================================
# MAIN
# ============================================

def main() -> None:
    parser = argparse.ArgumentParser(description="OAB benchmark evaluation")
    parser.add_argument("--model", "-m", required=True, help="Model path (HuggingFace or local)")
    parser.add_argument("--questions", "-q", default="training/data/oab_questions.jsonl", help="OAB questions JSONL")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    if not Path(args.questions).exists():
        print(f"Arquivo não encontrado: {args.questions}")
        return

    run_eval(
        model_path=args.model,
        questions_path=args.questions,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

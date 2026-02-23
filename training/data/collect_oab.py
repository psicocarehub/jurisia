"""
Coletar questoes OAB do HuggingFace (eduagarcia/oab_exams).

Baixa ~1.200 questoes de multipla escolha do Exame da Ordem e
converte para o formato JSONL usado pelo generate_cot.py.

Usage:
    python -m training.data.collect_oab --output training/data/questions_oab.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


AREA_MAP = {
    "Direito Constitucional": "constitucional",
    "Direito Civil": "civil",
    "Direito Penal": "penal",
    "Direito do Trabalho": "trabalhista",
    "Direito Administrativo": "administrativo",
    "Direito Tributário": "tributario",
    "Direito Processual Civil": "processual_civil",
    "Direito Processual Penal": "processual_penal",
    "Direito Empresarial": "empresarial",
    "Direito Ambiental": "ambiental",
    "Direitos Humanos": "direitos_humanos",
    "Direito Internacional": "internacional",
    "Ética Profissional": "etica",
    "Estatuto da Criança e do Adolescente": "eca",
    "Direito do Consumidor": "consumidor",
    "Filosofia do Direito": "filosofia",
}

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def collect_oab(output_file: str = "training/data/questions_oab.jsonl") -> int:
    """Download OAB exams dataset and convert to JSONL."""
    print("Baixando dataset eduagarcia/oab_exams...")
    ds = load_dataset("eduagarcia/oab_exams", split="train")

    questions: list[dict[str, Any]] = []

    for row in ds:
        question_text = row.get("question", "")
        choices = row.get("choices", {})
        answer_idx = row.get("answerKey", "")
        subject = row.get("subject", "")
        exam = row.get("exam_id", "")

        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not labels or not texts:
            continue

        options = {}
        for label, text in zip(labels, texts):
            options[label] = text

        if isinstance(answer_idx, int):
            correct = LETTER_MAP.get(answer_idx, str(answer_idx))
        else:
            correct = str(answer_idx).strip().upper()

        area_raw = subject.strip() if subject else ""
        area = AREA_MAP.get(area_raw, area_raw.lower().replace(" ", "_") if area_raw else "geral")

        questions.append({
            "question": question_text.strip(),
            "options": options,
            "correct_answer": correct,
            "area": area,
            "difficulty": "medium",
            "source": "oab_exam",
            "metadata": {"exam_id": exam, "subject_raw": area_raw},
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Coletadas {len(questions)} questoes OAB -> {output_file}")

    areas = {}
    for q in questions:
        a = q["area"]
        areas[a] = areas.get(a, 0) + 1
    for area, count in sorted(areas.items(), key=lambda x: -x[1]):
        print(f"  {area}: {count}")

    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OAB exam questions from HuggingFace")
    parser.add_argument("--output", "-o", default="training/data/questions_oab.jsonl")
    args = parser.parse_args()
    collect_oab(args.output)


if __name__ == "__main__":
    main()

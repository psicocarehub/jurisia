"""
Coletar decisoes STJ Open Data e gerar cenarios juridicos.

1. Baixa acordaos e decisoes monocraticas do STJ Open Data
2. Extrai ementas e dispositivos
3. Gera perguntas juridicas abertas a partir das ementas
4. Extrai patterns de tribunal (sumulas citadas, areas, teses)
5. Salva questions.jsonl + tribunal_patterns.json

Usage:
    python -m training.data.collect_stj_scenarios --output training/data/questions_stj.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

STJ_BASE = "https://dadosabertos.web.stj.jus.br"

DATASETS = [
    "acordaos",
    "decisoes_monocraticas",
]

QUESTION_TEMPLATES = [
    "Com base na ementa a seguir, qual é o entendimento do STJ sobre o tema?\n\n{ementa}",
    "Analise a seguinte decisão do STJ e explique seus fundamentos jurídicos:\n\n{ementa}",
    "A partir da ementa abaixo, identifique a tese jurídica firmada pelo STJ:\n\n{ementa}",
    "Considerando a decisão do STJ transcrita abaixo, quais artigos de lei e súmulas são aplicáveis?\n\n{ementa}",
]


def _fetch_stj_data(dataset: str, limit: int = 500) -> list[dict[str, Any]]:
    """Fetch decisions from STJ Open Data."""
    results: list[dict[str, Any]] = []

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            url = f"{STJ_BASE}/dataset/{dataset}"
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"  Aviso: {dataset} retornou {resp.status_code}")
                return results

            resource_url = f"{STJ_BASE}/dataset/{dataset}/resource/latest.csv"
            dl = client.get(resource_url)
            if dl.status_code != 200:
                print(f"  Aviso: CSV {dataset} nao disponivel")
                return results

            lines = dl.text.split("\n")
            if len(lines) < 2:
                return results

            header = lines[0].strip().split(",")
            for line in lines[1:limit + 1]:
                if not line.strip():
                    continue
                fields = line.strip().split(",")
                row = {}
                for i, h in enumerate(header):
                    if i < len(fields):
                        row[h.strip().strip('"')] = fields[i].strip().strip('"')
                if row:
                    results.append(row)

    except Exception as e:
        print(f"  Erro ao buscar {dataset}: {e}")

    return results


def _extract_ementa(row: dict[str, Any]) -> str:
    """Extract ementa from decision row."""
    for key in ["ementa", "EMENTA", "Ementa", "descricao", "texto"]:
        val = row.get(key, "")
        if val and len(val) > 50:
            return val
    return ""


def _extract_sumulas(text: str) -> list[str]:
    """Extract sumula references from text."""
    pattern = r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?(\d+)\s*(?:d[oe]\s+(STF|STJ|TST))?"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [f"Súmula {num} {trib}".strip() for num, trib in matches]


def _extract_articles(text: str) -> list[str]:
    """Extract article references from text."""
    pattern = r"Art\.\s*(\d+)(?:[°ºª])?\s*(?:,?\s*d[aoe]\s+)?(Lei|CF|CC|CPC|CPP|CLT|CDC|CTN|CP|ECA)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [f"Art. {num} {law}" for num, law in matches]


def _classify_area(text: str) -> str:
    """Classify legal area from text."""
    text_lower = text.lower()
    area_keywords = {
        "civil": ["contrato", "obrigação", "responsabilidade civil", "dano moral", "indenização"],
        "consumidor": ["consumidor", "cdc", "fornecedor", "produto", "serviço defeituoso"],
        "trabalhista": ["trabalhista", "clt", "empregado", "justa causa", "rescisão"],
        "tributario": ["tributário", "imposto", "icms", "iss", "contribuição"],
        "penal": ["crime", "penal", "pena", "réu", "absolvição", "condenação"],
        "processual_civil": ["recurso", "agravo", "apelação", "embargos", "tutela"],
        "administrativo": ["licitação", "concurso", "servidor público", "ato administrativo"],
        "previdenciario": ["previdência", "aposentadoria", "inss", "benefício"],
    }
    for area, keywords in area_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return area
    return "geral"


def collect_stj(
    output_file: str = "training/data/questions_stj.jsonl",
    patterns_file: str = "training/data/tribunal_patterns.json",
    limit_per_dataset: int = 500,
) -> tuple[int, dict[str, Any]]:
    """Collect STJ decisions and generate questions + patterns."""
    print("Coletando decisoes STJ Open Data...")

    all_decisions: list[dict[str, Any]] = []
    for dataset in DATASETS:
        print(f"  Baixando {dataset}...")
        rows = _fetch_stj_data(dataset, limit=limit_per_dataset)
        all_decisions.extend(rows)
        print(f"    {len(rows)} registros")

    questions: list[dict[str, Any]] = []
    sumula_counts: dict[str, int] = defaultdict(int)
    article_counts: dict[str, int] = defaultdict(int)
    area_counts: dict[str, int] = defaultdict(int)
    teses: list[str] = []

    template_idx = 0
    for row in all_decisions:
        ementa = _extract_ementa(row)
        if not ementa or len(ementa) < 100:
            continue

        area = _classify_area(ementa)
        area_counts[area] += 1

        for sumula in _extract_sumulas(ementa):
            sumula_counts[sumula] += 1

        for article in _extract_articles(ementa):
            article_counts[article] += 1

        tese_match = re.search(
            r"(?:Tese\s*:?\s*|firmou[- ]se\s+(?:a\s+)?tese\s*:?\s*)(.*?)(?:\.\s|$)",
            ementa, re.IGNORECASE | re.DOTALL,
        )
        if tese_match:
            teses.append(tese_match.group(1).strip()[:500])

        template = QUESTION_TEMPLATES[template_idx % len(QUESTION_TEMPLATES)]
        template_idx += 1

        ementa_truncated = ementa[:1500]

        questions.append({
            "question": template.format(ementa=ementa_truncated),
            "area": area,
            "difficulty": "hard",
            "source": "stj_opendata",
            "metadata": {
                "processo": row.get("numero", row.get("NUMERO", "")),
                "relator": row.get("relator", row.get("RELATOR", "")),
            },
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Geradas {len(questions)} questoes STJ -> {output_file}")

    patterns = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
        "tribunals": {
            "STJ": {
                "patterns": {
                    "sumula_citations": {
                        "top_sumulas": [
                            {"sumula": s, "count": c}
                            for s, c in sorted(sumula_counts.items(), key=lambda x: -x[1])[:30]
                        ],
                    },
                    "article_citations": {
                        "top_articles": [
                            {"article": a, "count": c}
                            for a, c in sorted(article_counts.items(), key=lambda x: -x[1])[:30]
                        ],
                    },
                    "area_distribution": dict(area_counts),
                    "teses_firmadas": teses[:50],
                },
                "meta": {
                    "total_decisions": len(all_decisions),
                    "total_questions": len(questions),
                },
            },
        },
    }

    Path(patterns_file).parent.mkdir(parents=True, exist_ok=True)
    with open(patterns_file, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    print(f"Patterns exportados -> {patterns_file}")
    for area, count in sorted(area_counts.items(), key=lambda x: -x[1]):
        print(f"  {area}: {count}")

    return len(questions), patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect STJ decisions and generate scenarios")
    parser.add_argument("--output", "-o", default="training/data/questions_stj.jsonl")
    parser.add_argument("--patterns", "-p", default="training/data/tribunal_patterns.json")
    parser.add_argument("--limit", "-l", type=int, default=500)
    args = parser.parse_args()
    collect_stj(args.output, args.patterns, args.limit)


if __name__ == "__main__":
    main()

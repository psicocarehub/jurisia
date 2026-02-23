"""
Gerar cenarios sinteticos de pratica juridica via DeepSeek V3.2.

Produz ~5K cenarios open-ended que complementam as questoes OAB (multipla escolha):
- Consultas de cliente
- Analise de contratos
- Estrategia processual
- Elaboracao de peticoes
- Calculos trabalhistas/previdenciarios
- Prazos processuais

Suporta enriquecimento com Nemotron-Personas-Brazil (1M personas demograficas
reais do Brasil, CC-BY-4.0) para diversidade regional, educacional e ocupacional.

Usage:
    python -m training.data.generate_scenarios --output training/data/questions_scenarios.jsonl --count 5000
    python -m training.data.generate_scenarios --count 5000 --personas  # com Nemotron
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

_personas_dataset = None


def _load_personas():
    """Load Nemotron-Personas-Brazil from HuggingFace (cached)."""
    global _personas_dataset
    if _personas_dataset is not None:
        return _personas_dataset

    try:
        from datasets import load_dataset
        print("Carregando Nemotron-Personas-Brazil do HuggingFace (~2.5GB)...")
        _personas_dataset = load_dataset(
            "nvidia/Nemotron-Personas-Brazil", split="train",
        )
        print(f"  {len(_personas_dataset)} personas carregadas")
        return _personas_dataset
    except Exception as e:
        print(f"  Aviso: Nao foi possivel carregar Nemotron-Personas: {e}")
        print("  Instale: pip install datasets")
        return None


def _get_random_persona(ds) -> dict[str, str] | None:
    """Pick a random persona from the dataset."""
    if ds is None:
        return None
    idx = random.randint(0, len(ds) - 1)
    row = ds[idx]
    return {
        "persona": row.get("persona", ""),
        "cultural_background": row.get("cultural_background", ""),
        "occupation": row.get("occupation", ""),
        "municipality": row.get("municipality", ""),
        "state": row.get("state", ""),
        "education_level": row.get("education_level", ""),
        "age": str(row.get("age", "")),
        "sex": row.get("sex", ""),
        "marital_status": row.get("marital_status", ""),
        "skills": row.get("skills_and_expertise", "")[:300],
    }


def _build_persona_context(persona: dict[str, str]) -> str:
    """Build persona context string for prompt injection."""
    return (
        f"Perfil do cliente:\n"
        f"- Descricao: {persona['persona']}\n"
        f"- Ocupacao: {persona['occupation']}\n"
        f"- Local: {persona['municipality']}/{persona['state']}\n"
        f"- Escolaridade: {persona['education_level']}\n"
        f"- Idade: {persona['age']} anos, {persona['sex']}, {persona['marital_status']}\n"
        f"- Contexto cultural: {persona['cultural_background'][:400]}\n"
    )


SCENARIO_CATEGORIES = [
    {
        "category": "consulta_cliente",
        "areas": ["civil", "consumidor", "trabalhista", "penal", "familia"],
        "prompt": """Gere uma consulta realista de um cliente para um advogado brasileiro.
A consulta deve ser sobre {area} e incluir:
- Situação fática detalhada (nomes fictícios)
- Pergunta específica do cliente
- Contexto relevante (datas, valores, documentos)

Formato: apenas a consulta do cliente, em primeira pessoa, 3-5 parágrafos.""",
    },
    {
        "category": "analise_contrato",
        "areas": ["civil", "empresarial", "consumidor", "trabalhista", "imobiliario"],
        "prompt": """Gere um cenário onde um advogado precisa analisar um contrato de {area}.
Inclua:
- Tipo de contrato e partes envolvidas
- Cláusulas problemáticas ou ambíguas
- Pergunta: "Quais são os riscos e recomendações?"

Formato: descrição do contrato + pergunta, 3-5 parágrafos.""",
    },
    {
        "category": "estrategia_processual",
        "areas": ["processual_civil", "trabalhista", "tributario", "penal"],
        "prompt": """Gere um cenário de estratégia processual em {area}.
Inclua:
- Descrição do caso (partes, fatos, valor da causa)
- Fase processual atual
- Pergunta: "Qual a melhor estratégia processual?"

Formato: descrição do caso + pergunta, 3-5 parágrafos.""",
    },
    {
        "category": "peticao",
        "areas": ["civil", "trabalhista", "consumidor", "administrativo", "tributario"],
        "prompt": """Gere um cenário onde é necessário elaborar uma petição de {area}.
Inclua:
- Tipo de petição (inicial, recurso, contestação, etc.)
- Fatos do caso
- Pergunta: "Como estruturar essa petição e quais fundamentos usar?"

Formato: descrição + pergunta, 3-5 parágrafos.""",
    },
    {
        "category": "calculo_juridico",
        "areas": ["trabalhista", "previdenciario", "tributario", "civil"],
        "prompt": """Gere um cenário que envolva cálculos jurídicos em {area}.
Inclua:
- Dados específicos (valores, datas, percentuais)
- Tipo de cálculo necessário
- Pergunta: "Como calcular o valor correto?"

Formato: dados do caso + pergunta, 2-4 parágrafos.""",
    },
    {
        "category": "prazo_recurso",
        "areas": ["processual_civil", "processual_penal", "trabalhista", "tributario"],
        "prompt": """Gere um cenário sobre prazos processuais em {area}.
Inclua:
- Tipo de decisão recebida
- Data da intimação
- Pergunta: "Qual o prazo e qual recurso cabível?"

Formato: situação + pergunta, 2-3 parágrafos.""",
    },
    {
        "category": "compliance",
        "areas": ["empresarial", "tributario", "trabalhista", "ambiental", "lgpd"],
        "prompt": """Gere um cenário de compliance/conformidade legal em {area}.
Inclua:
- Tipo de empresa e setor
- Situação que gera dúvida regulatória
- Pergunta: "A empresa está em conformidade? Quais riscos?"

Formato: descrição + pergunta, 3-4 parágrafos.""",
    },
]

SYSTEM_PROMPT = """Você é um gerador de cenários jurídicos realistas para treinamento de IA jurídica.
Gere cenários que pareçam reais, com nomes fictícios, datas e valores plausíveis.
SEMPRE em português brasileiro. Use linguagem natural, como um cliente ou advogado falaria.
NÃO inclua a resposta — apenas o cenário/pergunta."""


async def _generate_scenario(
    category: dict[str, Any],
    api_key: str,
    model: str = "deepseek-chat",
    persona: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Generate a single scenario via DeepSeek, optionally enriched with a persona."""
    area = random.choice(category["areas"])
    prompt = category["prompt"].format(area=area)

    if persona:
        persona_ctx = _build_persona_context(persona)
        prompt = f"{persona_ctx}\n---\n{prompt}\n\nIMPORTANTE: O cenário deve refletir o perfil acima — linguagem, contexto socioeconômico e região."

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.9,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

        metadata: dict[str, Any] = {"category": category["category"]}
        if persona:
            metadata["persona"] = {
                "municipality": persona["municipality"],
                "state": persona["state"],
                "occupation": persona["occupation"],
                "education_level": persona["education_level"],
                "age": persona["age"],
            }

        return {
            "question": content.strip(),
            "area": area,
            "difficulty": "hard",
            "source": f"synthetic_{category['category']}",
            "metadata": metadata,
        }
    except Exception as e:
        print(f"  Erro: {e}")
        return None


async def generate_scenarios(
    output_file: str = "training/data/questions_scenarios.jsonl",
    count: int = 5000,
    model: str = "deepseek-chat",
    batch_size: int = 10,
    api_key: str | None = None,
    use_personas: bool = False,
) -> int:
    """Generate synthetic legal scenarios, optionally enriched with Nemotron-Personas."""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY nao configurada")
        return 0

    personas_ds = _load_personas() if use_personas else None
    if use_personas and personas_ds is None:
        print("AVISO: Continuando SEM personas (fallback)")

    questions: list[dict[str, Any]] = []
    label = " (com Nemotron-Personas)" if personas_ds else ""
    print(f"Gerando {count} cenarios sinteticos{label}...")

    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        tasks = []
        for _ in range(batch_end - batch_start):
            category = random.choice(SCENARIO_CATEGORIES)
            persona = _get_random_persona(personas_ds) if personas_ds else None
            tasks.append(_generate_scenario(category, api_key, model, persona))

        results = await asyncio.gather(*tasks)
        for r in results:
            if r:
                questions.append(r)

        if len(questions) % 100 == 0:
            print(f"  {len(questions)}/{count} gerados...")

        await asyncio.sleep(0.1)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    persona_count = sum(1 for q in questions if q.get("metadata", {}).get("persona"))
    print(f"Gerados {len(questions)} cenarios ({persona_count} com persona) -> {output_file}")
    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic legal scenarios")
    parser.add_argument("--output", "-o", default="training/data/questions_scenarios.jsonl")
    parser.add_argument("--count", "-n", type=int, default=5000)
    parser.add_argument("--model", "-m", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--personas", action="store_true",
        help="Enriquecer cenarios com Nemotron-Personas-Brazil (requer `pip install datasets`)",
    )
    args = parser.parse_args()

    asyncio.run(generate_scenarios(
        output_file=args.output,
        count=args.count,
        model=args.model,
        batch_size=args.batch_size,
        use_personas=args.personas,
    ))


if __name__ == "__main__":
    main()

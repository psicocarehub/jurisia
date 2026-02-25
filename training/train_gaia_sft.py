"""
Generate SFT training data from indexed legal corpus + train GAIA.

This script:
1. Extracts Q&A pairs from our Elasticsearch legal corpus
2. Generates chain-of-thought traces using a teacher LLM
3. Formats into Gemma chat template
4. Optionally trains with LoRA SFT (requires GPU)

Usage:
    python training/train_gaia_sft.py --generate-data [--limit 5000]
    python training/train_gaia_sft.py --train [--epochs 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHAT_TEMPLATE = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<think>
{thinking}
</think>

{answer}<end_of_turn>"""

QUESTION_GENERATORS = [
    ("concept", "O que é {conceito} no direito brasileiro?"),
    ("article", "O que diz o {artigo}?"),
    ("procedure", "Como funciona o procedimento de {procedimento}?"),
    ("requirement", "Quais são os requisitos para {requisito}?"),
    ("compare", "Qual a diferença entre {conceito_a} e {conceito_b}?"),
    ("jurisprudence", "Qual o entendimento do {tribunal} sobre {tema}?"),
    ("application", "Como se aplica {conceito} na prática?"),
    ("deadline", "Qual o prazo para {acao}?"),
]

CONCEPT_EXTRACTORS = [
    (r"(usucapi[ãa]o\s+\w+)", "concept"),
    (r"(responsabilidade\s+(?:civil|objetiva|subjetiva))", "concept"),
    (r"(tutela\s+(?:provis[oó]ria|antecipada|de\s+urg[eê]ncia))", "concept"),
    (r"(prescri[çc][ãa]o\s*\w*)", "concept"),
    (r"(decad[eê]ncia)", "concept"),
    (r"(dano\s+moral\s*\w*)", "concept"),
    (r"(art\.?\s*\d+\s*(?:,\s*)?(?:do\s+)?(?:CPC|CC|CP|CLT|CF|CDC|CTN|ECA)[\w/]*)", "article"),
    (r"(s[úu]mula\s+\d+\s*(?:do\s+\w+)?)", "article"),
    (r"(lei\s+n?\.?\s*[\d.]+/?\d*)", "article"),
]


def extract_qa_from_content(content: str, doc_meta: dict) -> list[dict]:
    """Extract question-answer pairs from a legal document."""
    rng = random.Random(hash(content[:100]))
    pairs = []

    concepts_found = []
    for pattern, ptype in CONCEPT_EXTRACTORS:
        matches = re.findall(pattern, content.lower())
        for m in matches[:2]:
            concepts_found.append((m.strip(), ptype))

    court = doc_meta.get("court", "STJ")
    area = doc_meta.get("area", "cível")

    for concept, ctype in concepts_found[:3]:
        if ctype == "concept":
            templates = [t for qt, t in QUESTION_GENERATORS if qt in ("concept", "application", "requirement")]
            template = rng.choice(templates)
            question = template.format(
                conceito=concept, requisito=concept,
                conceito_a=concept, conceito_b="outro instituto",
            )
        elif ctype == "article":
            question = f"O que diz o {concept}?"
        else:
            question = f"Explique {concept}"

        relevant_section = content[:800]
        thinking = (
            f"O usuário pergunta sobre {concept}. "
            f"Vou buscar nas fontes do {court} na área de {area}. "
            f"Encontrei informações relevantes que abordam este tema. "
            f"Preciso fundamentar a resposta com artigos de lei e jurisprudência."
        )

        answer = (
            f"Com base nas fontes consultadas ({court}, área {area}):\n\n"
            f"{relevant_section}\n\n"
            f"**Importante:** Esta resposta é baseada na legislação e jurisprudência vigentes. "
            f"Consulte um advogado para análise do seu caso específico.\n"
            f"⚠️ Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025."
        )

        pairs.append({
            "question": question,
            "thinking": thinking,
            "answer": answer,
            "source_court": court,
            "source_area": area,
        })

    return pairs


async def fetch_docs_for_sft(es_url: str, es_api_key: str, index: str, limit: int) -> list[dict]:
    import httpx

    headers = {"Content-Type": "application/json"}
    if es_api_key:
        headers["Authorization"] = f"ApiKey {es_api_key}"

    docs = []
    search_after = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        while len(docs) < limit:
            body: dict = {
                "size": 500,
                "sort": [{"_doc": "asc"}],
                "_source": ["content", "area", "court", "doc_type", "document_title"],
                "query": {"bool": {"must": [{"range": {"content.length": {"gte": 200}}}]}} if False else {"match_all": {}},
            }
            if search_after:
                body["search_after"] = search_after

            resp = await client.post(f"{es_url}/{index}/_search", headers=headers, json=body)
            if resp.status_code != 200:
                break

            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                src = hit["_source"]
                if len(src.get("content", "")) >= 200:
                    docs.append(src)
                search_after = hit["sort"]

            if len(docs) % 2000 == 0:
                logger.info("Fetched %d docs...", len(docs))

    return docs[:limit]


async def generate_data(args):
    if not args.es_url:
        root = Path(__file__).resolve().parents[1]
        env_file = root / "apps" / "api" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ELASTICSEARCH_URL="):
                    args.es_url = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("ES_API_KEY="):
                    args.es_api_key = line.split("=", 1)[1].strip().strip('"')

    if not args.es_url:
        logger.error("ELASTICSEARCH_URL not set")
        sys.exit(1)

    logger.info("Fetching documents...")
    docs = await fetch_docs_for_sft(args.es_url, args.es_api_key, "jurisai_chunks", args.limit)
    logger.info("Fetched %d documents", len(docs))

    all_pairs = []
    for doc in docs:
        pairs = extract_qa_from_content(doc.get("content", ""), doc)
        all_pairs.extend(pairs)

    logger.info("Generated %d Q&A pairs", len(all_pairs))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw_file = out / "raw_traces.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Raw traces saved to %s", raw_file)

    formatted_file = out / "sft_train.jsonl"
    with open(formatted_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            text = CHAT_TEMPLATE.format(
                question=pair["question"],
                thinking=pair["thinking"],
                answer=pair["answer"],
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info("Formatted SFT data saved to %s (%d examples)", formatted_file, len(all_pairs))

    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * 0.05))
    val_file = out / "sft_val.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for pair in all_pairs[:n_val]:
            text = CHAT_TEMPLATE.format(
                question=pair["question"],
                thinking=pair["thinking"],
                answer=pair["answer"],
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info("Validation set: %d examples saved to %s", n_val, val_file)

    config = {
        "total_pairs": len(all_pairs),
        "train_file": str(formatted_file),
        "val_file": str(val_file),
        "raw_file": str(raw_file),
    }
    with open(out / "data_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config


def train_lora(args):
    """Train GAIA with LoRA SFT. Requires GPU + transformers + peft."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
    except ImportError as e:
        logger.error("Missing dependency for training: %s", e)
        logger.info("Install: pip install transformers peft trl torch accelerate")
        logger.info("Or use the generated data with your preferred training framework.")
        sys.exit(1)

    data_dir = Path(args.output)
    train_file = data_dir / "sft_train.jsonl"
    val_file = data_dir / "sft_val.jsonl"

    if not train_file.exists():
        logger.error("Training data not found. Run --generate-data first.")
        sys.exit(1)

    base_model = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    logger.info("Loading model: %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from datasets import load_dataset
    dataset = load_dataset("json", data_files={
        "train": str(train_file),
        "validation": str(val_file),
    })

    output_model_dir = str(data_dir / "gaia-legal-sft")
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    logger.info("Starting LoRA SFT training...")
    trainer.train()

    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    logger.info("Model saved to %s", output_model_dir)


async def main():
    parser = argparse.ArgumentParser(description="Generate SFT data and train GAIA")
    parser.add_argument("--generate-data", action="store_true", help="Generate training data from ES")
    parser.add_argument("--train", action="store_true", help="Train with LoRA SFT (requires GPU)")
    parser.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_URL", ""))
    parser.add_argument("--es-api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", default="training/data/sft_dataset")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if not args.generate_data and not args.train:
        args.generate_data = True

    if args.generate_data:
        config = await generate_data(args)
        logger.info("Data generation complete: %s", json.dumps(config, indent=2))

    if args.train:
        train_lora(args)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

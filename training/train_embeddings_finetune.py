"""
Generate training data for embedding fine-tuning on Brazilian legal domain.

Creates query-document pairs from existing indexed data for contrastive learning.
Exports in format compatible with sentence-transformers training.

Since Voyage AI is a hosted API and can't be fine-tuned directly, this script:
1. Generates training pairs from our legal corpus
2. Fine-tunes a local sentence-transformers model (multilingual-e5 or similar)
3. Saves the model for optional local deployment

Usage:
    python training/train_embeddings_finetune.py [--limit 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUERY_TEMPLATES = [
    "Qual a jurisprudência sobre {tema}?",
    "O que diz a lei sobre {tema}?",
    "Como funciona {tema} no direito brasileiro?",
    "Quais os requisitos para {tema}?",
    "Busque decisões sobre {tema}",
    "Qual o entendimento do {tribunal} sobre {tema}?",
    "Precedentes sobre {tema}",
    "{tema}",
]


def extract_themes_from_content(content: str) -> list[str]:
    themes = []

    art_matches = re.findall(r"(art\.?\s*\d+[\w\s,]*(?:lei|c[oó]digo|cpc|cc|cp|clt|cf|cdc)[\w\s/]*\d*)", content.lower())
    for m in art_matches[:3]:
        themes.append(m.strip()[:80])

    sumula_matches = re.findall(r"(s[úu]mula\s+\d+\s*(?:do\s+\w+)?)", content.lower())
    for m in sumula_matches[:2]:
        themes.append(m.strip())

    legal_concepts = re.findall(
        r"((?:responsabilidade\s+civil|dano\s+moral|tutela\s+(?:provisória|antecipada)|"
        r"prescrição|decadência|usucapião|despejo|alimentos|guarda|divórcio|"
        r"indenização|rescisão|nulidade|anulação|impugnação|execução|"
        r"reconvenção|litisconsórcio|denunciação|chamamento)[\w\s]*)",
        content.lower()
    )
    for m in legal_concepts[:3]:
        themes.append(m.strip()[:60])

    return themes


def generate_pairs(docs: list[dict]) -> list[dict]:
    """Generate (query, positive_doc, negative_doc) triplets."""
    import random
    rng = random.Random(42)

    pairs = []
    doc_contents = [(i, d.get("content", "")) for i, d in enumerate(docs) if len(d.get("content", "")) > 100]

    if len(doc_contents) < 10:
        logger.warning("Not enough documents for pair generation")
        return pairs

    for idx, (doc_idx, content) in enumerate(doc_contents):
        doc = docs[doc_idx]
        themes = extract_themes_from_content(content)
        court = doc.get("court", "STJ")

        for theme in themes[:2]:
            template = rng.choice(QUERY_TEMPLATES)
            query = template.format(tema=theme, tribunal=court)

            positive = content[:500]

            neg_idx = rng.choice([i for i, _ in doc_contents if i != doc_idx])
            negative = docs[neg_idx].get("content", "")[:500]

            pairs.append({
                "query": query,
                "positive": positive,
                "negative": negative,
                "area": doc.get("area", ""),
                "court": court,
            })

        if len(pairs) >= 50000:
            break

    logger.info("Generated %d training triplets", len(pairs))
    return pairs


async def fetch_docs(es_url: str, es_api_key: str, index: str, limit: int) -> list[dict]:
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
                "_source": ["content", "area", "court", "doc_type"],
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
                docs.append(hit["_source"])
                search_after = hit["sort"]

    return docs[:limit]


def train_sentence_transformer(pairs: list[dict], output_dir: str, epochs: int = 3) -> dict:
    """Fine-tune a sentence transformer on legal pairs."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.warning("sentence-transformers not installed, saving pairs only")
        return save_pairs_only(pairs, output_dir)

    model_name = "intfloat/multilingual-e5-base"
    logger.info("Loading base model: %s", model_name)

    try:
        model = SentenceTransformer(model_name)
    except Exception:
        logger.warning("Could not load %s, trying smaller model", model_name)
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)

    train_examples = []
    for pair in pairs:
        train_examples.append(InputExample(
            texts=[pair["query"], pair["positive"], pair["negative"]],
        ))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.TripletLoss(model=model)

    logger.info("Training with %d examples for %d epochs...", len(train_examples), epochs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=str(out),
        show_progress_bar=True,
    )

    config = {
        "type": "sentence_transformer",
        "base_model": model_name,
        "num_pairs": len(pairs),
        "epochs": epochs,
        "output_dir": str(out),
    }
    with open(out / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Model saved to %s", output_dir)
    return config


def save_pairs_only(pairs: list[dict], output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs_file = out / "training_pairs.jsonl"
    with open(pairs_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    config = {
        "type": "pairs_only",
        "num_pairs": len(pairs),
        "file": str(pairs_file),
        "note": "Install sentence-transformers to fine-tune: pip install sentence-transformers",
    }
    with open(out / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved %d training pairs to %s", len(pairs), pairs_file)
    return config


async def main():
    parser = argparse.ArgumentParser(description="Fine-tune embeddings for legal domain")
    parser.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_URL", ""))
    parser.add_argument("--es-api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--index", default="jurisai_chunks")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="training/models/embeddings")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

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

    logger.info("Fetching documents for pair generation...")
    docs = await fetch_docs(args.es_url, args.es_api_key, args.index, args.limit)
    logger.info("Fetched %d documents", len(docs))

    pairs = generate_pairs(docs)
    if not pairs:
        logger.error("No training pairs generated")
        sys.exit(1)

    metrics = train_sentence_transformer(pairs, args.output, epochs=args.epochs)
    logger.info("Complete! %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

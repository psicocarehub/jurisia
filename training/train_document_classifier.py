"""
Train a document classifier for Brazilian legal documents.

Uses a pre-trained Portuguese BERT and fine-tunes on labeled legal document data
from Elasticsearch + HuggingFace datasets.

Saves model to training/models/doc_classifier/

Usage:
    python training/train_document_classifier.py [--epochs 5] [--limit 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = [
    "peticao_inicial", "contestacao", "replica", "sentenca", "acordao",
    "decisao_monocratica", "recurso_apelacao", "recurso_agravo",
    "recurso_extraordinario", "recurso_especial", "certidao",
    "procuracao", "comprovante", "laudo_parecer", "despacho", "outros",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

DOC_TYPE_PATTERNS = {
    "peticao_inicial": [
        r"peti[çc][ãa]o\s+inicial", r"excelent[íi]ssimo.*juiz",
        r"dos\s+fatos", r"dos\s+pedidos", r"do\s+direito",
    ],
    "contestacao": [
        r"contesta[çc][ãa]o", r"preliminarmente", r"m[eé]rito\s+da\s+defesa",
    ],
    "sentenca": [
        r"senten[çc]a", r"julgo\s+(im)?procedente", r"dispositivo",
        r"extingo\s+o\s+processo", r"custas\s+pelo",
    ],
    "acordao": [
        r"ac[oó]rd[ãa]o", r"ementa", r"relat[oó]rio", r"voto\s+do\s+relator",
        r"turma\s+(?:julgadora|recursal)", r"c[aâ]mara",
    ],
    "decisao_monocratica": [
        r"decis[ãa]o\s+monocr[áa]tica", r"decido\s+monocraticamente",
    ],
    "recurso_apelacao": [
        r"apela[çc][ãa]o", r"recurso\s+de\s+apela",
    ],
    "recurso_agravo": [
        r"agravo", r"agravo\s+de\s+instrumento", r"agravo\s+interno",
    ],
    "recurso_extraordinario": [
        r"recurso\s+extraordin[áa]rio", r"repercuss[ãa]o\s+geral",
    ],
    "recurso_especial": [
        r"recurso\s+especial", r"resp\b",
    ],
    "certidao": [r"certid[ãa]o"],
    "procuracao": [r"procura[çc][ãa]o", r"outorgante", r"outorgado"],
    "laudo_parecer": [r"laudo", r"parecer\s+t[eé]cnico", r"per[íi]cia"],
    "despacho": [r"despacho", r"cite-se", r"intime-se"],
}


def label_from_patterns(text: str) -> tuple[str, float]:
    text_lower = text[:3000].lower()
    scores: dict[str, int] = {}
    for label, patterns in DOC_TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[label] = score
    if not scores:
        return "outros", 0.3
    best = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best] / 3.0, 1.0)
    return best, confidence


def label_from_doc_type(doc_type: str) -> str | None:
    if not doc_type:
        return None
    dt = doc_type.lower().strip()
    for label in LABELS:
        if label in dt or dt in label:
            return label
    mapping = {
        "jurisprudencia": "acordao",
        "jurisprudência": "acordao",
        "sumula": "acordao",
        "súmula": "acordao",
        "legislacao": "outros",
        "legislação": "outros",
    }
    return mapping.get(dt)


async def fetch_es_data(es_url: str, es_api_key: str, index: str, limit: int) -> list[dict]:
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
                "_source": ["content", "doc_type", "area", "court"],
            }
            if search_after:
                body["search_after"] = search_after

            resp = await client.post(f"{es_url}/{index}/_search", headers=headers, json=body)
            if resp.status_code != 200:
                logger.error("ES error %d: %s", resp.status_code, resp.text[:300])
                break

            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                docs.append(hit["_source"])
                search_after = hit["sort"]

            logger.info("Fetched %d docs...", len(docs))

    return docs[:limit]


def prepare_samples(raw_docs: list[dict]) -> tuple[list[str], list[int]]:
    texts = []
    labels = []

    for doc in raw_docs:
        content = doc.get("content", "") or ""
        if len(content) < 50:
            continue

        label = label_from_doc_type(doc.get("doc_type", ""))
        if label is None:
            label, conf = label_from_patterns(content)
            if conf < 0.5:
                continue

        texts.append(content[:1500])
        labels.append(LABEL2ID[label])

    return texts, labels


def train_with_sklearn(texts: list[str], labels: list[int], output_dir: str) -> dict:
    """Fallback: TF-IDF + Logistic Regression when transformers unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info("Training TF-IDF + LogReg classifier...")
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2),
        sublinear_tf=True, strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    used_labels = sorted(set(y_train) | set(y_test))
    target_names = [ID2LABEL[i] for i in used_labels]

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred, labels=used_labels, target_names=target_names))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, out / "classifier.joblib")
    joblib.dump(vectorizer, out / "vectorizer.joblib")

    config = {
        "type": "tfidf_logreg",
        "labels": LABELS,
        "label2id": LABEL2ID,
        "accuracy": accuracy,
        "num_train": len(X_train_text),
        "num_test": len(X_test_text),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Model saved to %s", output_dir)
    return config


def train_with_transformers(texts: list[str], labels: list[int], output_dir: str, epochs: int) -> dict:
    """Fine-tune a Portuguese BERT model."""
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import torch
    from torch.utils.data import Dataset

    class LegalDocDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    model_name = "neuralmind/bert-base-portuguese-cased"
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = LegalDocDataset(train_enc, train_labels)
    val_dataset = LegalDocDataset(val_enc, val_labels)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, lab = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(lab, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    config = {
        "type": "bert_classifier",
        "base_model": model_name,
        "labels": LABELS,
        "label2id": LABEL2ID,
        "accuracy": eval_result.get("eval_accuracy", 0),
        "num_train": len(train_texts),
        "num_val": len(val_texts),
        "epochs": epochs,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("BERT classifier saved to %s (acc=%.4f)", output_dir, config["accuracy"])
    return config


async def main():
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_URL", ""))
    parser.add_argument("--es-api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--index", default="jurisai_chunks")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--output", default="training/models/doc_classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--use-transformers", action="store_true",
                        help="Use BERT fine-tuning (requires GPU)")
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

    logger.info("Fetching data from Elasticsearch...")
    raw_docs = await fetch_es_data(args.es_url, args.es_api_key, args.index, args.limit)
    logger.info("Fetched %d documents", len(raw_docs))

    texts, labels = prepare_samples(raw_docs)
    logger.info("Prepared %d labeled samples", len(texts))
    logger.info("Label distribution: %s", {ID2LABEL[k]: v for k, v in Counter(labels).items()})

    if len(texts) < 50:
        logger.error("Not enough labeled data (%d). Need at least 50.", len(texts))
        sys.exit(1)

    if args.use_transformers:
        try:
            metrics = train_with_transformers(texts, labels, args.output, args.epochs)
        except ImportError:
            logger.warning("transformers not available, falling back to sklearn")
            metrics = train_with_sklearn(texts, labels, args.output)
    else:
        metrics = train_with_sklearn(texts, labels, args.output)

    logger.info("Training complete! Metrics: %s", json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

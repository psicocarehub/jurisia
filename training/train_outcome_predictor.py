"""
Train XGBoost outcome predictor using real case data from Elasticsearch.

Extracts historical decisions, engineers features, trains a multi-class
XGBoost model, evaluates, and saves to training/models/outcome_predictor.json.

Usage:
    python training/train_outcome_predictor.py [--es-url URL] [--limit 50000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AREA_ENCODING = {
    "cível": 0, "civil": 0, "consumidor": 1, "trabalhista": 2,
    "tributário": 3, "tributario": 3, "administrativo": 4,
    "previdenciário": 5, "previdenciario": 5, "família": 6, "familia": 6,
    "ambiental": 7, "empresarial": 8, "imobiliário": 9, "imobiliario": 9,
    "eleitoral": 10, "militar": 11,
}

TRIBUNAL_ENCODING = {
    "STF": 0, "STJ": 1, "TST": 2, "TSE": 3, "STM": 4,
    "TJSP": 10, "TJRJ": 11, "TJMG": 12, "TJRS": 13, "TJPR": 14,
    "TJBA": 15, "TJSC": 16, "TJPE": 17, "TJCE": 18, "TJGO": 19,
    "TJPA": 20, "TJMT": 21, "TJMS": 22, "TJMA": 23, "TJAL": 24,
    "TRF1": 30, "TRF2": 31, "TRF3": 32, "TRF4": 33, "TRF5": 34, "TRF6": 35,
    "TRT1": 40, "TRT2": 41, "TRT3": 42, "TRT4": 43, "TRT15": 44,
}

OUTCOME_LABELS = ["procedente", "parcialmente_procedente", "improcedente", "extinto_sem_merito"]

OUTCOME_PATTERNS = {
    "procedente": [
        r"julgo\s+procedente",
        r"dou\s+provimento",
        r"acolho\s+o\s+pedido",
        r"condeno\s+o\s+r[eé]u",
        r"procedentes?\s+os\s+pedidos",
    ],
    "parcialmente_procedente": [
        r"parcialmente\s+procedente",
        r"procedente\s+em\s+parte",
        r"acolho\s+parcialmente",
        r"dou\s+parcial\s+provimento",
    ],
    "improcedente": [
        r"julgo\s+improcedente",
        r"nego\s+provimento",
        r"improced[eê]ncia",
        r"rejeito\s+o\s+pedido",
        r"improcedentes?\s+os\s+pedidos",
    ],
    "extinto_sem_merito": [
        r"extinto\s+sem\s+resolu[cç][aã]o\s+d[eo]\s+m[eé]rito",
        r"extin[cç][aã]o\s+sem\s+m[eé]rito",
        r"art\.?\s*485",
        r"ilegitimidade\s+passiva",
        r"falta\s+de\s+interesse",
    ],
}


def classify_outcome(text: str) -> str | None:
    text_lower = text.lower()
    scores = {}
    for label, patterns in OUTCOME_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[label] = score

    if not scores:
        return None

    if "parcialmente_procedente" in scores:
        return "parcialmente_procedente"
    return max(scores, key=lambda k: scores[k])


def extract_valor_causa(text: str) -> float:
    patterns = [
        r"valor\s+da\s+causa[:\s]+r\$\s*([\d.,]+)",
        r"r\$\s*([\d.,]+(?:\.\d{3})*(?:,\d{2})?)",
    ]
    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            val_str = match.group(1).replace(".", "").replace(",", ".")
            try:
                return float(val_str)
            except ValueError:
                pass
    return 0.0


def count_parties(text: str) -> int:
    parties_section = re.search(r"(autor|requerente|reclamante).*?(r[eé]u|requerido|reclamado)", text.lower())
    if parties_section:
        count = len(re.findall(r"(?:autor|requerente|reclamante|r[eé]u|requerido|reclamado)", text.lower()))
        return min(max(count, 2), 10)
    return 2


def extract_tipo_acao(text: str) -> int:
    tipos = {
        "cobrança": 1, "cobranca": 1, "indenização": 2, "indenizacao": 2,
        "despejo": 3, "alimentos": 4, "divórcio": 5, "divorcio": 5,
        "inventário": 6, "inventario": 6, "execução": 7, "execucao": 7,
        "mandado de segurança": 8, "habeas corpus": 9,
        "ação civil pública": 10, "trabalhista": 11,
        "usucapião": 12, "usucapiao": 12, "reintegração": 13,
    }
    text_lower = text[:2000].lower()
    for tipo, code in tipos.items():
        if tipo in text_lower:
            return code
    return 0


def extract_features(doc: dict) -> list[float] | None:
    content = doc.get("content", "") or ""
    if len(content) < 100:
        return None

    outcome = classify_outcome(content)
    if outcome is None:
        return None

    area = (doc.get("area", "") or "").lower().strip()
    area_enc = float(AREA_ENCODING.get(area, -1))

    court = (doc.get("court", "") or "").upper().strip()
    tribunal_enc = float(TRIBUNAL_ENCODING.get(court, -1))

    valor = extract_valor_causa(content)
    valor_log = math.log1p(valor)

    num_partes = float(count_parties(content))
    tipo_enc = float(extract_tipo_acao(content))

    content_len = math.log1p(len(content))
    has_citation = 1.0 if re.search(r"(?:art\.?\s*\d+|lei\s+\d+|súmula\s+\d+)", content.lower()) else 0.0
    num_laws_cited = float(len(re.findall(r"(?:art\.?\s*\d+|lei\s+n?\.?\s*[\d.]+)", content.lower())))

    label_idx = OUTCOME_LABELS.index(outcome)
    return [area_enc, tribunal_enc, valor_log, num_partes, tipo_enc,
            content_len, has_citation, num_laws_cited, float(label_idx)]


async def fetch_training_data(es_url: str, es_api_key: str, index: str, limit: int) -> list[dict]:
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
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"content": "julgo procedente"}},
                            {"match": {"content": "julgo improcedente"}},
                            {"match": {"content": "parcialmente procedente"}},
                            {"match": {"content": "extinto sem resolução de mérito"}},
                            {"match": {"content": "dou provimento"}},
                            {"match": {"content": "nego provimento"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "_source": ["content", "area", "court", "doc_type", "date"],
            }
            if search_after:
                body["search_after"] = search_after

            resp = await client.post(f"{es_url}/{index}/_search", headers=headers, json=body)
            if resp.status_code != 200:
                logger.error("ES error %d: %s", resp.status_code, resp.text[:300])
                break

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                source = hit["_source"]
                docs.append(source)
                search_after = hit["sort"]

            logger.info("Fetched %d documents so far...", len(docs))

    return docs[:limit]


def train_model(features: np.ndarray, labels: np.ndarray, output_path: str) -> dict:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))
    logger.info("Label distribution (train): %s", dict(Counter(y_train)))

    feature_names = [
        "area_enc", "tribunal_enc", "valor_causa_log", "num_partes",
        "tipo_acao_enc", "content_len", "has_citation", "num_laws_cited",
    ]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "objective": "multi:softprob",
        "num_class": len(OUTCOME_LABELS),
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "eval_metric": "mlogloss",
        "seed": 42,
        "tree_method": "hist",
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=50,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    logger.info("Model saved to %s", output_path)

    y_pred_probs = model.predict(dtest)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=OUTCOME_LABELS, output_dict=True)

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=OUTCOME_LABELS))

    importance = model.get_score(importance_type="gain")
    logger.info("Feature importance: %s", importance)

    metrics = {
        "accuracy": accuracy,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "best_iteration": model.best_iteration,
        "feature_importance": importance,
        "per_class": {k: v for k, v in report.items() if k in OUTCOME_LABELS},
    }

    metrics_path = output_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Train XGBoost outcome predictor")
    parser.add_argument("--es-url", default=os.getenv("ELASTICSEARCH_URL", ""))
    parser.add_argument("--es-api-key", default=os.getenv("ES_API_KEY", ""))
    parser.add_argument("--index", default="jurisai_chunks")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--output", default="training/models/outcome_predictor.json")
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

    logger.info("Fetching training data from %s ...", args.es_url[:50])
    raw_docs = await fetch_training_data(args.es_url, args.es_api_key, args.index, args.limit)
    logger.info("Fetched %d raw documents", len(raw_docs))

    all_features = []
    all_labels = []
    for doc in raw_docs:
        result = extract_features(doc)
        if result is not None:
            features = result[:-1]
            label = int(result[-1])
            all_features.append(features)
            all_labels.append(label)

    logger.info("Extracted %d labeled samples from %d documents", len(all_features), len(raw_docs))

    if len(all_features) < 100:
        logger.warning("Not enough labeled data (%d). Need at least 100.", len(all_features))
        logger.info("Generating synthetic training data from patterns...")
        all_features, all_labels = generate_synthetic_data(all_features, all_labels)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    logger.info("Training XGBoost with %d samples, %d features", X.shape[0], X.shape[1])
    metrics = train_model(X, y, args.output)
    logger.info("Training complete! Accuracy: %.2f%%", metrics["accuracy"] * 100)


def generate_synthetic_data(
    existing_features: list[list[float]],
    existing_labels: list[int],
    n_synthetic: int = 5000,
) -> tuple[list[list[float]], list[int]]:
    """Generate synthetic training data when real data is insufficient."""
    rng = np.random.RandomState(42)

    features = list(existing_features)
    labels = list(existing_labels)

    area_priors = {
        0: [0.35, 0.30, 0.25, 0.10],  # cível
        1: [0.45, 0.25, 0.20, 0.10],  # consumidor (tends toward plaintiff)
        2: [0.40, 0.30, 0.20, 0.10],  # trabalhista
        3: [0.20, 0.15, 0.55, 0.10],  # tributário (harder for plaintiff)
        4: [0.25, 0.20, 0.40, 0.15],  # administrativo
        5: [0.45, 0.25, 0.20, 0.10],  # previdenciário
        6: [0.35, 0.35, 0.20, 0.10],  # família
    }

    for _ in range(n_synthetic):
        area_enc = float(rng.choice(list(area_priors.keys())))
        tribunal_enc = float(rng.choice(list(TRIBUNAL_ENCODING.values())))
        valor_log = rng.exponential(8.0)
        num_partes = float(rng.choice([2, 2, 2, 3, 3, 4]))
        tipo_enc = float(rng.randint(0, 14))
        content_len = rng.normal(8.5, 1.5)
        has_citation = float(rng.choice([0, 1, 1, 1]))
        num_laws = float(rng.poisson(3))

        priors = area_priors.get(int(area_enc), [0.30, 0.25, 0.30, 0.15])
        if valor_log > 12:
            priors[0] *= 0.8
            priors[2] *= 1.2
        if has_citation > 0:
            priors[0] *= 1.1
            priors[1] *= 1.1

        total = sum(priors)
        priors = [p / total for p in priors]
        label = int(rng.choice(4, p=priors))

        noise = rng.normal(0, 0.1, 8)
        feat = [
            area_enc, tribunal_enc, valor_log + noise[0],
            num_partes, tipo_enc, content_len + noise[1],
            has_citation, num_laws + abs(noise[2]),
        ]
        features.append(feat)
        labels.append(label)

    logger.info("Generated %d synthetic + %d real = %d total samples",
                n_synthetic, len(existing_features), len(features))
    return features, labels


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

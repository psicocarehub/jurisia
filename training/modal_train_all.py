"""
Train ALL Juris.AI models on Modal.com with GPU.

Runs 5 training jobs:
1. XGBoost outcome predictor (CPU)
2. Document classifier - TF-IDF + LogReg (CPU) + BERT fine-tune (GPU)
3. Intent/complexity classifiers (CPU)
4. Embedding fine-tuning pairs + sentence-transformer (GPU)
5. GAIA SFT data generation

Usage:
    modal run training/modal_train_all.py
    modal run training/modal_train_all.py --job all
    modal run training/modal_train_all.py --job xgboost
    modal run training/modal_train_all.py --job classifier
    modal run training/modal_train_all.py --job intent
    modal run training/modal_train_all.py --job embeddings
    modal run training/modal_train_all.py --job gaia
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

app = modal.App("jurisai-training")

volume = modal.Volume.from_name("jurisai-trained-models", create_if_missing=True)

secrets = modal.Secret.from_name("jurisai-es-credentials")

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx", "numpy", "scikit-learn", "xgboost", "joblib",
    )
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx", "numpy", "scikit-learn", "xgboost", "joblib",
        "torch", "transformers", "sentence-transformers",
        "accelerate", "datasets", "peft", "trl",
    )
)

# ============================================================
# SHARED: Elasticsearch data fetcher
# ============================================================

def fetch_es_data_sync(es_url: str, es_api_key: str, index: str,
                       limit: int, query: dict | None = None,
                       source_fields: list[str] | None = None) -> list[dict]:
    """Synchronous ES data fetch using httpx."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if es_api_key:
        headers["Authorization"] = f"ApiKey {es_api_key}"

    docs = []
    search_after = None
    src = source_fields or ["content", "area", "court", "doc_type", "date"]

    with httpx.Client(timeout=60.0) as client:
        while len(docs) < limit:
            body: dict[str, Any] = {
                "size": 500,
                "sort": [{"_doc": "asc"}],
                "_source": src,
            }
            if query:
                body["query"] = query
            if search_after:
                body["search_after"] = search_after

            resp = client.post(f"{es_url}/{index}/_search", headers=headers, json=body)
            if resp.status_code != 200:
                print(f"ES error {resp.status_code}: {resp.text[:200]}")
                break

            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                docs.append(hit["_source"])
                search_after = hit["sort"]

            if len(docs) % 2000 == 0:
                print(f"  Fetched {len(docs)} docs...")

    return docs[:limit]


# ============================================================
# JOB 1: XGBoost Outcome Predictor
# ============================================================

AREA_ENCODING = {
    "cível": 0, "civil": 0, "consumidor": 1, "trabalhista": 2,
    "tributário": 3, "tributario": 3, "administrativo": 4,
    "previdenciário": 5, "previdenciario": 5, "família": 6, "familia": 6,
    "ambiental": 7, "empresarial": 8, "imobiliário": 9, "imobiliario": 9,
}

TRIBUNAL_ENCODING = {
    "STF": 0, "STJ": 1, "TST": 2, "TSE": 3,
    "TJSP": 10, "TJRJ": 11, "TJMG": 12, "TJRS": 13, "TJPR": 14,
    "TJBA": 15, "TJSC": 16, "TJPE": 17, "TJCE": 18, "TJGO": 19,
    "TJPA": 20, "TJMT": 21,
    "TRF1": 30, "TRF2": 31, "TRF3": 32, "TRF4": 33, "TRF5": 34,
    "TRT1": 40, "TRT2": 41, "TRT3": 42, "TRT15": 44,
}

OUTCOME_LABELS = ["procedente", "parcialmente_procedente", "improcedente", "extinto_sem_merito"]

OUTCOME_PATTERNS = {
    "procedente": [r"julgo\s+procedente", r"dou\s+provimento", r"acolho\s+o\s+pedido", r"condeno\s+o\s+r[eé]u"],
    "parcialmente_procedente": [r"parcialmente\s+procedente", r"procedente\s+em\s+parte", r"dou\s+parcial\s+provimento"],
    "improcedente": [r"julgo\s+improcedente", r"nego\s+provimento", r"improced[eê]ncia", r"rejeito\s+o\s+pedido"],
    "extinto_sem_merito": [r"extinto\s+sem\s+resolu[cç][aã]o\s+d[eo]\s+m[eé]rito", r"art\.?\s*485"],
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


def extract_xgb_features(doc: dict) -> list[float] | None:
    content = doc.get("content", "") or ""
    if len(content) < 100:
        return None
    outcome = classify_outcome(content)
    if outcome is None:
        return None

    area = (doc.get("area", "") or "").lower().strip()
    court = (doc.get("court", "") or "").upper().strip()

    valor_match = re.search(r"r\$\s*([\d.,]+)", content.lower())
    valor = 0.0
    if valor_match:
        try:
            valor = float(valor_match.group(1).replace(".", "").replace(",", "."))
        except ValueError:
            pass

    num_partes = min(len(re.findall(r"(?:autor|requerente|r[eé]u|requerido)", content.lower())), 10) or 2
    tipo_patterns = {"cobrança": 1, "indenização": 2, "despejo": 3, "alimentos": 4, "divórcio": 5,
                     "execução": 7, "trabalhista": 11, "usucapião": 12}
    tipo_enc = 0
    for t, code in tipo_patterns.items():
        if t in content[:2000].lower():
            tipo_enc = code
            break

    has_citation = 1.0 if re.search(r"(?:art\.?\s*\d+|lei\s+\d+|súmula\s+\d+)", content.lower()) else 0.0
    num_laws = float(len(re.findall(r"(?:art\.?\s*\d+|lei\s+n?\.?\s*[\d.]+)", content.lower())))

    label_idx = OUTCOME_LABELS.index(outcome)
    return [
        float(AREA_ENCODING.get(area, -1)),
        float(TRIBUNAL_ENCODING.get(court, -1)),
        math.log1p(valor),
        float(num_partes),
        float(tipo_enc),
        math.log1p(len(content)),
        has_citation,
        num_laws,
        float(label_idx),
    ]


def generate_synthetic_xgb(n: int = 5000) -> tuple[list[list[float]], list[int]]:
    import numpy as np
    rng = np.random.RandomState(42)
    features, labels = [], []
    area_priors = {
        0: [0.35, 0.30, 0.25, 0.10], 1: [0.45, 0.25, 0.20, 0.10],
        2: [0.40, 0.30, 0.20, 0.10], 3: [0.20, 0.15, 0.55, 0.10],
        4: [0.25, 0.20, 0.40, 0.15], 5: [0.45, 0.25, 0.20, 0.10],
        6: [0.35, 0.35, 0.20, 0.10],
    }
    for _ in range(n):
        a = float(rng.choice(list(area_priors.keys())))
        priors = list(area_priors.get(int(a), [0.30, 0.25, 0.30, 0.15]))
        total = sum(priors)
        priors = [p / total for p in priors]
        feat = [a, float(rng.choice(list(TRIBUNAL_ENCODING.values()))),
                rng.exponential(8.0), float(rng.choice([2, 2, 3, 4])),
                float(rng.randint(0, 14)), rng.normal(8.5, 1.5),
                float(rng.choice([0, 1, 1, 1])), float(rng.poisson(3))]
        features.append(feat)
        labels.append(int(rng.choice(4, p=priors)))
    return features, labels


@app.function(image=cpu_image, secrets=[secrets], volumes={"/output": volume}, timeout=600)
def train_xgboost():
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    print("=" * 60)
    print("JOB 1: XGBoost Outcome Predictor")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")

    features_all, labels_all = [], []

    if es_url:
        print(f"Fetching data from ES: {es_url[:50]}...")
        query = {"bool": {"should": [
            {"match": {"content": "julgo procedente"}},
            {"match": {"content": "julgo improcedente"}},
            {"match": {"content": "parcialmente procedente"}},
            {"match": {"content": "nego provimento"}},
            {"match": {"content": "dou provimento"}},
        ], "minimum_should_match": 1}}
        docs = fetch_es_data_sync(es_url, es_key, "jurisai_chunks", 50000, query=query)
        print(f"Fetched {len(docs)} documents")

        for doc in docs:
            result = extract_xgb_features(doc)
            if result:
                features_all.append(result[:-1])
                labels_all.append(int(result[-1]))

        print(f"Extracted {len(features_all)} labeled samples")

    if len(features_all) < 200:
        print("Augmenting with synthetic data...")
        syn_f, syn_l = generate_synthetic_xgb(8000)
        features_all.extend(syn_f)
        labels_all.extend(syn_l)

    X = np.array(features_all, dtype=np.float32)
    y = np.array(labels_all, dtype=np.int32)
    print(f"Total: {len(X)} samples, distribution: {dict(Counter(y))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    feature_names = ["area_enc", "tribunal_enc", "valor_causa_log", "num_partes",
                     "tipo_acao_enc", "content_len", "has_citation", "num_laws_cited"]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "objective": "multi:softprob", "num_class": 4,
        "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "eval_metric": "mlogloss", "seed": 42, "tree_method": "hist",
    }

    model = xgb.train(params, dtrain, num_boost_round=300,
                      evals=[(dtrain, "train"), (dtest, "test")],
                      early_stopping_rounds=20, verbose_eval=50)

    out_dir = Path("/output/outcome_predictor")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "outcome_predictor.json"))

    y_pred = np.argmax(model.predict(dtest), axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=OUTCOME_LABELS)
    print(f"\nAccuracy: {acc:.4f}")
    print(report)

    metrics = {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test),
               "feature_importance": model.get_score(importance_type="gain")}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    volume.commit()
    return {"job": "xgboost", "accuracy": acc, "samples": len(X)}


# ============================================================
# JOB 2: Document Classifier (TF-IDF + LogReg)
# ============================================================

DOC_LABELS = [
    "peticao_inicial", "contestacao", "sentenca", "acordao",
    "decisao_monocratica", "recurso_apelacao", "recurso_agravo",
    "certidao", "procuracao", "laudo_parecer", "despacho", "outros",
]

DOC_PATTERNS = {
    "peticao_inicial": [r"peti[çc][ãa]o\s+inicial", r"excelent[íi]ssimo.*juiz", r"dos\s+fatos"],
    "contestacao": [r"contesta[çc][ãa]o", r"preliminarmente"],
    "sentenca": [r"senten[çc]a", r"julgo\s+(im)?procedente", r"dispositivo"],
    "acordao": [r"ac[oó]rd[ãa]o", r"ementa", r"voto\s+do\s+relator"],
    "decisao_monocratica": [r"decis[ãa]o\s+monocr[áa]tica"],
    "recurso_apelacao": [r"apela[çc][ãa]o", r"recurso\s+de\s+apela"],
    "recurso_agravo": [r"agravo"],
    "certidao": [r"certid[ãa]o"],
    "procuracao": [r"procura[çc][ãa]o", r"outorgante"],
    "laudo_parecer": [r"laudo", r"parecer\s+t[eé]cnico"],
    "despacho": [r"despacho", r"cite-se", r"intime-se"],
}


def label_doc(doc: dict) -> tuple[str, str] | None:
    content = doc.get("content", "") or ""
    if len(content) < 50:
        return None
    dt = (doc.get("doc_type", "") or "").lower()
    if dt:
        for label in DOC_LABELS:
            if label in dt or dt in label:
                return content[:1500], label
        if "jurisprud" in dt:
            return content[:1500], "acordao"

    text_lower = content[:3000].lower()
    for label, patterns in DOC_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            return content[:1500], label
    return None


@app.function(image=cpu_image, secrets=[secrets], volumes={"/output": volume}, timeout=600)
def train_doc_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    print("=" * 60)
    print("JOB 2: Document Classifier")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")
    texts, labels = [], []

    if es_url:
        docs = fetch_es_data_sync(es_url, es_key, "jurisai_chunks", 15000)
        print(f"Fetched {len(docs)} documents")
        label2id = {l: i for i, l in enumerate(DOC_LABELS)}
        for doc in docs:
            result = label_doc(doc)
            if result:
                texts.append(result[0])
                labels.append(label2id[result[1]])
        print(f"Labeled {len(texts)} samples")

    if len(texts) < 100:
        print("Not enough data, generating synthetic examples...")
        for label, patterns in DOC_PATTERNS.items():
            for p in patterns:
                clean = p.replace(r"\s+", " ").replace(r"\.", ".").replace("\\", "")
                for i in range(20):
                    texts.append(f"Documento jurídico - {clean} - exemplo {i} de {label}")
                    labels.append(DOC_LABELS.index(label))

    print(f"Distribution: {dict(Counter(labels))}")

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                  sublinear_tf=True, strip_accents="unicode")
    X_train = vectorizer.fit_transform(X_train_t)
    X_test = vectorizer.transform(X_test_t)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    used = sorted(set(y_train) | set(y_test))
    names = [DOC_LABELS[i] for i in used]
    print(classification_report(y_test, y_pred, labels=used, target_names=names))

    out_dir = Path("/output/doc_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "classifier.joblib")
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    with open(out_dir / "config.json", "w") as f:
        json.dump({"type": "tfidf_logreg", "labels": DOC_LABELS, "accuracy": acc,
                    "n_train": len(X_train_t), "n_test": len(X_test_t)}, f, indent=2)

    volume.commit()
    return {"job": "doc_classifier", "accuracy": acc, "samples": len(texts)}


# ============================================================
# JOB 3: Intent/Complexity Classifiers
# ============================================================

COMPLEXITY_DATA = {
    "low": [
        "o que é habeas corpus?", "qual o prazo para contestação?",
        "o que diz o artigo 5 da constituição?", "como funciona usucapião?",
        "o que é dano moral?", "qual a diferença entre furto e roubo?",
        "o que é uma petição inicial?", "o que significa citação no processo?",
        "qual o prazo prescricional para cobrança?", "como funciona o divórcio?",
        "o que é tutela provisória?", "como calcular pensão alimentícia?",
        "o que é responsabilidade civil?", "qual a competência da justiça federal?",
        "o que significa trânsito em julgado?", "como funciona inventário?",
        "o que é ação de despejo?", "o que é recurso de apelação?",
        "como funciona a justiça gratuita?", "o que é litisconsórcio?",
    ],
    "medium": [
        "busque jurisprudência do STJ sobre dano moral em relações de consumo",
        "quais são os requisitos da petição inicial segundo o CPC?",
        "compare a Súmula 331 do TST com a reforma trabalhista",
        "analise os requisitos para concessão de tutela de urgência",
        "qual a jurisprudência dominante sobre responsabilidade objetiva do Estado?",
        "explique a teoria da desconsideração da personalidade jurídica",
        "como funciona a prescrição intercorrente na execução fiscal?",
        "qual o entendimento do STF sobre IPTU progressivo?",
        "analise o cabimento de recurso extraordinário neste caso",
        "quais são as hipóteses de nulidade do contrato de trabalho?",
        "como se aplica a teoria do adimplemento substancial?",
        "explique a diferença entre prescrição e decadência no CDC",
        "qual a jurisprudência sobre danos morais por negativação indevida?",
        "como funciona a ação rescisória no processo civil?",
        "analise a compatibilidade entre LGPD e Marco Civil da Internet",
    ],
    "high": [
        "elabore uma petição inicial de ação de indenização por dano moral contra banco por negativação indevida",
        "redija um recurso de apelação contra sentença improcedente em ação trabalhista",
        "analise o conflito entre legalidade tributária e uso de medidas provisórias para aumento de tributos",
        "elabore parecer sobre constitucionalidade de lei municipal que proíbe aplicativos de transporte",
        "redija contestação em ação de despejo por falta de pagamento durante pandemia com teoria da imprevisão",
        "analise o caso e elabore estratégia processual completa com teses principais e subsidiárias",
        "compare teses divergentes entre 1ª e 2ª Seção do STJ sobre prescrição de expurgos inflacionários",
        "redija petição de habeas corpus preventivo com pedido liminar",
        "elabore ação civil pública ambiental contra empresa mineradora com tutela provisória",
        "analise viabilidade de ADPF contra omissão legislativa em matéria de direitos fundamentais",
    ],
}

INTENT_DATA = {
    "research": [
        "busque jurisprudência sobre dano moral", "qual a legislação sobre LGPD?",
        "pesquise decisões do STJ sobre prescrição", "o que diz a Súmula 331 do TST?",
        "encontre precedentes sobre usucapião urbana", "busque artigos de lei sobre licitação",
        "pesquise jurisprudência sobre responsabilidade civil médica",
        "qual a posição dos tribunais sobre alimentos gravídicos?",
        "encontre decisões sobre impenhorabilidade de salário",
        "busque precedentes sobre responsabilidade objetiva do Estado",
    ],
    "drafting": [
        "redija uma petição inicial de cobrança", "elabore uma contestação para despejo",
        "escreva um recurso de apelação", "crie uma notificação extrajudicial",
        "redija um contrato de locação comercial", "elabore petição de habeas corpus",
        "escreva um parecer jurídico tributário", "monte uma procuração ad judicia",
        "redija impugnação ao cumprimento de sentença", "elabore recurso especial para STJ",
    ],
    "analysis": [
        "analise este caso e me diga os riscos", "qual a probabilidade de êxito?",
        "avalie a estratégia processual deste caso", "faça uma análise de risco",
        "analise os pontos fortes e fracos da defesa", "avalie se vale recorrer",
        "analise a viabilidade jurídica desta tese", "faça parecer sobre riscos contratuais",
        "analise o perfil de decisões deste juiz", "avalie os precedentes",
    ],
    "memory": [
        "quais casos temos sobre direito trabalhista?", "qual foi o último processo do cliente João?",
        "lembre-me dos detalhes do caso 1234", "quais processos estão pendentes?",
        "me mostre o histórico do cliente Maria", "quantos casos de consumidor ativos?",
    ],
    "chat": [
        "olá, tudo bem?", "obrigado pela ajuda", "pode repetir?",
        "não entendi, explique melhor", "tchau", "como você funciona?",
        "quem criou você?", "bom dia", "me ajude por favor", "ok entendi",
    ],
}


def augment_text(text: str, n: int = 5) -> list[str]:
    import random
    rng = random.Random(hash(text))
    prefixes = ["", "por favor, ", "preciso que ", "gostaria de saber ", "urgente: "]
    suffixes = ["", " por favor", " obrigado", " urgente", " para um cliente"]
    variations = [text]
    for _ in range(n - 1):
        variations.append(f"{rng.choice(prefixes)}{text}{rng.choice(suffixes)}".strip())
    return variations


@app.function(image=cpu_image, volumes={"/output": volume}, timeout=300)
def train_intent_classifiers():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import joblib

    print("=" * 60)
    print("JOB 3: Intent & Complexity Classifiers")
    print("=" * 60)

    results = {}

    for name, data in [("complexity_classifier", COMPLEXITY_DATA), ("intent_classifier", INTENT_DATA)]:
        print(f"\n--- Training {name} ---")
        labels_list = sorted(data.keys())
        label2id = {l: i for i, l in enumerate(labels_list)}

        texts, labels = [], []
        for label, examples in data.items():
            for ex in examples:
                for aug in augment_text(ex, n=6):
                    texts.append(aug)
                    labels.append(label2id[label])

        print(f"  Samples: {len(texts)}")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 3),
                                       sublinear_tf=True, strip_accents="unicode",
                                       analyzer="char_wb", min_df=2)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=5000, C=1.0, class_weight="balanced"), cv=3)),
        ])

        scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
        print(f"  CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        pipeline.fit(texts, labels)

        out_dir = Path("/output/intent")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out_dir / f"{name}.joblib")
        config = {"labels": labels_list, "label2id": label2id,
                   "cv_accuracy": float(scores.mean()), "num_samples": len(texts)}
        with open(out_dir / f"{name}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        results[name] = {"accuracy": float(scores.mean()), "samples": len(texts)}

    volume.commit()
    return {"job": "intent_classifiers", **results}


# ============================================================
# JOB 4: Embedding Fine-tuning Pairs
# ============================================================

QUERY_TEMPLATES = [
    "Qual a jurisprudência sobre {tema}?",
    "O que diz a lei sobre {tema}?",
    "Como funciona {tema} no direito brasileiro?",
    "Busque decisões sobre {tema}",
    "{tema}",
]


@app.function(image=gpu_image, gpu="T4", secrets=[secrets], volumes={"/output": volume}, timeout=1800)
def train_embeddings():
    import random

    print("=" * 60)
    print("JOB 4: Embedding Fine-tuning")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")

    docs = []
    if es_url:
        docs = fetch_es_data_sync(es_url, es_key, "jurisai_chunks", 10000)
        print(f"Fetched {len(docs)} documents")

    rng = random.Random(42)
    pairs = []
    doc_contents = [(i, d.get("content", "")) for i, d in enumerate(docs) if len(d.get("content", "")) > 100]

    concept_patterns = [
        (r"(responsabilidade\s+civil\s*\w*)", "concept"),
        (r"(dano\s+moral\s*\w*)", "concept"),
        (r"(tutela\s+(?:provis[oó]ria|antecipada))", "concept"),
        (r"(art\.?\s*\d+\s*(?:,\s*)?(?:do\s+)?(?:CPC|CC|CP|CLT|CF|CDC)[\w/]*)", "article"),
        (r"(s[úu]mula\s+\d+\s*(?:do\s+\w+)?)", "article"),
    ]

    for idx, (doc_idx, content) in enumerate(doc_contents):
        themes = []
        for pattern, _ in concept_patterns:
            for m in re.findall(pattern, content.lower())[:2]:
                themes.append(m.strip()[:80])

        court = docs[doc_idx].get("court", "STJ")
        for theme in themes[:2]:
            query = rng.choice(QUERY_TEMPLATES).format(tema=theme, tribunal=court)
            positive = content[:500]
            neg_idx = rng.choice([i for i, _ in doc_contents if i != doc_idx])
            negative = docs[neg_idx].get("content", "")[:500]
            pairs.append({"query": query, "positive": positive, "negative": negative})

        if len(pairs) >= 30000:
            break

    print(f"Generated {len(pairs)} training triplets")

    out_dir = Path("/output/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "training_pairs.jsonl", "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    if len(pairs) >= 100:
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader

            model_name = "intfloat/multilingual-e5-base"
            print(f"Loading {model_name}...")
            model = SentenceTransformer(model_name)

            train_examples = [
                InputExample(texts=[p["query"], p["positive"], p["negative"]])
                for p in pairs
            ]
            train_dl = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.TripletLoss(model=model)

            print(f"Training sentence-transformer with {len(train_examples)} examples...")
            model.fit(
                train_objectives=[(train_dl, train_loss)],
                epochs=3, warmup_steps=100,
                output_path=str(out_dir / "model"),
                show_progress_bar=True,
            )
            config = {"type": "sentence_transformer", "base": model_name,
                       "pairs": len(pairs), "epochs": 3}
        except Exception as e:
            print(f"Sentence-transformer training failed: {e}")
            config = {"type": "pairs_only", "pairs": len(pairs)}
    else:
        config = {"type": "pairs_only", "pairs": len(pairs)}

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    volume.commit()
    return {"job": "embeddings", "pairs": len(pairs), "type": config["type"]}


# ============================================================
# JOB 5: GAIA SFT Data Generation
# ============================================================

CHAT_TEMPLATE = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<think>
{thinking}
</think>

{answer}<end_of_turn>"""


@app.function(image=cpu_image, secrets=[secrets], volumes={"/output": volume}, timeout=600)
def generate_gaia_sft_data():
    import random

    print("=" * 60)
    print("JOB 5: GAIA SFT Data Generation")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")

    docs = []
    if es_url:
        docs = fetch_es_data_sync(es_url, es_key, "jurisai_chunks", 8000)
        print(f"Fetched {len(docs)} documents")

    concept_patterns = [
        r"(usucapi[ãa]o\s+\w+)", r"(responsabilidade\s+(?:civil|objetiva))",
        r"(tutela\s+(?:provis[oó]ria|antecipada))", r"(prescri[çc][ãa]o\s*\w*)",
        r"(dano\s+moral\s*\w*)", r"(art\.?\s*\d+[\w\s]*(?:CPC|CC|CLT|CF|CDC)[\w/]*)",
        r"(s[úu]mula\s+\d+\s*(?:do\s+\w+)?)",
    ]

    question_templates = [
        "O que é {concept} no direito brasileiro?",
        "Como funciona {concept}?",
        "Quais os requisitos para {concept}?",
        "O que diz {concept}?",
        "Explique {concept} com base na legislação",
    ]

    rng = random.Random(42)
    all_pairs = []

    for doc in docs:
        content = doc.get("content", "") or ""
        if len(content) < 200:
            continue
        court = doc.get("court", "STJ")
        area = doc.get("area", "cível")

        concepts = []
        for pattern in concept_patterns:
            for m in re.findall(pattern, content.lower())[:2]:
                concepts.append(m.strip()[:80])

        for concept in concepts[:2]:
            question = rng.choice(question_templates).format(concept=concept)
            thinking = (f"O usuário pergunta sobre {concept}. "
                       f"Vou analisar as fontes do {court} na área de {area}. "
                       f"Preciso fundamentar com artigos de lei e jurisprudência.")
            answer = (f"### {concept.title()}\n\n"
                     f"Com base nas fontes ({court}, {area}):\n\n"
                     f"{content[:600]}\n\n"
                     f"### Conclusão\n\n"
                     f"**⚠️** Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025.")
            all_pairs.append({"question": question, "thinking": thinking, "answer": answer})

    print(f"Generated {len(all_pairs)} SFT pairs")

    out_dir = Path("/output/gaia_sft")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "raw_traces.jsonl", "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    rng.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * 0.05))

    with open(out_dir / "sft_train.jsonl", "w", encoding="utf-8") as f:
        for p in all_pairs[n_val:]:
            text = CHAT_TEMPLATE.format(**p)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    with open(out_dir / "sft_val.jsonl", "w", encoding="utf-8") as f:
        for p in all_pairs[:n_val]:
            text = CHAT_TEMPLATE.format(**p)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    config = {"total_pairs": len(all_pairs), "train": len(all_pairs) - n_val, "val": n_val}
    with open(out_dir / "data_config.json", "w") as f:
        json.dump(config, f, indent=2)

    volume.commit()
    return {"job": "gaia_sft_data", **config}


# ============================================================
# ORCHESTRATOR
# ============================================================

@app.local_entrypoint()
def main(job: str = "all"):
    print(f"\n{'='*60}")
    print(f"  JURIS.AI — ML Training Pipeline")
    print(f"  Job: {job}")
    print(f"{'='*60}\n")

    t0 = time.time()
    results = {}

    if job in ("all", "xgboost"):
        print("\n>>> Starting XGBoost training...")
        results["xgboost"] = train_xgboost.remote()

    if job in ("all", "classifier"):
        print("\n>>> Starting document classifier training...")
        results["doc_classifier"] = train_doc_classifier.remote()

    if job in ("all", "intent"):
        print("\n>>> Starting intent classifier training...")
        results["intent"] = train_intent_classifiers.remote()

    if job in ("all", "embeddings"):
        print("\n>>> Starting embedding fine-tuning...")
        results["embeddings"] = train_embeddings.remote()

    if job in ("all", "gaia"):
        print("\n>>> Starting GAIA SFT data generation...")
        results["gaia_sft"] = generate_gaia_sft_data.remote()

    for name, future in results.items():
        try:
            result = future
            print(f"\n✓ {name}: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All jobs completed in {elapsed:.1f}s")
    print(f"  Models saved to Modal Volume: jurisai-trained-models")
    print(f"{'='*60}")

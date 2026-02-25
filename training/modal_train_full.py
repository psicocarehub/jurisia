"""
Full ML training pipeline for Juris.AI on Modal.com.

Phase 1 (parallel):  BERT classifier, enhanced XGBoost, enhanced Intent/Complexity
Phase 2 (sequential): GAIA SFT → GRPO → OAB Evaluation

Usage:
    modal run training/modal_train_full.py
    modal run training/modal_train_full.py --phase 1
    modal run training/modal_train_full.py --phase 2
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

app = modal.App("jurisai-full-training")

volume = modal.Volume.from_name("jurisai-trained-models", create_if_missing=True)
es_secret = modal.Secret.from_name("jurisai-es-credentials")

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx", "numpy", "scikit-learn", "xgboost", "joblib", "scipy",
    )
)

gpu_bert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx", "numpy", "scikit-learn", "joblib",
        "torch", "transformers>=4.49", "accelerate", "datasets",
    )
)

gpu_gaia_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.49", "peft>=0.14", "trl>=0.12",
        "accelerate", "datasets", "bitsandbytes", "scipy",
        "huggingface_hub", "sentencepiece", "protobuf",
    )
)


# ============================================================
# SHARED: ES data fetcher
# ============================================================

def fetch_es_docs(es_url: str, es_key: str, index: str, limit: int,
                  query: dict | None = None,
                  fields: list[str] | None = None) -> list[dict]:
    import httpx
    headers = {"Content-Type": "application/json"}
    if es_key:
        headers["Authorization"] = f"ApiKey {es_key}"
    docs, search_after = [], None
    src = fields or ["content", "area", "court", "doc_type", "date"]
    with httpx.Client(timeout=60.0) as client:
        while len(docs) < limit:
            body: dict = {"size": 500, "sort": [{"_doc": "asc"}], "_source": src}
            if query:
                body["query"] = query
            if search_after:
                body["search_after"] = search_after
            resp = client.post(f"{es_url}/{index}/_search", headers=headers, json=body)
            if resp.status_code != 200:
                break
            hits = resp.json().get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                docs.append(h["_source"])
                search_after = h["sort"]
    return docs[:limit]


# ============================================================
# PHASE 1A: BERT Document Classifier
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


def label_document(doc: dict) -> tuple[str, str] | None:
    content = doc.get("content", "") or ""
    if len(content) < 80:
        return None
    dt = (doc.get("doc_type", "") or "").lower()
    if dt:
        for lbl in DOC_LABELS:
            if lbl in dt or dt in lbl:
                return content[:512], lbl
        if "jurisprud" in dt:
            return content[:512], "acordao"
    text_low = content[:3000].lower()
    for lbl, pats in DOC_PATTERNS.items():
        if any(re.search(p, text_low) for p in pats):
            return content[:512], lbl
    return None


@app.function(image=gpu_bert_image, gpu="T4", secrets=[es_secret],
              volumes={"/output": volume}, timeout=1200)
def train_bert_classifier():
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np

    print("=" * 60)
    print("PHASE 1A: BERT Document Classifier (BERTimbau)")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")

    texts, labels = [], []
    if es_url:
        docs = fetch_es_docs(es_url, es_key, "jurisai_chunks", 20000)
        print(f"Fetched {len(docs)} documents")
        lbl2id = {l: i for i, l in enumerate(DOC_LABELS)}
        for d in docs:
            r = label_document(d)
            if r:
                texts.append(r[0])
                labels.append(lbl2id[r[1]])
        print(f"Labeled {len(texts)} samples")

    if len(texts) < 200:
        print("Augmenting with synthetic data...")
        for lbl, pats in DOC_PATTERNS.items():
            lid = DOC_LABELS.index(lbl)
            for p in pats:
                clean = p.replace(r"\s+", " ").replace("\\", "")
                for i in range(30):
                    texts.append(f"Documento jurídico brasileiro - {clean} exemplo {i}")
                    labels.append(lid)

    print(f"Distribution: {dict(Counter(labels))}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None)

    model_name = "neuralmind/bert-base-portuguese-cased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(DOC_LABELS))

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=512)

    train_ds = Dataset.from_dict({
        "input_ids": train_enc["input_ids"],
        "attention_mask": train_enc["attention_mask"],
        "labels": y_train,
    })
    test_ds = Dataset.from_dict({
        "input_ids": test_enc["input_ids"],
        "attention_mask": test_enc["attention_mask"],
        "labels": y_test,
    })

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}

    training_args = TrainingArguments(
        output_dir="/tmp/bert_output",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        logging_steps=50,
        seed=42,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Training BERT classifier...")
    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    acc = accuracy_score(y_test, y_pred)

    used = sorted(set(y_train) | set(y_test))
    names = [DOC_LABELS[i] for i in used]
    report = classification_report(y_test, y_pred, labels=used, target_names=names)
    print(f"\nBERT Accuracy: {acc:.4f}")
    print(report)

    out_dir = Path("/output/doc_classifier_bert")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    with open(out_dir / "config.json", "w") as f:
        json.dump({"type": "bert", "base": model_name, "labels": DOC_LABELS,
                    "accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)

    volume.commit()
    return {"job": "bert_classifier", "accuracy": acc, "samples": len(texts)}


# ============================================================
# PHASE 1B: Enhanced XGBoost
# ============================================================

AREA_ENC = {
    "cível": 0, "civil": 0, "consumidor": 1, "trabalhista": 2,
    "tributário": 3, "tributario": 3, "administrativo": 4,
    "previdenciário": 5, "previdenciario": 5, "família": 6, "familia": 6,
    "ambiental": 7, "empresarial": 8, "imobiliário": 9, "imobiliario": 9,
}

TRIB_ENC = {
    "STF": 0, "STJ": 1, "TST": 2, "TSE": 3,
    "TJSP": 10, "TJRJ": 11, "TJMG": 12, "TJRS": 13, "TJPR": 14,
    "TJBA": 15, "TJSC": 16, "TJPE": 17, "TJCE": 18, "TJGO": 19,
    "TJPA": 20, "TJMT": 21,
    "TRF1": 30, "TRF2": 31, "TRF3": 32, "TRF4": 33, "TRF5": 34,
    "TRT1": 40, "TRT2": 41, "TRT3": 42, "TRT15": 44,
}

OUTCOME_LABELS = ["procedente", "parcialmente_procedente", "improcedente", "extinto_sem_merito"]

OUTCOME_PAT = {
    "procedente": [r"julgo\s+procedente", r"dou\s+provimento", r"acolho\s+o\s+pedido"],
    "parcialmente_procedente": [r"parcialmente\s+procedente", r"procedente\s+em\s+parte", r"dou\s+parcial\s+provimento"],
    "improcedente": [r"julgo\s+improcedente", r"nego\s+provimento", r"improced[eê]ncia"],
    "extinto_sem_merito": [r"extinto\s+sem\s+resolu[cç][aã]o\s+d[eo]\s+m[eé]rito", r"art\.?\s*485"],
}


def classify_outcome(text: str) -> str | None:
    low = text.lower()
    scores = {l: sum(1 for p in ps if re.search(p, low)) for l, ps in OUTCOME_PAT.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return None
    if "parcialmente_procedente" in scores:
        return "parcialmente_procedente"
    return max(scores, key=lambda k: scores[k])


def extract_features_v2(doc: dict) -> list[float] | None:
    content = doc.get("content", "") or ""
    if len(content) < 100:
        return None
    outcome = classify_outcome(content)
    if outcome is None:
        return None

    area = (doc.get("area", "") or "").lower().strip()
    court = (doc.get("court", "") or "").upper().strip()
    low = content.lower()

    valor_m = re.search(r"r\$\s*([\d.,]+)", low)
    valor = 0.0
    if valor_m:
        try:
            valor = float(valor_m.group(1).replace(".", "").replace(",", "."))
        except ValueError:
            pass

    num_partes = min(len(re.findall(r"(?:autor|requerente|r[eé]u|requerido)", low)), 10) or 2
    tipo_map = {"cobrança": 1, "indenização": 2, "despejo": 3, "alimentos": 4,
                "divórcio": 5, "execução": 7, "trabalhista": 11, "usucapião": 12}
    tipo = 0
    for t, c in tipo_map.items():
        if t in low[:2000]:
            tipo = c
            break

    has_cite = 1.0 if re.search(r"(?:art\.?\s*\d+|lei\s+\d+|súmula\s+\d+)", low) else 0.0
    n_laws = float(len(re.findall(r"(?:art\.?\s*\d+|lei\s+n?\.?\s*[\d.]+)", low)))
    n_jurisprud = float(len(re.findall(r"(?:STF|STJ|TST|TJ\w{2}|TRF\d)", content)))
    n_sumulas = float(len(re.findall(r"s[úu]mula\s+\d+", low)))
    has_tutela = 1.0 if re.search(r"tutela\s+(?:provis|antecipada|urg)", low) else 0.0
    content_words = len(content.split())
    has_recurso = 1.0 if re.search(r"recurso\s+(?:de\s+)?(?:apela|agravo|especial|extra)", low) else 0.0

    label_idx = OUTCOME_LABELS.index(outcome)
    return [
        float(AREA_ENC.get(area, -1)),
        float(TRIB_ENC.get(court, -1)),
        math.log1p(valor),
        float(num_partes),
        float(tipo),
        math.log1p(len(content)),
        has_cite,
        n_laws,
        n_jurisprud,
        n_sumulas,
        has_tutela,
        math.log1p(content_words),
        has_recurso,
        float(label_idx),
    ]


@app.function(image=cpu_image, secrets=[es_secret], volumes={"/output": volume}, timeout=900)
def train_xgboost_enhanced():
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score

    print("=" * 60)
    print("PHASE 1B: Enhanced XGBoost with GridSearch")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")
    features, labels = [], []

    if es_url:
        query = {"bool": {"should": [
            {"match": {"content": "julgo procedente"}},
            {"match": {"content": "julgo improcedente"}},
            {"match": {"content": "parcialmente procedente"}},
            {"match": {"content": "nego provimento"}},
            {"match": {"content": "dou provimento"}},
        ], "minimum_should_match": 1}}
        docs = fetch_es_docs(es_url, es_key, "jurisai_chunks", 80000, query=query)
        print(f"Fetched {len(docs)} documents")
        for d in docs:
            r = extract_features_v2(d)
            if r:
                features.append(r[:-1])
                labels.append(int(r[-1]))
        print(f"Extracted {len(features)} samples")

    if len(features) < 300:
        print("Augmenting with synthetic data...")
        rng = np.random.RandomState(42)
        priors_map = {0: [.35,.30,.25,.10], 1: [.45,.25,.20,.10], 2: [.40,.30,.20,.10],
                      3: [.20,.15,.55,.10], 4: [.25,.20,.40,.15], 5: [.45,.25,.20,.10]}
        for _ in range(10000):
            a = float(rng.choice(list(priors_map.keys())))
            pr = np.array(priors_map.get(int(a), [.30,.25,.30,.15]), dtype=np.float64)
            pr /= pr.sum()
            features.append([a, float(rng.choice(list(TRIB_ENC.values()))),
                           rng.exponential(8.0), float(rng.choice([2,2,3,4])),
                           float(rng.randint(0,14)), rng.normal(8.5,1.5),
                           float(rng.choice([0,1,1,1])), float(rng.poisson(3)),
                           float(rng.poisson(2)), float(rng.poisson(1)),
                           float(rng.choice([0,0,1])), rng.normal(7,1.5),
                           float(rng.choice([0,0,0,1]))])
            labels.append(int(rng.choice(4, p=pr)))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Total: {len(X)} samples")

    feature_names = ["area_enc", "tribunal_enc", "valor_causa_log", "num_partes",
                     "tipo_acao_enc", "content_len", "has_citation", "num_laws",
                     "n_jurisprud", "n_sumulas", "has_tutela", "content_words_log",
                     "has_recurso"]

    param_grid = [
        {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5},
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3},
        {"max_depth": 5, "learning_rate": 0.15, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 7},
        {"max_depth": 7, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 4},
    ]

    best_acc, best_params, best_model = 0.0, {}, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, params in enumerate(param_grid):
        fold_accs = []
        for train_idx, val_idx in skf.split(X, y):
            dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feature_names)
            dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feature_names)
            xgb_params = {**params, "objective": "multi:softprob", "num_class": 4,
                         "eval_metric": "mlogloss", "seed": 42, "tree_method": "hist"}
            m = xgb.train(xgb_params, dtrain, num_boost_round=500,
                         evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
            preds = np.argmax(m.predict(dval), axis=1)
            fold_accs.append(accuracy_score(y[val_idx], preds))

        mean_acc = np.mean(fold_accs)
        print(f"  Config {i+1}: CV acc = {mean_acc:.4f} (+/- {np.std(fold_accs):.4f})")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params

    print(f"\nBest params: {best_params} (CV acc: {best_acc:.4f})")

    dtrain_full = xgb.DMatrix(X, label=y, feature_names=feature_names)
    final_params = {**best_params, "objective": "multi:softprob", "num_class": 4,
                    "eval_metric": "mlogloss", "seed": 42, "tree_method": "hist"}
    best_model = xgb.train(final_params, dtrain_full, num_boost_round=500)

    out_dir = Path("/output/outcome_predictor")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(out_dir / "outcome_predictor.json"))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"accuracy_cv": best_acc, "params": best_params,
                   "n_samples": len(X), "features": feature_names}, f, indent=2, default=str)

    volume.commit()
    return {"job": "xgboost_enhanced", "cv_accuracy": best_acc, "samples": len(X)}


# ============================================================
# PHASE 1C: Enhanced Intent/Complexity
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
        "o que é mandado de segurança?", "qual o conceito de boa-fé?",
        "quais são os princípios do contraditório?", "o que é prescrição?",
        "o que é decadência?", "como funciona a mediação?",
        "o que é arbitragem?", "qual o prazo da apelação?",
        "o que é competência relativa?", "o que é competência absoluta?",
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
        "qual o entendimento sobre impenhorabilidade do bem de família?",
        "como funciona a responsabilidade civil do médico?",
        "quais os requisitos para usucapião extraordinária?",
        "explique as modalidades de intervenção de terceiros no CPC",
        "qual a diferença entre litispendência e coisa julgada?",
    ],
    "high": [
        "elabore uma petição inicial de ação de indenização por dano moral contra banco por negativação indevida com pedido de tutela antecipada",
        "redija um recurso de apelação contra sentença improcedente em ação trabalhista por assédio moral",
        "analise o conflito entre legalidade tributária e uso de medidas provisórias para aumento de tributos com análise constitucional",
        "elabore parecer sobre constitucionalidade de lei municipal que proíbe aplicativos de transporte fundamentado em livre iniciativa",
        "redija contestação em ação de despejo por falta de pagamento durante pandemia com defesa baseada na teoria da imprevisão e caso fortuito",
        "analise o caso e elabore estratégia processual completa com teses principais e subsidiárias para defesa em ação de improbidade administrativa",
        "compare teses divergentes entre 1ª e 2ª Seção do STJ sobre prescrição de expurgos inflacionários em caderneta de poupança",
        "redija petição de habeas corpus preventivo com pedido liminar por excesso de prazo na prisão preventiva",
        "elabore ação civil pública ambiental contra empresa mineradora com tutela provisória de urgência e pedido de reparação integral",
        "analise viabilidade de ADPF contra omissão legislativa em matéria de direitos fundamentais sociais e redija a peça",
        "elabore recurso especial ao STJ demonstrando divergência jurisprudencial entre tribunais sobre quantum de dano moral",
        "redija mandado de segurança contra ato de autoridade coatora com pedido de liminar e demonstração de direito líquido e certo",
        "elabore parecer jurídico completo sobre fusão de empresas abordando aspectos concorrenciais trabalhistas e tributários",
        "analise e redija defesa completa em ação de responsabilidade civil por erro médico incluindo laudo pericial",
        "elabore ação popular contra ato lesivo ao patrimônio público com pedido de medida cautelar e demonstração de cidadania",
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
        "pesquise sobre prescrição intercorrente", "busque jurisprudência do TRF4",
        "qual o entendimento do STF sobre ICMS na base do PIS?",
        "pesquise sobre direito de arrependimento no e-commerce",
        "qual a legislação sobre proteção de dados?",
    ],
    "drafting": [
        "redija uma petição inicial de cobrança", "elabore uma contestação para despejo",
        "escreva um recurso de apelação", "crie uma notificação extrajudicial",
        "redija um contrato de locação comercial", "elabore petição de habeas corpus",
        "escreva um parecer jurídico tributário", "monte uma procuração ad judicia",
        "redija impugnação ao cumprimento de sentença", "elabore recurso especial para STJ",
        "escreva uma réplica à contestação", "redija embargos de declaração",
        "elabore uma petição de alimentos", "faça um contrato de prestação de serviços",
        "redija uma ação de despejo por falta de pagamento",
    ],
    "analysis": [
        "analise este caso e me diga os riscos", "qual a probabilidade de êxito?",
        "avalie a estratégia processual deste caso", "faça uma análise de risco",
        "analise os pontos fortes e fracos da defesa", "avalie se vale recorrer",
        "analise a viabilidade jurídica desta tese", "faça parecer sobre riscos contratuais",
        "analise o perfil de decisões deste juiz", "avalie os precedentes",
        "analise a possibilidade de acordo", "avalie os riscos de prosseguir",
        "compare as opções processuais disponíveis", "analise o custo-benefício",
        "faça uma análise jurimetrica do caso",
    ],
    "memory": [
        "quais casos temos sobre direito trabalhista?", "qual foi o último processo do cliente João?",
        "lembre-me dos detalhes do caso 1234", "quais processos estão pendentes?",
        "me mostre o histórico do cliente Maria", "quantos casos de consumidor ativos?",
        "qual o status dos meus processos?", "liste os prazos desta semana",
        "quais documentos preciso preparar?", "mostre o andamento processual",
    ],
    "chat": [
        "olá, tudo bem?", "obrigado pela ajuda", "pode repetir?",
        "não entendi, explique melhor", "tchau", "como você funciona?",
        "quem criou você?", "bom dia", "me ajude por favor", "ok entendi",
        "boa tarde", "boa noite", "até logo", "valeu", "pode me ajudar?",
    ],
}


def augment_text_v2(text: str, n: int = 8) -> list[str]:
    import random
    rng = random.Random(hash(text))
    prefixes = ["", "por favor, ", "preciso que ", "gostaria de saber ", "urgente: ",
                "me ajude a ", "quero ", "poderia "]
    suffixes = ["", " por favor", " obrigado", " urgente", " para um cliente",
                " para meu caso", " é urgente", ""]
    cases = [str.lower, str.title, lambda s: s]
    variations = [text]
    for _ in range(n - 1):
        t = f"{rng.choice(prefixes)}{text}{rng.choice(suffixes)}".strip()
        variations.append(rng.choice(cases)(t))
    return variations


@app.function(image=cpu_image, volumes={"/output": volume}, timeout=600)
def train_intent_enhanced():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    import joblib

    print("=" * 60)
    print("PHASE 1C: Enhanced Intent & Complexity Classifiers")
    print("=" * 60)
    results = {}

    for name, data in [("complexity_classifier", COMPLEXITY_DATA), ("intent_classifier", INTENT_DATA)]:
        print(f"\n--- Training {name} ---")
        labels_list = sorted(data.keys())
        lbl2id = {l: i for i, l in enumerate(labels_list)}

        texts, labels = [], []
        for lbl, examples in data.items():
            for ex in examples:
                for aug in augment_text_v2(ex, n=10):
                    texts.append(aug)
                    labels.append(lbl2id[lbl])

        print(f"  Samples: {len(texts)}")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
        ])

        param_grid = {
            "tfidf__max_features": [10000, 20000, 30000],
            "tfidf__ngram_range": [(1, 3), (2, 4), (2, 5)],
            "tfidf__sublinear_tf": [True],
            "clf__estimator__C": [0.5, 1.0, 2.0],
            "clf__estimator__max_iter": [5000],
        }

        gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(texts, labels)

        print(f"  Best CV accuracy: {gs.best_score_:.4f}")
        print(f"  Best params: { {k: v for k, v in gs.best_params_.items()} }")

        out_dir = Path("/output/intent")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, out_dir / f"{name}.joblib")
        config = {"labels": labels_list, "label2id": lbl2id,
                  "cv_accuracy": float(gs.best_score_), "best_params": {str(k): str(v) for k, v in gs.best_params_.items()},
                  "num_samples": len(texts)}
        with open(out_dir / f"{name}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        results[name] = {"accuracy": float(gs.best_score_), "samples": len(texts)}

    volume.commit()
    return {"job": "intent_enhanced", **results}


# ============================================================
# PHASE 1D: Generate GRPO Problems + OAB Questions
# ============================================================

GRPO_TEMPLATES = [
    "Explique detalhadamente {concept} no contexto do direito {area} brasileiro, citando a legislação aplicável.",
    "Quais são os requisitos legais para {concept}? Fundamente com artigos de lei e jurisprudência.",
    "Compare as posições doutrinárias sobre {concept} e indique a corrente majoritária nos tribunais.",
    "Analise a seguinte situação jurídica envolvendo {concept} e indique a solução legal aplicável.",
    "Redija uma fundamentação jurídica sobre {concept} para uma peça processual.",
]

LEGAL_CONCEPTS = [
    ("usucapião extraordinária", "civil"), ("responsabilidade civil objetiva", "civil"),
    ("dano moral coletivo", "consumidor"), ("tutela provisória de urgência", "processual"),
    ("prescrição intercorrente", "processual"), ("desconsideração da personalidade jurídica", "empresarial"),
    ("estabilidade provisória da gestante", "trabalhista"), ("férias proporcionais", "trabalhista"),
    ("imunidade tributária recíproca", "tributário"), ("substituição tributária", "tributário"),
    ("improbidade administrativa", "administrativo"), ("licitação dispensa e inexigibilidade", "administrativo"),
    ("alimentos gravídicos", "família"), ("guarda compartilhada", "família"),
    ("usucapião urbana", "civil"), ("direito de arrependimento", "consumidor"),
    ("assédio moral no trabalho", "trabalhista"), ("hora extra e banco de horas", "trabalhista"),
    ("ICMS na base do PIS/COFINS", "tributário"), ("responsabilidade do Estado por omissão", "administrativo"),
    ("bem de família impenhorabilidade", "civil"), ("vício do produto e do serviço", "consumidor"),
    ("penhora online e BACENJUD", "processual"), ("recurso especial requisitos", "processual"),
    ("mandado de segurança coletivo", "processual"), ("ação civil pública legitimidade", "processual"),
    ("contrato de trabalho intermitente", "trabalhista"), ("terceirização lícita", "trabalhista"),
    ("execução fiscal e exceção de pré-executividade", "tributário"),
    ("habeas corpus preventivo", "penal"),
]

OAB_TEMPLATES = [
    {
        "question": "João comprou um produto com defeito pela internet e quer devolvê-lo. Qual o prazo legal para exercer o direito de arrependimento segundo o CDC?",
        "options": {"A": "7 dias", "B": "15 dias", "C": "30 dias", "D": "90 dias"},
        "correct_answer": "A", "area": "consumidor",
    },
    {
        "question": "Para que se configure a usucapião extraordinária, qual o prazo de posse ininterrupta exigido pelo Código Civil?",
        "options": {"A": "5 anos", "B": "10 anos", "C": "15 anos", "D": "20 anos"},
        "correct_answer": "C", "area": "civil",
    },
    {
        "question": "Qual é o prazo para o réu apresentar contestação no procedimento comum do CPC/2015?",
        "options": {"A": "5 dias", "B": "15 dias", "C": "20 dias", "D": "30 dias"},
        "correct_answer": "B", "area": "processual",
    },
    {
        "question": "Segundo a CLT, qual o período máximo de duração do contrato de experiência?",
        "options": {"A": "30 dias", "B": "60 dias", "C": "90 dias", "D": "120 dias"},
        "correct_answer": "C", "area": "trabalhista",
    },
    {
        "question": "O mandado de segurança deve ser impetrado no prazo de:",
        "options": {"A": "30 dias", "B": "60 dias", "C": "90 dias", "D": "120 dias"},
        "correct_answer": "D", "area": "processual",
    },
    {
        "question": "A desconsideração da personalidade jurídica está prevista em qual artigo do Código Civil?",
        "options": {"A": "Art. 28", "B": "Art. 50", "C": "Art. 186", "D": "Art. 927"},
        "correct_answer": "B", "area": "civil",
    },
    {
        "question": "O aviso prévio proporcional ao tempo de serviço pode chegar a no máximo:",
        "options": {"A": "30 dias", "B": "60 dias", "C": "90 dias", "D": "120 dias"},
        "correct_answer": "C", "area": "trabalhista",
    },
    {
        "question": "Qual o prazo prescricional para ação de reparação civil (Art. 206, §3º, V, CC)?",
        "options": {"A": "1 ano", "B": "3 anos", "C": "5 anos", "D": "10 anos"},
        "correct_answer": "B", "area": "civil",
    },
    {
        "question": "A responsabilidade civil objetiva está prevista no Código Civil no:",
        "options": {"A": "Art. 186", "B": "Art. 187", "C": "Art. 927, parágrafo único", "D": "Art. 944"},
        "correct_answer": "C", "area": "civil",
    },
    {
        "question": "Em relação aos alimentos, a revisão pode ser feita quando houver:",
        "options": {"A": "Apenas mudança de emprego", "B": "Mudança na situação financeira de qualquer das partes", "C": "Apenas quando o filho completar 18 anos", "D": "Apenas por decisão do Ministério Público"},
        "correct_answer": "B", "area": "família",
    },
    {
        "question": "O princípio da insignificância (bagatela) é aplicável quando:",
        "options": {"A": "O réu é primário", "B": "Há mínima ofensividade, nenhuma periculosidade social, reduzido grau de reprovabilidade e inexpressividade da lesão", "C": "O valor do bem é inferior a um salário mínimo", "D": "A vítima perdoa o réu"},
        "correct_answer": "B", "area": "penal",
    },
    {
        "question": "Na execução fiscal, a exceção de pré-executividade pode ser oposta:",
        "options": {"A": "Apenas antes da penhora", "B": "A qualquer tempo, para matérias de ordem pública", "C": "Apenas após garantia do juízo", "D": "Apenas no prazo de 30 dias"},
        "correct_answer": "B", "area": "tributário",
    },
]


@app.function(image=cpu_image, volumes={"/output": volume}, timeout=300)
def generate_training_data():
    import random

    print("=" * 60)
    print("PHASE 1D: Generate GRPO Problems + OAB Questions")
    print("=" * 60)

    rng = random.Random(42)
    grpo_problems = []
    for concept, area in LEGAL_CONCEPTS:
        for tmpl in GRPO_TEMPLATES:
            prompt = tmpl.format(concept=concept, area=area)
            grpo_problems.append({"prompt": [{"role": "user", "content": prompt}]})
    rng.shuffle(grpo_problems)
    print(f"Generated {len(grpo_problems)} GRPO problems")

    out_dir = Path("/output/grpo_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "grpo_problems.jsonl", "w", encoding="utf-8") as f:
        for p in grpo_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    oab_dir = Path("/output/oab_eval")
    oab_dir.mkdir(parents=True, exist_ok=True)
    with open(oab_dir / "oab_questions.jsonl", "w", encoding="utf-8") as f:
        for q in OAB_TEMPLATES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Generated {len(OAB_TEMPLATES)} OAB evaluation questions")

    volume.commit()
    return {"grpo_problems": len(grpo_problems), "oab_questions": len(OAB_TEMPLATES)}


# ============================================================
# PHASE 2A: GAIA SFT with LoRA (QLoRA)
# ============================================================

@app.function(image=gpu_gaia_image, gpu="A10G", secrets=[es_secret],
              volumes={"/output": volume}, timeout=3600)
def train_gaia_sft():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    print("=" * 60)
    print("PHASE 2A: GAIA SFT with LoRA (QLoRA 4-bit)")
    print("=" * 60)

    sft_path = Path("/output/gaia_sft/sft_train.jsonl")
    val_path = Path("/output/gaia_sft/sft_val.jsonl")

    if not sft_path.exists():
        print("ERROR: SFT data not found. Run phase 1 first.")
        return {"job": "gaia_sft", "error": "No SFT data"}

    train_ds = load_dataset("json", data_files=str(sft_path), split="train")
    val_ds = None
    if val_path.exists():
        val_ds = load_dataset("json", data_files=str(val_path), split="train")

    print(f"Training samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")

    model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    print(f"Loading {model_name} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    max_len = 1024
    pad_id = tokenizer.pad_token_id or 0

    def tokenize_fn(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_len,
                        padding="max_length", return_attention_mask=True)
        labels = [t if t != pad_id else -100 for t in enc["input_ids"]]
        token_type_ids = [0] * len(enc["input_ids"])
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels, "token_type_ids": token_type_ids}

    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    train_tokenized.set_format("torch")
    val_tokenized = None
    if val_ds:
        val_tokenized = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names)
        val_tokenized.set_format("torch")

    training_args = TrainingArguments(
        output_dir="/tmp/sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=default_data_collator,
    )

    print("Starting SFT training...")
    result = trainer.train()
    print(f"Training loss: {result.training_loss:.4f}")

    save_dir = Path("/output/gaia_sft_model")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump({"training_loss": result.training_loss,
                    "epochs": 3, "samples": len(train_ds),
                    "lora_r": 64, "lora_alpha": 32}, f, indent=2)

    print(f"SFT model saved to {save_dir}")
    volume.commit()
    return {"job": "gaia_sft", "training_loss": result.training_loss, "samples": len(train_ds)}


# ============================================================
# PHASE 2B: GAIA GRPO with Reward Functions
# ============================================================

KNOWN_ARTICLE_RANGES = {
    "CF": 250, "CC": 2046, "CPC": 1072, "CLT": 922,
    "CDC": 119, "CTN": 185, "CP": 361, "CPP": 643, "ECA": 267,
}


def grpo_format_reward(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = []
    for c in completions:
        score = 0.0
        if "<think>" in c and "</think>" in c:
            score += 0.4
            think = c.split("<think>")[1].split("</think>")[0]
            if len(think) > 50:
                score += 0.2
        if "###" in c or "**" in c:
            score += 0.2
        if "art." in c.lower() or "lei" in c.lower():
            score += 0.2
        rewards.append(score)
    return rewards


def grpo_citation_reward(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = []
    for c in completions:
        refs = re.findall(r"art\.?\s*(\d+)[\w\s,]*(?:d[oa]\s+)?(\w{2,4})", c.lower())
        valid, total = 0, 0
        for art_num, law in refs:
            total += 1
            law_up = law.upper()
            if law_up in KNOWN_ARTICLE_RANGES:
                try:
                    if 1 <= int(art_num) <= KNOWN_ARTICLE_RANGES[law_up]:
                        valid += 1
                except ValueError:
                    pass
            else:
                valid += 1
        if total == 0:
            rewards.append(-0.1)
        else:
            rewards.append(valid / total)
    return rewards


def grpo_language_reward(completions: list[str], **kwargs: Any) -> list[float]:
    rewards = []
    pt_markers = ["de", "do", "da", "dos", "das", "que", "para", "com", "por", "não", "em"]
    for c in completions:
        words = c.lower().split()[:200]
        if not words:
            rewards.append(0.0)
            continue
        pt_count = sum(1 for w in words if w in pt_markers)
        ratio = pt_count / len(words)
        rewards.append(min(1.0, ratio * 5))
    return rewards


@app.function(image=gpu_gaia_image, gpu="A10G",
              volumes={"/output": volume}, timeout=7200)
def train_gaia_grpo():
    """
    Reward-based refinement: generate completions, score with rewards,
    then do rejection sampling + SFT on the best completions.
    More robust than GRPO trainer which has API compatibility issues.
    """
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from peft import PeftModel
    from datasets import Dataset

    print("=" * 60)
    print("PHASE 2B: Reward-Based Refinement (Rejection Sampling)")
    print("=" * 60)

    sft_dir = Path("/output/gaia_sft_model")
    problems_path = Path("/output/grpo_data/grpo_problems.jsonl")

    if not sft_dir.exists():
        print("ERROR: SFT model not found. Run SFT first.")
        return {"job": "gaia_grpo", "error": "No SFT model"}

    if not problems_path.exists():
        print("ERROR: GRPO problems not found.")
        return {"job": "gaia_grpo", "error": "No GRPO data"}

    problems = []
    with open(problems_path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"GRPO problems: {len(problems)}")

    base_model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    print(f"Loading base model + SFT adapters...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Generating completions and scoring with rewards...")
    best_texts = []
    n_gens = 3

    for i, prob in enumerate(problems[:80]):
        prompt_text = prob["prompt"][0]["content"]
        full_prompt = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        completions = []
        for _ in range(n_gens):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=1024, temperature=0.8,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                    top_p=0.95)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            completions.append(text)

        fmt_scores = grpo_format_reward(completions)
        cit_scores = grpo_citation_reward(completions)
        lang_scores = grpo_language_reward(completions)
        total_scores = [f + c + l for f, c, l in zip(fmt_scores, cit_scores, lang_scores)]

        best_idx = max(range(len(total_scores)), key=lambda j: total_scores[j])
        best_completion = completions[best_idx]

        best_texts.append(full_prompt + best_completion + "<end_of_turn>")

        if (i + 1) % 10 == 0:
            avg_score = sum(total_scores) / len(total_scores)
            print(f"  [{i+1}/{min(80, len(problems))}] avg_reward={avg_score:.3f} best={total_scores[best_idx]:.3f}")

    print(f"\nCollected {len(best_texts)} high-reward completions")
    print("Fine-tuning on best completions (rejection sampling SFT)...")

    max_len = 1024
    pad_id = tokenizer.pad_token_id or 0

    def tokenize_fn(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_len,
                        padding="max_length", return_attention_mask=True)
        labels = [t if t != pad_id else -100 for t in enc["input_ids"]]
        token_type_ids = [0] * len(enc["input_ids"])
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels, "token_type_ids": token_type_ids}

    ds = Dataset.from_dict({"text": best_texts})
    ds_tok = ds.map(tokenize_fn, remove_columns=["text"])
    ds_tok.set_format("torch")

    training_args = TrainingArguments(
        output_dir="/tmp/grpo_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=ds_tok, data_collator=default_data_collator,
    )

    result = trainer.train()
    print(f"Refinement training loss: {result.training_loss:.4f}")

    save_dir = Path("/output/gaia_grpo_model")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump({"training_loss": result.training_loss,
                    "method": "rejection_sampling_sft",
                    "n_problems": len(problems), "n_best": len(best_texts),
                    "n_gens_per_problem": n_gens}, f, indent=2)

    volume.commit()
    return {"job": "gaia_grpo", "training_loss": result.training_loss, "best_completions": len(best_texts)}


# ============================================================
# PHASE 3: OAB Evaluation
# ============================================================

@app.function(image=gpu_gaia_image, gpu="A10G",
              volumes={"/output": volume}, timeout=1800)
def evaluate_oab():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("=" * 60)
    print("PHASE 3: OAB Benchmark Evaluation")
    print("=" * 60)

    questions_path = Path("/output/oab_eval/oab_questions.jsonl")
    if not questions_path.exists():
        return {"job": "oab_eval", "error": "No OAB questions"}

    questions = []
    with open(questions_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"OAB questions: {len(questions)}")

    grpo_dir = Path("/output/gaia_grpo_model")
    sft_dir = Path("/output/gaia_sft_model")
    model_dir = grpo_dir if grpo_dir.exists() else sft_dir

    if not model_dir.exists():
        print("No fine-tuned model found, evaluating base GAIA...")
        model_dir = None

    base_model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
        attn_implementation="eager",
    )

    if model_dir:
        print(f"Loading fine-tuned adapters from {model_dir}...")
        model = PeftModel.from_pretrained(model, str(model_dir))

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    correct = 0
    results = []
    area_stats: dict[str, dict[str, int]] = {}

    for i, q in enumerate(questions):
        options_text = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
        prompt = f"""<start_of_turn>user
{q['question']}

Alternativas:
{options_text}

Responda APENAS com a letra da alternativa correta (A, B, C ou D).<end_of_turn>
<start_of_turn>model
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = re.search(r"[A-D]", response.strip().upper()[:20])
        pred_letter = pred.group(0) if pred else "?"
        gold = q["correct_answer"]
        is_correct = pred_letter == gold

        if is_correct:
            correct += 1

        area = q.get("area", "geral")
        if area not in area_stats:
            area_stats[area] = {"correct": 0, "total": 0}
        area_stats[area]["total"] += 1
        if is_correct:
            area_stats[area]["correct"] += 1

        results.append({"q": i, "pred": pred_letter, "gold": gold, "ok": is_correct, "area": area})
        print(f"  [{i+1}/{len(questions)}] {pred_letter} vs {gold} {'OK' if is_correct else 'WRONG'}")

    accuracy = correct / len(questions) if questions else 0
    print(f"\nOAB Accuracy: {accuracy:.2%} ({correct}/{len(questions)})")
    for area, stats in sorted(area_stats.items()):
        a = stats["correct"] / stats["total"]
        print(f"  {area}: {a:.2%} ({stats['correct']}/{stats['total']})")

    out_dir = Path("/output/oab_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": len(questions),
                    "area_stats": area_stats, "model": str(model_dir or base_model_name),
                    "results": results}, f, indent=2, ensure_ascii=False)

    volume.commit()
    return {"job": "oab_eval", "accuracy": accuracy, "correct": correct, "total": len(questions)}


# ============================================================
# ORCHESTRATOR
# ============================================================

@app.local_entrypoint()
def main(phase: int = 0):
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  JURIS.AI — Full ML Training Pipeline")
    print(f"  Phase: {'ALL' if phase == 0 else phase}")
    print(f"{'='*60}\n")

    if phase in (0, 1):
        print(">>> PHASE 1: Parallel improvements")
        print("    Spawning 4 jobs in parallel...\n")

        bert_call = train_bert_classifier.spawn()
        xgb_call = train_xgboost_enhanced.spawn()
        intent_call = train_intent_enhanced.spawn()
        data_call = generate_training_data.spawn()

        data_result = data_call.get()
        print(f"\n  Data generation: {json.dumps(data_result, default=str)}")

        intent_result = intent_call.get()
        print(f"  Intent/Complexity: {json.dumps(intent_result, default=str)}")

        xgb_result = xgb_call.get()
        print(f"  XGBoost: {json.dumps(xgb_result, default=str)}")

        bert_result = bert_call.get()
        print(f"  BERT Classifier: {json.dumps(bert_result, default=str)}")

        print(f"\n  Phase 1 completed in {time.time()-t0:.0f}s")

    if phase in (0, 2):
        t1 = time.time()
        print("\n>>> PHASE 2: Sequential GAIA pipeline")

        print("\n  [2A] SFT Training...")
        sft_result = train_gaia_sft.remote()
        print(f"  SFT result: {json.dumps(sft_result, default=str)}")

        if "error" not in sft_result:
            print("\n  [2B] GRPO Training...")
            try:
                grpo_result = train_gaia_grpo.remote()
                print(f"  GRPO result: {json.dumps(grpo_result, default=str)}")
            except Exception as e:
                print(f"  GRPO failed: {e}")
                grpo_result = {"error": str(e)}
        else:
            grpo_result = {"skipped": "SFT failed"}

        print(f"\n  Phase 2 completed in {time.time()-t1:.0f}s")

    if phase in (0, 3):
        print("\n>>> PHASE 3: OAB Evaluation")
        eval_result = evaluate_oab.remote()
        print(f"  OAB result: {json.dumps(eval_result, default=str)}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Full pipeline completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Models saved to Modal Volume: jurisai-trained-models")
    print(f"{'='*60}")

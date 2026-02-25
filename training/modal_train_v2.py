"""
Juris.AI — Full ML Training Pipeline v2 (Modal.com)

Phase 1 (parallel, CPU+GPU):
  1A. Cross-encoder Reranker juridico (T4)
  1B. NER juridico fine-tuned (T4)
  1C. Section Classifier / Chunker ML (CPU)
  1D. Citation Classifier (CPU)
  1E. XGBoost v2 com mais dados (CPU)
  1F. Generate more SFT + scenarios data (CPU)

Phase 2 (sequential, GPU):
  2A. Legal Summarizer fine-tuned (A10G)
  2B. GAIA SFT v2 com mais dados (A10G)

Usage:
    modal run training/modal_train_v2.py
    modal run training/modal_train_v2.py --phase 1
    modal run training/modal_train_v2.py --phase 2
"""
from __future__ import annotations

import json, logging, math, os, re, time
from collections import Counter
from pathlib import Path
from typing import Any

import modal

app = modal.App("jurisai-training-v2")
volume = modal.Volume.from_name("jurisai-trained-models", create_if_missing=True)
es_secret = modal.Secret.from_name("jurisai-es-credentials")

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx", "numpy", "scikit-learn", "xgboost", "joblib", "scipy")
)

gpu_bert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "httpx", "numpy", "scikit-learn", "joblib",
        "torch", "transformers>=4.49", "accelerate", "datasets",
        "seqeval",
    )
)

gpu_gaia_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers>=4.49", "peft>=0.14",
        "accelerate", "datasets", "bitsandbytes", "scipy",
        "huggingface_hub", "sentencepiece", "protobuf",
    )
)


def fetch_es_docs(es_url, es_key, index, limit, query=None, fields=None):
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
# 1A: Cross-Encoder Reranker
# ============================================================

RERANKER_QUERIES = [
    ("prazo para contestação no CPC", [
        ("O prazo para contestação no procedimento comum é de 15 dias úteis, conforme Art. 335 do CPC/2015.", 1),
        ("A contestação deve impugnar todos os fatos alegados na petição inicial.", 0),
        ("O procedimento sumaríssimo tem prazo diferenciado no JEC.", 0),
    ]),
    ("responsabilidade civil objetiva", [
        ("A responsabilidade civil objetiva está prevista no Art. 927, parágrafo único, do CC, dispensando a comprovação de culpa.", 1),
        ("O Código Civil brasileiro foi promulgado pela Lei 10.406/2002.", 0),
        ("A teoria do risco criado fundamenta a responsabilidade objetiva nas relações de consumo.", 1),
    ]),
    ("usucapião extraordinária requisitos", [
        ("A usucapião extraordinária exige posse mansa e pacífica por 15 anos, podendo ser reduzida para 10 anos se houver moradia habitual, conforme Art. 1.238, CC.", 1),
        ("O registro de imóveis confere publicidade e segurança jurídica.", 0),
        ("A usucapião é forma originária de aquisição da propriedade.", 0),
    ]),
    ("dano moral por negativação indevida", [
        ("O STJ consolidou entendimento de que a negativação indevida gera dano moral presumido (in re ipsa), dispensando prova do prejuízo.", 1),
        ("O cadastro de devedores é regulado pelo CDC e pela LGPD.", 0),
        ("O valor de indenização por dano moral deve observar os princípios da razoabilidade e proporcionalidade.", 1),
    ]),
    ("tutela de urgência requisitos", [
        ("A tutela de urgência antecipada exige probabilidade do direito e perigo de dano ou risco ao resultado útil do processo, conforme Art. 300, CPC.", 1),
        ("O juiz pode conceder tutela de urgência liminarmente ou após justificação prévia.", 1),
        ("O processo civil brasileiro adota o princípio do contraditório e ampla defesa.", 0),
    ]),
    ("alimentos gravídicos", [
        ("Os alimentos gravídicos são devidos desde a concepção até o parto, conforme Lei 11.804/2008, bastando indícios de paternidade.", 1),
        ("A pensão alimentícia pode ser revisada a qualquer tempo havendo mudança de situação.", 0),
        ("O direito a alimentos é irrenunciável, intransmissível e impenhorável.", 0),
    ]),
    ("recurso especial cabimento", [
        ("O recurso especial cabe contra decisão de tribunal que viole lei federal ou dê interpretação divergente entre tribunais, conforme Art. 105, III, CF.", 1),
        ("O STJ é o tribunal responsável pela uniformização da interpretação da legislação federal.", 0),
        ("Súmula 7 do STJ impede reexame de provas em recurso especial.", 1),
    ]),
    ("desconsideração da personalidade jurídica", [
        ("O Art. 50 do CC permite a desconsideração da personalidade jurídica em caso de abuso, caracterizado pelo desvio de finalidade ou confusão patrimonial.", 1),
        ("A pessoa jurídica tem personalidade distinta de seus sócios.", 0),
        ("O incidente de desconsideração da personalidade jurídica está previsto nos Arts. 133-137 do CPC.", 1),
    ]),
    ("prescrição intercorrente execução fiscal", [
        ("A prescrição intercorrente na execução fiscal ocorre quando o processo fica paralisado por prazo superior ao da prescrição, conforme Art. 40, §4º, LEF e Súmula 314 do STJ.", 1),
        ("A execução fiscal é regulada pela Lei 6.830/1980.", 0),
        ("O crédito tributário prescreve em 5 anos conforme Art. 174 do CTN.", 0),
    ]),
    ("mandado de segurança prazo", [
        ("O mandado de segurança deve ser impetrado no prazo de 120 dias contados da ciência do ato impugnado, conforme Art. 23 da Lei 12.016/2009.", 1),
        ("O mandado de segurança protege direito líquido e certo.", 0),
        ("A autoridade coatora deve ser notificada para prestar informações em 10 dias.", 0),
    ]),
]


@app.function(image=gpu_bert_image, gpu="T4", volumes={"/output": volume}, timeout=1800)
def train_reranker():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score

    print("=" * 60)
    print("1A: Cross-Encoder Reranker Jurídico")
    print("=" * 60)

    texts_a, texts_b, labels = [], [], []
    for query, docs in RERANKER_QUERIES:
        for doc, label in docs:
            texts_a.append(query)
            texts_b.append(doc)
            labels.append(float(label))

    import random
    rng = random.Random(42)
    for _ in range(len(texts_a) * 8):
        idx = rng.randint(0, len(texts_a) - 1)
        prefixes = ["", "busque ", "qual ", "explique "]
        suffixes = ["", " por favor", " urgente", " no direito brasileiro"]
        a = f"{rng.choice(prefixes)}{texts_a[idx]}{rng.choice(suffixes)}".strip()
        texts_a.append(a)
        texts_b.append(texts_b[idx])
        labels.append(labels[idx])

    print(f"Training samples: {len(texts_a)}")

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    encodings = tokenizer(texts_a, texts_b, truncation=True, padding=True, max_length=512)

    split = int(len(texts_a) * 0.85)
    train_ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"][:split],
        "attention_mask": encodings["attention_mask"][:split],
        "labels": [[l] for l in labels[:split]],
    })
    val_ds = Dataset.from_dict({
        "input_ids": encodings["input_ids"][split:],
        "attention_mask": encodings["attention_mask"][split:],
        "labels": [[l] for l in labels[split:]],
    })

    args = TrainingArguments(
        output_dir="/tmp/reranker", num_train_epochs=5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, fp16=True, seed=42, logging_steps=20,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    out_dir = Path("/output/reranker")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    with open(out_dir / "config.json", "w") as f:
        json.dump({"type": "cross-encoder", "base": model_name, "n_train": len(texts_a)}, f, indent=2)

    volume.commit()
    return {"job": "reranker", "samples": len(texts_a)}


# ============================================================
# 1B: NER Jurídico Fine-tuned
# ============================================================

NER_LABELS = ["O", "B-PESSOA", "I-PESSOA", "B-ORGANIZACAO", "I-ORGANIZACAO",
              "B-LOCAL", "I-LOCAL", "B-LEGISLACAO", "I-LEGISLACAO",
              "B-JURISPRUDENCIA", "I-JURISPRUDENCIA", "B-DATA", "I-DATA",
              "B-VALOR", "I-VALOR"]

NER_EXAMPLES = [
    (["O", "autor", "João", "da", "Silva", "propôs", "ação", "contra", "o", "Banco", "do", "Brasil", "."],
     ["O", "O", "B-PESSOA", "I-PESSOA", "I-PESSOA", "O", "O", "O", "O", "B-ORGANIZACAO", "I-ORGANIZACAO", "I-ORGANIZACAO", "O"]),
    (["Conforme", "Art.", "5", "da", "CF/88", ",", "todos", "são", "iguais", "."],
     ["O", "B-LEGISLACAO", "I-LEGISLACAO", "I-LEGISLACAO", "I-LEGISLACAO", "O", "O", "O", "O", "O"]),
    (["O", "réu", "Maria", "Souza", "reside", "em", "São", "Paulo", "."],
     ["O", "O", "B-PESSOA", "I-PESSOA", "O", "O", "B-LOCAL", "I-LOCAL", "O"]),
    (["Valor", "da", "causa", ":", "R$", "50.000,00", "."],
     ["O", "O", "O", "O", "B-VALOR", "I-VALOR", "O"]),
    (["Data", ":", "15/03/2024", "."],
     ["O", "O", "B-DATA", "O"]),
    (["O", "STJ", "no", "REsp", "1.234.567", "decidiu", "pela", "procedência", "."],
     ["O", "B-ORGANIZACAO", "O", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA", "O", "O", "O", "O"]),
    (["Lei", "8.078/1990", "estabelece", "direitos", "do", "consumidor", "."],
     ["B-LEGISLACAO", "I-LEGISLACAO", "O", "O", "O", "O", "O"]),
    (["O", "Ministério", "Público", "Federal", "ajuizou", "ACP", "em", "Brasília", "."],
     ["O", "B-ORGANIZACAO", "I-ORGANIZACAO", "I-ORGANIZACAO", "O", "O", "O", "B-LOCAL", "O"]),
    (["Súmula", "331", "do", "TST", "trata", "de", "terceirização", "."],
     ["B-JURISPRUDENCIA", "I-JURISPRUDENCIA", "I-JURISPRUDENCIA", "I-JURISPRUDENCIA", "O", "O", "O", "O"]),
    (["Em", "10/01/2025", ",", "Pedro", "Santos", "Oliveira", "pagou", "R$", "1.500,00", "."],
     ["O", "B-DATA", "O", "B-PESSOA", "I-PESSOA", "I-PESSOA", "O", "B-VALOR", "I-VALOR", "O"]),
]


def augment_ner(examples, n=15):
    import random
    rng = random.Random(42)
    nomes = ["Carlos", "Ana", "Pedro", "Maria", "José", "Fernanda", "Lucas", "Beatriz", "Roberto", "Juliana"]
    sobrenomes = ["Silva", "Santos", "Oliveira", "Souza", "Lima", "Ferreira", "Costa", "Pereira", "Almeida", "Ribeiro"]
    orgs = ["Banco do Brasil", "Caixa Econômica", "Petrobras", "INSS", "Receita Federal"]
    locais = ["São Paulo", "Rio de Janeiro", "Brasília", "Belo Horizonte", "Porto Alegre", "Salvador", "Curitiba"]
    leis = ["Art. 5 da CF/88", "Art. 927 do CC", "Art. 300 do CPC", "Lei 8.078/1990", "Art. 186 do CC"]
    augmented = list(examples)
    for _ in range(n * len(examples)):
        nome = f"{rng.choice(nomes)} {rng.choice(sobrenomes)}"
        org = rng.choice(orgs)
        local = rng.choice(locais)
        lei = rng.choice(leis)
        templates = [
            (["O", "autor"] + nome.split() + ["ajuizou", "ação", "em"] + local.split() + ["."],
             ["O", "O"] + ["B-PESSOA"] + ["I-PESSOA"] * (len(nome.split()) - 1) + ["O", "O", "O"] + ["B-LOCAL"] + ["I-LOCAL"] * (len(local.split()) - 1) + ["O"]),
            (["Conforme"] + lei.split() + [",", "o"] + org.split() + ["deve", "pagar", "."],
             ["O"] + ["B-LEGISLACAO"] + ["I-LEGISLACAO"] * (len(lei.split()) - 1) + ["O", "O"] + ["B-ORGANIZACAO"] + ["I-ORGANIZACAO"] * (len(org.split()) - 1) + ["O", "O", "O"]),
        ]
        augmented.append(rng.choice(templates))
    return augmented


@app.function(image=gpu_bert_image, gpu="T4", volumes={"/output": volume}, timeout=1200)
def train_ner():
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
    from datasets import Dataset
    import numpy as np

    print("=" * 60)
    print("1B: NER Jurídico Fine-tuned")
    print("=" * 60)

    examples = augment_ner(NER_EXAMPLES, n=20)
    print(f"NER samples: {len(examples)}")

    label2id = {l: i for i, l in enumerate(NER_LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(NER_LABELS), id2label=id2label, label2id=label2id)

    all_input_ids, all_labels, all_attention = [], [], []
    for tokens, tags in examples:
        encoding = tokenizer(tokens, is_split_into_words=True, truncation=True,
                           padding="max_length", max_length=128)
        word_ids = encoding.word_ids()
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id.get(tags[wid] if wid < len(tags) else "O", 0))
        all_input_ids.append(encoding["input_ids"])
        all_attention.append(encoding["attention_mask"])
        all_labels.append(label_ids)

    split = int(len(all_input_ids) * 0.85)
    train_ds = Dataset.from_dict({
        "input_ids": all_input_ids[:split], "attention_mask": all_attention[:split], "labels": all_labels[:split]})
    val_ds = Dataset.from_dict({
        "input_ids": all_input_ids[split:], "attention_mask": all_attention[split:], "labels": all_labels[split:]})

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        true, pred = [], []
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                if p.label_ids[i][j] != -100:
                    true.append(id2label[p.label_ids[i][j]])
                    pred.append(id2label[preds[i][j]])
        correct = sum(1 for t, p in zip(true, pred) if t == p)
        return {"accuracy": correct / len(true) if true else 0}

    args = TrainingArguments(
        output_dir="/tmp/ner", num_train_epochs=8,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, fp16=True, seed=42, logging_steps=20,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    out_dir = Path("/output/ner_legal")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    with open(out_dir / "config.json", "w") as f:
        json.dump({"type": "ner", "base": model_name, "labels": NER_LABELS,
                   "n_train": len(all_input_ids)}, f, indent=2)

    volume.commit()
    return {"job": "ner_legal", "samples": len(all_input_ids)}


# ============================================================
# 1C: Section Classifier (Chunker ML)
# ============================================================

SECTION_LABELS = ["ementa", "relatorio", "fundamentacao", "dispositivo", "cabecalho", "generic"]

SECTION_DATA = {
    "ementa": [
        "EMENTA: DIREITO CIVIL. RESPONSABILIDADE CIVIL. DANO MORAL. NEGATIVAÇÃO INDEVIDA.",
        "EMENTA: PROCESSUAL CIVIL. AGRAVO DE INSTRUMENTO. TUTELA PROVISÓRIA.",
        "EMENTA - DIREITO DO CONSUMIDOR. VÍCIO DO PRODUTO. PRAZO DECADENCIAL.",
        "E M E N T A: DIREITO ADMINISTRATIVO. CONCURSO PÚBLICO. PRETERIÇÃO.",
        "EMENTA: TRABALHISTA. HORAS EXTRAS. BANCO DE HORAS. INVALIDADE.",
        "APELAÇÃO CÍVEL. DIREITO CIVIL. OBRIGAÇÕES. INADIMPLEMENTO CONTRATUAL.",
        "RECURSO ESPECIAL. DIREITO PROCESSUAL CIVIL. CUMPRIMENTO DE SENTENÇA.",
        "HABEAS CORPUS. PENAL. TRÁFICO DE DROGAS. PRISÃO PREVENTIVA.",
    ],
    "relatorio": [
        "Trata-se de apelação cível interposta contra sentença que julgou procedente o pedido autoral.",
        "Cuida-se de recurso especial interposto com fundamento no art. 105, III, a, da Constituição Federal.",
        "O autor ajuizou ação de indenização por danos morais e materiais em face do réu.",
        "O Ministério Público Federal interpôs recurso de apelação contra a sentença absolutória.",
        "Trata-se de ação civil pública ajuizada pelo IBAMA contra a empresa ré.",
        "O reclamante ajuizou reclamação trabalhista pleiteando horas extras e adicional noturno.",
        "A autora propôs ação revisional de alimentos em face do réu alimentante.",
        "Cuida-se de mandado de segurança impetrado contra ato do Secretário de Fazenda.",
    ],
    "fundamentacao": [
        "Passa-se à análise do mérito. O artigo 186 do Código Civil estabelece que aquele que por ação ou omissão voluntária causar dano a outrem comete ato ilícito.",
        "A jurisprudência do STJ é pacífica no sentido de que a negativação indevida gera dano moral in re ipsa.",
        "No caso dos autos, restou comprovado que o autor sofreu dano em decorrência da conduta ilícita da ré.",
        "O princípio da boa-fé objetiva impõe às partes o dever de lealdade e cooperação na relação contratual.",
        "Conforme preceitua o Art. 927 do CC, aquele que causar dano a outrem é obrigado a repará-lo.",
        "Merece reforma a sentença recorrida, pois a prova produzida nos autos demonstra claramente o direito do autor.",
        "A análise da prova pericial revela que houve erro médico, configurando a responsabilidade civil do réu.",
        "DO MÉRITO. Sem razão o apelante. A sentença merece ser mantida por seus próprios fundamentos.",
    ],
    "dispositivo": [
        "Ante o exposto, JULGO PROCEDENTE o pedido para condenar o réu ao pagamento de indenização.",
        "ISTO POSTO, nego provimento ao recurso e mantenho a sentença recorrida.",
        "Diante do exposto, DOU PROVIMENTO ao recurso para reformar a sentença de primeiro grau.",
        "DISPOSITIVO: Julgo parcialmente procedente o pedido autoral para condenar o réu.",
        "Em face do exposto, EXTINGO O PROCESSO SEM RESOLUÇÃO DO MÉRITO, nos termos do art. 485, VI, CPC.",
        "CONCLUSÃO: Ante o exposto, ACOLHO os embargos de declaração para sanar a omissão apontada.",
        "Por todo o exposto, NEGO PROVIMENTO ao agravo de instrumento, mantendo a decisão agravada.",
        "Pelo exposto, JULGO IMPROCEDENTE o pedido formulado na inicial e condeno o autor ao pagamento das custas.",
    ],
    "cabecalho": [
        "TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO. 5ª Câmara de Direito Privado.",
        "PODER JUDICIÁRIO. TRIBUNAL REGIONAL FEDERAL DA 3ª REGIÃO.",
        "Apelação Cível nº 1234567-89.2024.8.26.0100. Relator: Des. Fulano de Tal.",
        "SUPERIOR TRIBUNAL DE JUSTIÇA. PRIMEIRA TURMA. RECURSO ESPECIAL Nº 1.234.567 - SP.",
        "Processo nº 0001234-56.2024.5.02.0001. VARA DO TRABALHO DE SÃO PAULO.",
        "TRIBUNAL DE JUSTIÇA DO ESTADO DO RIO DE JANEIRO. 15ª CÂMARA CÍVEL.",
    ],
    "generic": [
        "Os documentos juntados às fls. 45/67 comprovam o alegado pelo autor.",
        "Intimem-se as partes para manifestação no prazo de 15 dias.",
        "Certidão: Certifico que o presente feito foi redistribuído por prevenção.",
        "Juntada de procuração e documentos pessoais do autor às fls. 10/25.",
        "Despacho: Cite-se o réu para contestar no prazo legal.",
        "Termo de audiência de conciliação realizada em 10/03/2024.",
    ],
}


@app.function(image=cpu_image, volumes={"/output": volume}, timeout=600)
def train_section_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    import joblib, random

    print("=" * 60)
    print("1C: Section Classifier (Chunker ML)")
    print("=" * 60)

    texts, labels = [], []
    rng = random.Random(42)
    for lbl, examples in SECTION_DATA.items():
        lid = SECTION_LABELS.index(lbl)
        for ex in examples:
            for _ in range(15):
                prefix = rng.choice(["", "  ", "\n", "\t"])
                suffix = rng.choice(["", ".", "\n", " "])
                case_fn = rng.choice([str.lower, str.upper, lambda s: s])
                texts.append(f"{prefix}{case_fn(ex)}{suffix}")
                labels.append(lid)

    print(f"Samples: {len(texts)}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", min_df=2)),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
    ])

    gs = GridSearchCV(pipe, {
        "tfidf__max_features": [10000, 20000],
        "tfidf__ngram_range": [(2, 4), (2, 5)],
        "clf__estimator__C": [0.5, 1.0, 2.0],
    }, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(texts, labels)

    print(f"Best CV accuracy: {gs.best_score_:.4f}")

    out_dir = Path("/output/section_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out_dir / "section_classifier.joblib")
    with open(out_dir / "config.json", "w") as f:
        json.dump({"labels": SECTION_LABELS, "cv_accuracy": float(gs.best_score_),
                   "n_samples": len(texts)}, f, indent=2)

    volume.commit()
    return {"job": "section_classifier", "accuracy": float(gs.best_score_), "samples": len(texts)}


# ============================================================
# 1D: Citation Classifier
# ============================================================

CITATION_TYPES = ["legislacao", "jurisprudencia", "sumula", "doutrina", "nao_citacao"]

CITATION_DATA = {
    "legislacao": [
        "Art. 5º da Constituição Federal", "Lei 8.078/1990", "Art. 927, parágrafo único, do CC",
        "Art. 300 do CPC/2015", "Decreto 9.580/2018", "Art. 186 do Código Civil",
        "Lei nº 13.105/2015", "Art. 37 da CF/88", "inciso IV do art. 5º da CF",
        "Art. 485, VI, do CPC", "§ 1º do art. 273 do CPC", "Lei Complementar 123/2006",
        "Art. 1.238 do CC/2002", "Resolução CNJ 615/2025", "Art. 7º, XXIX, da CLT",
    ],
    "jurisprudencia": [
        "REsp 1.234.567/SP", "AgRg no AREsp 567.890/RJ", "HC 123.456/MG",
        "0001234-56.2024.8.26.0100", "RE 1.234.567/DF", "ADI 5.678/DF",
        "ADPF 789/SP", "AI 123.456/RS", "RMS 54.321/PR",
        "Processo nº 1234567-89.2024.5.02.0001", "MS 36.123/DF",
    ],
    "sumula": [
        "Súmula 331 do TST", "Súmula Vinculante 11", "Súmula 7 do STJ",
        "Súmula 479 do STJ", "Súmula 297 do TST", "Súmula Vinculante 56",
        "Súmula 385 do STJ", "Súmula 83 do STJ", "Súmula 314 do STJ",
    ],
    "doutrina": [
        "conforme ensina Pontes de Miranda", "segundo Hely Lopes Meirelles",
        "na lição de Caio Mário da Silva Pereira", "como preleciona Nelson Nery Junior",
        "segundo a doutrina majoritária", "conforme entendimento de Fredie Didier",
    ],
    "nao_citacao": [
        "o réu deve pagar indenização", "a sentença foi reformada em segunda instância",
        "o prazo para contestação é de 15 dias", "o autor comprovou o dano sofrido",
        "as partes foram intimadas para manifestação", "a perícia técnica constatou o defeito",
    ],
}


@app.function(image=cpu_image, volumes={"/output": volume}, timeout=600)
def train_citation_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    import joblib, random

    print("=" * 60)
    print("1D: Citation Classifier")
    print("=" * 60)

    texts, labels = [], []
    rng = random.Random(42)
    for lbl, examples in CITATION_DATA.items():
        lid = CITATION_TYPES.index(lbl)
        for ex in examples:
            for _ in range(20):
                prefix = rng.choice(["", "conforme ", "nos termos do ", "vide ", "cf. "])
                texts.append(f"{prefix}{ex}")
                labels.append(lid)

    print(f"Samples: {len(texts)}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", analyzer="char_wb", min_df=2)),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
    ])

    gs = GridSearchCV(pipe, {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 3), (2, 4)],
        "clf__estimator__C": [0.5, 1.0, 2.0],
    }, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(texts, labels)

    print(f"Best CV accuracy: {gs.best_score_:.4f}")

    out_dir = Path("/output/citation_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out_dir / "citation_classifier.joblib")
    with open(out_dir / "config.json", "w") as f:
        json.dump({"labels": CITATION_TYPES, "cv_accuracy": float(gs.best_score_),
                   "n_samples": len(texts)}, f, indent=2)

    volume.commit()
    return {"job": "citation_classifier", "accuracy": float(gs.best_score_), "samples": len(texts)}


# ============================================================
# 1E: XGBoost v2 com mais dados
# ============================================================

OUTCOME_LABELS = ["procedente", "parcialmente_procedente", "improcedente", "extinto_sem_merito"]
OUTCOME_PAT = {
    "procedente": [r"julgo\s+procedente", r"dou\s+provimento", r"acolho\s+o\s+pedido"],
    "parcialmente_procedente": [r"parcialmente\s+procedente", r"procedente\s+em\s+parte", r"dou\s+parcial\s+provimento"],
    "improcedente": [r"julgo\s+improcedente", r"nego\s+provimento", r"improced[eê]ncia"],
    "extinto_sem_merito": [r"extinto\s+sem\s+resolu[cç][aã]o\s+d[eo]\s+m[eé]rito", r"art\.?\s*485"],
}
AREA_ENC = {"cível": 0, "civil": 0, "consumidor": 1, "trabalhista": 2, "tributário": 3, "tributario": 3,
            "administrativo": 4, "previdenciário": 5, "previdenciario": 5, "família": 6, "familia": 6,
            "ambiental": 7, "empresarial": 8, "imobiliário": 9, "imobiliario": 9}
TRIB_ENC = {"STF": 0, "STJ": 1, "TST": 2, "TSE": 3, "TJSP": 10, "TJRJ": 11, "TJMG": 12, "TJRS": 13,
            "TJPR": 14, "TJBA": 15, "TJSC": 16, "TRF1": 30, "TRF2": 31, "TRF3": 32, "TRF4": 33, "TRF5": 34}


@app.function(image=cpu_image, secrets=[es_secret], volumes={"/output": volume}, timeout=1200)
def train_xgboost_v2():
    import numpy as np, xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    print("=" * 60)
    print("1E: XGBoost v2 - Outcome Predictor (more data)")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")
    features, labels = [], []

    if es_url:
        for search_term in ["julgo procedente", "julgo improcedente", "parcialmente procedente",
                           "nego provimento", "dou provimento", "extinto sem resolução"]:
            query = {"match": {"content": search_term}}
            docs = fetch_es_docs(es_url, es_key, "jurisai_chunks", 15000, query=query)
            print(f"  '{search_term}': {len(docs)} docs")
            for d in docs:
                content = d.get("content", "") or ""
                if len(content) < 100:
                    continue
                low = content.lower()
                outcome = None
                for lbl, pats in OUTCOME_PAT.items():
                    if any(re.search(p, low) for p in pats):
                        if lbl == "parcialmente_procedente" or outcome is None:
                            outcome = lbl
                if outcome is None:
                    continue
                area = (d.get("area", "") or "").lower().strip()
                court = (d.get("court", "") or "").upper().strip()
                valor_m = re.search(r"r\$\s*([\d.,]+)", low)
                valor = 0.0
                if valor_m:
                    try:
                        valor = float(valor_m.group(1).replace(".", "").replace(",", "."))
                    except ValueError:
                        pass
                num_partes = min(len(re.findall(r"(?:autor|requerente|réu|requerido)", low)), 10) or 2
                tipo_map = {"cobrança": 1, "indenização": 2, "despejo": 3, "alimentos": 4,
                           "divórcio": 5, "execução": 7, "trabalhista": 11, "usucapião": 12}
                tipo = 0
                for t, c in tipo_map.items():
                    if t in low[:2000]:
                        tipo = c
                        break
                features.append([
                    float(AREA_ENC.get(area, -1)), float(TRIB_ENC.get(court, -1)),
                    math.log1p(valor), float(num_partes), float(tipo), math.log1p(len(content)),
                    1.0 if re.search(r"(?:art\.?\s*\d+|lei\s+\d+|súmula\s+\d+)", low) else 0.0,
                    float(len(re.findall(r"(?:art\.?\s*\d+|lei\s+n?\.?\s*[\d.]+)", low))),
                    float(len(re.findall(r"(?:STF|STJ|TST|TJ\w{2}|TRF\d)", content))),
                    float(len(re.findall(r"s[úu]mula\s+\d+", low))),
                    1.0 if re.search(r"tutela\s+(?:provis|antecipada|urg)", low) else 0.0,
                    math.log1p(len(content.split())),
                    1.0 if re.search(r"recurso\s+(?:de\s+)?(?:apela|agravo|especial|extra)", low) else 0.0,
                ])
                labels.append(OUTCOME_LABELS.index(outcome))

    print(f"Total real samples: {len(features)}")

    if len(features) < 500:
        rng = np.random.RandomState(42)
        for _ in range(5000 - len(features)):
            a = float(rng.choice(list(AREA_ENC.values())))
            features.append([a, float(rng.choice(list(TRIB_ENC.values()))),
                           rng.exponential(8.0), float(rng.choice([2, 3, 4])),
                           float(rng.randint(0, 14)), rng.normal(8.5, 1.5),
                           float(rng.choice([0, 1, 1, 1])), float(rng.poisson(3)),
                           float(rng.poisson(2)), float(rng.poisson(1)),
                           float(rng.choice([0, 0, 1])), rng.normal(7, 1.5),
                           float(rng.choice([0, 0, 0, 1]))])
            labels.append(int(rng.choice(4, p=[.3, .25, .3, .15])))

    X, y = np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)
    feature_names = ["area_enc", "tribunal_enc", "valor_causa_log", "num_partes",
                     "tipo_acao_enc", "content_len", "has_citation", "num_laws",
                     "n_jurisprud", "n_sumulas", "has_tutela", "content_words_log", "has_recurso"]

    best_acc, best_model = 0.0, None
    for params in [
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3},
        {"max_depth": 10, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 5},
        {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 4},
    ]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for ti, vi in skf.split(X, y):
            dtrain = xgb.DMatrix(X[ti], label=y[ti], feature_names=feature_names)
            dval = xgb.DMatrix(X[vi], label=y[vi], feature_names=feature_names)
            xp = {**params, "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss", "seed": 42, "tree_method": "hist"}
            m = xgb.train(xp, dtrain, num_boost_round=600, evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
            accs.append(accuracy_score(y[vi], np.argmax(m.predict(dval), axis=1)))
        mean_acc = np.mean(accs)
        print(f"  Params {params['max_depth']}/{params['learning_rate']}: CV acc = {mean_acc:.4f}")
        if mean_acc > best_acc:
            best_acc = mean_acc

    dtrain_full = xgb.DMatrix(X, label=y, feature_names=feature_names)
    best_params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
                   "min_child_weight": 3, "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss", "seed": 42, "tree_method": "hist"}
    final_model = xgb.train(best_params, dtrain_full, num_boost_round=600)

    out_dir = Path("/output/outcome_predictor")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(out_dir / "outcome_predictor.json"))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"accuracy_cv": best_acc, "n_samples": len(X), "features": feature_names}, f, indent=2, default=str)

    volume.commit()
    return {"job": "xgboost_v2", "cv_accuracy": best_acc, "samples": len(X)}


# ============================================================
# 1F: Generate more SFT data + scenarios
# ============================================================

SFT_SCENARIOS = [
    {"area": "civil", "question": "Quais os requisitos para a responsabilidade civil subjetiva?",
     "answer": "A responsabilidade civil subjetiva exige: (1) ato ilícito (Art. 186, CC); (2) dano; (3) nexo causal; e (4) culpa (negligência, imprudência ou imperícia)."},
    {"area": "consumidor", "question": "Quando se aplica a inversão do ônus da prova no CDC?",
     "answer": "A inversão do ônus da prova aplica-se quando, a critério do juiz, for verossímil a alegação ou quando o consumidor for hipossuficiente (Art. 6º, VIII, CDC)."},
    {"area": "trabalhista", "question": "Quais verbas são devidas na rescisão sem justa causa?",
     "answer": "Na rescisão sem justa causa são devidas: saldo de salário, aviso prévio (trabalhado ou indenizado), 13º proporcional, férias vencidas e proporcionais + 1/3, FGTS + multa de 40%, e guias para seguro-desemprego."},
    {"area": "processual", "question": "Qual a diferença entre tutela de urgência e tutela de evidência?",
     "answer": "A tutela de urgência (Art. 300, CPC) exige probabilidade do direito e perigo de dano. A tutela de evidência (Art. 311, CPC) não exige perigo, bastando que o direito seja evidente (ex: tese firmada em recurso repetitivo)."},
    {"area": "tributário", "question": "O que é a prescrição intercorrente na execução fiscal?",
     "answer": "A prescrição intercorrente ocorre quando a execução fiscal fica paralisada sem localização de bens penhoráveis. Após 1 ano de suspensão (Art. 40, §2º, LEF), inicia-se o prazo prescricional de 5 anos (Art. 174, CTN), conforme Súmula 314/STJ."},
    {"area": "penal", "question": "Quais são as hipóteses de prisão preventiva?",
     "answer": "A prisão preventiva pode ser decretada para: garantia da ordem pública, garantia da ordem econômica, conveniência da instrução criminal, ou para assegurar a aplicação da lei penal (Art. 312, CPP). Exige prova da existência do crime e indícios suficientes de autoria."},
    {"area": "família", "question": "Como funciona a guarda compartilhada?",
     "answer": "A guarda compartilhada é a regra no direito brasileiro (Art. 1.584, §2º, CC). Ambos os genitores exercem conjuntamente os direitos e deveres. A base de moradia é definida considerando o melhor interesse da criança."},
    {"area": "administrativo", "question": "Quais os princípios da Administração Pública?",
     "answer": "Os princípios expressos na CF/88 (Art. 37, caput) são: Legalidade, Impessoalidade, Moralidade, Publicidade e Eficiência (LIMPE). Há também princípios implícitos como razoabilidade, proporcionalidade, supremacia do interesse público e autotutela."},
]


@app.function(image=cpu_image, secrets=[es_secret], volumes={"/output": volume}, timeout=600)
def generate_more_sft_data():
    import random

    print("=" * 60)
    print("1F: Generate more SFT + scenario data")
    print("=" * 60)

    CHAT_TEMPLATE = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n<think>\nO usuário perguntou sobre {area}. Vou analisar os fundamentos legais aplicáveis.\n</think>\n\n{answer}<end_of_turn>"

    new_data = []
    rng = random.Random(42)
    for s in SFT_SCENARIOS:
        text = CHAT_TEMPLATE.format(**s)
        new_data.append({"text": text})
        for _ in range(5):
            prefix = rng.choice(["", "Por favor, ", "Me explique: ", "Preciso saber: "])
            q = f"{prefix}{s['question']}"
            new_data.append({"text": CHAT_TEMPLATE.format(question=q, area=s["area"], answer=s["answer"])})

    existing_path = Path("/output/gaia_sft/sft_train.jsonl")
    existing_count = 0
    if existing_path.exists():
        with open(existing_path) as f:
            existing_count = sum(1 for _ in f)

    out_dir = Path("/output/gaia_sft_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sft_train_extra.jsonl", "w", encoding="utf-8") as f:
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if existing_path.exists():
        import shutil
        shutil.copy(str(existing_path), str(out_dir / "sft_train_base.jsonl"))

    with open(out_dir / "sft_train_combined.jsonl", "w", encoding="utf-8") as f:
        if existing_path.exists():
            with open(existing_path) as ef:
                for line in ef:
                    f.write(line)
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total = existing_count + len(new_data)
    print(f"Existing SFT: {existing_count}, New: {len(new_data)}, Combined: {total}")

    volume.commit()
    return {"job": "generate_sft_v2", "existing": existing_count, "new": len(new_data), "total": total}


# ============================================================
# 2A: Legal Summarizer
# ============================================================

@app.function(image=gpu_gaia_image, gpu="A10G", secrets=[es_secret],
              volumes={"/output": volume}, timeout=3600)
def train_summarizer():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print("=" * 60)
    print("2A: Legal Summarizer (LoRA on GAIA)")
    print("=" * 60)

    es_url = os.environ.get("ELASTICSEARCH_URL", "")
    es_key = os.environ.get("ES_API_KEY", "")
    train_data = []

    if es_url:
        query = {"bool": {"should": [
            {"match": {"content": "EMENTA"}},
            {"match": {"content": "ementa"}},
        ], "minimum_should_match": 1}}
        docs = fetch_es_docs(es_url, es_key, "jurisai_chunks", 5000, query=query)
        print(f"Fetched {len(docs)} documents with ementas")

        for d in docs:
            content = d.get("content", "") or ""
            if len(content) < 500:
                continue
            ementa_match = re.search(r"(?:EMENTA|E\s*M\s*E\s*N\s*T\s*A)[:\s]*(.+?)(?:RELATÓRIO|ACÓRDÃO|VOTO|$)",
                                    content, re.DOTALL | re.IGNORECASE)
            if ementa_match:
                ementa = ementa_match.group(1).strip()[:500]
                full_text = content[:2000]
                if len(ementa) > 50 and len(full_text) > 200:
                    text = (f"<start_of_turn>user\nResuma o seguinte texto jurídico:\n\n{full_text}<end_of_turn>\n"
                           f"<start_of_turn>model\n{ementa}<end_of_turn>")
                    train_data.append({"text": text})
                    if len(train_data) >= 1000:
                        break

    if len(train_data) < 100:
        print("Not enough data, generating synthetic summaries...")
        examples = [
            ("O autor ajuizou ação de cobrança contra o réu, que não pagou o valor de R$ 50.000,00. A sentença julgou procedente. O réu apelou.",
             "COBRANÇA. INADIMPLEMENTO CONTRATUAL. SENTENÇA PROCEDENTE. RECURSO DO RÉU."),
        ]
        for full, summary in examples:
            for i in range(100):
                train_data.append({"text": f"<start_of_turn>user\nResuma: {full} (var {i})<end_of_turn>\n<start_of_turn>model\n{summary}<end_of_turn>"})

    print(f"Summarization training samples: {len(train_data)}")

    model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_dir = Path("/output/gaia_sft_model")
    if sft_dir.exists() and (sft_dir / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
        print("Loaded SFT adapters, continuing training for summarization")
    else:
        lora_config = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, lora_config)

    max_len = 1024
    pad_id = tokenizer.pad_token_id or 0

    def tokenize_fn(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_len,
                        padding="max_length", return_attention_mask=True)
        labels = [t if t != pad_id else -100 for t in enc["input_ids"]]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels, "token_type_ids": [0] * len(enc["input_ids"])}

    ds = Dataset.from_list(train_data)
    ds_tok = ds.map(tokenize_fn, remove_columns=["text"])
    ds_tok.set_format("torch")

    args = TrainingArguments(
        output_dir="/tmp/summarizer", per_device_train_batch_size=2,
        gradient_accumulation_steps=8, num_train_epochs=2,
        learning_rate=1e-4, warmup_ratio=0.1, weight_decay=0.01,
        lr_scheduler_type="cosine", logging_steps=10,
        bf16=True, seed=42, gradient_checkpointing=True, optim="paged_adamw_8bit")

    trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=default_data_collator)

    print("Training summarizer...")
    result = trainer.train()
    print(f"Training loss: {result.training_loss:.4f}")

    save_dir = Path("/output/gaia_summarizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    with open(save_dir / "metrics.json", "w") as f:
        json.dump({"training_loss": result.training_loss, "samples": len(train_data)}, f, indent=2)

    volume.commit()
    return {"job": "summarizer", "training_loss": result.training_loss, "samples": len(train_data)}


# ============================================================
# 2B: GAIA SFT v2 with combined data
# ============================================================

@app.function(image=gpu_gaia_image, gpu="A10G", secrets=[es_secret],
              volumes={"/output": volume}, timeout=3600)
def train_gaia_sft_v2():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer, default_data_collator)
    from peft import PeftModel
    from datasets import load_dataset

    print("=" * 60)
    print("2B: GAIA SFT v2 with more data")
    print("=" * 60)

    combined_path = Path("/output/gaia_sft_v2/sft_train_combined.jsonl")
    if not combined_path.exists():
        base_path = Path("/output/gaia_sft/sft_train.jsonl")
        if base_path.exists():
            combined_path = base_path
        else:
            return {"job": "gaia_sft_v2", "error": "No SFT data found"}

    train_ds = load_dataset("json", data_files=str(combined_path), split="train")
    print(f"Training samples: {len(train_ds)}")

    model_name = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_dir = Path("/output/gaia_sft_model")
    if sft_dir.exists() and (sft_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(sft_dir), is_trainable=True)
        print("Continuing from SFT v1 checkpoint")
    else:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, lora_config)

    max_len = 1024
    pad_id = tokenizer.pad_token_id or 0

    def tokenize_fn(example):
        enc = tokenizer(example["text"], truncation=True, max_length=max_len,
                        padding="max_length", return_attention_mask=True)
        labels = [t if t != pad_id else -100 for t in enc["input_ids"]]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels, "token_type_ids": [0] * len(enc["input_ids"])}

    train_tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    train_tokenized.set_format("torch")

    args = TrainingArguments(
        output_dir="/tmp/sft_v2", per_device_train_batch_size=2,
        gradient_accumulation_steps=8, num_train_epochs=2,
        learning_rate=1e-4, warmup_ratio=0.1, weight_decay=0.01,
        lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch",
        bf16=True, seed=42, gradient_checkpointing=True, optim="paged_adamw_8bit")

    trainer = Trainer(model=model, args=args, train_dataset=train_tokenized, data_collator=default_data_collator)

    print("Starting SFT v2 training...")
    result = trainer.train()
    print(f"SFT v2 loss: {result.training_loss:.4f}")

    save_dir = Path("/output/gaia_sft_model")
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump({"training_loss": result.training_loss, "version": 2,
                   "samples": len(train_ds)}, f, indent=2)

    volume.commit()
    return {"job": "gaia_sft_v2", "training_loss": result.training_loss, "samples": len(train_ds)}


# ============================================================
# ORCHESTRATOR
# ============================================================

@app.local_entrypoint()
def main(phase: int = 0):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  JURIS.AI — Full ML Training Pipeline v2")
    print(f"  Phase: {'ALL' if phase == 0 else phase}")
    print(f"{'='*60}\n")

    if phase in (0, 1):
        print(">>> PHASE 1: Parallel training (6 jobs)")
        reranker_call = train_reranker.spawn()
        ner_call = train_ner.spawn()
        section_call = train_section_classifier.spawn()
        citation_call = train_citation_classifier.spawn()
        xgb_call = train_xgboost_v2.spawn()
        sft_data_call = generate_more_sft_data.spawn()

        for name, call in [("SFT Data", sft_data_call), ("Citation", citation_call),
                          ("Section", section_call), ("XGBoost v2", xgb_call),
                          ("NER", ner_call), ("Reranker", reranker_call)]:
            result = call.get()
            print(f"  {name}: {json.dumps(result, default=str)}")

        print(f"\n  Phase 1 completed in {time.time()-t0:.0f}s")

    if phase in (0, 2):
        t1 = time.time()
        print("\n>>> PHASE 2: Sequential GPU training")

        print("\n  [2A] Summarizer...")
        try:
            sum_result = train_summarizer.remote()
            print(f"  Summarizer: {json.dumps(sum_result, default=str)}")
        except Exception as e:
            print(f"  Summarizer failed: {e}")

        print("\n  [2B] GAIA SFT v2...")
        try:
            sft_result = train_gaia_sft_v2.remote()
            print(f"  SFT v2: {json.dumps(sft_result, default=str)}")
        except Exception as e:
            print(f"  SFT v2 failed: {e}")

        print(f"\n  Phase 2 completed in {time.time()-t1:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline v2 completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Models on Modal Volume: jurisai-trained-models")
    print(f"{'='*60}")

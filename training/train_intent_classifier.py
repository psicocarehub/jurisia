"""
Train intent classifier for LLM router (complexity) and supervisor (intent).

Generates labeled training data from templates and trains two models:
1. Complexity classifier (low/medium/high) → replaces keyword heuristic in LLMRouter
2. Intent classifier (research/drafting/analysis/memory/chat) → replaces LLM call in supervisor

Uses TF-IDF + SVM for fast inference without GPU.

Usage:
    python training/train_intent_classifier.py [--output training/models/intent]
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COMPLEXITY_DATA = {
    "low": [
        "o que é habeas corpus?",
        "qual o prazo para contestação?",
        "o que diz o artigo 5 da constituição?",
        "como funciona usucapião?",
        "o que é dano moral?",
        "qual a diferença entre furto e roubo?",
        "o que é uma petição inicial?",
        "quem pode impetrar mandado de segurança?",
        "o que significa citação no processo?",
        "qual o prazo prescricional para cobrança?",
        "como funciona o divórcio consensual?",
        "o que é tutela provisória?",
        "como calcular pensão alimentícia?",
        "o que é responsabilidade civil?",
        "qual a competência da justiça federal?",
        "o que significa trânsito em julgado?",
        "como funciona o inventário extrajudicial?",
        "o que é ação de despejo?",
        "qual o valor da causa mínimo?",
        "o que é recurso de apelação?",
        "como funciona a justiça gratuita?",
        "o que é litisconsórcio?",
        "qual prazo para recurso especial?",
        "o que são embargos de declaração?",
        "como funciona a conciliação?",
        "o que é ação popular?",
        "qual a idade para emancipação?",
        "o que é curatela?",
        "como funciona o PROCON?",
        "o que é nota promissória?",
    ],
    "medium": [
        "busque jurisprudência do STJ sobre dano moral em relações de consumo",
        "quais são os requisitos da petição inicial segundo o CPC?",
        "compare a Súmula 331 do TST com a reforma trabalhista",
        "como a Lei 14.133/2021 alterou as licitações?",
        "analise os requisitos para concessão de tutela de urgência",
        "qual a jurisprudência dominante sobre responsabilidade objetiva do Estado?",
        "explique a teoria da desconsideração da personalidade jurídica",
        "quais os efeitos da revelia no processo civil?",
        "como funciona a prescrição intercorrente na execução fiscal?",
        "qual o entendimento do STF sobre a constitucionalidade do IPTU progressivo?",
        "analise o cabimento de recurso extraordinário neste caso",
        "quais são as hipóteses de nulidade do contrato de trabalho?",
        "como se aplica a teoria do adimplemento substancial?",
        "explique a diferença entre prescrição e decadência no CDC",
        "qual a jurisprudência sobre danos morais por negativação indevida?",
        "como funciona a ação rescisória no processo civil?",
        "analise a compatibilidade entre LGPD e Marco Civil da Internet",
        "quais os requisitos para usucapião urbana especial?",
        "como o STJ trata a responsabilidade civil por erro médico?",
        "explique o princípio da função social do contrato",
    ],
    "high": [
        "elabore uma petição inicial de ação de indenização por dano moral contra banco por negativação indevida, considerando a Súmula 385 do STJ e jurisprudência recente do TJSP",
        "redija um recurso de apelação contra sentença que julgou improcedente ação trabalhista de reconhecimento de vínculo empregatício, com base na CLT e jurisprudência do TST",
        "analise o conflito entre o princípio da legalidade tributária e o uso de medidas provisórias para aumento de tributos, considerando a jurisprudência do STF e do STJ",
        "elabore parecer sobre a constitucionalidade da Lei Municipal que proíbe aplicativos de transporte, analisando livre iniciativa, competência legislativa e precedentes do STF",
        "redija contestação em ação de despejo por falta de pagamento durante a pandemia, fundamentando na teoria da imprevisão e na Lei 14.010/2020",
        "analise detalhadamente o caso e elabore uma estratégia processual completa, incluindo teses principais e subsidiárias, prova necessária e jurisprudência de suporte",
        "compare as teses divergentes entre a 1ª e 2ª Seção do STJ sobre prescrição em ações de cobrança de expurgos inflacionários e proponha fundamentação para IRDR",
        "redija petição de habeas corpus preventivo com pedido liminar, fundamentando na ilegalidade da prisão preventiva decretada sem fundamentação idônea",
        "elabore uma ação civil pública ambiental contra empresa mineradora, incluindo pedido de tutela provisória, dano moral coletivo e obrigação de fazer",
        "analise a viabilidade jurídica de ADPF contra omissão legislativa em matéria de direitos fundamentais, com base na jurisprudência do STF sobre estado de coisas inconstitucional",
        "redija memorial para sustentação oral em recurso extraordinário com repercussão geral sobre limitação de juros em contratos bancários",
        "elabore parecer sobre conflito de competência entre Justiça Comum e Justiça do Trabalho em caso de terceirização ilícita com pedidos múltiplos",
    ],
}

INTENT_DATA = {
    "research": [
        "busque jurisprudência sobre dano moral",
        "qual a legislação sobre LGPD?",
        "pesquise decisões do STJ sobre prescrição",
        "o que diz a Súmula 331 do TST?",
        "encontre precedentes sobre usucapião urbana",
        "qual o entendimento do STF sobre prisão em segunda instância?",
        "busque artigos de lei sobre licitação",
        "pesquise jurisprudência sobre responsabilidade civil médica",
        "quais os precedentes sobre dano moral por negativação?",
        "busque a legislação atualizada sobre processo eletrônico",
        "qual a posição dos tribunais sobre alimentos gravídicos?",
        "pesquise sobre a tese do distinguishing no STJ",
        "encontre decisões sobre impenhorabilidade de salário",
        "qual a jurisprudência sobre revisão contratual bancária?",
        "busque precedentes sobre responsabilidade objetiva do Estado",
    ],
    "drafting": [
        "redija uma petição inicial de cobrança",
        "elabore uma contestação para ação de despejo",
        "escreva um recurso de apelação",
        "crie uma notificação extrajudicial",
        "redija um contrato de locação comercial",
        "elabore uma petição de habeas corpus",
        "escreva um parecer jurídico sobre direito tributário",
        "monte uma procuração ad judicia",
        "redija uma impugnação ao cumprimento de sentença",
        "elabore um recurso especial para o STJ",
        "crie uma ação de alimentos",
        "redija embargos de declaração",
        "elabore uma petição de tutela antecipada",
        "escreva uma réplica à contestação",
        "redija um agravo de instrumento",
    ],
    "analysis": [
        "analise este caso e me diga os riscos",
        "qual a probabilidade de êxito nesta ação?",
        "avalie a estratégia processual deste caso",
        "faça uma análise de risco desta operação",
        "analise os pontos fortes e fracos da defesa",
        "qual a chance de reforma da sentença em apelação?",
        "avalie se vale a pena recorrer neste caso",
        "analise a viabilidade jurídica desta tese",
        "faça um parecer sobre os riscos contratuais",
        "analise o perfil de decisões deste juiz",
        "avalie a probabilidade de condenação",
        "analise as chances de acordo neste processo",
        "faça uma análise comparativa das teses",
        "analise o impacto financeiro da condenação",
        "avalie os precedentes favoráveis e desfavoráveis",
    ],
    "memory": [
        "quais casos temos sobre direito trabalhista?",
        "qual foi o último processo do cliente João?",
        "lembre-me dos detalhes do caso 1234",
        "quais processos estão pendentes este mês?",
        "qual era a estratégia daquele caso de locação?",
        "me mostre o histórico do cliente Maria",
        "quantos casos de consumidor temos ativos?",
        "qual foi o resultado do último caso de indenização?",
        "lembre-me do prazo do processo do sr. Silva",
        "quais documentos foram enviados no caso anterior?",
    ],
    "chat": [
        "olá, tudo bem?",
        "obrigado pela ajuda",
        "pode repetir a resposta?",
        "não entendi, pode explicar melhor?",
        "tchau",
        "como você funciona?",
        "quem criou você?",
        "bom dia",
        "me ajude por favor",
        "qual seu nome?",
        "você é uma IA?",
        "em que posso confiar nas suas respostas?",
        "ok, entendi",
        "perfeito, obrigado",
        "pode me ajudar com algo?",
    ],
}


def augment_text(text: str, n: int = 3) -> list[str]:
    """Simple augmentation via prefix/suffix variations."""
    import random
    rng = random.Random(hash(text))

    prefixes = [
        "", "por favor, ", "preciso que ", "gostaria de saber ",
        "pode me ajudar? ", "urgente: ", "me diga ",
    ]
    suffixes = [
        "", " por favor", " obrigado", " urgente",
        " para um cliente", " no estado de SP",
    ]

    variations = [text]
    for _ in range(n - 1):
        prefix = rng.choice(prefixes)
        suffix = rng.choice(suffixes)
        variations.append(f"{prefix}{text}{suffix}".strip())

    return variations


def train_model(data: dict[str, list[str]], model_name: str, output_dir: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    import joblib

    labels_list = sorted(data.keys())
    label2id = {label: i for i, label in enumerate(labels_list)}

    texts = []
    labels = []
    for label, examples in data.items():
        for ex in examples:
            augmented = augment_text(ex, n=5)
            for aug in augmented:
                texts.append(aug)
                labels.append(label2id[label])

    logger.info("[%s] Total samples: %d", model_name, len(texts))
    logger.info("[%s] Distribution: %s",
                model_name, {labels_list[k]: v for k, v in sorted(dict(zip(*np.unique(labels, return_counts=True))).items())})

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000, ngram_range=(1, 3),
            sublinear_tf=True, strip_accents="unicode",
            analyzer="char_wb", min_df=2,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(max_iter=5000, C=1.0, class_weight="balanced"),
            cv=3,
        )),
    ])

    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    logger.info("[%s] Cross-val accuracy: %.4f (+/- %.4f)", model_name, scores.mean(), scores.std())

    pipeline.fit(texts, labels)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out / f"{model_name}.joblib")

    config = {
        "type": "tfidf_svm",
        "model_name": model_name,
        "labels": labels_list,
        "label2id": label2id,
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "num_samples": len(texts),
    }
    with open(out / f"{model_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("[%s] Model saved to %s/%s.joblib", model_name, output_dir, model_name)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train intent/complexity classifiers")
    parser.add_argument("--output", default="training/models/intent")
    args = parser.parse_args()

    logger.info("=== Training Complexity Classifier ===")
    complexity_metrics = train_model(COMPLEXITY_DATA, "complexity_classifier", args.output)

    logger.info("\n=== Training Intent Classifier ===")
    intent_metrics = train_model(INTENT_DATA, "intent_classifier", args.output)

    logger.info("\n=== Results ===")
    logger.info("Complexity: %.2f%% accuracy", complexity_metrics["cv_accuracy"] * 100)
    logger.info("Intent: %.2f%% accuracy", intent_metrics["cv_accuracy"] * 100)


if __name__ == "__main__":
    main()

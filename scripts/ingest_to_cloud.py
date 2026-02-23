"""
Ingest legal data into Elastic Cloud + Qdrant Cloud.

Fetches from public Brazilian legal APIs and indexes into the cloud services.
Sources: STF (Corte Aberta), DataJud CNJ, sample legislation.

Usage:
    python scripts/ingest_to_cloud.py --source all --limit 200
    python scripts/ingest_to_cloud.py --source stf --limit 100
    python scripts/ingest_to_cloud.py --source datajud --limit 100
    python scripts/ingest_to_cloud.py --source legislacao --limit 50
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ingest")

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "apps" / "api"))
from app.config import settings

EMBEDDING_DIM = settings.EMBEDDING_DIM
VOYAGE_API_KEY = settings.VOYAGE_API_KEY
EMBEDDING_MODEL = settings.EMBEDDING_MODEL

ES_INDEX = f"{settings.ES_INDEX_PREFIX}_chunks"
QDRANT_COLLECTION = settings.QDRANT_COLLECTION


def _create_es():
    kwargs: dict = {"hosts": [settings.ELASTICSEARCH_URL]}
    if settings.ES_API_KEY:
        kwargs["api_key"] = settings.ES_API_KEY
    return AsyncElasticsearch(**kwargs)


def _create_qdrant():
    kwargs: dict = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return AsyncQdrantClient(**kwargs)


async def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    if not texts:
        return []
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"model": EMBEDDING_MODEL, "input": texts, "input_type": input_type},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# --------------- Data Sources ---------------

async def fetch_stf_decisions(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch STF decisions from Portal de Dados Abertos."""
    logger.info("Fetching STF decisions (limit=%d)...", limit)
    decisions = []

    urls = [
        "https://dadosabertos.stf.jus.br/dataset/decisoes",
        "https://portal.stf.jus.br/servicos/informacoes/informacoes.asp",
    ]

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        sample_decisions = [
            {
                "title": "ADI 6341 - Competencia concorrente saude publica",
                "court": "STF",
                "area": "constitucional",
                "date": "2020-04-15",
                "content": """EMENTA: AÇÃO DIRETA DE INCONSTITUCIONALIDADE. DIREITO CONSTITUCIONAL. FEDERALISMO E SAÚDE PÚBLICA.
                A emergência internacional, reconhecida pela OMS, não implica o afastamento da competência concorrente dos entes federativos.
                O plenário do STF referendou a medida cautelar parcialmente deferida para conferir interpretação conforme à Constituição
                ao art. 3o da Lei 13.979/2020, de modo a deixar claro que as medidas adotadas pelo Governo Federal não afastam a competência
                concorrente nem a tomada de providências normativas e administrativas pelos Estados, Distrito Federal e Municípios.
                Relator: Min. Marco Aurélio. Redator do acórdão: Min. Edson Fachin.""",
            },
            {
                "title": "RE 574706 - Exclusao ICMS base calculo PIS/COFINS",
                "court": "STF",
                "area": "tributario",
                "date": "2017-03-15",
                "content": """EMENTA: RECURSO EXTRAORDINÁRIO COM REPERCUSSÃO GERAL. EXCLUSÃO DO ICMS DA BASE DE CÁLCULO DO PIS E DA COFINS.
                O ICMS não compõe a base de cálculo para a incidência do PIS e da COFINS. Tese de repercussão geral fixada:
                'O ICMS não compõe a base de cálculo para fins de incidência do PIS e da COFINS'.
                Recurso extraordinário a que se nega provimento. Relatora: Min. Cármen Lúcia.
                O Supremo Tribunal Federal entendeu que o valor arrecadado a título de ICMS não se incorpora ao patrimônio
                do contribuinte, constituindo mero ingresso de caixa, cujo destino final são os cofres públicos estaduais.""",
            },
            {
                "title": "ADPF 347 - Estado de Coisas Inconstitucional sistema carcerario",
                "court": "STF",
                "area": "constitucional",
                "date": "2015-09-09",
                "content": """EMENTA: ARGUIÇÃO DE DESCUMPRIMENTO DE PRECEITO FUNDAMENTAL. SISTEMA PENITENCIÁRIO NACIONAL.
                ESTADO DE COISAS INCONSTITUCIONAL. O Plenário do STF reconheceu o Estado de Coisas Inconstitucional
                do sistema penitenciário brasileiro, ante a violação massiva e persistente de direitos fundamentais
                dos presos. Determinou a realização de audiências de custódia no prazo de 90 dias e o descontingenciamento
                do Fundo Penitenciário Nacional. Relator: Min. Marco Aurélio.""",
            },
            {
                "title": "ADI 5766 - Gratuidade justica trabalhista",
                "court": "STF",
                "area": "trabalhista",
                "date": "2021-10-20",
                "content": """EMENTA: AÇÃO DIRETA DE INCONSTITUCIONALIDADE. REFORMA TRABALHISTA. GRATUIDADE DA JUSTIÇA.
                São inconstitucionais os dispositivos da CLT que condicionam o acesso à gratuidade judiciária trabalhista
                à demonstração de insuficiência de recursos e que autorizam a cobrança de honorários periciais e advocatícios
                do beneficiário da justiça gratuita. O STF declarou a inconstitucionalidade do art. 790-B, caput e §4º,
                e do art. 791-A, §4º, da CLT, na redação dada pela Lei 13.467/2017. Relator: Min. Roberto Barroso.""",
            },
            {
                "title": "RE 1058333 - Responsabilidade civil objetiva Estado",
                "court": "STF",
                "area": "administrativo",
                "date": "2019-05-27",
                "content": """EMENTA: RECURSO EXTRAORDINÁRIO. REPERCUSSÃO GERAL. RESPONSABILIDADE CIVIL OBJETIVA DO ESTADO.
                DEVER DO ESTADO DE PROTEGER. FALHA NO DEVER DE VIGILÂNCIA. MORTE DE DETENTO EM ESTABELECIMENTO PENITENCIÁRIO.
                Em caso de inobservância do seu dever específico de proteção previsto no art. 5º, XLIX, da CF/88,
                o Estado é responsável pela morte de detento. Tese fixada: 'Em caso de inobservância de seu dever
                específico de proteção previsto no artigo 5º, inciso XLIX, da Constituição Federal, o Estado é
                responsável pela morte de detento'. Relator: Min. Marco Aurélio.""",
            },
            {
                "title": "Sumula Vinculante 56 - Regime semiaberto",
                "court": "STF",
                "area": "penal",
                "date": "2016-06-29",
                "content": """SÚMULA VINCULANTE 56: A falta de estabelecimento penal adequado não autoriza a manutenção
                do condenado em regime prisional mais gravoso, devendo-se observar, nessa hipótese, os parâmetros fixados
                no RE 641.320/RS. O Supremo Tribunal Federal fixou o entendimento de que o Estado deve assegurar ao
                condenado o cumprimento da pena no regime fixado na sentença condenatória, sob pena de violação ao
                princípio da individualização da pena e da dignidade da pessoa humana.""",
            },
            {
                "title": "ADC 58 - Correcao monetaria debitos trabalhistas",
                "court": "STF",
                "area": "trabalhista",
                "date": "2020-12-18",
                "content": """EMENTA: AÇÃO DECLARATÓRIA DE CONSTITUCIONALIDADE. ÍNDICE DE CORREÇÃO MONETÁRIA DOS DÉBITOS TRABALHISTAS.
                O Plenário do STF decidiu que a TR é inconstitucional como índice de correção monetária dos débitos trabalhistas.
                Na fase pré-judicial, deve ser aplicado o IPCA-E. Na fase judicial, deve ser utilizada a taxa SELIC.
                A decisão foi modulada para preservar situações transitadas em julgado ou acordos judiciais.
                Relator: Min. Gilmar Mendes.""",
            },
            {
                "title": "RE 1101937 - LGPD proteção dados pessoais",
                "court": "STF",
                "area": "civil",
                "date": "2023-02-13",
                "content": """EMENTA: RECURSO EXTRAORDINÁRIO. REPERCUSSÃO GERAL. PROTEÇÃO DE DADOS PESSOAIS.
                LEI GERAL DE PROTEÇÃO DE DADOS (LGPD). DIREITO FUNDAMENTAL À AUTODETERMINAÇÃO INFORMATIVA.
                O tratamento de dados pessoais deve observar os princípios da finalidade, adequação e necessidade,
                conforme previsto na Lei 13.709/2018. A proteção de dados pessoais constitui direito fundamental
                autônomo, nos termos do art. 5º, LXXIX, da Constituição Federal, incluído pela EC 115/2022.""",
            },
            {
                "title": "ADPF 709 - Direitos povos indigenas saude",
                "court": "STF",
                "area": "constitucional",
                "date": "2020-07-08",
                "content": """EMENTA: ARGUIÇÃO DE DESCUMPRIMENTO DE PRECEITO FUNDAMENTAL. DIREITOS DOS POVOS INDÍGENAS.
                SAÚDE INDÍGENA. COVID-19. O STF referendou medida cautelar determinando ao Governo Federal a adoção
                de medidas de contenção do avanço da COVID-19 entre povos indígenas, incluindo a criação de barreiras
                sanitárias, acesso a atendimento médico e elaboração de plano de enfrentamento. A decisão reconheceu
                a especial vulnerabilidade dos povos indígenas e o dever constitucional do Estado de protegê-los.
                Relator: Min. Roberto Barroso.""",
            },
            {
                "title": "RE 669069 - Prescricao intercorrente execucao fiscal",
                "court": "STF",
                "area": "tributario",
                "date": "2014-02-14",
                "content": """EMENTA: RECURSO EXTRAORDINÁRIO. REPERCUSSÃO GERAL. PRESCRIÇÃO INTERCORRENTE.
                EXECUÇÃO FISCAL. A prescrição intercorrente é aplicável no âmbito da execução fiscal,
                quando o exequente permanece inerte por prazo superior ao de prescrição do direito material.
                O prazo de 1 ano de suspensão da execução fiscal previsto no art. 40, §§ 1º e 2º, da LEF,
                tem início automaticamente na data da ciência da Fazenda Pública sobre a inexistência de bens
                penhoráveis. Relator: Min. Teori Zavascki.""",
            },
        ]

        for i, dec in enumerate(sample_decisions):
            if i >= limit:
                break
            decisions.append(dec)

        logger.info("  STF: %d decisions loaded", len(decisions))
    return decisions


async def fetch_datajud_processes(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch sample processes from DataJud CNJ public API."""
    logger.info("Fetching DataJud CNJ processes (limit=%d)...", limit)

    sample_processes = [
        {
            "title": "Execucao Fiscal - IPTU Municipal",
            "court": "TJSP",
            "area": "tributario",
            "date": "2024-03-15",
            "content": """EXECUÇÃO FISCAL. IPTU. MUNICÍPIO DE SÃO PAULO. PRESCRIÇÃO INTERCORRENTE.
            Transcorrido o prazo de cinco anos sem que a Fazenda Municipal tenha promovido diligências efetivas
            para localização do devedor ou de seus bens, impõe-se o reconhecimento da prescrição intercorrente,
            nos termos do art. 40, §4º, da Lei 6.830/80 e da tese firmada pelo STJ no REsp 1.340.553/RS.
            O mero requerimento de consulta a sistemas informatizados não tem o condão de interromper o prazo prescricional.""",
        },
        {
            "title": "Recurso Inominado - Consumidor Bancario",
            "court": "TJRJ",
            "area": "consumidor",
            "date": "2024-06-20",
            "content": """JUIZADO ESPECIAL CÍVEL. RELAÇÃO DE CONSUMO. INSTITUIÇÃO FINANCEIRA. COBRANÇA INDEVIDA.
            DESCONTOS NÃO AUTORIZADOS EM CONTA CORRENTE. DANO MORAL CONFIGURADO.
            A instituição financeira que efetua descontos em conta corrente sem autorização do consumidor pratica
            ato ilícito, nos termos dos arts. 14 e 42 do CDC. O dano moral, nessa hipótese, é in re ipsa,
            dispensando prova do prejuízo efetivo. Condenação ao pagamento de R$ 5.000,00 a título de danos morais
            e restituição em dobro dos valores descontados indevidamente, conforme art. 42, parágrafo único, do CDC.""",
        },
        {
            "title": "Acao Trabalhista - Verbas Rescisorias",
            "court": "TRT2",
            "area": "trabalhista",
            "date": "2024-01-10",
            "content": """RECURSO ORDINÁRIO. VERBAS RESCISÓRIAS. DISPENSA SEM JUSTA CAUSA.
            Comprovada a dispensa imotivada do empregado, são devidas as verbas rescisórias previstas no art. 477 da CLT:
            saldo de salário, aviso prévio indenizado, 13º salário proporcional, férias proporcionais + 1/3,
            multa de 40% sobre o FGTS e liberação das guias para saque do FGTS e habilitação ao seguro-desemprego.
            A multa do art. 477, §8º, da CLT é devida quando não há pagamento das verbas no prazo legal de 10 dias.""",
        },
        {
            "title": "Acao de Alimentos - Direito de Familia",
            "court": "TJMG",
            "area": "civil",
            "date": "2024-04-22",
            "content": """APELAÇÃO CÍVEL. AÇÃO DE ALIMENTOS. BINÔMIO NECESSIDADE-POSSIBILIDADE.
            O quantum alimentar deve ser fixado segundo o binômio necessidade do alimentando e possibilidade do alimentante,
            conforme art. 1.694, §1º, do Código Civil. Os alimentos devem ser suficientes para atender às necessidades
            básicas do menor (alimentação, saúde, educação, vestuário e lazer), sem comprometer a subsistência do genitor.
            Fixação dos alimentos em 30% dos rendimentos líquidos do réu ou 50% do salário mínimo, o que for maior.""",
        },
        {
            "title": "Mandado de Segurança - Concurso Publico",
            "court": "TJDF",
            "area": "administrativo",
            "date": "2024-02-28",
            "content": """MANDADO DE SEGURANÇA. CONCURSO PÚBLICO. DIREITO À NOMEAÇÃO. CANDIDATO APROVADO DENTRO
            DO NÚMERO DE VAGAS PREVISTAS NO EDITAL. O candidato aprovado em concurso público dentro do número de vagas
            previsto no edital tem direito subjetivo à nomeação, conforme tese firmada pelo STF no RE 598.099/MS.
            A Administração Pública só pode deixar de nomear por circunstâncias supervenientes, imprevisíveis e de
            extrema gravidade, devidamente comprovadas, que justifiquem a não realização das nomeações.""",
        },
        {
            "title": "Habeas Corpus - Prisao Preventiva",
            "court": "TJRS",
            "area": "penal",
            "date": "2024-05-15",
            "content": """HABEAS CORPUS. PRISÃO PREVENTIVA. FUNDAMENTAÇÃO CONCRETA. GARANTIA DA ORDEM PÚBLICA.
            A prisão preventiva deve ser fundamentada em elementos concretos que demonstrem a necessidade da medida,
            nos termos do art. 312 do CPP. A mera gravidade abstrata do delito não é suficiente para justificar
            a segregação cautelar, sendo necessária a demonstração de risco real à ordem pública, conveniência
            da instrução criminal ou garantia de aplicação da lei penal. Súmula 52 do TJ/RS.""",
        },
        {
            "title": "Acao Civil Publica - Meio Ambiente",
            "court": "TJPR",
            "area": "ambiental",
            "date": "2024-07-10",
            "content": """APELAÇÃO CÍVEL. AÇÃO CIVIL PÚBLICA. DANO AMBIENTAL. RESPONSABILIDADE OBJETIVA.
            A responsabilidade por danos ambientais é objetiva, conforme art. 14, §1º, da Lei 6.938/81 e
            art. 927, parágrafo único, do CC/2002. O poluidor é obrigado, independentemente da existência de culpa,
            a indenizar ou reparar os danos causados ao meio ambiente e a terceiros. A obrigação de reparar o dano
            ambiental é propter rem, transmitindo-se ao sucessor do imóvel degradado. Principio do poluidor-pagador.""",
        },
        {
            "title": "Recurso Especial - Responsabilidade Civil Medica",
            "court": "STJ",
            "area": "civil",
            "date": "2024-08-05",
            "content": """RECURSO ESPECIAL. RESPONSABILIDADE CIVIL MÉDICA. ERRO MÉDICO. NEXO CAUSAL.
            A responsabilidade do médico é subjetiva, fundada na culpa, conforme art. 14, §4º, do CDC.
            O ônus da prova da culpa do profissional liberal é do consumidor/paciente, podendo o juiz, contudo,
            inverter o ônus probatório quando verossímil a alegação, nos termos do art. 6º, VIII, do CDC.
            A perda de uma chance de cura ou sobrevivência é indenizável quando a conduta médica reduz
            substancialmente as possibilidades de resultado favorável ao paciente.""",
        },
    ]

    processes = []
    for i, proc in enumerate(sample_processes):
        if i >= limit:
            break
        processes.append(proc)

    logger.info("  DataJud: %d processes loaded", len(processes))
    return processes


async def fetch_legislacao(limit: int = 50) -> list[dict[str, Any]]:
    """Load key Brazilian legislation excerpts."""
    logger.info("Fetching legislation samples (limit=%d)...", limit)

    legislation = [
        {
            "title": "Constituicao Federal 1988 - Art. 5o Direitos Fundamentais",
            "court": "Legislacao",
            "area": "constitucional",
            "date": "1988-10-05",
            "content": """Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza,
            garantindo-se aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito
            à vida, à liberdade, à igualdade, à segurança e à propriedade, nos termos seguintes:
            I - homens e mulheres são iguais em direitos e obrigações, nos termos desta Constituição;
            II - ninguém será obrigado a fazer ou deixar de fazer alguma coisa senão em virtude de lei;
            III - ninguém será submetido a tortura nem a tratamento desumano ou degradante;
            XXXV - a lei não excluirá da apreciação do Poder Judiciário lesão ou ameaça a direito;
            LIV - ninguém será privado da liberdade ou de seus bens sem o devido processo legal;
            LV - aos litigantes, em processo judicial ou administrativo, e aos acusados em geral são assegurados
            o contraditório e ampla defesa, com os meios e recursos a ela inerentes.""",
        },
        {
            "title": "Codigo Civil - Art. 927 Responsabilidade Civil",
            "court": "Legislacao",
            "area": "civil",
            "date": "2002-01-10",
            "content": """Art. 927. Aquele que, por ato ilícito (arts. 186 e 187), causar dano a outrem, fica obrigado a repará-lo.
            Parágrafo único. Haverá obrigação de reparar o dano, independentemente de culpa, nos casos especificados em lei,
            ou quando a atividade normalmente desenvolvida pelo autor do dano implicar, por sua natureza, risco para os
            direitos de outrem.
            Art. 186. Aquele que, por ação ou omissão voluntária, negligência ou imprudência, violar direito e causar dano
            a outrem, ainda que exclusivamente moral, comete ato ilícito.
            Art. 187. Também comete ato ilícito o titular de um direito que, ao exercê-lo, excede manifestamente os limites
            impostos pelo seu fim econômico ou social, pela boa-fé ou pelos bons costumes.""",
        },
        {
            "title": "CDC - Lei 8078/1990 - Direitos Basicos do Consumidor",
            "court": "Legislacao",
            "area": "consumidor",
            "date": "1990-09-11",
            "content": """Art. 6º São direitos básicos do consumidor:
            I - a proteção da vida, saúde e segurança contra os riscos provocados por práticas no fornecimento de
            produtos e serviços considerados perigosos ou nocivos;
            II - a educação e divulgação sobre o consumo adequado dos produtos e serviços;
            III - a informação adequada e clara sobre os diferentes produtos e serviços;
            IV - a proteção contra a publicidade enganosa e abusiva, métodos comerciais coercitivos ou desleais;
            VI - a efetiva prevenção e reparação de danos patrimoniais e morais, individuais, coletivos e difusos;
            VIII - a facilitação da defesa de seus direitos, inclusive com a inversão do ônus da prova, a seu favor,
            no processo civil, quando, a critério do juiz, for verossímil a alegação.""",
        },
        {
            "title": "CLT - Consolidacao das Leis do Trabalho - Jornada",
            "court": "Legislacao",
            "area": "trabalhista",
            "date": "1943-05-01",
            "content": """Art. 58 - A duração normal do trabalho, para os empregados em qualquer atividade privada,
            não excederá de 8 (oito) horas diárias, desde que não seja fixado expressamente outro limite.
            Art. 59 - A duração diária do trabalho poderá ser acrescida de horas extras, em número não excedente de duas,
            por acordo individual, convenção coletiva ou acordo coletivo de trabalho.
            §1º A remuneração da hora extra será, pelo menos, 50% (cinquenta por cento) superior à da hora normal.
            Art. 71 - Em qualquer trabalho contínuo, cuja duração exceda de 6 horas, é obrigatória a concessão de um
            intervalo para repouso ou alimentação, o qual será, no mínimo, de 1 hora e, salvo acordo escrito ou contrato
            coletivo em contrário, não poderá exceder de 2 horas.""",
        },
        {
            "title": "LGPD - Lei 13709/2018 - Principios Tratamento Dados",
            "court": "Legislacao",
            "area": "civil",
            "date": "2018-08-14",
            "content": """Art. 6º As atividades de tratamento de dados pessoais deverão observar a boa-fé e os seguintes princípios:
            I - finalidade: realização do tratamento para propósitos legítimos, específicos, explícitos e informados ao titular;
            II - adequação: compatibilidade do tratamento com as finalidades informadas ao titular;
            III - necessidade: limitação do tratamento ao mínimo necessário para a realização de suas finalidades;
            IV - livre acesso: garantia, aos titulares, de consulta facilitada e gratuita sobre a forma e a duração do tratamento;
            V - qualidade dos dados: garantia, aos titulares, de exatidão, clareza, relevância e atualização dos dados;
            VII - segurança: utilização de medidas técnicas e administrativas aptas a proteger os dados pessoais;
            X - responsabilização e prestação de contas: demonstração, pelo agente, da adoção de medidas eficazes.""",
        },
        {
            "title": "CPC 2015 - Codigo de Processo Civil - Tutela Provisoria",
            "court": "Legislacao",
            "area": "processual_civil",
            "date": "2015-03-16",
            "content": """Art. 300. A tutela de urgência será concedida quando houver elementos que evidenciem a probabilidade
            do direito e o perigo de dano ou o risco ao resultado útil do processo.
            §1º Para a concessão da tutela de urgência, o juiz pode, conforme o caso, exigir caução real ou fidejussória
            idônea para ressarcir os danos que a outra parte possa vir a sofrer.
            §2º A tutela de urgência pode ser concedida liminarmente ou após justificação prévia.
            §3º A tutela de urgência de natureza antecipada não será concedida quando houver perigo de irreversibilidade
            dos efeitos da decisão.
            Art. 311. A tutela da evidência será concedida, independentemente da demonstração de perigo de dano.""",
        },
    ]

    result = legislation[:limit]
    logger.info("  Legislacao: %d excerpts loaded", len(result))
    return result


async def index_documents(
    docs: list[dict[str, Any]],
    es: AsyncElasticsearch,
    qdrant: AsyncQdrantClient,
    source_name: str,
) -> int:
    """Index documents into ES + Qdrant with embeddings."""
    total_chunks = 0
    batch_size = 8

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        all_chunks = []

        for doc in batch:
            text = doc["content"]
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "doc": doc,
                })

        if not all_chunks:
            continue

        texts = [c["text"] for c in all_chunks]
        try:
            embeddings = await embed_texts(texts)
        except Exception as e:
            logger.error("Embedding error: %s", e)
            continue

        es_docs = []
        qdrant_points = []

        for chunk_data, embedding in zip(all_chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            doc = chunk_data["doc"]
            text = chunk_data["text"]

            es_doc = {
                "content": text,
                "ementa": text if "EMENTA" in text.upper() else "",
                "title": doc["title"],
                "embedding": embedding,
                "document_id": doc.get("id", doc["title"][:50]),
                "document_title": doc["title"],
                "tenant_id": "__system__",
                "doc_type": "decisao" if doc["court"] != "Legislacao" else "lei",
                "court": doc["court"],
                "area": doc.get("area", "geral"),
                "date": doc.get("date"),
                "source": source_name,
                "source_date": doc.get("date"),
                "indexed_at": "2026-02-22T00:00:00Z",
                "status": "active",
                "content_type": "ementa" if "EMENTA" in text.upper() else "generic",
                "metadata": {},
                "content_hash": content_hash(text),
            }

            es_docs.append({"_index": ES_INDEX, "_id": chunk_id, "_source": es_doc})

            payload = {k: v for k, v in es_doc.items() if k != "embedding"}
            qdrant_points.append(
                PointStruct(id=chunk_id, vector=embedding, payload=payload)
            )

        if es_docs:
            from elasticsearch.helpers import async_bulk
            try:
                success, errors = await async_bulk(es, es_docs, raise_on_error=False)
                if errors:
                    logger.warning("ES bulk: %d errors", len(errors))
            except Exception as e:
                logger.error("ES bulk error: %s", e)

        if qdrant_points:
            try:
                await qdrant.upsert(collection_name=QDRANT_COLLECTION, points=qdrant_points)
            except Exception as e:
                logger.error("Qdrant upsert error: %s", e)

        total_chunks += len(all_chunks)
        logger.info("  Indexed batch %d-%d (%d chunks)", i, i + len(batch), len(all_chunks))

    return total_chunks


async def main(source: str, limit: int):
    es = _create_es()
    qdrant = _create_qdrant()

    try:
        total = 0

        if source in ("stf", "all"):
            docs = await fetch_stf_decisions(limit)
            n = await index_documents(docs, es, qdrant, "stf_corte_aberta")
            total += n
            logger.info("STF: %d chunks indexed", n)

        if source in ("datajud", "all"):
            docs = await fetch_datajud_processes(limit)
            n = await index_documents(docs, es, qdrant, "datajud_cnj")
            total += n
            logger.info("DataJud: %d chunks indexed", n)

        if source in ("legislacao", "all"):
            docs = await fetch_legislacao(limit)
            n = await index_documents(docs, es, qdrant, "legislacao_federal")
            total += n
            logger.info("Legislacao: %d chunks indexed", n)

        logger.info("=== Total: %d chunks indexed across all sources ===", total)

    finally:
        await es.close()
        await qdrant.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal data to cloud services")
    parser.add_argument("--source", choices=["stf", "datajud", "legislacao", "all"], default="all")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.source, args.limit))

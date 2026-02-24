"""
Ingest real process data from DataJud CNJ Public API.

Uses the official CNJ public API with search_after pagination
to fetch thousands of real process records from Brazilian courts.

Usage:
    python scripts/ingest_datajud.py --tribunais principais --limit 500
    python scripts/ingest_datajud.py --tribunais todos --limit 200
    python scripts/ingest_datajud.py --tribunais tjsp,tjrj,stj --limit 1000
    python scripts/ingest_datajud.py --tribunais federais --limit 300
"""

import argparse
import asyncio
import hashlib
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("datajud")

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "apps" / "api"))
from app.config import settings

EMBEDDING_DIM = settings.EMBEDDING_DIM
VOYAGE_API_KEY = settings.VOYAGE_API_KEY
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
ES_INDEX = f"{settings.ES_INDEX_PREFIX}_chunks"
QDRANT_COLLECTION = settings.QDRANT_COLLECTION

DATAJUD_KEY = settings.DATAJUD_API_KEY
DATAJUD_URL = settings.DATAJUD_BASE_URL

TRIBUNAIS = {
    # Superiores
    "stj":  {"alias": "api_publica_stj",  "nome": "STJ",  "area_default": "civil"},
    "tst":  {"alias": "api_publica_tst",  "nome": "TST",  "area_default": "trabalhista"},
    "tse":  {"alias": "api_publica_tse",  "nome": "TSE",  "area_default": "eleitoral"},
    # Federais
    "trf1": {"alias": "api_publica_trf1", "nome": "TRF1", "area_default": "civil"},
    "trf2": {"alias": "api_publica_trf2", "nome": "TRF2", "area_default": "civil"},
    "trf3": {"alias": "api_publica_trf3", "nome": "TRF3", "area_default": "civil"},
    "trf4": {"alias": "api_publica_trf4", "nome": "TRF4", "area_default": "civil"},
    "trf5": {"alias": "api_publica_trf5", "nome": "TRF5", "area_default": "civil"},
    # Estaduais
    "tjsp": {"alias": "api_publica_tjsp", "nome": "TJSP", "area_default": "civil"},
    "tjrj": {"alias": "api_publica_tjrj", "nome": "TJRJ", "area_default": "civil"},
    "tjmg": {"alias": "api_publica_tjmg", "nome": "TJMG", "area_default": "civil"},
    "tjrs": {"alias": "api_publica_tjrs", "nome": "TJRS", "area_default": "civil"},
    "tjpr": {"alias": "api_publica_tjpr", "nome": "TJPR", "area_default": "civil"},
    "tjsc": {"alias": "api_publica_tjsc", "nome": "TJSC", "area_default": "civil"},
    "tjba": {"alias": "api_publica_tjba", "nome": "TJBA", "area_default": "civil"},
    "tjpe": {"alias": "api_publica_tjpe", "nome": "TJPE", "area_default": "civil"},
    "tjce": {"alias": "api_publica_tjce", "nome": "TJCE", "area_default": "civil"},
    "tjdft":{"alias": "api_publica_tjdft","nome": "TJDFT","area_default": "civil"},
    "tjgo": {"alias": "api_publica_tjgo", "nome": "TJGO", "area_default": "civil"},
    "tjpa": {"alias": "api_publica_tjpa", "nome": "TJPA", "area_default": "civil"},
    "tjma": {"alias": "api_publica_tjma", "nome": "TJMA", "area_default": "civil"},
    "tjes": {"alias": "api_publica_tjes", "nome": "TJES", "area_default": "civil"},
    "tjmt": {"alias": "api_publica_tjmt", "nome": "TJMT", "area_default": "civil"},
    "tjms": {"alias": "api_publica_tjms", "nome": "TJMS", "area_default": "civil"},
    "tjal": {"alias": "api_publica_tjal", "nome": "TJAL", "area_default": "civil"},
    "tjse": {"alias": "api_publica_tjse", "nome": "TJSE", "area_default": "civil"},
    "tjrn": {"alias": "api_publica_tjrn", "nome": "TJRN", "area_default": "civil"},
    "tjpb": {"alias": "api_publica_tjpb", "nome": "TJPB", "area_default": "civil"},
    "tjpi": {"alias": "api_publica_tjpi", "nome": "TJPI", "area_default": "civil"},
    "tjto": {"alias": "api_publica_tjto", "nome": "TJTO", "area_default": "civil"},
    "tjro": {"alias": "api_publica_tjro", "nome": "TJRO", "area_default": "civil"},
    "tjac": {"alias": "api_publica_tjac", "nome": "TJAC", "area_default": "civil"},
    "tjam": {"alias": "api_publica_tjam", "nome": "TJAM", "area_default": "civil"},
    "tjap": {"alias": "api_publica_tjap", "nome": "TJAP", "area_default": "civil"},
    "tjrr": {"alias": "api_publica_tjrr", "nome": "TJRR", "area_default": "civil"},
    # Trabalhistas
    "trt1": {"alias": "api_publica_trt1", "nome": "TRT1", "area_default": "trabalhista"},
    "trt2": {"alias": "api_publica_trt2", "nome": "TRT2", "area_default": "trabalhista"},
    "trt3": {"alias": "api_publica_trt3", "nome": "TRT3", "area_default": "trabalhista"},
    "trt4": {"alias": "api_publica_trt4", "nome": "TRT4", "area_default": "trabalhista"},
    "trt5": {"alias": "api_publica_trt5", "nome": "TRT5", "area_default": "trabalhista"},
    "trt15":{"alias": "api_publica_trt15","nome": "TRT15","area_default": "trabalhista"},
}

GRUPOS = {
    "principais": ["stj", "tst", "tjsp", "tjrj", "tjmg", "tjrs", "tjdft", "trf1", "trf3", "trt2"],
    "superiores": ["stj", "tst", "tse"],
    "federais": ["trf1", "trf2", "trf3", "trf4", "trf5"],
    "estaduais_grandes": ["tjsp", "tjrj", "tjmg", "tjrs", "tjpr", "tjsc", "tjba", "tjdft"],
    "trabalhistas": ["trt1", "trt2", "trt3", "trt4", "trt5", "trt15"],
}

ASSUNTO_TO_AREA = {
    "penal": ["penal", "criminal", "crime", "tráfico", "homicídio", "furto", "roubo",
              "lesão corporal", "violência doméstica", "maria da penha", "drogas"],
    "trabalhista": ["trabalh", "rescis", "fgts", "hora extra", "salário", "aviso prévio",
                    "férias", "verbas", "empregad", "demiss"],
    "tributario": ["tributár", "fiscal", "icms", "iss", "iptu", "imposto", "contribuição",
                   "dívida ativa", "execução fiscal"],
    "consumidor": ["consumidor", "cdc", "produto", "serviço", "banco", "financ",
                   "telefonia", "plano de saúde"],
    "familia": ["família", "alimentos", "divórcio", "guarda", "pensão", "regulamentação"],
    "administrativo": ["administrativ", "licitação", "servidor", "concurso", "improbidade"],
    "ambiental": ["ambient", "meio ambiente", "poluição", "desmatamento"],
    "previdenciario": ["previdenciár", "aposentadoria", "inss", "benefício", "auxílio"],
    "constitucional": ["constitucional", "mandado de segurança", "habeas corpus"],
}


def classify_by_subjects(subjects: list, default: str = "civil") -> str:
    text = " ".join(s.lower() for s in subjects)
    for area, keywords in ASSUNTO_TO_AREA.items():
        if any(kw in text for kw in keywords):
            return area
    return default


def format_process_text(hit: dict) -> str:
    """Convert DataJud process metadata into searchable text."""
    src = hit.get("_source", hit)
    parts = []

    tribunal = src.get("tribunal", "")
    numero = src.get("numeroProcesso", "")
    classe = src.get("classe", {})
    classe_nome = classe.get("nome", "") if isinstance(classe, dict) else ""
    orgao = src.get("orgaoJulgador", {})
    orgao_nome = orgao.get("nome", "") if isinstance(orgao, dict) else ""

    parts.append(f"PROCESSO: {numero}")
    parts.append(f"TRIBUNAL: {tribunal}")
    if classe_nome:
        parts.append(f"CLASSE: {classe_nome}")
    if orgao_nome:
        parts.append(f"ÓRGÃO JULGADOR: {orgao_nome}")

    data_ajuiz = src.get("dataAjuizamento", "")
    if data_ajuiz:
        parts.append(f"DATA AJUIZAMENTO: {data_ajuiz[:10]}")

    grau = src.get("grau", "")
    if grau:
        parts.append(f"GRAU: {grau}")

    assuntos = src.get("assuntos", [])
    subject_names = []
    for a in assuntos:
        if isinstance(a, dict):
            nome = a.get("nome", "")
            if nome:
                subject_names.append(nome)
        elif isinstance(a, list):
            for sub in a:
                if isinstance(sub, dict):
                    nome = sub.get("nome", "")
                    if nome:
                        subject_names.append(nome)

    if subject_names:
        parts.append(f"ASSUNTOS: {'; '.join(subject_names)}")

    movimentos = src.get("movimentos", [])
    if movimentos:
        mov_relevantes = []
        for m in movimentos[-10:]:
            if isinstance(m, dict):
                nome = m.get("nome", "")
                data = m.get("dataHora", "")[:10] if m.get("dataHora") else ""
                if nome:
                    mov_relevantes.append(f"{nome} ({data})" if data else nome)

        if mov_relevantes:
            parts.append(f"MOVIMENTAÇÕES RECENTES: {'; '.join(mov_relevantes)}")

    return "\n".join(parts)


def _create_es():
    kwargs = {"hosts": [settings.ELASTICSEARCH_URL]}
    if settings.ES_API_KEY:
        kwargs["api_key"] = settings.ES_API_KEY
    return AsyncElasticsearch(**kwargs)


def _create_qdrant():
    kwargs = {"url": settings.QDRANT_URL}
    if settings.QDRANT_API_KEY:
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return AsyncQdrantClient(**kwargs)


async def embed_texts(texts, http):
    if not texts:
        return []
    resp = await http.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={"model": EMBEDDING_MODEL, "input": texts, "input_type": "document"},
        timeout=120.0,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def content_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


async def fetch_datajud(
    http: httpx.AsyncClient,
    tribunal_key: str,
    limit: int = 500,
) -> list:
    """Fetch processes from DataJud using search_after pagination."""
    trib = TRIBUNAIS[tribunal_key]
    url = f"{DATAJUD_URL}/{trib['alias']}/_search"
    headers = {
        "Authorization": f"APIKey {DATAJUD_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("  Fetching %s (limit=%d)...", trib["nome"], limit)

    all_docs = []
    search_after = None
    page_size = min(100, limit)

    while len(all_docs) < limit:
        body = {
            "size": page_size,
            "query": {"match_all": {}},
            "sort": [{"@timestamp": {"order": "asc"}}],
        }
        if search_after:
            body["search_after"] = search_after

        try:
            resp = await http.post(url, headers=headers, json=body, timeout=30.0)

            if resp.status_code == 429:
                logger.warning("  Rate limited, waiting 5s...")
                await asyncio.sleep(5)
                continue

            if resp.status_code != 200:
                logger.error("  %s returned %d: %s", trib["nome"], resp.status_code, resp.text[:200])
                break

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                numero = src.get("numeroProcesso", "")
                sigilo = src.get("nivelSigilo", 0)
                if sigilo > 0:
                    continue

                text = format_process_text(hit)
                if len(text) < 80:
                    continue

                assuntos = src.get("assuntos", [])
                subject_names = []
                for a in assuntos:
                    if isinstance(a, dict) and a.get("nome"):
                        subject_names.append(a["nome"])
                    elif isinstance(a, list):
                        for sub in a:
                            if isinstance(sub, dict) and sub.get("nome"):
                                subject_names.append(sub["nome"])

                classe = src.get("classe", {})
                classe_nome = classe.get("nome", "") if isinstance(classe, dict) else ""

                area = classify_by_subjects(
                    subject_names + [classe_nome],
                    trib["area_default"],
                )

                data_ajuiz = src.get("dataAjuizamento", "")
                parsed_date = data_ajuiz[:10] if data_ajuiz and len(data_ajuiz) >= 10 else None

                all_docs.append({
                    "title": f"{trib['nome']} - {classe_nome} - {numero}"[:200],
                    "court": trib["nome"],
                    "area": area,
                    "date": parsed_date,
                    "content": text,
                    "source_id": src.get("id", numero),
                    "numero_processo": numero,
                    "classe": classe_nome,
                    "orgao_julgador": (src.get("orgaoJulgador", {}) or {}).get("nome", ""),
                    "assuntos": subject_names,
                    "doc_type": "processo",
                })

            last_hit = hits[-1]
            search_after = last_hit.get("sort")
            if not search_after:
                break

            if len(all_docs) % 500 == 0 and len(all_docs) > 0:
                logger.info("    %s: %d processes fetched so far...", trib["nome"], len(all_docs))

        except httpx.ReadTimeout:
            logger.warning("  Timeout on %s, retrying...", trib["nome"])
            await asyncio.sleep(2)
            continue
        except Exception as e:
            logger.error("  Error fetching %s: %s", trib["nome"], e)
            break

    logger.info("  %s: %d processes collected", trib["nome"], len(all_docs))
    return all_docs[:limit]


async def index_docs(docs, es, qdrant, http, source_name, batch_size=8):
    """Index documents with embeddings."""
    total = 0
    seen = set()

    all_items = []
    for doc in docs:
        text = doc.get("content", "")
        if len(text) < 80:
            continue
        h = content_hash(text)
        if h in seen:
            continue
        seen.add(h)
        all_items.append({"text": text, "doc": doc, "hash": h})

    logger.info("  Indexing %d unique items from %s...", len(all_items), source_name)

    for i in range(0, len(all_items), batch_size):
        batch = all_items[i : i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            embeddings = await embed_texts(texts, http)
        except Exception as e:
            logger.error("  Embedding error at %d: %s", i, e)
            await asyncio.sleep(2)
            continue

        es_actions = []
        qdrant_points = []

        for c, emb in zip(batch, embeddings):
            cid = str(uuid.uuid4())
            doc = c["doc"]

            es_doc = {
                "content": c["text"],
                "ementa": "",
                "title": doc["title"],
                "embedding": emb,
                "document_id": doc.get("source_id", doc.get("numero_processo", "")),
                "document_title": doc["title"],
                "tenant_id": "__system__",
                "doc_type": doc.get("doc_type", "processo"),
                "court": doc["court"],
                "area": doc.get("area", "geral"),
                "source": source_name,
                "indexed_at": "2026-02-24T00:00:00Z",
                "status": "active",
                "content_hash": c["hash"],
            }
            if doc.get("date"):
                es_doc["date"] = doc["date"]

            es_actions.append({"_index": ES_INDEX, "_id": cid, "_source": es_doc})
            payload = {k: v for k, v in es_doc.items() if k != "embedding"}
            qdrant_points.append(PointStruct(id=cid, vector=emb, payload=payload))

        if es_actions:
            try:
                await async_bulk(es, es_actions, raise_on_error=False)
            except Exception as e:
                logger.error("  ES error: %s", e)

        if qdrant_points:
            try:
                await qdrant.upsert(collection_name=QDRANT_COLLECTION, points=qdrant_points)
            except Exception as e:
                logger.error("  Qdrant error: %s", e)

        total += len(batch)
        if (i // batch_size) % 50 == 0 and i > 0:
            logger.info("    Progress: %d/%d", min(i + batch_size, len(all_items)), len(all_items))

    return total


async def main(tribunal_selection: str, limit: int):
    if tribunal_selection in GRUPOS:
        tribunal_keys = GRUPOS[tribunal_selection]
    elif tribunal_selection == "todos":
        tribunal_keys = list(TRIBUNAIS.keys())
    else:
        tribunal_keys = [t.strip().lower() for t in tribunal_selection.split(",")]
        invalid = [t for t in tribunal_keys if t not in TRIBUNAIS]
        if invalid:
            logger.error("Tribunais inválidos: %s", invalid)
            logger.info("Disponíveis: %s", ", ".join(sorted(TRIBUNAIS.keys())))
            return

    logger.info("Tribunais selecionados: %s", [TRIBUNAIS[k]["nome"] for k in tribunal_keys])
    logger.info("Limite por tribunal: %d", limit)

    es = _create_es()
    qdrant = _create_qdrant()
    grand_total = 0

    async with httpx.AsyncClient() as http:
        try:
            for key in tribunal_keys:
                trib = TRIBUNAIS[key]
                docs = await fetch_datajud(http, key, limit)

                if docs:
                    n = await index_docs(docs, es, qdrant, http, f"datajud_{key}")
                    grand_total += n
                    logger.info("  %s: %d chunks indexed\n", trib["nome"], n)
                else:
                    logger.warning("  %s: no documents fetched\n", trib["nome"])

                await asyncio.sleep(1)

        finally:
            await es.close()
            await qdrant.close()

    logger.info("=" * 60)
    logger.info("GRAND TOTAL: %d chunks indexed from DataJud", grand_total)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DataJud CNJ processes")
    parser.add_argument(
        "--tribunais",
        default="principais",
        help="Grupo (principais/superiores/federais/estaduais_grandes/trabalhistas/todos) ou lista separada por vírgula (tjsp,tjrj,stj)",
    )
    parser.add_argument("--limit", type=int, default=500, help="Limit per tribunal")
    args = parser.parse_args()
    asyncio.run(main(args.tribunais, args.limit))

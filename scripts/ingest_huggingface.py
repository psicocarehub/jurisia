"""
Ingest massive Brazilian legal data from HuggingFace datasets.

Sources (celsowm collection):
  - Jurisprudência: STF, STJ, TJSP, TJRJ, TJMG, TJDF, TRF2, TRT1, TJPR
  - Súmulas: STF (vinculantes + ordinárias), STJ
  - Legislação: CF/88, CC, CPC, CDC, CLT, CP, CPP, CTN, ECA, Lei Licitações
  - Modelos de petições
  - Leis ordinárias federais (6.5k+)
  - Verbetes vademecum jurídico (14.6k+)

Usage:
    python scripts/ingest_huggingface.py --category all --limit 3000
    python scripts/ingest_huggingface.py --category jurisprudencia --limit 10000
    python scripts/ingest_huggingface.py --category sumulas
    python scripts/ingest_huggingface.py --category legislacao
    python scripts/ingest_huggingface.py --category peticoes
    python scripts/ingest_huggingface.py --category vademecum
    python scripts/ingest_huggingface.py --category leis_ordinarias --limit 6500
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
from datasets import load_dataset
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("hf_ingest")

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "apps" / "api"))
from app.config import settings

EMBEDDING_DIM = settings.EMBEDDING_DIM
VOYAGE_API_KEY = settings.VOYAGE_API_KEY
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
ES_INDEX = f"{settings.ES_INDEX_PREFIX}_chunks"
QDRANT_COLLECTION = settings.QDRANT_COLLECTION


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


def chunk_text(text, max_chars=1500, overlap=200):
    if not text or len(text.strip()) < 50:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + max_chars].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def content_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def classify_area(text):
    t = text.lower()
    for area, kws in {
        "tributario": ["icms", "pis", "cofins", "tributar", "fiscal", "imposto"],
        "trabalhista": ["clt", "trabalh", "empregad", "rescis", "salário", "fgts"],
        "penal": ["penal", "crime", "pena", "prisão", "habeas corpus", "tráfico"],
        "constitucional": ["constitui", "fundamental", "adi ", "adpf", "repercussão geral"],
        "consumidor": ["consumidor", "cdc", "fornecedor", "defeito do produto"],
        "administrativo": ["administrat", "licitação", "concurso", "improbidade"],
        "ambiental": ["ambiental", "meio ambiente", "poluição", "fauna"],
        "civil": ["civil", "contrato", "responsabilidade", "dano moral", "família", "usucapião"],
        "processual_civil": ["processo civil", "recurso", "tutela", "agravo", "apelação"],
        "previdenciario": ["previdenciár", "aposentadoria", "inss", "benefício"],
    }.items():
        if any(kw in t for kw in kws):
            return area
    return "geral"


# ============ DATASET LOADERS ============

JURISPRUDENCIA_DATASETS = {
    "STJ": ("celsowm/jurisprudencias_stj", 2000),
    "STF": ("celsowm/jurisprudencias_stf", 1500),
    "TJSP": ("celsowm/jurisprudencias_tjsp", 2000),
    "TJRJ": ("celsowm/jurisprudencias_tjrj", 1000),
    "TJMG": ("celsowm/jurisprudencias_tjmg", 2000),
    "TJDF": ("celsowm/jurisprudencias_tjdf", 500),
    "TRF2": ("celsowm/jurisprudencias_trf2", 500),
    "TRT1": ("celsowm/jurisprudencias_trt1", 2000),
    "TJPR": ("celsowm/jurisprudencias_tjpr", 500),
    "TJMT": ("celsowm/jurisprudencias_tjmt", 500),
    "TJPA": ("celsowm/jurisprudencias_tjpa", 500),
}

SUMULA_DATASETS = {
    "STF_vinculante": "celsowm/sumulas_vinculantes_stf",
    "STJ": "celsowm/sumulas_stj",
    "STF": "celsowm/sumulas_stf",
}

LEGISLACAO_DATASETS = {
    "CF/1988": "celsowm/constituicao_br_1988",
    "Código Civil": "celsowm/codigo_civil_brasileiro_lei_10406_2002",
    "CPC/2015": "celsowm/cpc_2015",
    "CDC": "celsowm/cdc_lei_8078_1990",
    "CLT": "celsowm/clt_consolidacao_leis_trabalho_decreto_lei_5452",
    "Código Penal": "celsowm/codigo_penal_brasileiro_lei_2848_1940",
    "CPP": "celsowm/cpc_codigo_processo_penal_lei_3689_1941",
    "CTN": "celsowm/codigo_tributario_lei_5172_1966",
    "ECA": "celsowm/eca_lei_13_07_1990",
    "Lei Licitações": "celsowm/lei_licitacoes_14133",
    "CTB": "celsowm/ctb_codigo_de_transito_brasileiro_lei_9_503_1997",
    "LEP": "celsowm/lei_execucao_penal_7_210_11_07_1984",
    "Código Florestal": "celsowm/codigo_florestal_lei_25_05_12",
    "Código Eleitoral": "celsowm/codigo_eleitoral_lei_15_07_95",
    "Lei Migração": "celsowm/lei_migracao_L13445",
    "Ética OAB": "celsowm/codigo_etica_e_disciplina_oab",
}


def load_jurisprudencia(court, dataset_name, limit):
    logger.info("  Loading %s (limit=%d)...", court, limit)
    docs = []
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            ementa = row.get("ementa_texto", row.get("ementa", ""))
            if not ementa or len(ementa) < 100:
                continue
            processo = row.get("processo", row.get("identificacao", ""))
            docs.append({
                "title": f"{court} - {processo}"[:200] if processo else f"{court} #{i}",
                "court": court,
                "area": classify_area(ementa),
                "date": row.get("data_julgamento", row.get("data_publicacao_fonte", "")),
                "content": ementa,
                "source_id": processo,
                "relator": row.get("relator", ""),
                "orgao": row.get("orgao_julgador", ""),
            })
    except Exception as e:
        logger.error("  Error loading %s: %s", dataset_name, e)
    logger.info("    %s: %d documents loaded", court, len(docs))
    return docs


def load_sumulas(name, dataset_name):
    logger.info("  Loading súmulas %s...", name)
    docs = []
    try:
        ds = load_dataset(dataset_name, split="train")
        for row in ds:
            titulo = row.get("titulo", row.get("nome", f"Súmula {name}"))
            texto = row.get("texto_sem_formatacao", row.get("texto", row.get("enunciado", "")))
            if not texto or len(texto) < 20:
                continue
            court = "STF" if "stf" in name.lower() else "STJ"
            docs.append({
                "title": titulo[:200],
                "court": court,
                "area": classify_area(texto),
                "date": "",
                "content": texto,
                "source_id": titulo,
                "doc_type": "sumula",
            })
    except Exception as e:
        logger.error("  Error loading %s: %s", dataset_name, e)
    logger.info("    Súmulas %s: %d loaded", name, len(docs))
    return docs


def load_legislacao(name, dataset_name):
    logger.info("  Loading legislação %s...", name)
    docs = []
    try:
        ds = load_dataset(dataset_name, split="train")
        for row in ds:
            artigo = row.get("artigo", row.get("numero", ""))
            texto = row.get("texto", row.get("conteudo", row.get("texto_sem_formatacao", "")))
            if not texto or len(texto) < 20:
                continue
            titulo_full = f"{name} - Art. {artigo}" if artigo else name
            docs.append({
                "title": titulo_full[:200],
                "court": "Legislacao",
                "area": classify_area(f"{name} {texto}"),
                "date": "",
                "content": texto,
                "source_id": titulo_full,
                "doc_type": "lei",
            })
    except Exception as e:
        logger.error("  Error loading %s: %s", dataset_name, e)
    logger.info("    %s: %d articles loaded", name, len(docs))
    return docs


def load_peticoes():
    logger.info("  Loading modelos de petições...")
    docs = []
    try:
        ds = load_dataset("celsowm/modelos_peticoes", split="train")
        for row in ds:
            titulo = row.get("titulo", "Modelo de petição")
            conteudo = row.get("conteudo", "")
            if not conteudo or len(conteudo) < 100:
                continue
            docs.append({
                "title": f"Modelo: {titulo}"[:200],
                "court": "Modelo",
                "area": classify_area(f"{titulo} {conteudo[:500]}"),
                "date": "",
                "content": conteudo[:10000],
                "source_id": titulo,
                "doc_type": "modelo_peticao",
            })
    except Exception as e:
        logger.error("  Error loading petições: %s", e)
    logger.info("    Petições: %d models loaded", len(docs))
    return docs


def load_leis_ordinarias(limit=3000):
    logger.info("  Loading leis ordinárias federais (limit=%d)...", limit)
    docs = []
    try:
        ds = load_dataset("celsowm/leis_ordinarias_1988_2024", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            titulo = row.get("titulo", row.get("nome", f"Lei #{i}"))
            texto = row.get("texto", row.get("ementa", row.get("conteudo", "")))
            if not texto or len(texto) < 50:
                continue
            docs.append({
                "title": titulo[:200],
                "court": "Legislacao",
                "area": classify_area(f"{titulo} {texto[:500]}"),
                "date": row.get("data", row.get("data_publicacao", "")),
                "content": texto[:10000],
                "source_id": titulo,
                "doc_type": "lei",
            })
    except Exception as e:
        logger.error("  Error loading leis ordinárias: %s", e)
    logger.info("    Leis ordinárias: %d loaded", len(docs))
    return docs


def load_vademecum(limit=5000):
    """Load verbetes from vademecum jurídico dataset (14.6k+ entries)."""
    logger.info("  Loading vademecum jurídico (limit=%d)...", limit)
    docs = []
    try:
        ds = load_dataset("celsowm/verbetes_vademecum_direito", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            verbete = row.get("verbete", row.get("titulo", row.get("nome", "")))
            texto = row.get("significado", row.get("texto", row.get("descricao", "")))
            if not texto or len(texto) < 20:
                continue
            full_text = f"{verbete}: {texto}" if verbete else texto
            docs.append({
                "title": f"Vademecum: {verbete}"[:200] if verbete else f"Vademecum #{i}",
                "court": "Doutrina",
                "area": classify_area(full_text[:500]),
                "date": "",
                "content": full_text[:8000],
                "source_id": verbete or f"vademecum_{i}",
                "doc_type": "doutrina",
            })
    except Exception as e:
        logger.error("  Error loading vademecum: %s", e)
    logger.info("    Vademecum: %d loaded", len(docs))
    return docs


def load_leis_estaduais_rj(limit=3000):
    """Load leis estaduais do RJ (8.8k+ entries)."""
    logger.info("  Loading leis estaduais RJ (limit=%d)...", limit)
    docs = []
    try:
        ds = load_dataset("celsowm/leis_estaduais_rj", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            lei = row.get("lei", f"Lei RJ #{i}")
            texto = row.get("texto") or ""
            artigos = row.get("artigos", [])

            if not texto and artigos:
                parts = []
                for art in artigos[:50]:
                    if isinstance(art, dict):
                        num = art.get("numero", "")
                        txt = art.get("texto", "")
                        if txt:
                            parts.append(f"Art. {num}: {txt}")
                texto = "\n".join(parts)

            if not texto or len(texto) < 50:
                continue

            docs.append({
                "title": f"Lei RJ {lei}"[:200],
                "court": "Legislacao",
                "area": classify_area(texto[:500]),
                "date": "",
                "content": texto[:10000],
                "source_id": f"lei_rj_{lei}",
                "doc_type": "lei_estadual",
            })
    except Exception as e:
        logger.error("  Error loading leis estaduais RJ: %s", e)
    logger.info("    Leis estaduais RJ: %d loaded", len(docs))
    return docs


def load_simulado_oab(limit=1000):
    """Load questões do simulado OAB (1k+ entries)."""
    logger.info("  Loading simulado OAB (limit=%d)...", limit)
    docs = []
    try:
        ds = load_dataset("celsowm/simulado_oab", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            enunciado = row.get("enunciado", "")
            if not enunciado or len(enunciado) < 30:
                continue
            disciplina = row.get("disciplina", "")
            alternativas = row.get("alternativas", {})
            resposta = row.get("resposta_correta", "")

            full = f"QUESTÃO OAB - {disciplina}:\n{enunciado}"
            if isinstance(alternativas, dict):
                for letra, texto in alternativas.items():
                    full += f"\n{letra}) {texto}"
            if resposta:
                full += f"\n\nRESPOSTA CORRETA: {resposta}"

            docs.append({
                "title": f"OAB Simulado - {disciplina} #{i+1}"[:200],
                "court": "OAB",
                "area": classify_area(f"{disciplina} {enunciado}"),
                "date": "",
                "content": full[:8000],
                "source_id": f"oab_simulado_{i}",
                "doc_type": "simulado",
            })
    except Exception as e:
        logger.error("  Error loading simulado OAB: %s", e)
    logger.info("    Simulado OAB: %d loaded", len(docs))
    return docs


# ============ INDEXING ============

async def index_batch(docs, es, qdrant, http, source_name, batch_size=8):
    """Index documents with embeddings into ES + Qdrant."""
    total = 0
    seen = set()

    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc.get("content", "")):
            h = content_hash(chunk)
            if h in seen:
                continue
            seen.add(h)
            all_chunks.append({"text": chunk, "doc": doc, "hash": h})

    logger.info("  Indexing %d unique chunks from %d docs (%s)...", len(all_chunks), len(docs), source_name)

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
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
            raw_date = doc.get("date", "")
            parsed_date = raw_date if raw_date and len(raw_date) >= 8 else None

            es_doc = {
                "content": c["text"],
                "ementa": c["text"] if "EMENTA" in c["text"].upper() else "",
                "title": doc["title"],
                "embedding": emb,
                "document_id": doc.get("source_id", doc["title"][:50]),
                "document_title": doc["title"],
                "tenant_id": "__system__",
                "doc_type": doc.get("doc_type", "decisao"),
                "court": doc["court"],
                "area": doc.get("area", "geral"),
                "source": source_name,
                "indexed_at": "2026-02-23T00:00:00Z",
                "status": "active",
                "content_hash": c["hash"],
            }
            if parsed_date:
                es_doc["date"] = parsed_date
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
        if (i // batch_size) % 25 == 0:
            logger.info("    Progress: %d/%d chunks", min(i + batch_size, len(all_chunks)), len(all_chunks))

    return total


# ============ MAIN ============

async def main(category, limit):
    es = _create_es()
    qdrant = _create_qdrant()
    grand_total = 0

    async with httpx.AsyncClient(timeout=120.0) as http:
        try:
            # Jurisprudência
            if category in ("jurisprudencia", "all"):
                logger.info("\n=== JURISPRUDÊNCIA ===")
                for court, (ds_name, default_limit) in JURISPRUDENCIA_DATASETS.items():
                    court_limit = min(limit, default_limit) if category == "all" else limit
                    docs = load_jurisprudencia(court, ds_name, court_limit)
                    if docs:
                        n = await index_batch(docs, es, qdrant, http, f"hf_{court.lower()}")
                        grand_total += n
                        logger.info("  %s: %d chunks indexed", court, n)

            # Súmulas
            if category in ("sumulas", "all"):
                logger.info("\n=== SÚMULAS ===")
                for name, ds_name in SUMULA_DATASETS.items():
                    docs = load_sumulas(name, ds_name)
                    if docs:
                        n = await index_batch(docs, es, qdrant, http, f"hf_sumula_{name.lower()}")
                        grand_total += n
                        logger.info("  Súmulas %s: %d chunks indexed", name, n)

            # Legislação
            if category in ("legislacao", "all"):
                logger.info("\n=== LEGISLAÇÃO ===")
                for name, ds_name in LEGISLACAO_DATASETS.items():
                    docs = load_legislacao(name, ds_name)
                    if docs:
                        n = await index_batch(docs, es, qdrant, http, f"hf_leg_{name.lower().replace('/', '_')}")
                        grand_total += n
                        logger.info("  %s: %d chunks indexed", name, n)

            # Leis ordinárias
            if category in ("leis_ordinarias", "legislacao", "all"):
                logger.info("\n=== LEIS ORDINÁRIAS ===")
                docs = load_leis_ordinarias(limit=limit if category == "leis_ordinarias" else 3000)
                if docs:
                    n = await index_batch(docs, es, qdrant, http, "hf_leis_ordinarias")
                    grand_total += n
                    logger.info("  Leis ordinárias: %d chunks indexed", n)

            # Petições
            if category in ("peticoes", "all"):
                logger.info("\n=== MODELOS DE PETIÇÕES ===")
                docs = load_peticoes()
                if docs:
                    n = await index_batch(docs, es, qdrant, http, "hf_modelos_peticoes")
                    grand_total += n
                    logger.info("  Petições: %d chunks indexed", n)

            # Vademecum
            if category in ("vademecum", "all"):
                logger.info("\n=== VADEMECUM JURÍDICO ===")
                docs = load_vademecum(limit=limit if category == "vademecum" else 5000)
                if docs:
                    n = await index_batch(docs, es, qdrant, http, "hf_vademecum")
                    grand_total += n
                    logger.info("  Vademecum: %d chunks indexed", n)

            # Leis estaduais RJ
            if category in ("leis_estaduais", "all"):
                logger.info("\n=== LEIS ESTADUAIS RJ ===")
                docs = load_leis_estaduais_rj(limit=limit if category == "leis_estaduais" else 3000)
                if docs:
                    n = await index_batch(docs, es, qdrant, http, "hf_leis_estaduais_rj")
                    grand_total += n
                    logger.info("  Leis estaduais RJ: %d chunks indexed", n)

            # Simulado OAB
            if category in ("simulado_oab", "all"):
                logger.info("\n=== SIMULADO OAB ===")
                docs = load_simulado_oab(limit=limit if category == "simulado_oab" else 1000)
                if docs:
                    n = await index_batch(docs, es, qdrant, http, "hf_simulado_oab")
                    grand_total += n
                    logger.info("  Simulado OAB: %d chunks indexed", n)

            logger.info("\n" + "=" * 60)
            logger.info("GRAND TOTAL: %d chunks indexed", grand_total)
            logger.info("=" * 60)

        finally:
            await es.close()
            await qdrant.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HuggingFace legal datasets")
    parser.add_argument(
        "--category",
        choices=[
            "jurisprudencia", "sumulas", "legislacao", "leis_ordinarias",
            "peticoes", "vademecum", "leis_estaduais", "simulado_oab", "all",
        ],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    asyncio.run(main(args.category, args.limit))

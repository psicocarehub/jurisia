"""
DAG: Atualizacao mensal da base CNPJ da Receita Federal.

Baixa os arquivos mensais de empresas, estabelecimentos e socios,
parseia e indexa em Elasticsearch para entity resolution.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger("jurisai.dag.cnpj")

default_args = {
    "owner": "jurisai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(hours=1),
}


def _download_empresas(**context):
    from app.services.ingestion.cnpj import CNPJClient
    client = CNPJClient()
    files = asyncio.run(client.download_latest("Empresas"))
    logger.info("Downloaded %d empresa files", len(files))
    return [str(f) for f in files]


def _download_estabelecimentos(**context):
    from app.services.ingestion.cnpj import CNPJClient
    client = CNPJClient()
    files = asyncio.run(client.download_latest("Estabelecimentos"))
    logger.info("Downloaded %d estabelecimento files", len(files))
    return [str(f) for f in files]


def _download_socios(**context):
    from app.services.ingestion.cnpj import CNPJClient
    client = CNPJClient()
    files = asyncio.run(client.download_latest("Socios"))
    logger.info("Downloaded %d socio files", len(files))
    return [str(f) for f in files]


def _parse_and_index(**context):
    """Parse all downloaded files and index in Elasticsearch."""
    from pathlib import Path

    from elasticsearch import Elasticsearch

    from app.config import settings
    from app.services.ingestion.cnpj import CNPJClient

    client = CNPJClient()
    es = Elasticsearch(settings.ELASTICSEARCH_URL)
    index_name = f"{settings.ES_INDEX_PREFIX}_cnpj"

    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "cnpj": {"type": "keyword"},
                        "cnpj_basico": {"type": "keyword"},
                        "razao_social": {"type": "text", "analyzer": "brazilian"},
                        "nome_fantasia": {"type": "text", "analyzer": "brazilian"},
                        "natureza_juridica": {"type": "keyword"},
                        "situacao_cadastral": {"type": "keyword"},
                        "uf": {"type": "keyword"},
                        "municipio": {"type": "keyword"},
                        "cnae_principal": {"type": "keyword"},
                        "capital_social": {"type": "keyword"},
                        "porte_empresa": {"type": "keyword"},
                    }
                },
                "settings": {"number_of_shards": 3, "number_of_replicas": 0},
            },
        )

    data_dir = Path(settings.CNPJ_DATA_DIR)
    total_indexed = 0

    for zip_path in sorted(data_dir.glob("*Empresas*.zip")):
        records = client.parse_empresas(zip_path)
        for batch_start in range(0, len(records), 500):
            batch = records[batch_start : batch_start + 500]
            actions = []
            for r in batch:
                actions.append({"index": {"_index": index_name, "_id": r["cnpj_basico"]}})
                actions.append(r)
            if actions:
                es.bulk(body=actions)
                total_indexed += len(batch)

    for zip_path in sorted(data_dir.glob("*Estabelecimentos*.zip")):
        records = client.parse_estabelecimentos(zip_path)
        for batch_start in range(0, len(records), 500):
            batch = records[batch_start : batch_start + 500]
            actions = []
            for r in batch:
                doc_id = r.get("cnpj", r.get("cnpj_basico", ""))
                actions.append({"update": {"_index": index_name, "_id": doc_id}})
                actions.append({"doc": r, "doc_as_upsert": True})
            if actions:
                es.bulk(body=actions)

    logger.info("CNPJ indexing complete: %d empresas", total_indexed)
    es.close()


with DAG(
    dag_id="update_cnpj",
    default_args=default_args,
    description="Atualizacao mensal da base CNPJ da Receita Federal",
    schedule_interval="0 2 1 * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "cnpj", "compliance"],
    max_active_runs=1,
) as dag:
    download_emp = PythonOperator(task_id="download_empresas", python_callable=_download_empresas)
    download_est = PythonOperator(task_id="download_estabelecimentos", python_callable=_download_estabelecimentos)
    download_soc = PythonOperator(task_id="download_socios", python_callable=_download_socios)
    parse_index = PythonOperator(task_id="parse_and_index", python_callable=_parse_and_index)

    [download_emp, download_est, download_soc] >> parse_index

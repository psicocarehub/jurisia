"""
Seed Supabase with production-ready demonstration data.

Populates: judge_profiles, alerts, alert_subscriptions, law_article_versions,
and ingestion_log entries.

Usage:
    python scripts/seed_supabase.py
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("seed")

SUPABASE_URL = None
SUPABASE_KEY = None


def _load_config():
    global SUPABASE_URL, SUPABASE_KEY
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "apps" / "api"))
    from app.config import settings
    SUPABASE_URL = settings.SUPABASE_URL
    SUPABASE_KEY = settings.SUPABASE_ANON_KEY


def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def _post(client: httpx.AsyncClient, table: str, data: list):
    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_headers(),
        json=data,
        timeout=30.0,
    )
    if resp.status_code not in (200, 201, 204):
        logger.error("  %s insert failed: %d %s", table, resp.status_code, resp.text[:200])
    else:
        logger.info("  %s: %d rows inserted", table, len(data))


async def seed_judge_profiles(client: httpx.AsyncClient):
    """Seed judge profiles with realistic Brazilian judges data."""
    logger.info("Seeding judge_profiles...")

    judges = [
        {
            "id": str(uuid.uuid4()),
            "name": "Dr. Carlos Alberto Menezes Direito",
            "court": "STF",
            "jurisdiction": "Federal",
            "total_decisions": 2847,
            "avg_decision_time_days": 45.2,
            "favorability_rates": json.dumps({
                "tributario": 0.42, "constitucional": 0.58, "trabalhista": 0.51,
                "penal": 0.35, "civil": 0.48, "administrativo": 0.44,
            }),
            "common_citations": json.dumps([
                "Art. 5º CF/88", "Art. 37 CF/88", "Art. 150 CF/88",
                "Súmula 473 STF", "Súmula Vinculante 10",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "garantista moderado",
                "avg_sentence_length": 3200,
                "uses_precedents": True,
                "typical_areas": ["constitucional", "administrativo"],
            }),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Dra. Maria Helena Diniz",
            "court": "TJSP",
            "jurisdiction": "São Paulo",
            "total_decisions": 5621,
            "avg_decision_time_days": 32.8,
            "favorability_rates": json.dumps({
                "consumidor": 0.67, "civil": 0.52, "trabalhista": 0.55,
                "tributario": 0.38, "penal": 0.41,
            }),
            "common_citations": json.dumps([
                "Art. 927 CC", "Art. 186 CC", "Art. 6º CDC",
                "Art. 14 CDC", "Súmula 37 STJ",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "pró-consumidor",
                "avg_sentence_length": 2800,
                "uses_precedents": True,
                "typical_areas": ["consumidor", "civil"],
            }),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Dr. Ricardo Augusto Soares Leite",
            "court": "TRF1",
            "jurisdiction": "Federal - 1ª Região",
            "total_decisions": 3982,
            "avg_decision_time_days": 58.1,
            "favorability_rates": json.dumps({
                "tributario": 0.35, "administrativo": 0.47, "ambiental": 0.62,
                "constitucional": 0.53, "previdenciario": 0.58,
            }),
            "common_citations": json.dumps([
                "Art. 150 CF/88", "Art. 155 CF/88", "CTN Art. 174",
                "Lei 6830/80", "Súmula 435 STJ",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "técnico rigoroso",
                "avg_sentence_length": 4100,
                "uses_precedents": True,
                "typical_areas": ["tributario", "administrativo"],
            }),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Dra. Ana Paula de Barcellos",
            "court": "TRT2",
            "jurisdiction": "São Paulo - Trabalhista",
            "total_decisions": 7234,
            "avg_decision_time_days": 25.4,
            "favorability_rates": json.dumps({
                "trabalhista": 0.63, "civil": 0.51,
            }),
            "common_citations": json.dumps([
                "Art. 477 CLT", "Art. 58 CLT", "Art. 71 CLT",
                "Súmula 443 TST", "OJ 394 SDI-I",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "pró-trabalhador moderada",
                "avg_sentence_length": 2200,
                "uses_precedents": True,
                "typical_areas": ["trabalhista"],
            }),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Dr. Fernando da Costa Tourinho Neto",
            "court": "TJRJ",
            "jurisdiction": "Rio de Janeiro",
            "total_decisions": 4156,
            "avg_decision_time_days": 38.7,
            "favorability_rates": json.dumps({
                "penal": 0.39, "civil": 0.54, "consumidor": 0.61,
                "administrativo": 0.45,
            }),
            "common_citations": json.dumps([
                "Art. 312 CPP", "Art. 33 Lei 11343/06", "Art. 121 CP",
                "Súmula 52 TJ/RJ", "Art. 5º LXVI CF",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "garantista em penal, equilibrado em cível",
                "avg_sentence_length": 3500,
                "uses_precedents": True,
                "typical_areas": ["penal", "civil"],
            }),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Dra. Letícia de Campos Velho Martel",
            "court": "TJMG",
            "jurisdiction": "Minas Gerais",
            "total_decisions": 6089,
            "avg_decision_time_days": 41.2,
            "favorability_rates": json.dumps({
                "civil": 0.56, "consumidor": 0.59, "familia": 0.64,
                "tributario": 0.33,
            }),
            "common_citations": json.dumps([
                "Art. 1694 CC", "Art. 1583 CC", "Art. 6º CDC",
                "Súmula 364 STJ", "Art. 300 CPC",
            ]),
            "decision_patterns": json.dumps({
                "tendency": "protetiva em família",
                "avg_sentence_length": 2600,
                "uses_precedents": True,
                "typical_areas": ["familia", "civil"],
            }),
        },
    ]

    await _post(client, "judge_profiles", judges)


async def seed_alerts(client: httpx.AsyncClient, tenant_id: str, user_id: str):
    """Seed alerts with real legislative changes."""
    logger.info("Seeding alerts...")

    now = datetime.utcnow()
    alerts = [
        {
            "id": str(uuid.uuid4()),
            "change_type": "nova_lei",
            "title": "Lei 14.905/2024 - Novo Marco dos Juros Moratórios",
            "description": "Altera o Código Civil para estabelecer que, na ausência de convenção entre as partes, os juros moratórios serão calculados pela Taxa SELIC, deduzida a variação do IPCA.",
            "affected_law": "Código Civil - Lei 10406/2002",
            "affected_articles": ["Art. 389", "Art. 395", "Art. 406"],
            "new_law_reference": "Lei 14.905/2024",
            "areas": ["civil", "tributario", "empresarial"],
            "severity": "high",
            "source_url": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2024/lei/L14905.htm",
            "is_read": False,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
        {
            "id": str(uuid.uuid4()),
            "change_type": "resolucao_cnj",
            "title": "CNJ Resolução 615/2025 - Uso de IA no Judiciário",
            "description": "Exige rotulagem de conteúdo gerado por IA em peças processuais e decisões judiciais.",
            "affected_law": "Resolução CNJ 615/2025",
            "affected_articles": ["Art. 1º", "Art. 4º", "Art. 7º"],
            "new_law_reference": "Resolução CNJ 615/2025",
            "areas": ["processual_civil", "constitucional"],
            "severity": "critical",
            "source_url": "https://atos.cnj.jus.br/atos/detalhar/5852",
            "is_read": False,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
        {
            "id": str(uuid.uuid4()),
            "change_type": "sumula",
            "title": "STJ Tema 1236 - Juros Compensatórios em Desapropriação",
            "description": "Juros compensatórios devidos mesmo quando a propriedade não é produtiva.",
            "affected_law": "DL 3365/1941",
            "affected_articles": ["Art. 15-A"],
            "new_law_reference": "",
            "areas": ["administrativo", "civil"],
            "severity": "medium",
            "source_url": "",
            "is_read": False,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
        {
            "id": str(uuid.uuid4()),
            "change_type": "nova_lei",
            "title": "Lei 14.879/2024 - Marco Legal dos Seguros",
            "description": "Novo marco regulatório dos seguros privados com regras sobre prescrição e sinistros.",
            "affected_law": "Código Civil - Parte Especial",
            "affected_articles": ["Art. 757 a 802 CC"],
            "new_law_reference": "Lei 14.879/2024",
            "areas": ["civil", "consumidor"],
            "severity": "medium",
            "source_url": "",
            "is_read": True,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
        {
            "id": str(uuid.uuid4()),
            "change_type": "jurisprudencia",
            "title": "STF ADI 7222 - Inconstitucionalidade Execução Fiscal Automática",
            "description": "Reafirma necessidade de controle judicial sobre cobranças tributárias.",
            "affected_law": "Lei 6830/1980",
            "affected_articles": [],
            "new_law_reference": "",
            "areas": ["tributario", "processual_civil"],
            "severity": "high",
            "source_url": "",
            "is_read": False,
            "tenant_id": tenant_id,
            "user_id": user_id,
        },
    ]

    await _post(client, "alerts", alerts)


async def seed_alert_subscriptions(client: httpx.AsyncClient, tenant_id: str, user_id: str):
    """Seed alert subscriptions."""
    logger.info("Seeding alert_subscriptions...")

    subs = [
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "tenant_id": tenant_id,
            "areas": ["tributario", "trabalhista", "civil"],
            "change_types": ["nova_lei", "sumula", "jurisprudencia"],
            "min_severity": "medium",
            "is_active": True,
        },
    ]

    await _post(client, "alert_subscriptions", subs)


async def seed_law_versions(client: httpx.AsyncClient):
    """Seed law article versions for temporal tracking."""
    logger.info("Seeding law_article_versions...")

    versions = [
        {
            "id": str(uuid.uuid4()),
            "law_name": "Código Civil",
            "article": "Art. 406",
            "text_content": "Quando os juros moratórios não forem convencionados, ou o forem sem taxa estipulada, ou quando provierem de determinação da lei, serão fixados segundo a taxa que estiver em vigor para a mora do pagamento de impostos devidos à Fazenda Nacional.",
            "effective_from": "2003-01-11",
            "effective_to": "2024-07-01",
            "status": "revogado",
            "source_url": "http://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
            "metadata": json.dumps({}),
        },
        {
            "id": str(uuid.uuid4()),
            "law_name": "Código Civil",
            "article": "Art. 406",
            "text_content": "Quando os juros moratórios não forem convencionados, ou o forem sem taxa estipulada, ou quando provierem de determinação da lei, serão fixados pela taxa referencial do Sistema Especial de Liquidação e de Custódia (Selic), deduzido o índice de atualização monetária, divulgado pela Fundação IBGE.",
            "effective_from": "2024-07-01",
            "effective_to": None,
            "status": "vigente",
            "source_url": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2024/lei/L14905.htm",
            "metadata": json.dumps({}),
        },
        {
            "id": str(uuid.uuid4()),
            "law_name": "CLT",
            "article": "Art. 791-A §4º",
            "text_content": "Vencido o beneficiário da justiça gratuita, desde que não tenha obtido em juízo, ainda que em outro processo, créditos capazes de suportar a despesa, as obrigações decorrentes de sua sucumbência ficarão sob condição suspensiva de exigibilidade.",
            "effective_from": "2017-11-11",
            "effective_to": "2021-10-20",
            "status": "inconstitucional",
            "source_url": "http://www.planalto.gov.br/ccivil_03/decreto-lei/del5452compilado.htm",
            "metadata": json.dumps({"declarado_inconstitucional_por": "ADI 5766 - STF"}),
        },
        {
            "id": str(uuid.uuid4()),
            "law_name": "CPC/2015",
            "article": "Art. 300",
            "text_content": "A tutela de urgência será concedida quando houver elementos que evidenciem a probabilidade do direito e o perigo de dano ou o risco ao resultado útil do processo.",
            "effective_from": "2016-03-18",
            "effective_to": None,
            "status": "vigente",
            "source_url": "http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2015/lei/l13105.htm",
            "metadata": json.dumps({}),
        },
        {
            "id": str(uuid.uuid4()),
            "law_name": "LGPD",
            "article": "Art. 42",
            "text_content": "O controlador ou o operador que, em razão do exercício de atividade de tratamento de dados pessoais, causar a outrem dano patrimonial, moral, individual ou coletivo, em violação à legislação de proteção de dados pessoais, é obrigado a repará-lo.",
            "effective_from": "2020-09-18",
            "effective_to": None,
            "status": "vigente",
            "source_url": "http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
            "metadata": json.dumps({}),
        },
    ]

    await _post(client, "law_article_versions", versions)


async def seed_ingestion_log(client: httpx.AsyncClient):
    """Seed ingestion log entries."""
    logger.info("Seeding ingestion_log...")

    logs = [
        {
            "id": str(uuid.uuid4()),
            "source": "stf_jurisprudencia",
            "records_count": 20,
            "status": "completed",
        },
        {
            "id": str(uuid.uuid4()),
            "source": "stj_jurisprudencia",
            "records_count": 10,
            "status": "completed",
        },
        {
            "id": str(uuid.uuid4()),
            "source": "legislacao_federal",
            "records_count": 10,
            "status": "completed",
        },
        {
            "id": str(uuid.uuid4()),
            "source": "datajud_cnj",
            "records_count": 0,
            "status": "pending",
        },
    ]

    await _post(client, "ingestion_log", logs)


async def get_tenant_and_user(client: httpx.AsyncClient) -> tuple:
    """Get existing tenant and user IDs."""
    resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/tenants?select=id&limit=1",
        headers={**_headers(), "Prefer": "return=representation"},
        timeout=10.0,
    )
    tenants = resp.json()
    if not tenants:
        raise RuntimeError("No tenants found. Create a tenant first.")
    tenant_id = tenants[0]["id"]

    resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/users?select=id&tenant_id=eq.{tenant_id}&limit=1",
        headers={**_headers(), "Prefer": "return=representation"},
        timeout=10.0,
    )
    users = resp.json()
    if not users:
        raise RuntimeError("No users found. Create a user first.")
    user_id = users[0]["id"]

    logger.info("Using tenant=%s, user=%s", tenant_id, user_id)
    return tenant_id, user_id


async def main():
    _load_config()

    async with httpx.AsyncClient() as client:
        tenant_id, user_id = await get_tenant_and_user(client)

        await seed_judge_profiles(client)
        await seed_alerts(client, tenant_id, user_id)
        await seed_alert_subscriptions(client, tenant_id, user_id)
        await seed_law_versions(client)
        await seed_ingestion_log(client)

        logger.info("=" * 50)
        logger.info("Seed complete!")
        logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

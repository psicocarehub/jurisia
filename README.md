# Juris.AI

Copiloto de IA para advogados brasileiros. Combina busca semântica em jurisprudência, geração de petições, análise de processos (Raio-X), perfil de juízes e chat jurídico com RAG — tudo em conformidade com a Resolução CNJ 615/2025 e LGPD.

## Funcionalidades

- **Chat Jurídico** — Chat com IA especializada em direito brasileiro, com streaming e citação de fontes via RAG
- **Raio-X do Processo** — Análise completa: brechas, estratégias, perfil do juiz, predição estatística e jurisprudência similar
- **Busca Semântica Híbrida** — BM25 (Elasticsearch) + Dense (Qdrant) com RRF fusion e reranking (Jina-ColBERT-v2)
- **Geração de Petições** — Petições iniciais, recursos e contestações com label de IA obrigatório
- **Perfil de Juízes** — Jurimetria via agregações ES: favorabilidade, tempo médio, leis mais citadas
- **Feed de Atualizações** — Legislação, jurisprudência e normativos recentes com alertas personalizados
- **Multi-tenant** — Isolamento por tenant_id com RLS no PostgreSQL
- **Dark Mode** — Interface com suporte completo a modo escuro

## Tech Stack

| Camada        | Tecnologia                                                       |
|---------------|------------------------------------------------------------------|
| Frontend      | Next.js 15, React 19, Tailwind CSS, Lucide Icons, Zustand        |
| Backend       | FastAPI, Python 3.11+, Pydantic v2                               |
| Banco de Dados| PostgreSQL 16 (pgvector), Supabase (PostgREST)                   |
| Busca Vetorial| Elasticsearch 8.16 (BM25), Qdrant (Dense)                        |
| Cache         | Redis 7                                                          |
| LLM           | LiteLLM (multi-provider): DeepSeek V3.2, Claude Sonnet 4, GPT-5.2 Pro, Qwen 3.5, Kimi K2.5 |
| Orquestração  | LangGraph (multi-agent: Supervisor, Research, Drafting, Analysis)  |
| OCR           | PaddleOCR                                                        |
| Observability | Sentry, Langfuse                                                 |

## Quick Start

### Pré-requisitos

- Docker + Docker Compose
- Node.js 18+
- Python 3.11+

### 1. Clone e configure

```bash
git clone https://github.com/seu-org/jurisai.git
cd jurisai
cp .env.example .env
# Edite .env com suas API keys
```

### 2. Suba os serviços

```bash
docker compose up -d
```

Isso inicia: PostgreSQL, Redis, Elasticsearch, Qdrant, Langfuse, API (porta 8000) e Web (porta 3000).

### 3. Inicialize o banco

```bash
docker compose exec postgres psql -U jurisai -d jurisai -f /docker-entrypoint-initdb.d/01-init.sql
```

### 4. Acesse

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Langfuse**: http://localhost:3001

## Estrutura do Projeto

```
openlaw/
├── apps/
│   ├── api/                    # Backend FastAPI
│   │   ├── app/
│   │   │   ├── api/v1/         # Endpoints REST
│   │   │   ├── core/           # Auth, middleware, validators, rate limiting
│   │   │   ├── services/       # Lógica de negócio
│   │   │   │   ├── llm/        # Router multi-tier, prompts, providers
│   │   │   │   ├── rag/        # Retriever híbrido, embeddings, reranker
│   │   │   │   ├── jurimetrics/# Judge profile, predictor
│   │   │   │   ├── ingestion/  # DataJud, STJ, Diários, JUIT
│   │   │   │   └── agents/     # LangGraph multi-agent
│   │   │   └── config.py       # Settings (env vars)
│   │   └── tests/              # Pytest
│   └── web/                    # Frontend Next.js
│       ├── app/
│       │   ├── (dashboard)/    # Layout com sidebar
│       │   │   ├── cases/      # Gestão de processos + Raio-X
│       │   │   ├── chat/       # Chat jurídico
│       │   │   ├── documents/  # Upload e OCR
│       │   │   ├── petitions/  # Editor de petições
│       │   │   └── ...
│       │   └── login/          # Autenticação
│       ├── components/         # Componentes reutilizáveis
│       └── lib/                # Hooks, API client, auth
├── training/                   # Pipeline de treinamento GAIA
│   └── data/                   # Geração de CoT traces
├── scripts/
│   ├── init_db.sql             # Schema completo (dev)
│   └── init_db_secure.sql      # RLS restritivo (produção)
├── docker-compose.yml
├── BLUEPRINT.md                # Arquitetura detalhada
└── docs/
    └── ARCHITECTURE.md         # Visão geral
```

## Endpoints Principais

| Método | Endpoint                          | Descrição                              |
|--------|-----------------------------------|----------------------------------------|
| POST   | `/api/v1/chat/completions`        | Chat com streaming SSE                 |
| GET    | `/api/v1/cases`                   | Listar processos (paginado)            |
| POST   | `/api/v1/cases/{id}/analyze`      | Raio-X do processo                     |
| GET    | `/api/v1/cases/{id}/analyze/stream` | Raio-X com progresso SSE             |
| POST   | `/api/v1/documents/upload`        | Upload + OCR + classificação           |
| POST   | `/api/v1/search`                  | Busca semântica híbrida                |
| POST   | `/api/v1/petitions`               | Gerar petição com IA                   |
| GET    | `/api/v1/jurimetrics/judge/{name}`| Perfil do juiz                         |
| GET    | `/api/v1/updates/feed`            | Feed de atualizações                   |

## Variáveis de Ambiente

Veja `.env.example` para a lista completa. As principais:

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
SECRET_KEY=
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=
REDIS_URL=redis://localhost:6379
ES_URL=http://localhost:9200
QDRANT_URL=http://localhost:6333
```

## Licença

Proprietário. Todos os direitos reservados.

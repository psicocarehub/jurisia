# JURIS.AI — Blueprint Técnico Completo

> Documento de referência para implementação no Cursor. Contém toda a arquitetura, stack tecnológico, modelos de dados, pipelines, código de referência e decisões técnicas do projeto.

**Projeto**: Plataforma SaaS B2B de IA jurídica para escritórios de advocacia brasileiros
**Modelo**: Múltiplas áreas do direito desde o início
**Infra**: Híbrido (GAIA local + APIs para OCR/embeddings)
**Autor**: Marcio Estevam — MeridianAI Labs
**Data**: Fevereiro 2026

---

## ÍNDICE

1. [Visão Geral e Arquitetura](#1-visão-geral-e-arquitetura)
2. [Tech Stack Completo](#2-tech-stack-completo)
3. [Estrutura de Diretórios](#3-estrutura-de-diretórios)
4. [Backend — FastAPI](#4-backend--fastapi)
5. [Banco de Dados e Schemas](#5-banco-de-dados-e-schemas)
6. [Sistema de RAG Jurídico (Busca Semântica)](#6-sistema-de-rag-jurídico-busca-semântica)
7. [Pipeline de Ingestão de Dados (DataJud, STJ, STF)](#7-pipeline-de-ingestão-de-dados)
8. [OCR de Alta Performance](#8-ocr-de-alta-performance)
9. [Classificação de Documentos e NER](#9-classificação-de-documentos-e-ner)
10. [Sistema de Memória Persistente](#10-sistema-de-memória-persistente)
11. [Jurimetria e Perfil de Juízes](#11-jurimetria-e-perfil-de-juízes)
12. [Geração de Petições](#12-geração-de-petições)
13. [Orquestração Multi-Agente (LangGraph)](#13-orquestração-multi-agente-langgraph)
14. [Reasoning Model — GAIA Fine-Tuned](#14-reasoning-model--gaia-fine-tuned)
15. [Pipeline de Treinamento GRPO](#15-pipeline-de-treinamento-grpo)
16. [Infraestrutura GPU (Modal/RunPod)](#16-infraestrutura-gpu-modalrunpod)
17. [Roteamento Multi-Modelo](#17-roteamento-multi-modelo)
18. [Frontend — Next.js](#18-frontend--nextjs)
19. [Autenticação e Multi-Tenancy](#19-autenticação-e-multi-tenancy)
20. [LGPD, OAB e Compliance](#20-lgpd-oab-e-compliance)
21. [Observabilidade (Langfuse)](#21-observabilidade-langfuse)
22. [Deploy e CI/CD](#22-deploy-e-cicd)
23. [Roadmap de Implementação](#23-roadmap-de-implementação)

---

## 1. VISÃO GERAL E ARQUITETURA

### O que estamos construindo

Um copiloto de IA para advogados brasileiros que entrega:

1. **Busca semântica contextual** com dados sempre atualizados (DataJud, STF, STJ, TJs)
2. **Super memória** — contexto persistente por cliente, caso e sessão
3. **GAIA fine-tuned** como reasoning model jurídico brasileiro (4B params, self-hosted)
4. **OCR de alta qualidade** para documentos judiciais escaneados
5. **Classificação e tagueamento** automático de documentos
6. **Compreensão de juízes/varas/tribunais** — jurimetria integrada
7. **Assistência na montagem de petições** com verificação de citações

### Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│  Next.js 14+ (App Router) + Vercel AI SDK + Tiptap Editor       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ SSE/WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                      API GATEWAY                                 │
│  FastAPI (Python 3.12+) + Pydantic v2 + SSE Streaming           │
│  LiteLLM (router multi-modelo)                                   │
└──────┬─────────┬──────────┬──────────┬─────────┬───────────────┘
       │         │          │          │         │
┌──────▼───┐ ┌──▼────┐ ┌───▼───┐ ┌───▼────┐ ┌──▼──────────────┐
│ LangGraph│ │  RAG  │ │  OCR  │ │Memória │ │  Jurimetria     │
│ Multi-   │ │Híbrido│ │Paddle │ │4-Tier  │ │  Analytics      │
│ Agent    │ │BM25+  │ │OCR +  │ │Letta + │ │  Judge Profile  │
│ Orchestr.│ │Dense  │ │Surya  │ │Graphiti│ │  + Predictions  │
└──────┬───┘ └──┬────┘ └───┬───┘ └───┬────┘ └──┬──────────────┘
       │        │          │         │          │
┌──────▼────────▼──────────▼─────────▼──────────▼─────────────────┐
│                      DATA LAYER                                   │
│  PostgreSQL 16 (pgvector + RLS) │ Elasticsearch 8.16+             │
│  Redis 7+ (cache/sessions)      │ Qdrant (vector DB dedicado)     │
│  Neo4j/Apache AGE (knowledge)   │ S3/MinIO (docs originais)       │
└──────┬──────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                      LLM LAYER                                    │
│  GAIA 4B Fine-Tuned (vLLM no Modal) ← Tier 1 (70-80% queries)  │
│  DeepSeek V3.2 / Qwen 3.5 API      ← Tier 2 (moderado)        │
│  Claude Sonnet 4 / Gemini 3 Pro     ← Tier 3 (complexo)        │
│  Kimi K2.5 / MiniMax M2.5          ← Tier 2 alt (agentes)     │
│  Sabiá-3 API (Maritaca)            ← Tier 2 alt (PT-BR nativo)│
│  RouteLLM (classifier-based router)                              │
└──────┬──────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                  INGESTION PIPELINE                                │
│  Apache Airflow → DataJud API + STJ + STF + TJs + DOU            │
│  S3 Raw → dbt Transform → Dedup (MinHash-LSH) → Index           │
└─────────────────────────────────────────────────────────────────┘
```

### Padrão Arquitetural

**Modular monolith → selective microservices**. Começar monolítico com separação clara de módulos. Extrair para microserviços apenas quando escala exigir (OCR processing é primeiro candidato).

---

## 2. TECH STACK COMPLETO

### Backend
| Componente | Tecnologia | Versão | Justificativa |
|-----------|-----------|--------|---------------|
| Framework | FastAPI | 0.110+ | Async nativo, Pydantic v2, SSE streaming |
| Runtime | Python | 3.12+ | Performance, typing melhorado |
| Task Queue | Celery + Redis | 5.4+ | Jobs assíncronos (OCR, ingestão) |
| Scheduler | Apache Airflow | 2.8+ | DAGs de ingestão de dados |
| LLM Orchestration | LangGraph | 1.0+ | Multi-agent, human-in-the-loop, checkpointing |
| LLM Router | LiteLLM | latest | 100+ modelos, OpenAI-compatible, fallbacks |
| LLM Observability | Langfuse | latest | Open-source, self-hostable, tracing completo |

### Databases
| Componente | Tecnologia | Versão | Justificativa |
|-----------|-----------|--------|---------------|
| Primary DB | PostgreSQL | 16+ | pgvector, RLS multi-tenancy, JSONB |
| Vector DB | Qdrant | latest | Rust, HNSW pre-filtering, self-hostable |
| Search Engine | Elasticsearch | 8.16+ | Native RRF, mesma tech do CNJ DataJud |
| Knowledge Graph | Apache AGE | latest | Extensão PostgreSQL, migrar p/ Neo4j depois |
| Cache/Sessions | Redis | 7+ | Pub/sub, caching, session store |
| Object Storage | MinIO / S3 | latest | PDFs originais, backups |

### Frontend
| Componente | Tecnologia | Versão | Justificativa |
|-----------|-----------|--------|---------------|
| Framework | Next.js | 14+ | App Router, Server Components |
| AI Chat | Vercel AI SDK | latest | useChat(), streaming nativo |
| Editor | Tiptap | latest | ProseMirror-based, CRDT, extensível |
| Styling | Tailwind CSS | 3+ | Utility-first |
| State | Zustand | latest | Leve, simples |
| Auth | NextAuth.js / Clerk | latest | OAuth, magic link |

### AI/ML
| Componente | Tecnologia | Versão | Justificativa |
|-----------|-----------|--------|---------------|
| Reasoning Model | GAIA 4B fine-tuned | custom | Modelo brasileiro, self-hosted (base Gemma 3 4B) |
| Teacher Models | DeepSeek V3.2 / Qwen 3.5 | latest | Distilação: melhor custo-benefício 2026 |
| LLM Serving | vLLM | latest | Continuous batching, PagedAttention |
| Training | TRL + Unsloth | latest | GRPO, QLoRA, 80% menos VRAM |
| Embeddings | voyage-law-2 / bge-m3 | latest | Legal benchmark SOTA / self-hosted multilingual |
| Reranking | Jina-ColBERT-v2 | latest | 89 idiomas, 8K tokens |
| OCR | PaddleOCR-VL 1.5 | latest | 94.5% precision, selos/carimbos |
| NER | pierreguillou/ner-bert-large-cased-pt-lenerbr | latest | F1 0.908 em LeNER-Br |
| Classification | RoBERTaLexPT | latest | Pré-treinado em corpus jurídico PT |
| Memory | Graphiti + Mem0 | latest | Knowledge graph temporal + fact extraction |

### Infra
| Componente | Tecnologia | Justificativa |
|-----------|-----------|---------------|
| GPU Training | RunPod / Modal | A100 80GB $1.19-$2.50/hr |
| GPU Inference 24/7 | Modal | Serverless, scale-to-zero, L4 $0.80/hr |
| App Hosting | AWS sa-east-1 / Railway | Data residency Brasil (LGPD) |
| CI/CD | GitHub Actions | Standard |
| Containers | Docker + Docker Compose | Dev local |
| Monitoring | Grafana + Prometheus | Métricas de infra |

---

## 3. ESTRUTURA DE DIRETÓRIOS

```
juris-ai/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── deploy-api.yml
│       └── deploy-web.yml
├── apps/
│   ├── api/                          # FastAPI Backend
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # FastAPI app entry
│   │   │   ├── config.py             # Settings (pydantic-settings)
│   │   │   ├── dependencies.py       # Dependency injection
│   │   │   │
│   │   │   ├── api/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── v1/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── router.py     # Main v1 router
│   │   │   │   │   ├── chat.py       # Chat/conversation endpoints
│   │   │   │   │   ├── documents.py  # Document upload/management
│   │   │   │   │   ├── search.py     # Semantic search endpoints
│   │   │   │   │   ├── cases.py      # Case management
│   │   │   │   │   ├── petitions.py  # Petition generation
│   │   │   │   │   ├── jurimetrics.py # Judge profiles, analytics
│   │   │   │   │   ├── memory.py     # Memory management
│   │   │   │   │   └── admin.py      # Admin/tenant management
│   │   │   │
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py           # JWT + API key auth
│   │   │   │   ├── middleware.py     # Tenant isolation, logging
│   │   │   │   ├── exceptions.py    # Custom exceptions
│   │   │   │   └── security.py      # Encryption, LGPD utils
│   │   │   │
│   │   │   ├── db/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── session.py        # SQLAlchemy async session
│   │   │   │   ├── models/           # SQLAlchemy ORM models
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── base.py       # Base model with tenant_id
│   │   │   │   │   ├── tenant.py
│   │   │   │   │   ├── user.py
│   │   │   │   │   ├── document.py
│   │   │   │   │   ├── case.py
│   │   │   │   │   ├── conversation.py
│   │   │   │   │   ├── petition.py
│   │   │   │   │   └── audit_log.py
│   │   │   │   ├── migrations/       # Alembic
│   │   │   │   │   ├── env.py
│   │   │   │   │   └── versions/
│   │   │   │   └── seed.py
│   │   │   │
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── llm/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── router.py     # RouteLLM / LiteLLM routing
│   │   │   │   │   ├── gaia.py       # GAIA inference client
│   │   │   │   │   ├── providers.py  # DeepSeek V3.2, Claude, Qwen 3.5, Kimi K2.5
│   │   │   │   │   └── prompts.py    # System prompts por tarefa
│   │   │   │   │
│   │   │   │   ├── rag/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── retriever.py  # Hybrid BM25 + Dense + RRF
│   │   │   │   │   ├── reranker.py   # Jina-ColBERT-v2
│   │   │   │   │   ├── embeddings.py # voyage-law-2 / bge-m3
│   │   │   │   │   ├── chunker.py    # Hierarchical chunking
│   │   │   │   │   └── indexer.py    # Elasticsearch + Qdrant
│   │   │   │   │
│   │   │   │   ├── ocr/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── paddle.py     # PaddleOCR-VL 1.5
│   │   │   │   │   ├── surya.py      # Surya + Marker pipeline
│   │   │   │   │   ├── postprocess.py # Correção + NER pós-OCR
│   │   │   │   │   └── azure.py      # Azure Doc Intelligence (fallback)
│   │   │   │   │
│   │   │   │   ├── nlp/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── ner.py        # NER jurídico (BERTimbau + RegEx)
│   │   │   │   │   ├── classifier.py # Classificação de documentos
│   │   │   │   │   └── entities.py   # Extração de entidades estruturadas
│   │   │   │   │
│   │   │   │   ├── memory/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── manager.py    # Orchestrador de memória 4-tier
│   │   │   │   │   ├── graphiti.py   # Knowledge graph (Neo4j/AGE)
│   │   │   │   │   ├── mem0.py       # Fact extraction/updates
│   │   │   │   │   └── checkpointer.py # LangGraph checkpointing
│   │   │   │   │
│   │   │   │   ├── jurimetrics/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── judge_profile.py
│   │   │   │   │   ├── court_stats.py
│   │   │   │   │   ├── predictor.py  # XGBoost prediction
│   │   │   │   │   └── data_lawyer.py # API integration
│   │   │   │   │
│   │   │   │   ├── petition/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── generator.py  # Multi-step generation
│   │   │   │   │   ├── templates.py  # Template management
│   │   │   │   │   ├── citation_verifier.py  # Verificação de citações
│   │   │   │   │   └── formatter.py  # Formatação ABNT/OAB
│   │   │   │   │
│   │   │   │   └── ingestion/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── datajud.py    # CNJ DataJud API
│   │   │   │       ├── stj.py        # STJ Open Data
│   │   │   │       ├── stf.py        # STF API
│   │   │   │       ├── lexml.py      # LexML
│   │   │   │       ├── esaj.py       # e-SAJ (TJSP etc)
│   │   │   │       ├── querido_diario.py # Diários oficiais
│   │   │   │       └── deduplicator.py   # MinHash-LSH
│   │   │   │
│   │   │   └── agents/
│   │   │       ├── __init__.py
│   │   │       ├── supervisor.py     # Supervisor/router agent
│   │   │       ├── research.py       # Legal research agent
│   │   │       ├── drafting.py       # Petition drafting agent
│   │   │       ├── analysis.py       # Case analysis agent
│   │   │       ├── memory_agent.py   # Memory management agent
│   │   │       └── graph.py          # LangGraph workflow definition
│   │   │
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── alembic.ini
│   │
│   └── web/                          # Next.js Frontend
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   ├── (auth)/
│       │   │   ├── login/page.tsx
│       │   │   └── register/page.tsx
│       │   ├── (dashboard)/
│       │   │   ├── layout.tsx
│       │   │   ├── chat/
│       │   │   │   ├── page.tsx       # Chat principal
│       │   │   │   └── [id]/page.tsx  # Conversa específica
│       │   │   ├── cases/
│       │   │   │   ├── page.tsx       # Lista de casos
│       │   │   │   └── [id]/page.tsx  # Caso específico
│       │   │   ├── documents/
│       │   │   │   └── page.tsx
│       │   │   ├── petitions/
│       │   │   │   ├── page.tsx
│       │   │   │   └── [id]/
│       │   │   │       └── editor/page.tsx  # Tiptap editor
│       │   │   ├── jurimetrics/
│       │   │   │   └── page.tsx
│       │   │   └── settings/
│       │   │       └── page.tsx
│       │   └── api/                   # API routes (BFF)
│       │       └── chat/route.ts
│       ├── components/
│       │   ├── chat/
│       │   │   ├── ChatInterface.tsx
│       │   │   ├── MessageBubble.tsx
│       │   │   ├── SourceCitation.tsx
│       │   │   └── ThinkingIndicator.tsx
│       │   ├── editor/
│       │   │   ├── TiptapEditor.tsx
│       │   │   ├── LegalToolbar.tsx
│       │   │   └── CitationPlugin.tsx
│       │   ├── search/
│       │   │   ├── SearchBar.tsx
│       │   │   └── ResultCard.tsx
│       │   └── ui/                    # shadcn/ui components
│       ├── lib/
│       │   ├── api.ts                 # API client
│       │   ├── auth.ts
│       │   └── utils.ts
│       ├── package.json
│       ├── next.config.js
│       ├── tailwind.config.ts
│       └── tsconfig.json
│
├── packages/
│   └── shared/                        # Tipos compartilhados
│       ├── types.ts
│       └── constants.ts
│
├── infra/
│   ├── docker-compose.yml             # Dev local
│   ├── docker-compose.prod.yml
│   ├── nginx/
│   ├── terraform/                     # AWS sa-east-1
│   └── k8s/                          # Kubernetes configs
│
├── training/                          # Pipeline de treinamento do GAIA
│   ├── data/
│   │   ├── collect_oab.py            # Scraping questões OAB
│   │   ├── collect_stf.py            # Decisões STF
│   │   ├── collect_stj.py            # Decisões STJ
│   │   ├── generate_cot.py           # Geração de CoT via teacher model
│   │   ├── filter_quality.py         # Rejection sampling + filtragem
│   │   └── prepare_dataset.py        # Formatação final
│   ├── sft/
│   │   ├── train_sft.py             # SFT com LoRA no GAIA
│   │   └── config_sft.yaml
│   ├── grpo/
│   │   ├── train_grpo.py            # GRPO com reward functions
│   │   ├── rewards.py               # Reward functions jurídicas
│   │   └── config_grpo.yaml
│   ├── eval/
│   │   ├── eval_oab.py              # Benchmark OAB
│   │   ├── eval_citation.py         # Verificação de citações
│   │   └── eval_reasoning.py        # Qualidade de reasoning
│   ├── serve/
│   │   ├── modal_serve.py           # Deploy no Modal
│   │   └── vllm_config.yaml
│   └── merge/
│       ├── merge_config.yaml         # mergekit config
│       └── merge_models.py
│
├── airflow/
│   ├── dags/
│   │   ├── ingest_datajud.py
│   │   ├── ingest_stj.py
│   │   ├── ingest_stf.py
│   │   ├── ingest_dou.py
│   │   └── reindex.py
│   └── plugins/
│
├── scripts/
│   ├── setup_db.sh
│   ├── seed_data.sh
│   └── run_migrations.sh
│
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── LGPD_COMPLIANCE.md
│
├── .env.example
├── .gitignore
├── README.md
├── Makefile
└── docker-compose.yml
```

---

## 4. BACKEND — FastAPI

### Entry Point (`apps/api/app/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.v1.router import api_router
from app.core.middleware import TenantMiddleware, AuditMiddleware
from app.db.session import engine, init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(
    title="Juris.AI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TenantMiddleware)
app.add_middleware(AuditMiddleware)

app.include_router(api_router, prefix="/api/v1")
```

### Configuration (`apps/api/app/config.py`)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Juris.AI"
    DEBUG: bool = False
    SECRET_KEY: str
    API_KEY_HEADER: str = "X-API-Key"

    # Database
    DATABASE_URL: str  # postgresql+asyncpg://user:pass@host:5432/jurisai
    REDIS_URL: str     # redis://localhost:6379/0

    # Elasticsearch
    ELASTICSEARCH_URL: str  # http://localhost:9200
    ES_INDEX_PREFIX: str = "jurisai"

    # Qdrant
    QDRANT_URL: str  # http://localhost:6333
    QDRANT_COLLECTION: str = "legal_documents"

    # LLM Providers
    GAIA_BASE_URL: str      # Modal vLLM endpoint
    GAIA_MODEL_NAME: str = "gaia-legal-reasoning-v1"
    DEEPSEEK_API_KEY: str = ""       # DeepSeek V3.2 ($0.28/$1.10 por M tokens)
    ANTHROPIC_API_KEY: str = ""      # Claude Sonnet 4 (complexo)
    OPENAI_API_KEY: str = ""         # GPT-5.2 (fallback)
    QWEN_API_KEY: str = ""           # Qwen 3.5 ($0.50/$2.00) — Apache 2.0, ótimo custo-benefício
    KIMI_API_KEY: str = ""           # Kimi K2.5 ($0.45/$2.20) — líder em agentic tasks
    MINIMAX_API_KEY: str = ""        # MiniMax M2.5 ($0.30/$1.10) — agentes e produtividade
    MARITACA_API_KEY: str = ""       # Sabiá-3 (PT-BR nativo)
    GOOGLE_API_KEY: str = ""         # Gemini 3 Pro ($1.25/$5.00) — melhor preço/performance frontier

    # Embeddings
    VOYAGE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "voyage-law-2"
    EMBEDDING_DIM: int = 1024

    # OCR
    PADDLE_OCR_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""

    # Storage
    S3_BUCKET: str = "jurisai-documents"
    S3_ENDPOINT: str = ""  # MinIO endpoint
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3001"

    # Routing thresholds
    ROUTER_THRESHOLD_TIER1: float = 0.6  # Abaixo -> GAIA
    ROUTER_THRESHOLD_TIER2: float = 0.85 # Abaixo -> DeepSeek V3.2/Qwen 3.5, Acima -> Claude/Gemini

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
```

### Chat Endpoint com SSE Streaming (`apps/api/app/api/v1/chat.py`)

```python
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import asyncio

from app.dependencies import get_current_user, get_tenant_id
from app.services.llm.router import LLMRouter
from app.services.rag.retriever import HybridRetriever
from app.services.memory.manager import MemoryManager
from app.agents.graph import legal_agent_graph

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = True
    use_rag: bool = True
    use_memory: bool = True

class ChatResponse(BaseModel):
    message: ChatMessage
    sources: List[dict] = []
    thinking: Optional[str] = None
    model_used: str = ""

@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    user=Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    if request.stream:
        return StreamingResponse(
            stream_chat(request, user, tenant_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        result = await run_chat(request, user, tenant_id)
        return result

async def stream_chat(request: ChatRequest, user, tenant_id: str):
    """SSE streaming via LangGraph agent."""
    config = {
        "configurable": {
            "thread_id": request.session_id or f"{tenant_id}_{user.id}",
            "tenant_id": tenant_id,
            "user_id": user.id,
            "case_id": request.case_id,
        }
    }

    input_state = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "use_rag": request.use_rag,
        "use_memory": request.use_memory,
        "tenant_id": tenant_id,
    }

    async for event in legal_agent_graph.astream_events(
        input_state, config=config, version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                data = json.dumps({
                    "type": "token",
                    "content": chunk.content,
                })
                yield f"data: {data}\n\n"

        elif kind == "on_tool_start":
            data = json.dumps({
                "type": "tool_start",
                "tool": event["name"],
            })
            yield f"data: {data}\n\n"

        elif kind == "on_tool_end":
            data = json.dumps({
                "type": "tool_end",
                "tool": event["name"],
                "output": str(event["data"].get("output", ""))[:500],
            })
            yield f"data: {data}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
```

---

## 5. BANCO DE DADOS E SCHEMAS

### PostgreSQL Schema Principal

```sql
-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Trigram para busca fuzzy

-- ============================================
-- MULTI-TENANCY COM ROW LEVEL SECURITY
-- ============================================

-- Tenants (escritórios de advocacia)
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'starter',  -- starter, professional, enterprise
    settings JSONB DEFAULT '{}',
    lgpd_consent_template TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Usuários
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'lawyer',  -- admin, lawyer, paralegal, viewer
    oab_number VARCHAR(20),             -- Registro OAB
    hashed_password VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    ai_consent_given BOOLEAN DEFAULT false,  -- OAB Item 4.4.3
    ai_consent_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, email)
);

-- ============================================
-- GESTÃO DE CASOS
-- ============================================

CREATE TABLE cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    cnj_number VARCHAR(25),  -- Formato: NNNNNNN-DD.AAAA.J.TR.OOOO
    title VARCHAR(500) NOT NULL,
    description TEXT,
    area VARCHAR(100),       -- civil, trabalhista, criminal, tributário, etc.
    status VARCHAR(50) DEFAULT 'active',
    client_name VARCHAR(255),
    client_document VARCHAR(20),  -- CPF/CNPJ (criptografado)
    opposing_party VARCHAR(255),
    court VARCHAR(255),           -- Ex: "2ª Vara Cível de São Paulo"
    judge_name VARCHAR(255),
    judge_id UUID,                -- FK para judge_profiles
    estimated_value DECIMAL(15,2),
    metadata JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cases_tenant ON cases(tenant_id);
CREATE INDEX idx_cases_cnj ON cases(cnj_number);
CREATE INDEX idx_cases_area ON cases(tenant_id, area);

-- ============================================
-- DOCUMENTOS
-- ============================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500) NOT NULL,
    doc_type VARCHAR(100),       -- peticao_inicial, contestacao, sentenca, acordao, etc.
    source VARCHAR(100),         -- upload, datajud, stj, stf, esaj, scraping
    file_path VARCHAR(500),      -- S3 path
    file_size INTEGER,
    mime_type VARCHAR(100),
    ocr_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    ocr_text TEXT,               -- Texto extraído pelo OCR
    ocr_confidence FLOAT,
    classification_label VARCHAR(100),  -- Label do classificador
    classification_confidence FLOAT,
    ner_entities JSONB DEFAULT '[]',    -- Entidades extraídas
    metadata JSONB DEFAULT '{}',
    uploaded_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_docs_tenant ON documents(tenant_id);
CREATE INDEX idx_docs_case ON documents(case_id);
CREATE INDEX idx_docs_type ON documents(tenant_id, doc_type);

-- Chunks de documentos para RAG
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50),  -- ementa, fundamentacao, dispositivo, artigo
    token_count INTEGER,
    embedding vector(1024),    -- voyage-law-2 dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_doc ON document_chunks(document_id);
CREATE INDEX idx_chunks_tenant ON document_chunks(tenant_id);
CREATE INDEX idx_chunks_embedding ON document_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================
-- CONVERSAS E CHAT
-- ============================================

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID NOT NULL REFERENCES users(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500),
    model_used VARCHAR(100),
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    role VARCHAR(20) NOT NULL,  -- user, assistant, system, tool
    content TEXT NOT NULL,
    thinking TEXT,              -- Reasoning trace (hidden from user by default)
    sources JSONB DEFAULT '[]', -- Fontes citadas
    model_used VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    feedback VARCHAR(20),      -- thumbs_up, thumbs_down
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_messages_conv ON messages(conversation_id);
CREATE INDEX idx_messages_tenant ON messages(tenant_id);

-- ============================================
-- PETIÇÕES
-- ============================================

CREATE TABLE petitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500) NOT NULL,
    petition_type VARCHAR(100),  -- inicial, contestacao, recurso, agravo, etc.
    content TEXT,                -- Conteúdo completo
    tiptap_json JSONB,          -- Tiptap document state
    status VARCHAR(50) DEFAULT 'draft',  -- draft, review, approved, filed
    version INTEGER DEFAULT 1,
    citations JSONB DEFAULT '[]',  -- Citações verificadas
    citations_verified BOOLEAN DEFAULT false,
    ai_generated BOOLEAN DEFAULT true,
    ai_label TEXT DEFAULT 'Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025',
    created_by UUID REFERENCES users(id),
    reviewed_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- JURIMETRIA
-- ============================================

CREATE TABLE judge_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    court VARCHAR(255),
    jurisdiction VARCHAR(255),
    total_decisions INTEGER DEFAULT 0,
    avg_decision_time_days FLOAT,
    favorability_rates JSONB DEFAULT '{}',  -- {area: {plaintiff: %, defendant: %}}
    common_citations JSONB DEFAULT '[]',     -- Legislação mais citada
    decision_patterns JSONB DEFAULT '{}',
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_judges_name ON judge_profiles USING gin(name gin_trgm_ops);
CREATE INDEX idx_judges_court ON judge_profiles(court);

-- ============================================
-- AUDIT LOG (LGPD COMPLIANCE)
-- ============================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,  -- create, read, update, delete, ai_query, export
    resource_type VARCHAR(100),    -- document, case, petition, conversation
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_tenant ON audit_logs(tenant_id, created_at);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE petitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Política: cada tenant só acessa seus dados
CREATE POLICY tenant_isolation_users ON users
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_cases ON cases
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_chunks ON document_chunks
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_conversations ON conversations
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_messages ON messages
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_petitions ON petitions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
CREATE POLICY tenant_isolation_audit ON audit_logs
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Função para setar tenant no início de cada request
CREATE OR REPLACE FUNCTION set_tenant(tenant_uuid UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant_id', tenant_uuid::TEXT, true);
END;
$$ LANGUAGE plpgsql;
```

### Middleware de Tenant (`apps/api/app/core/middleware.py`)

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.db.session import AsyncSessionLocal

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            request.state.tenant_id = tenant_id
            # Set RLS context for this request
            async with AsyncSessionLocal() as session:
                await session.execute(
                    f"SELECT set_tenant('{tenant_id}'::UUID)"
                )
        response = await call_next(request)
        return response
```

---

## 6. SISTEMA DE RAG JURÍDICO (BUSCA SEMÂNTICA)

### Arquitetura: Hybrid Retrieval com RRF

```
Query do advogado
        │
        ▼
┌───────────────────┐
│  Query Expansion  │ ← LLM expande termos jurídicos
│  + Reformulation  │   Ex: "dano moral" → "dano moral extrapatrimonial indenização"
└───────┬───────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────────┐
│ BM25 │  │  Dense   │
│ ES   │  │  Qdrant  │
│      │  │ voyage-  │
│      │  │ law-2    │
└──┬───┘  └────┬─────┘
   │           │
   ▼           ▼
┌─────────────────────┐
│  Reciprocal Rank    │
│  Fusion (RRF)       │
│  k=60 (ES nativo)   │
└───────┬─────────────┘
        │ Top 20-30
        ▼
┌─────────────────────┐
│  Reranker           │
│  Jina-ColBERT-v2    │
│  ou bge-reranker-   │
│  v2-m3              │
└───────┬─────────────┘
        │ Top 5-10
        ▼
┌─────────────────────┐
│  Context Assembly   │
│  + Metadata         │
│  (tribunal, data,   │
│   juiz, área)       │
└───────┬─────────────┘
        │
        ▼
   LLM Generation
```

### Retriever Híbrido (`apps/api/app/services/rag/retriever.py`)

```python
from typing import List, Optional
from dataclasses import dataclass
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.reranker import RerankerService

@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float
    document_id: str
    document_title: str
    doc_type: str
    court: str
    date: str
    metadata: dict

class HybridRetriever:
    def __init__(self):
        self.es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        self.qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.embedder = EmbeddingService()
        self.reranker = RerankerService()

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        use_reranker: bool = True,
    ) -> List[RetrievedChunk]:
        # 1. Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # 2. Hybrid search with RRF in Elasticsearch
        es_results = await self._es_hybrid_search(
            query=query,
            embedding=query_embedding,
            tenant_id=tenant_id,
            top_k=top_k * 3,  # Over-retrieve for reranking
            filters=filters,
        )

        # 3. Optionally also search Qdrant (for user-uploaded docs)
        qdrant_results = await self._qdrant_search(
            embedding=query_embedding,
            tenant_id=tenant_id,
            top_k=top_k * 2,
            filters=filters,
        )

        # 4. Merge and deduplicate
        all_results = self._merge_results(es_results, qdrant_results)

        # 5. Rerank
        if use_reranker and all_results:
            all_results = await self.reranker.rerank(
                query=query,
                chunks=all_results,
                top_k=top_k,
            )
        else:
            all_results = all_results[:top_k]

        return all_results

    async def _es_hybrid_search(
        self, query: str, embedding: List[float],
        tenant_id: str, top_k: int, filters: Optional[dict]
    ) -> List[RetrievedChunk]:
        """Elasticsearch 8.16+ native RRF with BM25 + kNN."""

        must_clauses = [
            {"term": {"tenant_id": tenant_id}}
        ]
        if filters:
            if filters.get("area"):
                must_clauses.append({"term": {"area": filters["area"]}})
            if filters.get("court"):
                must_clauses.append({"term": {"court": filters["court"]}})
            if filters.get("date_from"):
                must_clauses.append({
                    "range": {"date": {"gte": filters["date_from"]}}
                })

        body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "bool": {
                                        "must": must_clauses,
                                        "should": [
                                            {
                                                "multi_match": {
                                                    "query": query,
                                                    "fields": [
                                                        "content^1",
                                                        "ementa^3",
                                                        "title^2"
                                                    ],
                                                    "type": "best_fields"
                                                }
                                            }
                                        ]
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": embedding,
                                "k": top_k,
                                "num_candidates": top_k * 5,
                                "filter": {
                                    "bool": {"must": must_clauses}
                                }
                            }
                        }
                    ],
                    "rank_constant": 60,
                    "rank_window_size": top_k
                }
            },
            "size": top_k,
            "_source": [
                "content", "document_id", "document_title",
                "doc_type", "court", "date", "metadata"
            ]
        }

        result = await self.es.search(
            index=f"{settings.ES_INDEX_PREFIX}_chunks",
            body=body,
        )

        return [
            RetrievedChunk(
                id=hit["_id"],
                content=hit["_source"]["content"],
                score=hit["_score"],
                document_id=hit["_source"]["document_id"],
                document_title=hit["_source"].get("document_title", ""),
                doc_type=hit["_source"].get("doc_type", ""),
                court=hit["_source"].get("court", ""),
                date=hit["_source"].get("date", ""),
                metadata=hit["_source"].get("metadata", {}),
            )
            for hit in result["hits"]["hits"]
        ]

    async def _qdrant_search(
        self, embedding: List[float], tenant_id: str,
        top_k: int, filters: Optional[dict]
    ) -> List[RetrievedChunk]:
        """Qdrant search for user-uploaded documents."""
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
            ]
        )

        results = await self.qdrant.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                id=str(r.id),
                content=r.payload.get("content", ""),
                score=r.score,
                document_id=r.payload.get("document_id", ""),
                document_title=r.payload.get("document_title", ""),
                doc_type=r.payload.get("doc_type", ""),
                court=r.payload.get("court", ""),
                date=r.payload.get("date", ""),
                metadata=r.payload.get("metadata", {}),
            )
            for r in results
        ]

    def _merge_results(
        self, es_results: List[RetrievedChunk],
        qdrant_results: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Merge and deduplicate results from multiple sources."""
        seen = set()
        merged = []
        for chunk in es_results + qdrant_results:
            key = f"{chunk.document_id}_{chunk.content[:100]}"
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        return merged
```

### Embeddings Service (`apps/api/app/services/rag/embeddings.py`)

```python
import httpx
from typing import List
from app.config import settings

class EmbeddingService:
    """Voyage AI voyage-law-2 para embeddings jurídicos."""

    def __init__(self):
        self.model = settings.EMBEDDING_MODEL
        self.api_key = settings.VOYAGE_API_KEY
        self.base_url = "https://api.voyageai.com/v1"

    async def embed_query(self, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": [text],
                    "input_type": "query",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": texts,
                    "input_type": "document",
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [d["embedding"] for d in data]
```

### Hierarchical Chunking (`apps/api/app/services/rag/chunker.py`)

```python
import re
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    content_type: str  # ementa, fundamentacao, dispositivo, artigo, paragrafo
    metadata: dict
    token_count: int

class LegalChunker:
    """Chunking hierárquico para documentos jurídicos brasileiros."""

    # Padrões de seção em decisões judiciais
    SECTION_PATTERNS = {
        "ementa": r"(?i)(EMENTA|E\s*M\s*E\s*N\s*T\s*A)[:\s]",
        "relatorio": r"(?i)(RELATÓRIO|R\s*E\s*L\s*A\s*T\s*Ó\s*R\s*I\s*O)[:\s]",
        "fundamentacao": r"(?i)(FUNDAMENTAÇÃO|VOTO|DO MÉRITO|FUNDAMENTAÇÃO JURÍDICA)",
        "dispositivo": r"(?i)(DISPOSITIVO|DECISÃO|CONCLUSÃO|ISTO POSTO|ANTE O EXPOSTO)",
    }

    # Padrão CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO
    CNJ_PATTERN = r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_tokens: int = 64,  # ~12% overlap
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, text: str, doc_type: str = "generic") -> List[Chunk]:
        """Chunk a legal document using hierarchical strategy."""

        if doc_type in ("acordao", "sentenca", "decisao"):
            return self._chunk_judicial_decision(text)
        elif doc_type in ("lei", "decreto", "resolucao"):
            return self._chunk_legislation(text)
        else:
            return self._chunk_generic(text)

    def _chunk_judicial_decision(self, text: str) -> List[Chunk]:
        """Chunk judicial decisions by section then paragraphs."""
        chunks = []

        # Split into sections
        sections = self._split_sections(text)

        for section_type, section_text in sections:
            if not section_text.strip():
                continue

            # Ementa sempre em chunk único (geralmente curta)
            if section_type == "ementa":
                chunks.append(Chunk(
                    content=section_text.strip(),
                    content_type="ementa",
                    metadata={"section": "ementa"},
                    token_count=self._estimate_tokens(section_text),
                ))
                continue

            # Outras seções: chunk por parágrafos com overlap
            paragraphs = self._split_paragraphs(section_text)
            section_chunks = self._merge_paragraphs_into_chunks(
                paragraphs, section_type
            )
            chunks.extend(section_chunks)

        return chunks

    def _chunk_legislation(self, text: str) -> List[Chunk]:
        """Chunk legislation by articles."""
        chunks = []
        # Split by "Art." pattern
        articles = re.split(r"(?=Art\.\s*\d+)", text)
        for article in articles:
            if not article.strip():
                continue
            if self._estimate_tokens(article) <= self.max_chunk_tokens:
                chunks.append(Chunk(
                    content=article.strip(),
                    content_type="artigo",
                    metadata=self._extract_article_number(article),
                    token_count=self._estimate_tokens(article),
                ))
            else:
                # Article too long, split by paragraphs
                sub_chunks = self._chunk_generic(article)
                for sc in sub_chunks:
                    sc.content_type = "artigo"
                chunks.extend(sub_chunks)
        return chunks

    def _chunk_generic(self, text: str) -> List[Chunk]:
        """Fallback: recursive character splitting with overlap."""
        chunks = []
        paragraphs = self._split_paragraphs(text)
        chunks = self._merge_paragraphs_into_chunks(paragraphs, "generic")
        return chunks

    def _split_sections(self, text: str):
        """Split judicial decision into named sections."""
        positions = []
        for section_type, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text):
                positions.append((match.start(), section_type))

        if not positions:
            return [("generic", text)]

        positions.sort(key=lambda x: x[0])
        sections = []
        for i, (pos, stype) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections.append((stype, text[pos:end]))

        return sections

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_paragraphs_into_chunks(
        self, paragraphs: List[str], content_type: str
    ) -> List[Chunk]:
        """Merge small paragraphs into chunks respecting max_tokens."""
        chunks = []
        current = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > self.max_chunk_tokens and current:
                chunk_text = "\n\n".join(current)
                chunks.append(Chunk(
                    content=chunk_text,
                    content_type=content_type,
                    metadata={},
                    token_count=current_tokens,
                ))
                # Overlap: keep last paragraph
                if self.overlap_tokens > 0 and current:
                    last = current[-1]
                    current = [last]
                    current_tokens = self._estimate_tokens(last)
                else:
                    current = []
                    current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunks.append(Chunk(
                content="\n\n".join(current),
                content_type=content_type,
                metadata={},
                token_count=current_tokens,
            ))

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (Portuguese: ~1 token per 4 chars)."""
        return len(text) // 4

    @staticmethod
    def _extract_article_number(text: str) -> dict:
        match = re.search(r"Art\.\s*(\d+)", text)
        return {"article_number": match.group(1)} if match else {}
```

---

## 7. PIPELINE DE INGESTÃO DE DADOS

### DataJud — API do CNJ (`apps/api/app/services/ingestion/datajud.py`)

```python
"""
DataJud API - CNJ Resolution 331/2020
Endpoint: https://api-publica.datajud.cnj.jus.br/api_publica_{tribunal}/_search

Tribunais disponíveis:
- stf, stj, tst, tse, stm
- trf1 a trf6
- trt1 a trt24
- tjac, tjal, tjam, tjap, tjba, tjce, tjdft, tjes, tjgo, tjma,
  tjmg, tjms, tjmt, tjpa, tjpb, tjpe, tjpi, tjpr, tjrj, tjrn,
  tjro, tjrr, tjrs, tjsc, tjse, tjsp, tjto

Headers obrigatórios:
- Authorization: APIKey {chave}
- Content-Type: application/json

IMPORTANTE: DataJud retorna METADADOS e MOVIMENTAÇÕES, não texto integral das decisões.
Para texto integral, usar STJ Open Data ou scraping direto dos tribunais.
"""

import httpx
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class DataJudMovement(BaseModel):
    code: int
    name: str
    date: datetime

class DataJudProcess(BaseModel):
    cnj_number: str
    court: str
    class_name: str  # Ação Civil Pública, Mandado de Segurança, etc.
    subject: str
    filing_date: datetime
    judge: Optional[str]
    movements: List[DataJudMovement]
    parties: List[dict]
    metadata: dict

class DataJudClient:
    BASE_URL = "https://api-publica.datajud.cnj.jus.br"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"APIKey {api_key}",
            "Content-Type": "application/json",
        }

    async def search_processes(
        self,
        tribunal: str,       # Ex: "tjsp", "stj", "trf3"
        query: str = "*",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        class_code: Optional[int] = None,
        size: int = 100,
        from_offset: int = 0,
    ) -> List[DataJudProcess]:
        url = f"{self.BASE_URL}/api_publica_{tribunal}/_search"

        must_clauses = []
        if query != "*":
            must_clauses.append({
                "query_string": {"query": query}
            })
        if class_code:
            must_clauses.append({
                "term": {"classe.codigo": class_code}
            })
        if date_from or date_to:
            range_clause = {"range": {"dataAjuizamento": {}}}
            if date_from:
                range_clause["range"]["dataAjuizamento"]["gte"] = date_from
            if date_to:
                range_clause["range"]["dataAjuizamento"]["lte"] = date_to
            must_clauses.append(range_clause)

        body = {
            "size": size,
            "from": from_offset,
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "sort": [{"dataAjuizamento": {"order": "desc"}}]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=self.headers, json=body)
            response.raise_for_status()
            data = response.json()

        processes = []
        for hit in data.get("hits", {}).get("hits", []):
            src = hit["_source"]
            processes.append(DataJudProcess(
                cnj_number=src.get("numeroProcesso", ""),
                court=tribunal.upper(),
                class_name=src.get("classe", {}).get("nome", ""),
                subject=src.get("assuntos", [{}])[0].get("nome", "") if src.get("assuntos") else "",
                filing_date=src.get("dataAjuizamento", ""),
                judge=src.get("orgaoJulgador", {}).get("nomeJuiz", ""),
                movements=[
                    DataJudMovement(
                        code=m.get("codigo", 0),
                        name=m.get("nome", ""),
                        date=m.get("dataHora", ""),
                    )
                    for m in src.get("movimentos", [])
                ],
                parties=[
                    {"name": p.get("nome", ""), "type": p.get("tipo", "")}
                    for p in src.get("partes", [])
                    if not p.get("sigiloso", False)  # LGPD: excluir sigilosos
                ],
                metadata=src,
            ))

        return processes

    async def get_all_movements(
        self, tribunal: str, cnj_number: str
    ) -> List[DataJudMovement]:
        """Get all movements for a specific process."""
        processes = await self.search_processes(
            tribunal=tribunal,
            query=f'"{cnj_number}"',
            size=1,
        )
        if processes:
            return processes[0].movements
        return []
```

### STJ Open Data (`apps/api/app/services/ingestion/stj.py`)

```python
"""
STJ Dados Abertos - dadosabertos.web.stj.jus.br
19 datasets, updated daily, CC-BY license.
Includes FULL DECISION TEXT unlike DataJud.

Key datasets:
- decisoes_monocraticas: single-judge decisions with full text
- acordaos: panel decisions with full text (ementa + voto)
- processos: process metadata
"""

import httpx
import csv
import io
from typing import AsyncGenerator
from datetime import date

class STJOpenDataClient:
    BASE_URL = "https://dadosabertos.web.stj.jus.br"

    async def stream_decisions(
        self,
        dataset: str = "acordaos",  # acordaos, decisoes_monocraticas
        date_from: date = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream decisions from STJ Open Data Portal."""

        url = f"{self.BASE_URL}/dataset/{dataset}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Get dataset metadata to find latest CSV/JSON download URL
            response = await client.get(url)
            # Parse download links and iterate
            # STJ provides CSV and JSON formats

            # Example: direct CSV download
            csv_url = f"{self.BASE_URL}/dataset/{dataset}/resource/latest.csv"
            async with client.stream("GET", csv_url) as stream:
                buffer = ""
                async for chunk in stream.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        # Parse CSV line
                        reader = csv.DictReader(io.StringIO(line))
                        for row in reader:
                            yield row
```

### Airflow DAG de Ingestão (`airflow/dags/ingest_datajud.py`)

```python
"""
DAG de ingestão diária do DataJud.
Executa para cada tribunal configurado.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta

TRIBUNAIS = [
    "stf", "stj", "tst",
    "trf1", "trf2", "trf3", "trf4", "trf5", "trf6",
    "tjsp", "tjrj", "tjmg", "tjrs", "tjpr", "tjba",
    # Adicionar mais conforme necessário
]

default_args = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ingest_datajud_daily",
    default_args=default_args,
    schedule_interval="0 3 * * *",  # 3am daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "datajud"],
) as dag:

    def ingest_tribunal(tribunal: str, **kwargs):
        """Ingest latest data from a specific tribunal."""
        from app.services.ingestion.datajud import DataJudClient
        from app.services.rag.indexer import Indexer

        client = DataJudClient(api_key="{{var.value.datajud_api_key}}")
        indexer = Indexer()

        # Get last ingestion date from metadata
        hook = PostgresHook(postgres_conn_id="jurisai_db")
        last_date = hook.get_first(
            f"SELECT MAX(filing_date) FROM ingestion_log WHERE source = 'datajud_{tribunal}'"
        )

        # Fetch new data
        processes = client.search_processes(
            tribunal=tribunal,
            date_from=last_date[0].isoformat() if last_date[0] else "2024-01-01",
            size=1000,
        )

        # Index into Elasticsearch + Qdrant
        for process in processes:
            indexer.index_process(process)

        # Log ingestion
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, ingested_at) VALUES (%s, %s, NOW())",
            parameters=(f"datajud_{tribunal}", len(processes)),
        )

    for tribunal in TRIBUNAIS:
        PythonOperator(
            task_id=f"ingest_{tribunal}",
            python_callable=ingest_tribunal,
            op_kwargs={"tribunal": tribunal},
        )
```

---

## 8. OCR DE ALTA PERFORMANCE

### PaddleOCR Pipeline (`apps/api/app/services/ocr/paddle.py`)

```python
"""
PaddleOCR-VL 1.5:
- 94.5% precision (OmniDocBench v1.5)
- Reconhece selos, carimbos, protocolos (crítico para docs judiciais BR)
- PP-StructureV3: converte docs complexos para Markdown/JSON
- 25 categorias de layout (tabelas, fórmulas, headers)
- Todos os modelos <100M params (resource-efficient)
- Self-hosted cost: ~R$0.09/1,000 páginas em A100
"""

import httpx
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class OCRResult(BaseModel):
    text: str
    confidence: float
    pages: int
    structured_output: Optional[dict] = None  # Markdown ou JSON

class PaddleOCRService:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def process_pdf(
        self,
        file_path: str,
        output_format: str = "markdown",  # markdown, json
        enable_seal_recognition: bool = True,
    ) -> OCRResult:
        """Process a PDF through PaddleOCR-VL 1.5."""

        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"{self.endpoint}/predict",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={
                        "output_format": output_format,
                        "enable_seal": str(enable_seal_recognition).lower(),
                        "enable_table": "true",
                        "lang": "pt",
                    },
                )
                response.raise_for_status()
                result = response.json()

        return OCRResult(
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.0),
            pages=result.get("pages", 0),
            structured_output=result.get("structured", None),
        )


class SuryaOCRService:
    """Surya + Marker pipeline (Vik Paruchuri/Datalab).
    - 90+ languages including Portuguese
    - Reading order detection (essential for multi-column judicial decisions)
    - 25 pages/second on H100 in batch
    - --use_llm flag with Gemini API improves multi-page table accuracy
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def process_pdf(self, file_path: str) -> OCRResult:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"{self.endpoint}/convert",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={"output_format": "markdown"},
                )
                response.raise_for_status()
                result = response.json()

        return OCRResult(
            text=result.get("markdown", ""),
            confidence=result.get("confidence", 0.0),
            pages=result.get("pages", 0),
        )
```

### Pós-processamento OCR com NER (`apps/api/app/services/ocr/postprocess.py`)

```python
"""
Pipeline pós-OCR:
1. Correção ortográfica via dicionário jurídico custom (Hunspell pt_BR + termos legais)
2. NER com BERTimbau para extração de entidades
3. RegEx para entidades estruturadas (CNJ, valores, legislação)
"""

import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class LegalEntity:
    text: str
    label: str  # PESSOA, ORGANIZACAO, LEGISLACAO, JURISPRUDENCIA, VALOR, CNJ_NUMBER, DATA
    start: int
    end: int
    confidence: float

class LegalPostProcessor:
    # RegEx patterns para entidades jurídicas brasileiras
    PATTERNS = {
        "cnj_number": re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"),
        "monetary_value": re.compile(r"R\$\s?[\d.,]+(?:\s?(?:mil|milhões?|bilhões?))?"),
        "legislation": re.compile(
            r"(?:Lei|Decreto|Resolução|Portaria|Instrução Normativa|Medida Provisória|Emenda Constitucional)"
            r"\s+(?:n[ºo°]?\s*)?[\d.,/]+(?:\s*/\s*\d{4})?"
        ),
        "article_ref": re.compile(
            r"(?:Art(?:igo)?\.?\s*\d+(?:[°ºª])?(?:\s*,\s*(?:§\s*\d+[°ºª]?|inciso\s+[IVXLCDM]+|alínea\s+[a-z]))*)"
        ),
        "sumula": re.compile(
            r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?\d+(?:\s+do\s+(?:STF|STJ|TST|TSE))?"
        ),
        "date_br": re.compile(
            r"\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}"
        ),
        "cpf": re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
        "cnpj": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
        "oab": re.compile(r"OAB[/\s]*[A-Z]{2}[\s/]*\d+"),
    }

    def extract_entities(self, text: str) -> List[LegalEntity]:
        """Extract structured legal entities using RegEx."""
        entities = []

        for label, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append(LegalEntity(
                    text=match.group(),
                    label=label.upper(),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,  # RegEx matches are deterministic
                ))

        return sorted(entities, key=lambda e: e.start)

    def clean_ocr_text(self, text: str) -> str:
        """Clean common OCR artifacts in Brazilian legal documents."""
        # Fix common OCR errors
        replacements = {
            "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
            " .": ".", " ,": ",", " ;": ";",
            "\u00a0": " ",  # Non-breaking space
            "  ": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Fix broken hyphenation (common in scanned docs)
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
```

---

## 9. CLASSIFICAÇÃO DE DOCUMENTOS E NER

### Classificador de Documentos (`apps/api/app/services/nlp/classifier.py`)

```python
"""
Classificação de documentos jurídicos.

Modelos base:
- RoBERTaLexPT (PROPOR 2024, CC BY 4.0): pré-treinado em corpus jurídico PT
- LegalBert-pt: treinado em 1.5M docs de 10 tribunais brasileiros
- Pierre Guillou: 97.95% accuracy classificando textos TCU

Classes de documentos (baseado no VICTOR/STF):
- petição inicial, contestação, réplica
- sentença, acórdão, decisão monocrática
- recurso (apelação, agravo, RE, REsp)
- certidão, procuração, comprovante
- laudo/parecer
- despacho
- outros
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

class DocumentClassifier:
    LABELS = [
        "peticao_inicial", "contestacao", "replica",
        "sentenca", "acordao", "decisao_monocratica",
        "recurso_apelacao", "recurso_agravo", "recurso_extraordinario",
        "recurso_especial", "certidao", "procuracao",
        "comprovante", "laudo_parecer", "despacho", "outros"
    ]

    def __init__(self, model_name: str = "pfreitag/roberta-legal-pt-classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify document type from first ~1000 tokens (most informative)."""
        # First 1000 tokens are highly informative (VICTOR finding)
        inputs = self.tokenizer(
            text[:4000],  # ~1000 tokens
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs[0][pred_idx].item()

        return self.LABELS[pred_idx], confidence
```

### NER Jurídico (`apps/api/app/services/nlp/ner.py`)

```python
"""
NER Jurídico combinando:
1. Transformer: pierreguillou/ner-bert-large-cased-pt-lenerbr (F1 0.908)
   Labels: PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA
2. RegEx: entidades estruturadas (CNJ, valores monetários, OAB)
3. spaCy EntityRuler: regras customizadas

Dataset LeNER-Br: 70 docs de STF, STJ, TJMG, TCU com anotações manuais.
"""

from transformers import pipeline
from typing import List
from app.services.ocr.postprocess import LegalPostProcessor, LegalEntity

class LegalNERService:
    def __init__(self):
        self.ner_pipeline = pipeline(
            "ner",
            model="pierreguillou/ner-bert-large-cased-pt-lenerbr",
            aggregation_strategy="simple",
        )
        self.regex_extractor = LegalPostProcessor()

    def extract_entities(self, text: str) -> List[LegalEntity]:
        """Combined NER: Transformer + RegEx."""

        entities = []

        # 1. Transformer NER (chunks of 512 tokens)
        chunks = [text[i:i+2000] for i in range(0, len(text), 1800)]  # ~overlap
        for chunk in chunks:
            try:
                ner_results = self.ner_pipeline(chunk)
                for ent in ner_results:
                    entities.append(LegalEntity(
                        text=ent["word"],
                        label=ent["entity_group"],
                        start=ent["start"],
                        end=ent["end"],
                        confidence=ent["score"],
                    ))
            except Exception:
                continue

        # 2. RegEx entities
        regex_entities = self.regex_extractor.extract_entities(text)
        entities.extend(regex_entities)

        # 3. Deduplicate
        entities = self._deduplicate(entities)

        return entities

    def _deduplicate(self, entities: List[LegalEntity]) -> List[LegalEntity]:
        """Remove overlapping entities, preferring higher confidence."""
        if not entities:
            return []

        entities.sort(key=lambda e: (e.start, -e.confidence))
        result = [entities[0]]

        for ent in entities[1:]:
            last = result[-1]
            if ent.start >= last.end:
                result.append(ent)
            elif ent.confidence > last.confidence:
                result[-1] = ent

        return result
```

---

## 10. SISTEMA DE MEMÓRIA PERSISTENTE

### Arquitetura 4-Tier

```
Tier 1: Active Context (Letta/MemGPT-style)
   └── Editable memory blocks within LLM context window
   └── Updated every conversation turn

Tier 2: Session State (LangGraph Checkpointers)
   └── PostgresSaver: full state per thread
   └── Thread ID: {tenant_id}_{case_id}_{session_id}
   └── Time-travel debugging

Tier 3: Structured Knowledge (Graphiti + Mem0)
   └── Graphiti: temporal knowledge graph in Neo4j/AGE
   └── Entities: clients, cases, judges, courts, legislation, precedents
   └── Relationships with bi-temporal model
   └── Mem0: fact extraction with ADD/UPDATE/DELETE/NO-OP

Tier 4: Document Archive (Vector DB + S3)
   └── pgvector/Qdrant for document chunks
   └── S3/MinIO for original PDFs
```

### Memory Manager (`apps/api/app/services/memory/manager.py`)

```python
"""
4-Tier Memory System orchestrator.

Tier 1: Active context blocks (core_memory in Letta style)
Tier 2: LangGraph checkpoints (PostgresSaver)
Tier 3: Knowledge graph (Graphiti) + fact store (Mem0)
Tier 4: Document vectors (Qdrant) + files (S3)
"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class MemoryContext:
    """Assembled memory context for LLM."""
    active_memory: str = ""        # Tier 1: core facts about current case/client
    session_summary: str = ""      # Tier 2: summary of recent session
    knowledge_facts: List[str] = field(default_factory=list)  # Tier 3: relevant facts
    relevant_docs: List[dict] = field(default_factory=list)    # Tier 4: doc snippets

    def to_system_prompt_section(self) -> str:
        """Format memory for injection into system prompt."""
        sections = []

        if self.active_memory:
            sections.append(f"## Memória Ativa (Caso/Cliente Atual)\n{self.active_memory}")

        if self.session_summary:
            sections.append(f"## Contexto da Sessão\n{self.session_summary}")

        if self.knowledge_facts:
            facts = "\n".join(f"- {f}" for f in self.knowledge_facts[:20])
            sections.append(f"## Fatos Relevantes do Knowledge Graph\n{facts}")

        return "\n\n".join(sections)


class MemoryManager:
    def __init__(self, graphiti_client, mem0_client, checkpointer):
        self.graphiti = graphiti_client
        self.mem0 = mem0_client
        self.checkpointer = checkpointer

    async def assemble_context(
        self,
        tenant_id: str,
        user_id: str,
        case_id: Optional[str] = None,
        query: str = "",
    ) -> MemoryContext:
        """Assemble memory context from all 4 tiers."""

        context = MemoryContext()

        # Tier 1: Active memory (from Mem0 fact store)
        if case_id:
            facts = await self.mem0.search(
                query=query,
                user_id=f"{tenant_id}_{case_id}",
                limit=10,
            )
            context.active_memory = "\n".join(
                f"- {f['memory']}" for f in facts.get("results", [])
            )

        # Tier 3: Knowledge graph search
        if query:
            kg_results = await self.graphiti.search(
                query=query,
                namespace=tenant_id,
                limit=15,
            )
            context.knowledge_facts = [
                f"{r['fact']} (fonte: {r.get('source', 'N/A')})"
                for r in kg_results
            ]

        return context

    async def store_fact(
        self,
        tenant_id: str,
        case_id: str,
        fact: str,
        source: str = "conversation",
    ):
        """Store a new fact in Mem0 + Graphiti."""
        # Mem0: add/update fact
        await self.mem0.add(
            messages=[{"role": "assistant", "content": fact}],
            user_id=f"{tenant_id}_{case_id}",
            metadata={"source": source, "tenant_id": tenant_id},
        )

        # Graphiti: add to knowledge graph
        await self.graphiti.add_episode(
            name=f"fact_{case_id}",
            episode_body=fact,
            source_description=source,
            namespace=tenant_id,
        )
```

---

## 11. JURIMETRIA E PERFIL DE JUÍZES

### Judge Profiler (`apps/api/app/services/jurimetrics/judge_profile.py`)

```python
"""
Jurimetria: análise estatística de decisões judiciais.

Fontes:
- ABJ (Associação Brasileira de Jurimetria): github.com/abjur (109 repos)
- Justiça em Números 2025: 44.6M casos julgados em 2024
- Data Lawyer: 45M+ decisões trabalhistas e federais
- Turivius: 130M+ decisões
- JUDIT: 90+ tribunais via API

Modelos preditivos:
- XGBoost: ~80.2% F1-score (Lage-Freitas et al., 4.043 decisões TJAL)
- Bayesian Networks: 92% accuracy em TRT-3

⚠️ CNJ Resolução 615/2025: modelos preditivos em matéria CRIMINAL são desencorajados (Art. 23)
"""

from typing import Dict, Optional
from pydantic import BaseModel

class JudgeProfile(BaseModel):
    name: str
    court: str
    jurisdiction: str
    total_decisions: int
    avg_decision_time_days: float
    favorability: Dict[str, Dict[str, float]]  # {area: {autor: %, réu: %}}
    top_citations: list  # Legislação mais citada
    decision_patterns: Dict[str, any]
    conciliation_rate: float
    reform_rate: float  # % de decisões reformadas em instância superior

class JurimetricsService:
    async def get_judge_profile(
        self, judge_name: str, court: Optional[str] = None
    ) -> Optional[JudgeProfile]:
        """Build comprehensive judge profile from multiple sources."""
        # 1. Check local DB cache
        # 2. Query external APIs (Data Lawyer, Turivius, JUDIT)
        # 3. Aggregate and return
        pass

    async def predict_outcome(
        self,
        case_area: str,
        judge_name: str,
        case_features: dict,
    ) -> Dict[str, float]:
        """
        Predict case outcome probability.

        ⚠️ IMPORTANTE: NÃO usar para matéria criminal (CNJ Res. 615/2025 Art. 23)
        ⚠️ Resultado é indicativo, NUNCA determinístico
        ⚠️ Sempre apresentar com disclaimer ao advogado
        """
        if case_area.lower() in ("criminal", "penal"):
            raise ValueError(
                "Predição em matéria criminal é desencorajada pela CNJ Resolução 615/2025 Art. 23"
            )

        # XGBoost model trained on historical decisions
        # Features: area, judge, court, case_type, monetary_value, etc.
        pass
```

---

## 12. GERAÇÃO DE PETIÇÕES

### Citation Verifier (`apps/api/app/services/petition/citation_verifier.py`)

```python
"""
Verificação de citações jurídicas.
Diferencial competitivo: apenas JusBrasil e Turivius verificam citações no Brasil.

Verifica:
1. Legislação: artigo existe, lei está em vigor (não revogada)
2. Jurisprudência: súmula/decisão existe, número correto, tribunal correto
3. Doutrina: autor e obra existem (best-effort)

Fontes de verificação:
- Planalto.gov.br: legislação federal
- STF/STJ APIs: jurisprudência
- LexML: URN LEX persistentes
"""

import re
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel

class CitationStatus(str, Enum):
    VERIFIED = "verified"          # Confirmada na base
    NOT_FOUND = "not_found"        # Não encontrada
    REVOKED = "revoked"            # Legislação revogada
    OUTDATED = "outdated"          # Versão desatualizada
    UNCHECKED = "unchecked"        # Não foi possível verificar

class Citation(BaseModel):
    text: str                      # Texto original da citação
    type: str                      # legislacao, jurisprudencia, doutrina
    status: CitationStatus
    verified_text: Optional[str]   # Texto correto (se diferir)
    source_url: Optional[str]      # URL da fonte
    confidence: float

class CitationVerifier:
    async def verify_all(self, text: str) -> List[Citation]:
        """Extract and verify all citations in a legal text."""
        citations = []

        # 1. Extract legislation references
        leg_refs = self._extract_legislation_refs(text)
        for ref in leg_refs:
            result = await self._verify_legislation(ref)
            citations.append(result)

        # 2. Extract jurisprudence references
        juris_refs = self._extract_jurisprudence_refs(text)
        for ref in juris_refs:
            result = await self._verify_jurisprudence(ref)
            citations.append(result)

        # 3. Extract súmulas
        sumula_refs = self._extract_sumula_refs(text)
        for ref in sumula_refs:
            result = await self._verify_sumula(ref)
            citations.append(result)

        return citations

    def _extract_legislation_refs(self, text: str) -> List[str]:
        pattern = r"(?:Art(?:igo)?\.?\s*\d+(?:[°ºª])?(?:\s*(?:,|e)\s*(?:§\s*\d+[°ºª]?|inciso\s+[IVXLCDM]+|alínea\s+[\"']?[a-z][\"']?))*\s*(?:,?\s*d[aoe]\s+)?(?:Lei|Decreto|Resolução|CF|Código\s+\w+)(?:\s+(?:n[ºo°]?\s*)?[\d.,/]+)?(?:\s*/\s*\d{4})?)"
        return re.findall(pattern, text, re.IGNORECASE)

    def _extract_jurisprudence_refs(self, text: str) -> List[str]:
        # CNJ number pattern
        cnj = r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"
        # Informal refs: "RE 123.456", "AgRg no REsp 1.234.567"
        informal = r"(?:RE|REsp|AgRg|AI|HC|MS|ADI|ADPF|RMS)\s+[\d.,]+"
        results = re.findall(f"({cnj}|{informal})", text)
        return results

    def _extract_sumula_refs(self, text: str) -> List[str]:
        pattern = r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?\d+(?:\s+d[oe]\s+(?:STF|STJ|TST|TSE))?"
        return re.findall(pattern, text, re.IGNORECASE)

    async def _verify_legislation(self, ref: str) -> Citation:
        """Verify legislation against Planalto.gov.br / LexML."""
        # Implementation: query LexML API or local legislation DB
        # Check if law exists, is in force, article number is valid
        return Citation(
            text=ref,
            type="legislacao",
            status=CitationStatus.UNCHECKED,
            confidence=0.0,
        )

    async def _verify_jurisprudence(self, ref: str) -> Citation:
        """Verify jurisprudence against STF/STJ APIs."""
        return Citation(
            text=ref,
            type="jurisprudencia",
            status=CitationStatus.UNCHECKED,
            confidence=0.0,
        )

    async def _verify_sumula(self, ref: str) -> Citation:
        """Verify súmula against official sources."""
        return Citation(
            text=ref,
            type="jurisprudencia",
            status=CitationStatus.UNCHECKED,
            confidence=0.0,
        )
```

---

## 13. ORQUESTRAÇÃO MULTI-AGENTE (LANGGRAPH)

### Graph Definition (`apps/api/app/agents/graph.py`)

```python
"""
LangGraph 1.0 Multi-Agent Orchestration.

Agents:
- Supervisor: router que direciona para o agent adequado
- Research Agent: RAG + busca em bases oficiais
- Drafting Agent: geração de petições com templates
- Analysis Agent: timeline, NER, risk assessment
- Memory Agent: gestão de memória cross-session

Key features:
- Human-in-the-loop via interrupt() primitive
- Checkpointing com PostgresSaver (audit trail completo)
- Time-travel debugging
- SSE streaming de output + intermediate steps
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tenant_id: str
    user_id: str
    case_id: str | None
    use_rag: bool
    use_memory: bool
    current_agent: str
    rag_results: list
    memory_context: str
    citations: list
    thinking: str  # Reasoning trace

# ============================================
# AGENT NODES
# ============================================

async def supervisor_node(state: AgentState) -> dict:
    """Route to the appropriate agent based on user intent."""
    from app.services.llm.router import LLMRouter

    router = LLMRouter()
    last_message = state["messages"][-1].content

    # Classify intent
    classification_prompt = f"""Classifique a intenção do usuário em uma das categorias:
    - research: pesquisa jurídica, busca de jurisprudência, legislação
    - draft: redação de petição, documento jurídico
    - analyze: análise de caso, risk assessment, timeline
    - chat: conversa geral, dúvidas simples
    - memory: consulta sobre casos/clientes anteriores

    Mensagem: {last_message}

    Responda APENAS com a categoria."""

    intent = await router.quick_classify(classification_prompt)

    return {"current_agent": intent.strip().lower()}

async def research_node(state: AgentState) -> dict:
    """Legal research agent with RAG."""
    from app.services.rag.retriever import HybridRetriever
    from app.services.llm.router import LLMRouter

    retriever = HybridRetriever()
    router = LLMRouter()

    query = state["messages"][-1].content

    # RAG retrieval
    if state.get("use_rag", True):
        chunks = await retriever.retrieve(
            query=query,
            tenant_id=state["tenant_id"],
            top_k=10,
        )
        context = "\n\n---\n\n".join([
            f"[{c.doc_type} | {c.court} | {c.date}]\n{c.content}"
            for c in chunks
        ])
    else:
        context = ""
        chunks = []

    # Memory context
    memory_section = state.get("memory_context", "")

    system_prompt = f"""Você é um assistente jurídico especializado no direito brasileiro.
Use as fontes fornecidas para fundamentar sua resposta.
SEMPRE cite a fonte (tribunal, número do processo, data) quando referenciar jurisprudência.
Se não encontrar informação nas fontes, diga explicitamente.

{memory_section}

## Fontes Recuperadas
{context if context else "Nenhuma fonte encontrada para esta consulta."}

## Regras
- Cite artigos de lei com precisão (Art. X, Lei Y/Z)
- Indique súmulas relevantes
- Diferencie jurisprudência dominante de posições isoladas
- Se houver divergência entre tribunais, mencione
- ⚠️ Este conteúdo é gerado com auxílio de IA (CNJ Res. 615/2025)"""

    response = await router.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            *[{"role": m.type, "content": m.content} for m in state["messages"]],
        ],
        stream=False,
    )

    return {
        "messages": [AIMessage(content=response["content"])],
        "rag_results": [{"id": c.id, "title": c.document_title} for c in chunks],
        "thinking": response.get("thinking", ""),
    }

async def drafting_node(state: AgentState) -> dict:
    """Petition drafting agent."""
    from app.services.llm.router import LLMRouter
    from app.services.petition.citation_verifier import CitationVerifier

    router = LLMRouter()
    verifier = CitationVerifier()

    system_prompt = """Você é um redator jurídico especializado.
Redija petições e documentos jurídicos seguindo:
1. Estrutura formal (endereçamento, qualificação, fatos, direito, pedidos)
2. Linguagem técnica adequada ao tipo de peça
3. Fundamentação legal com artigos específicos
4. Jurisprudência relevante e atualizada
5. Formatação ABNT quando aplicável

⚠️ Conteúdo gerado com auxílio de IA — CNJ Resolução 615/2025
⚠️ OBRIGATÓRIO: revisão pelo advogado antes de protocolar (OAB Item 3.7)"""

    response = await router.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            *[{"role": m.type, "content": m.content} for m in state["messages"]],
        ],
        stream=False,
        tier="high",  # Sempre usar modelo forte para redação
    )

    # Verify citations in generated text
    citations = await verifier.verify_all(response["content"])

    return {
        "messages": [AIMessage(content=response["content"])],
        "citations": [c.model_dump() for c in citations],
    }

async def analysis_node(state: AgentState) -> dict:
    """Case analysis agent."""
    # Similar structure: analyze case, build timeline, risk assessment
    pass

async def memory_node(state: AgentState) -> dict:
    """Memory management agent."""
    from app.services.memory.manager import MemoryManager
    # Query and update memory based on conversation
    pass

# ============================================
# ROUTING
# ============================================

def route_by_intent(state: AgentState) -> Literal[
    "research", "drafting", "analysis", "memory", "chat"
]:
    agent = state.get("current_agent", "chat")
    if agent in ("research", "drafting", "analysis", "memory"):
        return agent
    return "chat"

# ============================================
# GRAPH CONSTRUCTION
# ============================================

def build_legal_agent_graph(checkpointer):
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("drafting", drafting_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("memory", memory_node)
    builder.add_node("chat", research_node)  # Default: same as research

    # Entry point
    builder.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "research": "research",
            "drafting": "drafting",
            "analysis": "analysis",
            "memory": "memory",
            "chat": "chat",
        }
    )

    # All agents end
    builder.add_edge("research", END)
    builder.add_edge("drafting", END)
    builder.add_edge("analysis", END)
    builder.add_edge("memory", END)
    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)

# Initialize graph with PostgreSQL checkpointer
async def get_legal_agent_graph():
    from app.config import settings
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
    await checkpointer.setup()
    return build_legal_agent_graph(checkpointer)
```

---

## 14. REASONING MODEL — GAIA FINE-TUNED

### Overview

```
GAIA (Gemma-3-Gaia-PT-BR-4b-it)
├── Base: Gemma 3 4B (Google DeepMind)
├── Training: ~13B tokens PT-BR contínuo + instruction residuals
├── Benchmarks: ENEM 70%, OAB 44.16%
├── HuggingFace: CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it
├── License: Gemma (commercial OK)
└── VRAM: ~8.64 GB (BF16), ~4-5 GB (INT8/FP8)

Pipeline de treinamento (estilo DeepSeek-R1, atualizado Fev/2026):
1. Coletar dados jurídicos (OAB, STF, STJ, legislação)
2. Gerar 50K-100K reasoning traces via teacher model (DeepSeek V3.2 / Qwen 3.5)
3. SFT com LoRA nos reasoning traces
4. GRPO com reward functions jurídicas
5. Avaliação (OAB benchmark, citações)
6. Deploy no Modal com vLLM

Teacher models disponíveis (Fev/2026, em ordem de custo-benefício):
├── DeepSeek V3.2    $0.28/$1.10/M — open-source MIT, 85.9% MMLU-Pro
├── Qwen 3.5         $0.50/$2.00/M — Apache 2.0, 397B MoE, ótimo p/ jurídico
├── Kimi K2.5        $0.45/$2.20/M — 1T params, líder em agentic tasks
├── MiniMax M2.5     $0.30/$1.10/M — 80.2% SWE-Bench, bom p/ agentes
├── Gemini 3 Pro     $1.25/$5.00/M — 89.8% MMLU-Pro, melhor frontier value
├── Claude Sonnet 4  $3.00/$15.0/M — melhor coding (72.5% SWE-Bench)
└── GPT-5.2 Pro      $$$/M         — 88.7% MMLU-Pro, mais caro
```

### Reasoning Trace Format

```
<think>
1. IDENTIFICAÇÃO DO TEMA: [área jurídica e questão específica]
2. LEGISLAÇÃO APLICÁVEL: [artigos relevantes da CF, códigos, leis]
3. JURISPRUDÊNCIA: [súmulas e decisões relevantes]
4. ANÁLISE: [aplicação da lei aos fatos, técnicas de interpretação]
5. ELIMINAÇÃO: [por que alternativas incorretas falham]
6. CONCLUSÃO: [resposta fundamentada]
</think>
Resposta: [conclusão final com fundamentação]
```

---

## 15. PIPELINE DE TREINAMENTO GRPO

### Geração de Dados CoT (`training/data/generate_cot.py`)

```python
"""
Gerar reasoning traces jurídicos usando teacher model.
Meta: 50K-100K traces de alta qualidade.

Teacher models (em ordem de custo-benefício — Fev/2026):
1. DeepSeek V3.2 API ($0.28/$1.10 por M tokens) — melhor custo-benefício, open-source MIT
2. Qwen 3.5 API ($0.50/$2.00 por M tokens) — Apache 2.0, ótimo em jurídico
3. Kimi K2.5 API ($0.45/$2.20 por M tokens) — líder em agentic tasks
4. Claude Sonnet 4 API ($3/$15 por M tokens) — melhor qualidade geral
5. Gemini 3 Pro API ($1.25/$5.00 por M tokens) — melhor frontier value

Pipeline:
1. Carregar questões OAB + cenários jurídicos
2. Para cada questão: gerar 5-10 traces com teacher
3. Rejection sampling: manter apenas onde resposta = gabarito
4. Self-consistency: selecionar trace mais comum
5. Filtragem de qualidade: verificar artigos/leis citados
"""

import asyncio
import json
import httpx
from pathlib import Path
from typing import List, Dict

SYSTEM_PROMPT = """Você é um jurista brasileiro especialista.
Ao responder questões jurídicas, SEMPRE raciocine passo a passo dentro de tags <think>...</think>.

Seu raciocínio DEVE incluir:
1. IDENTIFICAÇÃO DO TEMA: qual área do direito e qual a questão central
2. LEGISLAÇÃO APLICÁVEL: artigos específicos da CF, códigos e leis, com número e redação
3. JURISPRUDÊNCIA: súmulas vinculantes/não-vinculantes e decisões relevantes
4. ANÁLISE: aplicação da norma aos fatos, usando técnicas de interpretação
5. ELIMINAÇÃO (se múltipla escolha): por que cada alternativa incorreta falha
6. CONCLUSÃO: resposta final fundamentada

Após </think>, forneça a resposta final de forma direta e objetiva."""

async def generate_traces(
    questions: List[Dict],
    teacher_model: str = "deepseek-chat",
    traces_per_question: int = 5,
    output_file: str = "training/data/raw_traces.jsonl",
):
    """Generate reasoning traces using teacher model (DeepSeek V3.2 default)."""

    results = []

    for q in questions:
        prompt = q["question"]
        if q.get("options"):
            options = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
            prompt += f"\n\nAlternativas:\n{options}"

        traces = []
        for _ in range(traces_per_question):
            response = await call_teacher(
                model=teacher_model,
                system=SYSTEM_PROMPT,
                user=prompt,
            )

            # Extract thinking and answer
            thinking, answer = parse_response(response)

            # Check if answer matches correct answer (rejection sampling)
            if q.get("correct_answer"):
                is_correct = normalize_answer(answer) == normalize_answer(q["correct_answer"])
            else:
                is_correct = True  # Open-ended question

            traces.append({
                "thinking": thinking,
                "answer": answer,
                "is_correct": is_correct,
                "full_response": response,
            })

        # Keep only correct traces
        correct_traces = [t for t in traces if t["is_correct"]]

        if correct_traces:
            # Self-consistency: pick the most common answer
            best_trace = select_best_trace(correct_traces)

            results.append({
                "question": q["question"],
                "options": q.get("options"),
                "correct_answer": q.get("correct_answer"),
                "area": q.get("area", "geral"),
                "difficulty": q.get("difficulty", "medium"),
                "thinking": best_trace["thinking"],
                "answer": best_trace["answer"],
                "full_response": best_trace["full_response"],
                "num_correct": len(correct_traces),
                "num_total": traces_per_question,
            })

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(results)} traces ({len(results)/len(questions)*100:.1f}% success rate)")

async def call_teacher(model: str, system: str, user: str) -> str:
    """Call teacher model API (DeepSeek V3.2 / Qwen 3.5 / Claude / Gemini)."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        if "deepseek" in model:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
            )
        elif "qwen" in model:
            api_key = os.environ.get("QWEN_API_KEY")
            response = await client.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
            )

        return response.json()["choices"][0]["message"]["content"]

def parse_response(response: str):
    """Extract thinking and answer from response."""
    import re
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Answer is everything after </think>
    if "</think>" in response:
        answer = response.split("</think>", 1)[1].strip()
    else:
        answer = response.strip()

    return thinking, answer

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().upper()
    # Extract letter from multiple choice
    import re
    match = re.search(r"[A-E]", answer[:10])
    return match.group(0) if match else answer[:50]

def select_best_trace(traces: list) -> dict:
    """Select best trace via self-consistency (most common answer)."""
    from collections import Counter
    answers = [normalize_answer(t["answer"]) for t in traces]
    most_common = Counter(answers).most_common(1)[0][0]
    for t in traces:
        if normalize_answer(t["answer"]) == most_common:
            return t
    return traces[0]
```

### SFT com LoRA (`training/sft/train_sft.py`)

```python
"""
SFT (Supervised Fine-Tuning) com LoRA no GAIA.

Configs:
- LoRA rank: 64, alpha: 32
- Learning rate: 4e-4
- Batch size: 4 (gradient accumulation 4 = effective 16)
- Epochs: 2-3
- Max seq length: 4096-8192
- VRAM: ~12-16 GB com LoRA (RTX 4090 OK)
- Tempo: 1-3 dias em A100 80GB

Usando Unsloth para 80% menos VRAM e 2x mais velocidade.
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ============================================
# 1. LOAD MODEL
# ============================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    max_seq_length=8192,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # QLoRA
)

# ============================================
# 2. CONFIGURE LoRA
# ============================================

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimization
)

# ============================================
# 3. PREPARE DATASET
# ============================================

# Chat template for GAIA (Gemma format)
CHAT_TEMPLATE = """<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
<think>
{thinking}
</think>

{answer}<end_of_turn>"""

def format_example(example):
    return {
        "text": CHAT_TEMPLATE.format(
            question=example["question"],
            thinking=example["thinking"],
            answer=example["answer"],
        )
    }

dataset = load_dataset("json", data_files="training/data/raw_traces.jsonl")
dataset = dataset["train"].map(format_example)

# ============================================
# 4. TRAIN
# ============================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    args=TrainingArguments(
        output_dir="training/sft/checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=2,
        learning_rate=4e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
    ),
)

trainer.train()

# ============================================
# 5. SAVE
# ============================================

model.save_pretrained("training/sft/gaia-legal-sft")
tokenizer.save_pretrained("training/sft/gaia-legal-sft")

# Export to GGUF for vLLM/Ollama
model.save_pretrained_gguf(
    "training/sft/gaia-legal-sft-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### GRPO Training (`training/grpo/train_grpo.py`)

```python
"""
GRPO (Group Relative Policy Optimization) para reasoning jurídico.

Hyperparams (baseado no DeepSeek-R1/V3.2):
- Learning rate: 3e-6
- KL coefficient: 0.001
- Temperature: 1.0
- Group size: 8 outputs por questão
- Max length: 8192 tokens
- Clip ratio: 10 (incomum - PPO padrão usa ~0.2)

Reward functions (rule-based, não neural):
1. Format reward: presença de <think>...</think>, estrutura argumentativa
2. Citation accuracy: artigos/leis existem e estão corretos
3. Answer correctness: resposta correta (quando verificável)
4. Language consistency: português, sem mixing
"""

from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
from datasets import load_dataset
import re

# ============================================
# 1. LOAD SFT MODEL
# ============================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="training/sft/gaia-legal-sft",
    max_seq_length=8192,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# ============================================
# 2. REWARD FUNCTIONS
# ============================================

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for proper reasoning format."""
    rewards = []
    for text in completions:
        score = 0.0

        # Has <think>...</think> tags
        if "<think>" in text and "</think>" in text:
            score += 0.2

            # Has numbered steps inside thinking
            think_content = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if think_content:
                content = think_content.group(1)
                # Check for structured reasoning
                if re.search(r"(?:IDENTIFICAÇÃO|LEGISLAÇÃO|JURISPRUDÊNCIA|ANÁLISE|CONCLUSÃO)", content, re.IGNORECASE):
                    score += 0.2
                # Has specific article references
                if re.search(r"Art\.\s*\d+", content):
                    score += 0.1

        rewards.append(score)
    return rewards

def citation_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for correct legal citations."""
    rewards = []
    for text in completions:
        score = 0.0

        # Extract cited articles
        articles = re.findall(
            r"Art\.\s*(\d+)(?:[°ºª])?\s*(?:,?\s*d[aoe]\s+)?(Lei|CF|CC|CPC|CPP|CLT|CDC|CTN|CP|ECA)\s*(?:n[ºo°]?\s*)?([\d./]*)",
            text, re.IGNORECASE
        )

        if articles:
            score += min(len(articles) * 0.3, 1.5)  # Up to 1.5 for citations

            # Verify against known valid articles (simplified)
            # In production: check against legislation database
            for art_num, law, law_num in articles:
                if is_valid_article(art_num, law, law_num):
                    score += 0.1

        # Súmulas
        sumulas = re.findall(r"Súmula\s+(?:Vinculante\s+)?(\d+)", text)
        if sumulas:
            score += min(len(sumulas) * 0.2, 0.5)

        rewards.append(min(score, 2.0))  # Cap at 2.0
    return rewards

def correctness_reward(completions: list[str], correct_answer: list[str], **kwargs) -> list[float]:
    """Reward for correct final answer (when verifiable)."""
    rewards = []
    for completion, answer in zip(completions, correct_answer):
        if not answer:
            rewards.append(0.0)
            continue

        # Extract answer after </think>
        if "</think>" in completion:
            response = completion.split("</think>", 1)[1].strip()
        else:
            response = completion.strip()

        # Normalize and compare
        pred = response[:100].strip().upper()
        gold = answer.strip().upper()

        if gold in pred or pred.startswith(gold):
            rewards.append(2.0)  # Full credit
        else:
            rewards.append(0.0)

    return rewards

def language_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward for consistent Portuguese language (no mixing)."""
    rewards = []
    for text in completions:
        # Simple heuristic: check for common English words that indicate mixing
        english_markers = len(re.findall(
            r"\b(?:the|is|are|was|were|have|has|been|will|would|could|should)\b",
            text, re.IGNORECASE
        ))
        if english_markers > 5:
            rewards.append(-0.5)  # Penalty for language mixing
        elif english_markers > 0:
            rewards.append(0.0)
        else:
            rewards.append(0.3)
    return rewards

def is_valid_article(art_num: str, law: str, law_num: str) -> bool:
    """Check if article exists (simplified — expand with real DB)."""
    # Known ranges (simplified)
    known_ranges = {
        "CF": 250,
        "CC": 2046,
        "CPC": 1072,
        "CLT": 922,
        "CDC": 119,
        "CP": 361,
        "ECA": 267,
    }
    law_upper = law.upper()
    if law_upper in known_ranges:
        try:
            return 1 <= int(art_num) <= known_ranges[law_upper]
        except ValueError:
            return False
    return True  # Unknown law, assume valid

# ============================================
# 3. PREPARE DATASET
# ============================================

dataset = load_dataset("json", data_files="training/data/grpo_problems.jsonl")

# Format: {"prompt": str, "correct_answer": str (optional)}
def format_prompt(example):
    return {
        "prompt": [
            {"role": "user", "content": example["question"]},
        ],
        "correct_answer": example.get("correct_answer", ""),
    }

dataset = dataset["train"].map(format_prompt)

# ============================================
# 4. TRAIN WITH GRPO
# ============================================

config = GRPOConfig(
    output_dir="training/grpo/checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-6,
    num_generations=8,         # Group size (G outputs per prompt)
    max_completion_length=8192,
    max_prompt_length=2048,
    temperature=1.0,
    beta=0.001,                # KL coefficient
    logging_steps=5,
    save_steps=100,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward,
        citation_reward,
        correctness_reward,
        language_reward,
    ],
    config=config,
    train_dataset=dataset,
)

trainer.train()

# Save
model.save_pretrained("training/grpo/gaia-legal-reasoning")
tokenizer.save_pretrained("training/grpo/gaia-legal-reasoning")
```

---

## 16. INFRAESTRUTURA GPU (MODAL / RUNPOD)

### Modal — Inferência 24/7 (`training/serve/modal_serve.py`)

```python
"""
Deploy do GAIA Legal Reasoning no Modal.com

Modal:
- Serverless, pay per second
- H100 $3.95/hr, A100 $2.50/hr, L4 $0.80/hr
- Auto-scale 0 → N GPUs
- Volumes persistentes para weights
- Cold start: ~30s (mitigável com min_containers=1)

Para GAIA 4B quantizado (INT8): L4 24GB é suficiente
Para GAIA 4B full (BF16): A10G 24GB ou A100 40GB
"""

import modal

app = modal.App("jurisai-gaia-serve")

# Persistent volume for model weights
volume = modal.Volume.from_name("gaia-weights", create_if_missing=True)

# Docker image with vLLM
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.4.0",
        "transformers",
        "huggingface_hub",
    )
)

@app.cls(
    image=image,
    gpu="L4",              # L4 24GB for INT8 GAIA 4B
    volumes={"/models": volume},
    container_idle_timeout=300,   # Keep warm 5 min
    allow_concurrent_inputs=10,   # Handle 10 concurrent requests
    min_containers=1,             # Always-on (eliminates cold start)
    max_containers=5,             # Scale up to 5 for peaks
)
class GaiaServe:
    @modal.enter()
    def load_model(self):
        """Load model on container start."""
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        model_path = "/models/gaia-legal-reasoning"

        # Download model if not cached
        if not Path(model_path).exists():
            from huggingface_hub import snapshot_download
            snapshot_download(
                "your-org/gaia-legal-reasoning-v1",
                local_dir=model_path,
            )

        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="auto",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            quantization="awq",  # or "gptq" for quantized
            enforce_eager=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7):
        """Generate completion."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        request_id = f"req-{id(prompt)}"
        results = []
        async for output in self.engine.generate(prompt, params, request_id):
            results.append(output)

        final = results[-1]
        return {
            "text": final.outputs[0].text,
            "tokens": len(final.outputs[0].token_ids),
            "finish_reason": final.outputs[0].finish_reason,
        }

    @modal.method()
    async def generate_stream(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7):
        """Stream tokens one by one."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        request_id = f"req-{id(prompt)}"
        prev_text = ""
        async for output in self.engine.generate(prompt, params, request_id):
            new_text = output.outputs[0].text[len(prev_text):]
            prev_text = output.outputs[0].text
            if new_text:
                yield new_text

# ============================================
# OpenAI-compatible API endpoint
# ============================================

@app.function(image=image)
@modal.web_endpoint(method="POST")
async def v1_chat_completions(request: dict):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    model = GaiaServe()

    messages = request.get("messages", [])
    # Format messages into prompt
    prompt = format_chat_prompt(messages)

    if request.get("stream", False):
        # SSE streaming
        async def stream():
            async for token in model.generate_stream.remote_gen(
                prompt=prompt,
                max_tokens=request.get("max_tokens", 4096),
                temperature=request.get("temperature", 0.7),
            ):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return stream()
    else:
        result = await model.generate.remote(
            prompt=prompt,
            max_tokens=request.get("max_tokens", 4096),
            temperature=request.get("temperature", 0.7),
        )
        return {
            "choices": [{"message": {"role": "assistant", "content": result["text"]}}],
            "usage": {"completion_tokens": result["tokens"]},
        }

def format_chat_prompt(messages: list) -> str:
    """Format messages into Gemma chat template."""
    prompt = ""
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt
```

### Modal — Training Job (`training/grpo/modal_train.py`)

```python
"""
Run GRPO training on Modal.
A100 80GB for training.
"""

import modal

app = modal.App("jurisai-gaia-train")

volume = modal.Volume.from_name("gaia-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
        "trl>=0.14.0",
        "transformers",
        "datasets",
        "torch>=2.4.0",
        "wandb",
    )
)

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/training": volume},
    timeout=86400,  # 24 hours max
)
def run_grpo_training():
    """Run GRPO training job."""
    import subprocess
    subprocess.run(
        ["python", "/training/grpo/train_grpo.py"],
        check=True,
    )

@app.local_entrypoint()
def main():
    # Upload training scripts and data to volume
    volume = modal.Volume.from_name("gaia-training")
    # ... upload files ...
    run_grpo_training.remote()
```

---

## 17. ROTEAMENTO MULTI-MODELO

### LLM Router (`apps/api/app/services/llm/router.py`)

```python
"""
Roteamento inteligente entre modelos.

Tiers (atualizado Fev/2026):
- Tier 0: Cache/RAG (custo zero)
- Tier 1: GAIA 4B Local (70-80% queries) — R$0 (infra fixa)
- Tier 2: DeepSeek V3.2 ($0.28/$1.10/M) / Qwen 3.5 ($0.50/$2.00/M) / Sabiá-3 (R$5-10/M)
- Tier 2 alt: Kimi K2.5 ($0.45/$2.20/M) / MiniMax M2.5 ($0.30/$1.10/M) — agentes
- Tier 3: Claude Sonnet 4 ($3/$15/M) / Gemini 3 Pro ($1.25/$5.00/M)

RouteLLM (github.com/lm-sys/RouteLLM):
- BERT classifier, >85% redução de custo
- Transferência robusta entre pares de modelos

LiteLLM: proxy unificado, OpenAI-compatible, 100+ modelos
"""

from litellm import acompletion
from typing import Optional, Dict, List
from app.config import settings

class LLMRouter:
    MODEL_TIERS = {
        "low": {
            "model": f"openai/{settings.GAIA_MODEL_NAME}",
            "api_base": settings.GAIA_BASE_URL,
            "api_key": "dummy",
        },
        "medium": {
            "model": "deepseek/deepseek-chat",       # DeepSeek V3.2 — $0.28/$1.10/M
            "api_key": settings.DEEPSEEK_API_KEY,
        },
        "medium_qwen": {
            "model": "qwen/qwen3.5",                  # Qwen 3.5 — $0.50/$2.00/M, Apache 2.0
            "api_key": settings.QWEN_API_KEY,
        },
        "medium_pt": {
            "model": "maritaca/sabia-3",
            "api_key": settings.MARITACA_API_KEY,
        },
        "medium_agent": {
            "model": "kimi/moonshot-v1-k2.5",         # Kimi K2.5 — líder em agentic tasks
            "api_key": settings.KIMI_API_KEY,
        },
        "high": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": settings.ANTHROPIC_API_KEY,
        },
        "high_value": {
            "model": "gemini/gemini-3-pro",            # Gemini 3 Pro — melhor frontier value
            "api_key": settings.GOOGLE_API_KEY,
        },
    }

    def _select_tier(self, query: str, requested_tier: Optional[str] = None) -> str:
        """Select model tier based on query complexity."""
        if requested_tier:
            return requested_tier

        # Simple heuristic (replace with trained classifier later)
        complexity_indicators = [
            "analise", "análise", "compare", "redija", "elabore",
            "petição", "recurso", "fundamentação", "tese",
            "constitucional", "conflito de normas", "precedente",
        ]

        query_lower = query.lower()
        complexity_score = sum(
            1 for indicator in complexity_indicators
            if indicator in query_lower
        )

        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"

    async def generate(
        self,
        messages: List[Dict],
        stream: bool = False,
        tier: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Generate completion using appropriate model tier."""

        # Select tier
        query = messages[-1]["content"] if messages else ""
        selected_tier = self._select_tier(query, tier)
        model_config = self.MODEL_TIERS[selected_tier]

        # Call via LiteLLM
        response = await acompletion(
            model=model_config["model"],
            messages=messages,
            api_base=model_config.get("api_base"),
            api_key=model_config.get("api_key"),
            stream=stream,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
        )

        if stream:
            return response  # AsyncGenerator

        return {
            "content": response.choices[0].message.content,
            "model": model_config["model"],
            "tier": selected_tier,
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
            "thinking": getattr(response.choices[0].message, "reasoning_content", ""),
        }

    async def quick_classify(self, prompt: str) -> str:
        """Quick classification using Tier 1 (GAIA)."""
        result = await self.generate(
            messages=[{"role": "user", "content": prompt}],
            tier="low",
            max_tokens=50,
            temperature=0.1,
        )
        return result["content"]
```

---

## 18. FRONTEND — NEXT.JS

### Chat Interface (`apps/web/components/chat/ChatInterface.tsx`)

```typescript
"use client";

import { useChat } from "ai/react";
import { useState } from "react";
import { MessageBubble } from "./MessageBubble";
import { ThinkingIndicator } from "./ThinkingIndicator";
import { SourceCitation } from "./SourceCitation";

interface ChatInterfaceProps {
  caseId?: string;
  sessionId?: string;
}

export function ChatInterface({ caseId, sessionId }: ChatInterfaceProps) {
  const [sources, setSources] = useState<any[]>([]);

  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat({
      api: "/api/chat",
      body: {
        case_id: caseId,
        session_id: sessionId,
        use_rag: true,
        use_memory: true,
      },
      onResponse(response) {
        // Handle custom SSE events
      },
      onFinish(message) {
        // Extract sources from message metadata
        if (message.data?.sources) {
          setSources(message.data.sources);
        }
      },
    });

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {isLoading && <ThinkingIndicator />}
      </div>

      {/* Sources sidebar */}
      {sources.length > 0 && (
        <div className="border-t p-3 bg-gray-50 max-h-32 overflow-y-auto">
          <p className="text-xs font-semibold text-gray-500 mb-1">Fontes:</p>
          {sources.map((source, i) => (
            <SourceCitation key={i} source={source} />
          ))}
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Faça uma pergunta jurídica..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Enviar
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1">
          ⚠️ Conteúdo gerado com IA — revise antes de usar (OAB/CNJ)
        </p>
      </form>
    </div>
  );
}
```

### BFF API Route (`apps/web/app/api/chat/route.ts`)

```typescript
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const response = await fetch(
    `${process.env.API_URL}/api/v1/chat/completions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-ID": req.headers.get("x-tenant-id") || "",
        Authorization: req.headers.get("authorization") || "",
      },
      body: JSON.stringify({
        ...body,
        stream: true,
      }),
    }
  );

  // Forward SSE stream to client
  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
```

---

## 19. AUTENTICAÇÃO E MULTI-TENANCY

### JWT Auth (`apps/api/app/core/auth.py`)

```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        payload = jwt.decode(
            credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        if not user_id or not tenant_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": user_id, "tenant_id": tenant_id, "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_tenant_id(user=Depends(get_current_user)) -> str:
    return user["tenant_id"]
```

---

## 20. LGPD, OAB E COMPLIANCE

### Checklist de Compliance

```markdown
## LGPD (Lei 13.709/2018)
- [ ] Base legal: Art. 7, VI (exercício regular de direitos em processo judicial)
- [ ] Privacy by Design (Art. 46, §2)
- [ ] DPO nomeado (Art. 41)
- [ ] DPIA para processamento de alto risco (Art. 38)
- [ ] Direito à revisão de decisões automatizadas (Art. 20)
- [ ] Criptografia AES-256 at rest + TLS 1.3 in transit
- [ ] Dados em território brasileiro (AWS sa-east-1)
- [ ] Log de todas interações com IA
- [ ] Política de zero treinamento em dados de clientes

## OAB (Proposição 49.0000.2024.007325-9/COP, Nov 2024)
- [ ] Item 4.4.1: Formalizar por escrito intenção de usar IA ao cliente
- [ ] Item 4.4.3: Obter consentimento explícito ASSINADO
- [ ] Item 2.3: IA NÃO treinada com dados de clientes
- [ ] Item 3.7: Advogado REVISA todo output antes de protocolar
- [ ] Funcionalidade para gerar/rastrear/arquivar termos de consentimento

## CNJ Resolução 615/2025
- [ ] Classificação de risco: baixo (extração de dados) vs alto (análise comportamental)
- [ ] Tags de transparência em TODO conteúdo gerado por IA
- [ ] 12 meses para compliance (tribunais)
- [ ] Art. 23: modelos preditivos em matéria criminal DESENCORAJADOS
- [ ] Supervisão humana obrigatória

## PL 2338/2023 (Marco Legal da IA)
- [ ] IA jurídica = provavelmente ALTO RISCO
- [ ] Impact assessments
- [ ] Transparência sobre capabilities/limitations
- [ ] Human review mechanisms
- [ ] Audit trails completos
```

### AI Label Component

```python
# Sempre incluir em qualquer output de IA
AI_DISCLAIMER = """
⚠️ Conteúdo gerado com auxílio de inteligência artificial.
Conforme CNJ Resolução 615/2025 e recomendações da OAB (Proposição 49.0000.2024.007325-9/COP),
este conteúdo deve ser revisado por advogado habilitado antes de qualquer uso em processo judicial.
"""
```

---

## 21. OBSERVABILIDADE (LANGFUSE)

### Langfuse Setup

```python
"""
Langfuse: observabilidade completa para LLMs.
- Open-source, self-hostable, MIT license
- Tracing de inputs/outputs/tokens/latency/custos
- Prompt versioning
- LLM-as-judge evaluators
"""

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from app.config import settings

langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)

@observe()
async def traced_llm_call(messages, model, **kwargs):
    """All LLM calls are automatically traced."""
    langfuse_context.update_current_observation(
        metadata={
            "tenant_id": kwargs.get("tenant_id"),
            "case_id": kwargs.get("case_id"),
        }
    )
    # ... LLM call ...
```

---

## 22. DEPLOY E CI/CD

### Docker Compose (Dev Local)

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: ./apps/api
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres
      - redis
      - elasticsearch
      - qdrant
    volumes:
      - ./apps/api:/app

  web:
    build: ./apps/web
    ports:
      - "3000:3000"
    env_file: .env
    depends_on:
      - api

  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: jurisai
      POSTGRES_USER: jurisai
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.16.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    volumes:
      - esdata:/usr/share/elasticsearch/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrantdata:/qdrant/storage

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3001:3000"
    environment:
      DATABASE_URL: postgresql://jurisai:${DB_PASSWORD}@postgres:5432/langfuse
      NEXTAUTH_SECRET: ${SECRET_KEY}
      SALT: ${SALT}
    depends_on:
      - postgres

volumes:
  pgdata:
  esdata:
  qdrantdata:
```

---

## 23. ROADMAP DE IMPLEMENTAÇÃO

### Fase 1 — Fundação (Meses 1-3)

```
Semana 1-2: Setup
├── Repo, Docker Compose, CI/CD
├── PostgreSQL + pgvector + RLS
├── FastAPI scaffold + auth
└── Next.js scaffold + Vercel AI SDK

Semana 3-4: Chat Básico
├── Chat endpoint com SSE streaming
├── LiteLLM integration (DeepSeek V3.2/Claude/Qwen 3.5)
├── Conversation persistence
└── Basic chat UI

Semana 5-8: RAG v1
├── Elasticsearch setup + BM25
├── Voyage-law-2 embeddings
├── Qdrant vector search
├── Hybrid retrieval (RRF)
├── Hierarchical chunking
└── Document upload + OCR (PaddleOCR)

Semana 9-12: Core Features
├── Case management CRUD
├── Document classification (RoBERTaLexPT)
├── NER jurídico (BERTimbau + RegEx)
├── LangGraph multi-agent v1 (supervisor + research)
├── Basic Langfuse tracing
└── Tenant onboarding flow
```

### Fase 2 — Inteligência (Meses 4-6)

```
Semana 13-16: Memory & Agents
├── Mem0 integration (fact extraction)
├── LangGraph checkpointing (PostgresSaver)
├── Drafting agent
├── Analysis agent
├── Reranker (Jina-ColBERT-v2)
└── Citation verifier v1

Semana 17-20: Petition & Editor
├── Tiptap editor integration
├── Legal document templates
├── Multi-step petition generation
├── Citation verification UI
├── AI label compliance
└── Consent management (OAB)

Semana 21-24: Ingestion Pipeline
├── Airflow DAGs (DataJud, STJ, STF)
├── Deduplication (MinHash-LSH)
├── Incremental indexing
├── Querido Diário integration
└── LexML integration
```

### Fase 3 — Reasoning Model (Meses 7-9)

```
Semana 25-28: Data Collection
├── OAB exam dataset preparation
├── STF/STJ decision scraping
├── CoT generation via teacher model
├── Rejection sampling + quality filtering
└── Legal expert review (10-15% sample)

Semana 29-32: Training
├── SFT com LoRA no GAIA (Unsloth)
├── GRPO com reward functions jurídicas
├── OAB benchmark evaluation
├── Citation accuracy evaluation
└── A/B testing vs DeepSeek V3.2/Claude/Gemini 3 Pro

Semana 33-36: Deploy & Integration
├── Modal deploy (vLLM + GAIA)
├── RouteLLM classifier training
├── Tier routing integration
├── Speculative decoding setup
└── Continuous retraining pipeline
```

### Fase 4 — Escala (Meses 10-12)

```
Semana 37-40: Jurimetrics
├── Judge profiling system
├── Decision pattern analysis
├── XGBoost prediction models
├── Data Lawyer/Turivius API integration
└── Jurimetrics dashboard

Semana 41-44: Enterprise
├── Knowledge graph (Apache AGE → Neo4j)
├── Graphiti temporal knowledge
├── Advanced analytics dashboard
├── API for third-party integrations
└── SOC 2 preparation

Semana 45-48: Polish
├── Performance optimization
├── Load testing
├── Security audit
├── LGPD compliance audit
├── Documentation
└── Beta launch
```

---

## CUSTOS ESTIMADOS MENSAIS (PÓS-LAUNCH)

| Componente | Custo | Notas |
|-----------|-------|-------|
| Modal (GAIA inference 24/7) | $300-600 | L4, 1 container min |
| Modal (training mensal) | $60-150 | A100 48-120h |
| Voyage AI (embeddings) | $50-200 | Depende do volume |
| DeepSeek V3.2 API (Tier 2) | $15-80 | $0.28/$1.10 por M tokens |
| Qwen 3.5 / Kimi K2.5 (Tier 2 alt) | $10-60 | $0.30-0.50/M input |
| Claude Sonnet 4 (Tier 3) | $30-200 | Uso seletivo, complexo |
| Gemini 3 Pro (Tier 3 alt) | $20-150 | $1.25/$5.00/M, melhor value frontier |
| AWS sa-east-1 (infra) | $200-500 | RDS, EC2, S3 |
| Elasticsearch Cloud | $100-300 | Ou self-hosted |
| Langfuse Cloud | $0-100 | Ou self-hosted |
| **TOTAL** | **$785-2.340/mês** | Custo Tier 2 ~40% menor com DeepSeek V3.2 |

---

## REFERÊNCIAS CHAVE

### Papers
- DeepSeek-R1: arXiv:2501.12948 (pipeline completo de reasoning)
- DeepSeek V3.2: largo.dev/tutorials (frontier reasoning 6x menor custo, IMO/IOI gold medals 2025)
- GRPO: arXiv:2402.03300 (DeepSeekMath)
- Legal-R1: arXiv:2503.16040 (primeiro CoT jurídico via destilação)
- LegalEval-Q: arXiv:2505.24826 (benchmark qualidade texto legal — Qwen3 como optimal choice)
- VICTOR/STF: 45.532 processos, 692K documentos classificados
- JurisTCU: arXiv:2503.08379 (Summary-Augmented Chunking)
- RCP-Merging: arXiv:2508.03140 (merge reasoning + domain)

### Modelos (Fev/2026)
| Modelo | Custo (input/output por M tokens) | MMLU-Pro | Destaque |
|--------|-----------------------------------|----------|----------|
| DeepSeek V3.2 | $0.28 / $1.10 | 85.9% | Open-source MIT, 77.8% SWE-Bench |
| Qwen 3.5 | $0.50 / $2.00 | 84.6% | Apache 2.0, 397B MoE, 262K ctx |
| MiniMax M2.5 | $0.30 / $1.10 | — | 80.2% SWE-Bench, agentes |
| Kimi K2.5 | $0.45 / $2.20 | — | 1T params, líder agentic tasks |
| Gemini 3 Pro | $1.25 / $5.00 | 89.8% | Melhor value frontier |
| Claude Sonnet 4 | $3.00 / $15.00 | 88.2% | Melhor coding (72.5% SWE-Bench) |
| GPT-5.2 Pro | $$$ | 88.7% | Melhor overall, mais caro |

### Repos GitHub
- GAIA: `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it`
- Qwen 3.5: `QwenLM/Qwen3.5` (Apache 2.0)
- DeepSeek V3.2: `deepseek-ai/DeepSeek-V3` (MIT)
- TRL: `huggingface/trl` (GRPOTrainer)
- Unsloth: `unslothai/unsloth` (80% menos VRAM)
- Open-R1: `huggingface/open-r1`
- LangGraph: `langchain-ai/langgraph`
- mergekit: `arcee-ai/mergekit`
- RouteLLM: `lm-sys/RouteLLM`
- ABJ: `github.com/abjur` (109 repos jurimetria)
- LeNER-Br: NER jurídico brasileiro
- PaddleOCR: `PaddlePaddle/PaddleOCR`

### Datasets HuggingFace
- `eduagarcia/oab_exams` — questões OAB
- `joelniklaus/brazilian_court_decisions` — 4.043 decisões TJAL
- `unicamp-dl/BR-TaxQA-R` — QA tributário
- `pierreguillou/ner-bert-large-cased-pt-lenerbr` — NER model
- `open-r1/Mixture-of-Thoughts` — 350K reasoning traces

---

*Documento atualizado em Fevereiro 2026. Modelos e preços refletem o estado do mercado em Fev/2026.*

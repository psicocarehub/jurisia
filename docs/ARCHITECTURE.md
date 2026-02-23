# Arquitetura Juris.AI

Visão geral da arquitetura do sistema Juris.AI.

---

## Diagrama de Alto Nível

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
│                      DATA LAYER                                  │
│  PostgreSQL 16 (pgvector + RLS) │ Elasticsearch 8.16+             │
│  Redis 7+ (cache/sessions)      │ Qdrant (vector DB dedicado)     │
│  Neo4j/Apache AGE (knowledge)   │ S3/MinIO (docs originais)       │
└──────┬──────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                      LLM LAYER                                    │
│  GAIA 4B Fine-Tuned (vLLM no Modal) ← Tier 1 (70-80% queries)  │
│  DeepSeek R1 API                     ← Tier 2 (moderado)        │
│  Claude/GPT-4 API                    ← Tier 3 (complexo)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes

### API Gateway (FastAPI)
- Roteamento de requisições
- Autenticação JWT + multi-tenancy
- SSE streaming para chat
- Middleware de auditoria e tenant isolation

### LangGraph (Orquestração Multi-Agente)
- Supervisor que roteia para agentes especializados
- Agentes: research, drafting, analysis, memory
- Checkpointing PostgreSQL para estado persistente
- Human-in-the-loop quando necessário

### RAG Híbrido
- BM25 (Elasticsearch) para busca lexical
- Dense (Qdrant) para busca semântica
- RRF para fusão de resultados
- Jina-ColBERT-v2 para reranking

### OCR
- PaddleOCR-VL 1.5: alta precisão, selos/carimbos
- Surya + Marker: ordem de leitura, multi-coluna
- Azure Document Intelligence: fallback

### Memória 4-Tier
- Letta: memória de sessão (curto prazo)
- Mem0: fatos por usuário
- Graphiti: grafo de conhecimento (Neo4j/AGE)
- Checkpointer LangGraph: estado do agente

### Jurimetria
- Perfil de juízes
- Estatísticas por tribunal e área
- Preditor de resultado (XGBoost, guard para criminal)
- Integração Data Lawyer API

---

## Fluxo de Dados

1. **Chat**: Usuário → API → Supervisor → Agente → RAG/Memória → LLM → Resposta (SSE)
2. **Documentos**: Upload → S3 → Trigger OCR → Indexação Elasticsearch + Qdrant
3. **Petição**: Caso + templates → Agente drafting → Citation verifier → Formatter ABNT/OAB → AI label

---

## Isolamento Multi-Tenant

- `tenant_id` em todas as tabelas principais
- RLS no PostgreSQL por tenant
- Namespace em Graphiti/Mem0 por tenant
- Middleware injeta `tenant_id` do token JWT

---

## Infraestrutura

- **Dev**: Docker Compose local (volumes de código)
- **Prod**: Docker Compose sem volumes, resource limits, restart policies
- **GPU**: Modal (GAIA inference), RunPod (treinamento)
- **Hosting**: AWS sa-east-1 / Railway (LGPD: dados no Brasil)

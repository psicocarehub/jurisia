-- ============================================
-- Juris.AI — Complete Database Schema
-- ============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================
-- TENANTS
-- ============================================
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    plan VARCHAR(50) DEFAULT 'starter',
    settings JSONB DEFAULT '{}',
    lgpd_consent_template TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- USERS
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'lawyer',
    oab_number VARCHAR(20),
    hashed_password VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    ai_consent_given BOOLEAN DEFAULT FALSE,
    ai_consent_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, email)
);
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ============================================
-- CASES
-- ============================================
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    cnj_number VARCHAR(25),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    area VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    client_name VARCHAR(255),
    client_document VARCHAR(20),
    opposing_party VARCHAR(255),
    court VARCHAR(255),
    judge_name VARCHAR(255),
    judge_id UUID,
    estimated_value NUMERIC(15,2),
    metadata JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cases_tenant ON cases(tenant_id);
CREATE INDEX IF NOT EXISTS idx_cases_cnj ON cases(cnj_number);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);

-- ============================================
-- DOCUMENTS
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500) NOT NULL,
    doc_type VARCHAR(100),
    source VARCHAR(100),
    file_path VARCHAR(500),
    file_size INTEGER,
    mime_type VARCHAR(100),
    ocr_status VARCHAR(50) DEFAULT 'pending',
    ocr_text TEXT,
    ocr_confidence FLOAT,
    classification_label VARCHAR(100),
    classification_confidence FLOAT,
    ner_entities JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    uploaded_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_case ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_ocr_status ON documents(ocr_status);

-- ============================================
-- DOCUMENT CHUNKS
-- ============================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50),
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON document_chunks(tenant_id);

-- ============================================
-- CONVERSATIONS
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID NOT NULL REFERENCES users(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500),
    model_used VARCHAR(100),
    total_tokens INTEGER DEFAULT 0,
    total_cost NUMERIC(10,6) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);

-- ============================================
-- MESSAGES
-- ============================================
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    thinking TEXT,
    sources JSONB DEFAULT '[]',
    model_used VARCHAR(100),
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    feedback VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

-- ============================================
-- PETITIONS
-- ============================================
CREATE TABLE IF NOT EXISTS petitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(500) NOT NULL,
    petition_type VARCHAR(100),
    content TEXT,
    tiptap_json JSONB,
    status VARCHAR(50) DEFAULT 'draft',
    version INTEGER DEFAULT 1,
    citations JSONB DEFAULT '[]',
    citations_verified BOOLEAN DEFAULT FALSE,
    ai_generated BOOLEAN DEFAULT TRUE,
    ai_label TEXT DEFAULT 'Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025',
    created_by UUID REFERENCES users(id),
    reviewed_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_petitions_tenant ON petitions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_petitions_case ON petitions(case_id);
CREATE INDEX IF NOT EXISTS idx_petitions_status ON petitions(status);

-- ============================================
-- AUDIT LOGS
-- ============================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);

-- ============================================
-- JUDGE PROFILES (shared, no tenant isolation)
-- ============================================
CREATE TABLE IF NOT EXISTS judge_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    court VARCHAR(255),
    jurisdiction VARCHAR(255),
    total_decisions INTEGER DEFAULT 0,
    avg_decision_time_days FLOAT,
    favorability_rates JSONB DEFAULT '{}',
    common_citations JSONB DEFAULT '[]',
    decision_patterns JSONB DEFAULT '{}',
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_judges_name ON judge_profiles USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_judges_court ON judge_profiles(court);

-- ============================================
-- INGESTION LOG
-- ============================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(100) NOT NULL,
    records_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'completed',
    error_message TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- KNOWLEDGE GRAPH (simplified with PostgreSQL jsonb)
-- ============================================
CREATE TABLE IF NOT EXISTS kg_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(100) DEFAULT 'default',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_ns ON kg_nodes(namespace);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_content ON kg_nodes USING gin(to_tsvector('portuguese', content));

CREATE TABLE IF NOT EXISTS kg_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    relation VARCHAR(200) DEFAULT 'related',
    namespace VARCHAR(100) DEFAULT 'default',
    metadata JSONB DEFAULT '{}',
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_ns ON kg_edges(namespace);

-- ============================================
-- ALERTS (legislative/jurisprudential changes)
-- ============================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    change_type VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    affected_law TEXT,
    affected_articles TEXT[] DEFAULT '{}',
    new_law_reference TEXT,
    areas TEXT[] DEFAULT '{}',
    severity VARCHAR(20) DEFAULT 'medium',
    source_url TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    tenant_id UUID,
    user_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_tenant ON alerts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_areas ON alerts USING GIN(areas);
CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);

-- ============================================
-- ALERT SUBSCRIPTIONS
-- ============================================
CREATE TABLE IF NOT EXISTS alert_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    areas TEXT[] DEFAULT '{}',
    change_types TEXT[] DEFAULT '{}',
    min_severity VARCHAR(20) DEFAULT 'medium',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alert_sub_user ON alert_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_sub_tenant ON alert_subscriptions(tenant_id);

-- ============================================
-- FEEDBACK LOG (user feedback on AI responses)
-- ============================================
CREATE TABLE IF NOT EXISTS feedback_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_type VARCHAR(30) NOT NULL,
    message_id VARCHAR(100),
    conversation_id VARCHAR(100),
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    original_response TEXT,
    edited_response TEXT,
    source_ids TEXT[] DEFAULT '{}',
    query TEXT,
    area VARCHAR(100),
    comment TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback_log(user_id);
CREATE INDEX IF NOT EXISTS idx_fb_type ON feedback_log(feedback_type);
CREATE INDEX IF NOT EXISTS idx_fb_processed ON feedback_log(processed);
CREATE INDEX IF NOT EXISTS idx_fb_tenant ON feedback_log(tenant_id);

-- ============================================
-- SOURCE QUALITY SCORES (RAG reranking feedback)
-- ============================================
CREATE TABLE IF NOT EXISTS source_quality_scores (
    source_id VARCHAR(200) PRIMARY KEY,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 0.5,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- LAW ARTICLE VERSIONS (temporal versioning)
-- ============================================
CREATE TABLE IF NOT EXISTS law_article_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    law_name VARCHAR(500) NOT NULL,
    article VARCHAR(100) NOT NULL,
    text_content TEXT NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    status VARCHAR(30) DEFAULT 'vigente',
    source_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_law_versions_law ON law_article_versions(law_name);
CREATE INDEX IF NOT EXISTS idx_law_versions_article ON law_article_versions(article);
CREATE INDEX IF NOT EXISTS idx_law_versions_status ON law_article_versions(status);

-- ============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE petitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_edges ENABLE ROW LEVEL SECURITY;

-- Allow all access via anon/service role (permissive for dev; tighten for prod)
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_tenants') THEN
    CREATE POLICY "allow_all_tenants" ON tenants FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_users') THEN
    CREATE POLICY "allow_all_users" ON users FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_cases') THEN
    CREATE POLICY "allow_all_cases" ON cases FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_documents') THEN
    CREATE POLICY "allow_all_documents" ON documents FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_chunks') THEN
    CREATE POLICY "allow_all_chunks" ON document_chunks FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_conversations') THEN
    CREATE POLICY "allow_all_conversations" ON conversations FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_messages') THEN
    CREATE POLICY "allow_all_messages" ON messages FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_petitions') THEN
    CREATE POLICY "allow_all_petitions" ON petitions FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_audit') THEN
    CREATE POLICY "allow_all_audit" ON audit_logs FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_alerts') THEN
    CREATE POLICY "allow_all_alerts" ON alerts FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_alert_subs') THEN
    CREATE POLICY "allow_all_alert_subs" ON alert_subscriptions FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_feedback') THEN
    CREATE POLICY "allow_all_feedback" ON feedback_log FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_kg_nodes') THEN
    CREATE POLICY "allow_all_kg_nodes" ON kg_nodes FOR ALL USING (true);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'allow_all_kg_edges') THEN
    CREATE POLICY "allow_all_kg_edges" ON kg_edges FOR ALL USING (true);
  END IF;
END $$;

-- Public tables (no RLS needed)
-- judge_profiles, ingestion_log, source_quality_scores, law_article_versions, content_updates

-- ============================================
-- CONTENT UPDATES (unified feed for all new legal content)
-- ============================================
CREATE TABLE IF NOT EXISTS content_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(100),
    title TEXT NOT NULL,
    summary TEXT,
    content_preview TEXT,
    areas TEXT[] DEFAULT '{}',
    court_or_organ VARCHAR(200),
    territory VARCHAR(200),
    publication_date DATE,
    source_url TEXT,
    relevance_score FLOAT DEFAULT 0.5,
    is_verified BOOLEAN DEFAULT FALSE,
    verification_details JSONB DEFAULT '{}',
    document_id UUID,
    metadata JSONB DEFAULT '{}',
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT valid_category CHECK (category IN (
        'legislacao', 'jurisprudencia', 'doutrina', 'normativo', 'parecer', 'sumula', 'outro'
    ))
);
CREATE INDEX IF NOT EXISTS idx_cu_captured ON content_updates(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_cu_category ON content_updates(category);
CREATE INDEX IF NOT EXISTS idx_cu_source ON content_updates(source);
CREATE INDEX IF NOT EXISTS idx_cu_areas ON content_updates USING GIN(areas);
CREATE INDEX IF NOT EXISTS idx_cu_territory ON content_updates(territory);
CREATE INDEX IF NOT EXISTS idx_cu_pub_date ON content_updates(publication_date DESC);
CREATE INDEX IF NOT EXISTS idx_cu_title ON content_updates USING gin(to_tsvector('portuguese', title));
CREATE INDEX IF NOT EXISTS idx_cu_relevance ON content_updates(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_cu_feed ON content_updates(captured_at DESC, category);

-- ============================================
-- SEED: Default tenant for development
-- ============================================
INSERT INTO tenants (id, name, slug, plan)
VALUES ('00000000-0000-0000-0000-000000000001', 'JurisAI Dev', 'jurisai-dev', 'professional')
ON CONFLICT (id) DO NOTHING;

INSERT INTO users (id, tenant_id, email, name, role, is_active, ai_consent_given)
VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    'dev@jurisai.com.br',
    'Desenvolvedor',
    'admin',
    TRUE,
    TRUE
)
ON CONFLICT DO NOTHING;

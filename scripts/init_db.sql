-- Juris.AI Database Initialization
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================
-- ROW LEVEL SECURITY HELPER
-- ============================================

CREATE OR REPLACE FUNCTION set_tenant(tenant_uuid UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant_id', tenant_uuid::TEXT, true);
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- JUDGE PROFILES (no tenant isolation — shared data)
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
-- LANGFUSE DATABASE
-- ============================================

CREATE DATABASE langfuse;

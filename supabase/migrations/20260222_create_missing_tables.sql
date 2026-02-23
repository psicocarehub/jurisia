-- Knowledge Graph tables
CREATE TABLE IF NOT EXISTS kg_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR(100) DEFAULT 'default',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_ns ON kg_nodes(namespace);

CREATE TABLE IF NOT EXISTS kg_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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

-- Alerts
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

-- Alert Subscriptions
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

-- Feedback Log
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

-- Source Quality Scores
CREATE TABLE IF NOT EXISTS source_quality_scores (
    source_id VARCHAR(200) PRIMARY KEY,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 0.5,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Law Article Versions
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

-- RLS
ALTER TABLE kg_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback_log ENABLE ROW LEVEL SECURITY;

-- Policies (permissive for dev)
CREATE POLICY "allow_all_kg_nodes" ON kg_nodes FOR ALL USING (true);
CREATE POLICY "allow_all_kg_edges" ON kg_edges FOR ALL USING (true);
CREATE POLICY "allow_all_alerts" ON alerts FOR ALL USING (true);
CREATE POLICY "allow_all_alert_subs" ON alert_subscriptions FOR ALL USING (true);
CREATE POLICY "allow_all_feedback" ON feedback_log FOR ALL USING (true);

"""Create missing tables in Supabase using the Python SDK."""
import os
os.environ.setdefault("SUPABASE_URL", "https://hpyorwegajcxdzkfywmi.supabase.co")
os.environ.setdefault(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhweW9yd2VnYWpjeGR6a2Z5d21pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4MTA1MTMsImV4cCI6MjA4NzM4NjUxM30.FiBJcnFTkXq4tJbowkPd2W-_UcRFUGrr2vK3WuHohhg",
)

from supabase import create_client

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]

supabase = create_client(url, key)

STATEMENTS = [
    # kg_nodes
    """CREATE TABLE IF NOT EXISTS kg_nodes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        namespace VARCHAR(100) DEFAULT 'default',
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}',
        ts TIMESTAMPTZ DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_ns ON kg_nodes(namespace)",
    # kg_edges
    """CREATE TABLE IF NOT EXISTS kg_edges (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        target_id UUID NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        relation VARCHAR(200) DEFAULT 'related',
        namespace VARCHAR(100) DEFAULT 'default',
        metadata JSONB DEFAULT '{}',
        ts TIMESTAMPTZ DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_ns ON kg_edges(namespace)",
    # alerts
    """CREATE TABLE IF NOT EXISTS alerts (
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
    )""",
    "CREATE INDEX IF NOT EXISTS idx_alerts_tenant ON alerts(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_areas ON alerts USING GIN(areas)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)",
    # alert_subscriptions
    """CREATE TABLE IF NOT EXISTS alert_subscriptions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL,
        tenant_id UUID NOT NULL,
        areas TEXT[] DEFAULT '{}',
        change_types TEXT[] DEFAULT '{}',
        min_severity VARCHAR(20) DEFAULT 'medium',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_alert_sub_user ON alert_subscriptions(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_alert_sub_tenant ON alert_subscriptions(tenant_id)",
    # feedback_log
    """CREATE TABLE IF NOT EXISTS feedback_log (
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
    )""",
    "CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback_log(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_fb_type ON feedback_log(feedback_type)",
    "CREATE INDEX IF NOT EXISTS idx_fb_processed ON feedback_log(processed)",
    "CREATE INDEX IF NOT EXISTS idx_fb_tenant ON feedback_log(tenant_id)",
    # source_quality_scores
    """CREATE TABLE IF NOT EXISTS source_quality_scores (
        source_id VARCHAR(200) PRIMARY KEY,
        positive_count INTEGER DEFAULT 0,
        negative_count INTEGER DEFAULT 0,
        quality_score FLOAT DEFAULT 0.5,
        last_updated TIMESTAMPTZ DEFAULT NOW()
    )""",
    # law_article_versions
    """CREATE TABLE IF NOT EXISTS law_article_versions (
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
    )""",
    "CREATE INDEX IF NOT EXISTS idx_law_versions_law ON law_article_versions(law_name)",
    "CREATE INDEX IF NOT EXISTS idx_law_versions_article ON law_article_versions(article)",
    "CREATE INDEX IF NOT EXISTS idx_law_versions_status ON law_article_versions(status)",
]

RLS_STATEMENTS = [
    "ALTER TABLE kg_nodes ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE kg_edges ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE alerts ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE alert_subscriptions ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE feedback_log ENABLE ROW LEVEL SECURITY",
]

POLICY_STATEMENTS = [
    ("allow_all_kg_nodes", "kg_nodes"),
    ("allow_all_kg_edges", "kg_edges"),
    ("allow_all_alerts", "alerts"),
    ("allow_all_alert_subs", "alert_subscriptions"),
    ("allow_all_feedback", "feedback_log"),
]


def main():
    print("Connecting to Supabase...")
    print(f"URL: {url}")

    # Try using supabase.rpc or postgrest
    # The supabase Python SDK doesn't support raw SQL directly,
    # so we'll try using the postgrest-py execute method
    try:
        # Test connection
        result = supabase.table("tenants").select("id").limit(1).execute()
        print(f"Connection OK. Tenants found: {len(result.data)}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Try rpc call for SQL execution
    for i, sql in enumerate(STATEMENTS):
        try:
            supabase.rpc("exec_sql", {"query": sql}).execute()
            print(f"  [{i+1}/{len(STATEMENTS)}] OK")
        except Exception as e:
            print(f"  [{i+1}/{len(STATEMENTS)}] FAILED: {e}")

    print("\nSetting up RLS...")
    for sql in RLS_STATEMENTS:
        try:
            supabase.rpc("exec_sql", {"query": sql}).execute()
            print(f"  RLS OK: {sql[:60]}...")
        except Exception as e:
            print(f"  RLS FAILED: {e}")

    print("\nCreating policies...")
    for name, table in POLICY_STATEMENTS:
        try:
            sql = f'CREATE POLICY "{name}" ON {table} FOR ALL USING (true)'
            supabase.rpc("exec_sql", {"query": sql}).execute()
            print(f"  Policy OK: {name}")
        except Exception as e:
            err_str = str(e)
            if "already exists" in err_str:
                print(f"  Policy exists: {name}")
            else:
                print(f"  Policy FAILED ({name}): {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()

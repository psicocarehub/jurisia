-- Content Updates: unified table for all new legal content captured by ingestion pipelines.
-- Powers the "Novidades" portal and cross-source feed.

CREATE TABLE IF NOT EXISTS content_updates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
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

-- Composite index for the main feed query (date range + category)
CREATE INDEX IF NOT EXISTS idx_cu_feed ON content_updates(captured_at DESC, category);

-- Deduplication: prevent duplicate entries from the same source
CREATE UNIQUE INDEX IF NOT EXISTS idx_cu_dedup
  ON content_updates(source, md5(title), publication_date)
  WHERE publication_date IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_cu_dedup_null_date
  ON content_updates(source, md5(title))
  WHERE publication_date IS NULL;

-- Relevance score must be between 0 and 1
ALTER TABLE content_updates ADD CONSTRAINT valid_relevance
  CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0);

-- Partial index for cleanup operations
CREATE INDEX IF NOT EXISTS idx_cu_cleanup
  ON content_updates(captured_at) WHERE relevance_score < 0.7;

-- Index for unverified items
CREATE INDEX IF NOT EXISTS idx_cu_unverified
  ON content_updates(captured_at DESC) WHERE is_verified = FALSE;

-- User bookmarks for content updates
CREATE TABLE IF NOT EXISTS user_bookmarks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    content_update_id UUID NOT NULL REFERENCES content_updates(id) ON DELETE CASCADE,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, content_update_id)
);

CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON user_bookmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_bookmarks_tenant ON user_bookmarks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_bookmarks_content ON user_bookmarks(content_update_id);

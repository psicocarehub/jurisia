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

-- State tracking tables for regional and specialized ingestion DAGs

CREATE TABLE IF NOT EXISTS ingestion_diarios_state (
    territory_id VARCHAR(20) PRIMARY KEY,
    last_date DATE,
    last_ingested_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ingestion_receita_state (
    source VARCHAR(100) PRIMARY KEY,
    last_date DATE,
    last_ingested_at TIMESTAMPTZ
);

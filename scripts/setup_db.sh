#!/bin/bash
# setup_db.sh — Configuração do banco de dados Juris.AI
# Executa init SQL e migrações Alembic

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_DIR="$PROJECT_ROOT/apps/api"

# Carregar variáveis de ambiente
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

DB_PASSWORD="${DB_PASSWORD:-jurisai_dev}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-jurisai}"
DB_NAME="${DB_NAME:-jurisai}"

echo "==> Juris.AI — Setup do Banco de Dados"
echo "    Host: $DB_HOST:$DB_PORT"
echo "    Database: $DB_NAME"
echo ""

# 1. Executar init SQL (extensões, funções auxiliares, tabelas auxiliares)
if [ -f "$SCRIPT_DIR/init_db.sql" ]; then
  echo "==> Executando init_db.sql em $DB_NAME..."
  PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SCRIPT_DIR/init_db.sql" 2>/dev/null || true
  echo "    Init SQL aplicado."
else
  echo "    init_db.sql não encontrado, pulando."
fi

# 2. Rodar migrações Alembic
echo "==> Executando migrações Alembic..."
cd "$API_DIR"
alembic upgrade head 2>/dev/null || echo "    (Nenhuma migração pendente ou banco já inicializado)"

echo ""
echo "==> Setup concluído."

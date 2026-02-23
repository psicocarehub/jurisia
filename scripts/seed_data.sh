#!/bin/bash
# seed_data.sh — Popula o banco com dados iniciais para desenvolvimento

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_DIR="$PROJECT_ROOT/apps/api"

echo "==> Juris.AI — Seed de Dados"

# Executar seed Python
cd "$API_DIR"
python -c "
import asyncio
from app.db.seed import seed

asyncio.run(seed())
"

echo "==> Seed concluído."
echo "    Usuários de demonstração:"
echo "    - admin@demo.juris.ai / admin123"
echo "    - advogado@demo.juris.ai / lawyer123"
echo "    Tenant: demo"

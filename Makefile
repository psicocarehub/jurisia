.PHONY: dev dev-api dev-web up down logs migrate seed test lint

dev: up
	@echo "All services running. API: http://localhost:8000 | Web: http://localhost:3000"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

dev-api:
	cd apps/api && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-web:
	cd apps/web && npm run dev

migrate:
	cd apps/api && alembic upgrade head

migrate-create:
	cd apps/api && alembic revision --autogenerate -m "$(msg)"

seed:
	cd apps/api && python -m app.db.seed

test:
	cd apps/api && pytest -v
	cd apps/web && npm test

test-api:
	cd apps/api && pytest -v

lint:
	cd apps/api && ruff check .
	cd apps/web && npm run lint

format:
	cd apps/api && ruff format .

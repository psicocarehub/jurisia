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

# ============================================
# Training Pipeline
# ============================================

collect-oab:
	python -m training.data.collect_oab -o training/data/questions_oab.jsonl

collect-stj:
	python -m training.data.collect_stj_scenarios -o training/data/questions_stj.jsonl -p training/data/tribunal_patterns.json

generate-scenarios:
	python -m training.data.generate_scenarios -o training/data/questions_scenarios.jsonl -n 5000 --personas

merge-questions:
	cat training/data/questions_oab.jsonl training/data/questions_stj.jsonl training/data/questions_scenarios.jsonl > training/data/questions.jsonl
	wc -l training/data/questions.jsonl

generate-cot-simple:
	python -m training.data.generate_cot -q training/data/questions.jsonl -o training/data/raw_traces.jsonl --mode simple

generate-cot-debate:
	python -m training.data.generate_cot -q training/data/questions.jsonl -o training/data/raw_traces.jsonl --mode debate

filter-traces:
	python -m training.data.filter_quality -i training/data/raw_traces.jsonl -o training/data/filtered_traces.jsonl -r training/data/quality_report.json

prepare-dataset:
	python -m training.data.prepare_dataset -i training/data/filtered_traces.jsonl -o training/data/sft_dataset

collect-personas:
	python -c "from datasets import load_dataset; ds=load_dataset('nvidia/Nemotron-Personas-Brazil', split='train'); print(f'{len(ds)} personas cached')"

collect-pretraining:
	python -m training.data.collect_pretraining -o training/data/pretraining_corpus

pretrain-gaia:
	python -m training.pretraining.continued_pretrain --corpus-dir training/data/pretraining_corpus --output-dir training/checkpoints/pretrained

train-pipeline: collect-oab collect-stj generate-scenarios merge-questions generate-cot-debate filter-traces prepare-dataset
	@echo "Pipeline completo! Dataset pronto em training/data/sft_dataset/"

full-pipeline: collect-pretraining pretrain-gaia train-pipeline
	@echo "Pipeline completo: pre-training + SFT"

.PHONY: install dq pipeline train build run test lint up down genai

install:
	pip install -r requirements.txt

## Standalone DQ check -- runs against raw CSVs directly, no DB needed.
dq:
	python dq/dq_check.py --input data/raw --output reports/dq_report.json

## raw -> curated -> serving pipeline (idempotent).
pipeline:
	python pipeline/build_pipeline.py --raw data/raw --db data/warehouse.duckdb

## Train and save the delay-risk model artefact.
train:
	python ml/train_model.py --db data/warehouse.duckdb --out ml/model.joblib --metrics-out ml/metrics.json

## Full local build: DQ report + warehouse + model, in the correct order.
build: dq pipeline train

## Run the API locally (expects `make build` to have been run at least once).
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	ruff check .

## Full containerised stack: builds data, then starts the API.
up:
	docker compose up --build

down:
	docker compose down -v

## Example: make genai Q="What is the on-time rate for NLRTM to PKKAR?"
genai:
	python genai/assistant.py "$(Q)"

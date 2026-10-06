.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
UV          := uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the virtualenv and install runtime + dev dependencies
	$(UV) venv $(VENV) --allow-existing
	$(UV) pip install -e . --python $(PY)
	$(UV) pip install --group dev --python $(PY)

.PHONY: install-local-embeddings
install-local-embeddings: ## Add the optional sentence-transformers extra
	$(UV) pip install -e ".[local-embeddings]" --python $(PY)

.PHONY: env
env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "created .env")

.PHONY: run
run: ## Run the API locally with autoreload
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: test
test: ## Run the test suite
	$(VENV)/bin/pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage
	$(VENV)/bin/pytest --cov=app --cov-report=term-missing --cov-report=xml

.PHONY: lint
lint: ## Lint with ruff
	$(VENV)/bin/ruff check app tests

.PHONY: format
format: ## Auto-fix lint issues and format
	$(VENV)/bin/ruff check --fix app tests
	$(VENV)/bin/ruff format app tests

.PHONY: typecheck
typecheck: ## Static type checking with mypy
	$(VENV)/bin/mypy app

.PHONY: check
check: lint typecheck test ## Run the full quality gate

.PHONY: up
up: ## Start the full stack (api, qdrant, redis, postgres)
	docker compose up --build -d

.PHONY: down
down: ## Stop the stack and remove volumes
	docker compose down -v

.PHONY: logs
logs: ## Tail the API logs
	docker compose logs -f api

.PHONY: smoke
smoke: ## Health-check the running stack
	curl -fsS http://localhost:8000/api/v1/health | python3 -m json.tool

.PHONY: openapi
openapi: ## Print the generated OpenAPI schema
	curl -fsS http://localhost:8000/openapi.json | python3 -m json.tool

.PHONY: install fmt lint typecheck test check paper dashboard

install:
	uv sync

fmt:
	uv run ruff format . && uv run ruff check --fix .

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest -q --cov --cov-report=term-missing

check: lint typecheck test

paper:
	uv run fund run --mode paper

dashboard:
	uv run fund dashboard --out site

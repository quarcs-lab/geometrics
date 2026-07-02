# Development tasks for geometrics. Everything runs through uv.

.PHONY: sync test test-network lint format typecheck notebooks llms docs build

sync:
	uv sync --locked --all-extras --group dev --group docs

test:
	uv run pytest -n auto --cov=geometrics --cov-report=term-missing -m "not network"

test-network:
	uv run pytest -m network

lint:
	uv run ruff check src tests tools
	uv run ruff format --check src tests tools

format:
	uv run ruff format src tests tools
	uv run ruff check --fix src tests tools

typecheck:
	uv run mypy src

notebooks:
	uv run python tools/build_quickstart_notebook.py

llms:
	uv run python tools/build_llms_txt.py

docs: notebooks
	uv run quartodoc build --config docs/_quarto.yml
	uv run python -m ipykernel install --user --name geometrics
	uv run quarto render docs
	uv run python tools/build_llms_txt.py

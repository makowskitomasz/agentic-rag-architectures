.PHONY: help sync test lint

help:
	@echo "Available commands:"
	@echo "  make sync       - Sync uv environment"
	@echo "  make test       - Run unit tests"
	@echo "  make lint       - Run ruff"

sync:
	uv sync

lint:
	uv run ruff check . --output-format=full --fix

test:
	uv run pytest -v

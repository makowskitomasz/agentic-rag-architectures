ENV_NAME=ara
ENV_FILE=environment.yml

.PHONY: help env update activate run test clean lint notebooks

help:
	@echo "Available commands:"
	@echo "  make env        - Create Conda environment"
	@echo "  make update     - Update Conda environment"
	@echo "  make activate   - Show activation command"
	@echo "  make test       - Run unit tests"
	@echo "  make lint       - Run ruff"

env:
	conda env create -f $(ENV_FILE)

update:
	conda env update --file $(ENV_FILE) --prune

activate:
	@echo "To activate the environment, run:"
	@echo "  conda activate $(ENV_NAME)"

lint:
	ruff check . --output-format=full --fix

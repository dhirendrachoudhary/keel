.PHONY: help dev test lint fmt typecheck check

help:
	@echo "make dev        install the package + dev tools (editable)"
	@echo "make test       run the test suite"
	@echo "make lint       ruff lint"
	@echo "make fmt        ruff format"
	@echo "make typecheck  mypy src/keel"
	@echo "make check      lint + typecheck + test (what CI runs)"

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy src/keel

check: lint typecheck test

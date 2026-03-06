.PHONY: setup test test-cov lint format docs-serve docs-build build clean

setup:
	uv sync --all-extras
	uv run pre-commit install
	uv run nbstripout --install
	uv run nbdime config-git --enable

test:
	uv run pytest

test-cov:
	uv run pytest --cov=xaitimesynth --cov-report=term-missing

lint:
	uv run ruff check xaitimesynth/ tests/

format:
	uv run ruff format xaitimesynth/ tests/

docs-serve:
	JUPYTER_PLATFORM_DIRS=1 uv run mkdocs serve

docs-build:
	JUPYTER_PLATFORM_DIRS=1 uv run mkdocs build

build:
	uv build

clean:
	rm -rf dist/ build/ site/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

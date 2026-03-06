# Development Setup

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Clone and Install

```bash
git clone https://github.com/gregorbaer/xaitimesynth.git
cd xaitimesynth
```

Install all dependencies (runtime + dev + test + docs + notebooks):

```bash
make setup
```

This runs `uv sync --all-extras` and sets up pre-commit hooks, nbstripout, and nbdime.

## Common Commands

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Auto-format
make format

# Serve docs locally (localhost:8000)
make docs-serve

# Build docs
make docs-build

# Build package (wheel + sdist)
make build
```

## Project Structure

```
xaitimesynth/
├── xaitimesynth/      # Main package source
├── tests/             # Test suite
├── docs/              # Documentation source
├── notebooks/         # Example Jupyter notebooks
├── scripts/           # Utility scripts
├── pyproject.toml     # Package configuration
└── Makefile           # Development commands
```

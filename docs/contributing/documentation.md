# Documentation

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/) to build API documentation from docstrings.

## Setup

Install documentation dependencies:

```bash
uv sync --extra docs
```

## Common Commands

```bash
# Preview locally with live reload at localhost:8000
make docs-serve

# Build static site to site/ folder
make docs-build
```

## Structure

```
docs/
├── index.md              # Home page
├── guides/               # Conceptual guides
└── api/                  # Auto-generated API reference
    ├── builder.md
    ├── components.md
    ├── visualization.md
    ├── metrics.md
    └── registry.md
mkdocs.yml                # MkDocs configuration
```

The `api/*.md` files use mkdocstrings directives (e.g. `::: xaitimesynth.builder.TimeSeriesBuilder`) that are automatically expanded into full API documentation from docstrings at build time.

## Hosting on GitHub Pages

Documentation is automatically deployed to GitHub Pages on every push to `main` via the `.github/workflows/docs.yml` workflow.

The docs are served at [gregorbaer.github.io/xaitimesynth](https://gregorbaer.github.io/xaitimesynth/).

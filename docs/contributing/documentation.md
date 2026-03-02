# Documentation

This project uses [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/) to build API documentation from docstrings.

## Setup

Install documentation dependencies:

```bash
uv pip install mkdocs mkdocs-material "mkdocstrings[python]"
```

## Common Commands

```bash
# Preview locally with live reload at localhost:8000
.venv/bin/mkdocs serve

# Build static site to site/ folder
.venv/bin/mkdocs build

# Deploy to GitHub Pages (pushes to gh-pages branch)
.venv/bin/mkdocs gh-deploy
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

### Manual deploy

Run `mkdocs gh-deploy` from your local machine. This builds the site and pushes it to the `gh-pages` branch, which GitHub serves at `https://<your-username>.github.io/xaitimesynth`.

Before deploying, update `site_url` and `repo_url` in `mkdocs.yml` with your actual GitHub repository URL.

### Automated deploy via GitHub Actions

Create `.github/workflows/docs.yml` to deploy automatically on every push to `main`:

```yaml
name: Deploy documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install mkdocs mkdocs-material "mkdocstrings[python]"
      - run: mkdocs gh-deploy --force
```

!!! note
    GitHub Pages requires a **public repository** on the free GitHub plan. Private repositories require a paid plan (Pro/Team/Enterprise).

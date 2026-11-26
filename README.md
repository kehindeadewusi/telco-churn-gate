# Telco Churn Gate

Predicting customer churn for a telecom operator: identifying which customers are likely to close their subscription.

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-purple)
![Ruff](https://img.shields.io/badge/code%20style-ruff-black)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)

## Table of contents

- [Overview](#overview)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [Project activity](#project-activity)

## Overview

**Churn** is when customers leave, closing their subscription. This project explores telecom customer data and builds an end-to-end pipeline to predict churn.

Developed as the group project for the Enterprise Data Science Bootcamp (EDSB26, Group 3).

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python and dependencies)
- [VS Code](https://code.visualstudio.com/) with the Python extension (recommended)

### Installation

```bash
uv sync                     # create the virtual environment (.venv) from pyproject.toml
uv run pre-commit install   # enable the formatting and notebook-sync hooks
code .
```

## Usage

All code lives in plain `.py` files. Blocks separated by `# %%` markers can be run cell by cell in VS Code (**Run Cell**), just like a Jupyter notebook.

Run a full script from the terminal:

```bash
uv run python scripts/sandbox.py
```

To generate the `.ipynb` version of a script in `scripts/`:

```bash
uv run jupytext --sync scripts/<name>.py
```

## Project structure

```text
telco-churn-gate/
├── telco/                    # Project source code (end-to-end pipeline)
├── scripts/                  # Exploratory analysis scripts (paired with notebooks)
├── notebooks/                # Local .ipynb copies (git-ignored)
├── tests/                    # Automated tests
├── data/ models/ reports/ figures/   # Generated outputs (git-ignored)
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── jupytext.toml             # Pairing between notebooks/*.ipynb and scripts/*.py
└── .pre-commit-config.yaml   # Hooks run on every commit
```

| Path | Description |
|---|---|
| `pyproject.toml` | Python version and package dependencies |
| `uv.lock` | Exact dependency versions, so everyone gets the same environment. After changing dependencies, run `uv sync` or `uv add <pkg>` |
| `.venv/` | Local virtual environment. Git-ignored; do not commit |
| `scripts/` | Exploratory analysis code, committed to the repo |
| `notebooks/` | Local scratch notebooks. Git-ignored; the committed version is the paired `.py` file in `scripts/` |
| `jupytext.toml` | Keeps `notebooks/*.ipynb` and `scripts/*.py` in sync |
| `.pre-commit-config.yaml` | Formats code and syncs notebooks on every commit |
| `telco/` | Main project code |
| `tests/` | Automated tests |
| `data/`, `models/`, `reports/`, `figures/` | Generated outputs. Git-ignored; large files are never committed |

## Contributing

1. Create a branch for your change.
2. Commit as usual. `git commit` runs the pre-commit hooks automatically; if a hook fails, the output explains what to fix (e.g. a lint error or an unformatted file).
3. Open a pull request into `main`.

Run all checks manually at any time:

```bash
uv run pre-commit run -a
```

## Project activity

![Commits last 30 days](https://img.shields.io/github/commit-activity/m/kehindeadewusi/telco-churn-gate)
![Last commit](https://img.shields.io/github/last-commit/kehindeadewusi/telco-churn-gate)
![Open PRs](https://img.shields.io/github/issues-pr/kehindeadewusi/telco-churn-gate)
![Closed PRs](https://img.shields.io/github/issues-pr-closed/kehindeadewusi/telco-churn-gate)
![Contributors](https://img.shields.io/github/contributors/kehindeadewusi/telco-churn-gate)

### Contributors

[![Contributors](https://contrib.rocks/image?repo=kehindeadewusi/telco-churn-gate)](https://github.com/kehindeadewusi/telco-churn-gate/graphs/contributors)

[Commits](https://github.com/kehindeadewusi/telco-churn-gate/commits) ·
[Pull requests](https://github.com/kehindeadewusi/telco-churn-gate/pulls) ·
[Pulse (last 30 days)](https://github.com/kehindeadewusi/telco-churn-gate/pulse/monthly) ·
[Contributor graph](https://github.com/kehindeadewusi/telco-churn-gate/graphs/contributors)

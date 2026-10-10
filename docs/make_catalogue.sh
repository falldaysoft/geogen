#!/bin/bash
# Regenerate the browsable catalogue in docs/catalogue/ (pages + thumbnails).
# Thumbnails are only re-rendered for assets whose YAML changed; pass `all` to
# redo every one (e.g. after changing a material or generator).
# Run from the repository root: docs/make_catalogue.sh [changed|all|none]
set -euo pipefail

PY=${PYTHON:-.venv/bin/python}
"$PY" -m geogen.main --describe --markdown docs/catalogue --thumbnails "${1:-changed}"

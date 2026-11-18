#!/usr/bin/env bash
# One-time local setup: corpus -> index -> graph -> triage -> manifests
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
scripts/download_models.sh
$PY scripts/quantize_models.py
$PY scripts/build_corpus.py
$PY scripts/build_index.py
PYTHONPATH=src $PY -m ua.graphify build ./src ./knowledge_vault --output graph.json
$PY scripts/train_triage.py
$PY scripts/sign_manifests.py

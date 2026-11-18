#!/usr/bin/env bash
# Dev-profile ONNX models (CPU). Spec-profile models (bge-large-en-v1.5, bge-reranker-v2-m3 FP16)
# are selected via config/profile paths; see design.md ADR-002.
set -euo pipefail
cd "$(dirname "$0")/.."
H=https://huggingface.co
dl() { [ -s "$2" ] || curl -sSL --fail -o "$2" "$1"; }
dl $H/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx models/bge-small-en-v1.5/model.onnx
dl $H/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json models/bge-small-en-v1.5/tokenizer.json
dl $H/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx models/ms-marco-MiniLM-L-6-v2/model.onnx
dl $H/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json models/ms-marco-MiniLM-L-6-v2/tokenizer.json
echo done

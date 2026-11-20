#!/usr/bin/env bash
# One-time setup: a virtualenv with CPU-only torch, the Python deps, and the
# distilgpt2 weights (fp16 shards, ~165 MB) into models/distilgpt2/.
# Safe to re-run: each step is skipped when already done.
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS_REPO="promptster-ai/compress-eval"
WEIGHTS_REV="019ec2d2a3b6685062f088073ba2610f5fec0606"  # the `weights` branch

[ -x .venv/bin/python ] || "${PYTHON:-python3}" -m venv .venv
# PyPI's Linux torch is the multi-GB CUDA build; the CPU wheel is ~190 MB.
.venv/bin/python -c "import torch" 2>/dev/null || \
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt

if ! ls models/distilgpt2/model-*.safetensors >/dev/null 2>&1; then
  mkdir -p models/distilgpt2
  curl -fsSL "https://codeload.github.com/${WEIGHTS_REPO}/tar.gz/${WEIGHTS_REV}" \
    | tar -xz --strip-components=1 -C models/distilgpt2
  rm -f models/distilgpt2/README.md  # the weights branch's own README
fi
echo "setup done"

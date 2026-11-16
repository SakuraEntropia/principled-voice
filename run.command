#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export HF_HOME="$PWD/.cache/huggingface"
if [[ ! -x .venv/bin/pv ]]; then
  uv sync --locked
fi
if [[ ! -f models/kokoro/kokoro-v1_0.pth ]]; then
  .venv/bin/python scripts/download_models.py
fi
.venv/bin/pv render examples/baseline.json --out "renders/manual_$(date +%Y%m%d_%H%M%S)"

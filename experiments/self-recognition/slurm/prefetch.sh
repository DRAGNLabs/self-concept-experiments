#!/bin/bash
# Run on a login node: completes the stage 1 models in the shared HF cache (a no-op for complete snapshots).
set -euo pipefail
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
source "$(dirname "${BASH_SOURCE[0]}")/../../../activate.sh"
for model in zai-org/GLM-4.5-Air-FP8 google/gemma-4-31B-it openai/gpt-oss-120b Qwen/Qwen3-32B; do
    python -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$model"
done

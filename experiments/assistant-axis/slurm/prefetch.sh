#!/bin/bash --login
# Run on a LOGIN node (needs internet), from the repo root: slurm/prefetch.sh is not
# needed -- run `bash experiments/assistant-axis/slurm/prefetch.sh`. Downloads the
# pipeline's model weights into the shared HF cache so compute nodes (which run
# offline) find them. google/gemma-2-27b-it is GATED: accept its license on
# HuggingFace and authenticate (`huggingface-cli login` or export HF_TOKEN) first,
# or its download fails with 401/403.
set -euo pipefail

mamba activate "$HOME/experiments/self-concept-experiments/.env"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"  # shared cache (matches byutils/CLAUDE.md)
mkdir -p "$HF_HOME"

for MODEL in \
    "google/gemma-2-27b-it" \
    "Qwen/Qwen2.5-7B-Instruct"; do
    echo "=== Prefetching $MODEL ==="
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$MODEL"
done

echo "=== Prefetch complete. HF_HOME=$HF_HOME ==="

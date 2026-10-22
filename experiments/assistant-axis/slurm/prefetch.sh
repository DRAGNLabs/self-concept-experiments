#!/bin/bash --login
# Run on a LOGIN node (needs internet), from the repo root: slurm/prefetch.sh is not
# needed -- run `bash experiments/assistant-axis/slurm/prefetch.sh`. Downloads the
# pipeline's model weights and datasets into the shared HF cache so compute nodes
# (which run offline) find them. google/gemma-2-27b-it and lmsys/lmsys-chat-1m are
# GATED: accept their licenses on HuggingFace and authenticate (`huggingface-cli login`
# or export HF_TOKEN) first, or their downloads fail with 401/403.
set -euo pipefail

source "$HOME/experiments/self-concept-experiments/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"  # shared cache (matches byutils/CLAUDE.md)
mkdir -p "$HF_HOME"

for MODEL in \
    "google/gemma-2-27b-it" \
    "Qwen/Qwen2.5-7B-Instruct" \
    "allenai/Olmo-3.1-32B-Think"; do
    echo "=== Prefetching $MODEL ==="
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$MODEL"
done

echo "=== Prefetching dataset lmsys/lmsys-chat-1m ==="
python -c "from byutils import load_dataset; load_dataset('lmsys/lmsys-chat-1m', split='train', allow_download=True)"

echo "=== Prefetch complete. HF_HOME=$HF_HOME ==="

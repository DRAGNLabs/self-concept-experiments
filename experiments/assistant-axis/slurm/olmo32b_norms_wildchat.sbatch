#!/bin/bash --login
#SBATCH --job-name=axis-norms-olmo-wildchat
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=128G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Per-layer mean post-MLP residual norms for OLMo-3.1-32B-Think, the steering scale of
# 6_steered_traces / 7_steered_lcb. Same code and settings as olmo32b_norms.sbatch
# (1000 English conversations, max_length 2048), but on allenai/WildChat-1M: lmsys-chat-1m
# is gated and this account is not on its access list. Both are real user-chatbot logs.
# Usage (from repo root): sbatch experiments/assistant-axis/slurm/olmo32b_norms_wildchat.sbatch OUT_DIR

OUT="${1:?pass the output directory}"
REPO="$(cd ../.. && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
source "$REPO/.venv-313/bin/activate"
nvidia-smi
mkdir -p "$OUT"

python scripts/steering/activation_norms.py --config configs/steering/activation_norms.yaml \
    --run.model "allenai/Olmo-3.1-32B-Think" --run.dataset allenai/WildChat-1M --run.num_conversations 1000 \
    --run.output "$OUT/layer_norms.pt"

echo "=== olmo32b wildchat norms done ==="

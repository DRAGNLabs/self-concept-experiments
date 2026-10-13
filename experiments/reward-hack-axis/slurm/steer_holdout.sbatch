#!/bin/bash --login
#SBATCH --job-name=rh-axis-steer-holdout
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/reward-hack-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Add the hack axis at layer 18 while gpt-oss-120b answers the held-out prompts,
# then grade each reply. Same launch as extract_response_mean.sbatch.
# 5 prompts x 2 coefficients x 8 samples. The ±0.5 run is already in the jsonl;
# this launch only adds ±0.1. A standby requeue resumes from the jsonl.
# Usage (from the repo root):
#   sbatch experiments/reward-hack-axis/slurm/steer_holdout.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
ATTENTION="kernels-community/vllm-flash-attn3"  # gpt-oss has no sdpa path; eager materializes every attention score
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis}"
COEFFICIENTS="${COEFFICIENTS:--0.1,0.1}"
SAMPLES="${SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

source ../../activate.sh

python scripts/steer_holdout.py \
    --model "$MODEL" \
    --attn-implementation "$ATTENTION" \
    --output-dir "$ROOT" \
    --coefficients="$COEFFICIENTS" \
    --samples "$SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS"

echo "=== steer holdout done: $ROOT/steered_holdout_summary.json ==="

#!/bin/bash --login
#SBATCH --job-name=rh-axis-response-mean
#SBATCH --time=04:00:00
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

# Mean post-MLP residual over every reply token, at the middle layer.
# Same launch as experiments/assistant-axis/slurm/gptoss120b_thinking.sbatch:
# gpt-oss-120b MXFP4 fits one H200, and the HF cache is already local.
# Usage (from the repo root):
#   sbatch experiments/reward-hack-axis/slurm/extract_response_mean.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

MODEL="openai/gpt-oss-120b"
ATTENTION="kernels-community/vllm-flash-attn3"  # gpt-oss has no sdpa path; eager materializes every attention score
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis}"

source ../../activate.sh

python scripts/extract_response_mean.py --model "$MODEL" --attn-implementation "$ATTENTION" --output-dir "$ROOT"

echo "=== response mean done: $ROOT/response_mean.pt ==="

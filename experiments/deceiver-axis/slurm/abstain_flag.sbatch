#!/bin/bash --login
#SBATCH --job-name=abstain-flag
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=cs
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail
# Abstain-flag logprobs on one arm's finished episodes (scripts/abstain_flag_logprobs.py).
# Usage (from the repo root): export MODEL DATA TAG RUN ARM SEEDS OUT CHAT_KWARGS [STEER_DIRECTION [STEERING_FROM]] [N] [MAX_LENGTH]
#   sbatch [--array=0-3] -p <partition> --export=ALL experiments/deceiver-axis/slurm/abstain_flag.sbatch
export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOO_CHAT_KWARGS="${CHAT_KWARGS:?}"
SEEDS="${SEEDS:?}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then  # an array task takes every n-th seed (n = array size)
    SEEDS="$(echo "$SEEDS" | tr ' ' '\n' | awk -v n="$SLURM_ARRAY_TASK_COUNT" -v i="$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))" '(NR - 1) % n == i' | tr '\n' ' ')"
fi
read -r -a SEED_LIST <<< "$SEEDS"
EXTRA=()
if [ -n "${STEER_DIRECTION:-}" ]; then EXTRA+=(--steer-direction "$STEER_DIRECTION"); fi
if [ -n "${STEERING_FROM:-}" ]; then EXTRA+=(--steering-from "$STEERING_FROM"); fi
if [ -n "${N:-}" ]; then EXTRA+=(--n "$N"); fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$SLURM_SUBMIT_DIR/.venv-313}"
source ../../activate.sh
export PYTHONPATH="$PWD/scripts:$PWD/../../src:${PYTHONPATH:-}"
python scripts/abstain_flag_logprobs.py --model "${MODEL:?}" --data "${DATA:?}" --tag "${TAG:?}" --run "${RUN:?}" --arm "${ARM:?}" \
    --seeds "${SEED_LIST[@]}" --max-length "${MAX_LENGTH:-40960}" "${EXTRA[@]}" --out "${OUT:?}"

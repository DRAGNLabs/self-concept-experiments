#!/bin/bash --login
#SBATCH --job-name=abstain-resample
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=cs
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
# EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.
set -euo pipefail
# Decision-turn resampling with the abstain instruction (scripts/abstain_resample.py), one arm per job.
# Usage (from the repo root): export MODEL DATA TAG SLUG ARM OUT CHAT_KWARGS [STEER_DIRECTION STEERING_FROM] [PREFIXES] [K]
#   sbatch -p <partition> --export=ALL experiments/deceiver-axis/slurm/abstain_resample.sbatch
export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS="${CHAT_KWARGS:?}"
EXTRA=()
if [ -n "${STEER_DIRECTION:-}" ]; then EXTRA+=(--steer-direction "$STEER_DIRECTION" --steering-from "${STEERING_FROM:?}"); fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$SLURM_SUBMIT_DIR/.venv-313}"
source ../../activate.sh
export PYTHONPATH="$PWD/scripts:$PWD/../../src:${PYTHONPATH:-}"
python scripts/abstain_resample.py --model "${MODEL:?}" --data "${DATA:?}" --tag "${TAG:?}" --slug "${SLUG:?}" --arm "${ARM:?}" \
    --prefixes "${PREFIXES:-20}" --k "${K:-8}" "${EXTRA[@]}" --out "${OUT:?}"

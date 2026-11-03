#!/bin/bash --login
#SBATCH --job-name=rh-stagec-axis
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/reward-hack-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# One forward pass over a Stage C pair file, then the axis
# mean(train hack) - mean(train honest) at the middle layer.
# The previous 68-reply gpt-oss extraction finished in about 3 minutes on one H200.
# These files are smaller or a 30B model, so one hour covers a cold load.
# H200 is required: gpt-oss has no sdpa path, and Muse-Glimmer-30B is ~60 GB in bf16.
#
# Usage (from the repo root). Arguments go after the script name:
#   sbatch experiments/reward-hack-axis/slurm/extract_stagec_axis.sbatch MODEL PAIRS OUT [REVISION] [ATTENTION]

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Positional args are the submission path. Flags placed after the script name
# are not sbatch options, which is how the first two jobs lost MODEL.
if [ $# -ge 3 ]; then
    MODEL=$1
    PAIRS=$2
    OUT_NAME=$3
    REVISION=${4:-}
    ATTENTION=${5:-sdpa}
else
    MODEL="${MODEL:?set MODEL}"
    PAIRS="${PAIRS:?set PAIRS}"
    OUT_NAME="${OUT:?set OUT}"
    REVISION="${REVISION:-}"
    ATTENTION="${ATTENTION:-sdpa}"
fi
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis/$OUT_NAME}"

source ../../activate.sh

ARGS=(--model "$MODEL" --pairs "$PAIRS" --attn-implementation "$ATTENTION" --output-dir "$ROOT" --write-axis)
if [ -n "$REVISION" ]; then
    ARGS+=(--revision "$REVISION")
fi
python scripts/extract_response_mean.py "${ARGS[@]}"

echo "=== stage C axis done: $ROOT ==="

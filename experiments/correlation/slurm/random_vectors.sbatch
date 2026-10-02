#!/bin/bash --login
#SBATCH --job-name=rh-random-v3
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem=100G
#SBATCH --output=slurm-logs/%x-%A_%a.out
set -euo pipefail
# Submit from repository root; export TRANSCRIPTS, OUT, REVISION, PYTHON.
export OPENSSL_CONF=/dev/null
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/.runtime:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PYTHON")")/lib/python3.13/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
args=()
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "$LIMIT"); fi
"$PYTHON" -u -m selfconcept.correlation.random_vectors --transcripts "$TRANSCRIPTS" \
    --out "$OUT/shard_$SHARD" --revision "$REVISION" --shard "$SHARD" --shards "$SHARDS" "${args[@]}"

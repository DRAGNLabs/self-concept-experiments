#!/bin/bash --login
#SBATCH --job-name=axis-acts
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Unsteered-episode activations for the direction comparison (axis_acts_extract.py): finished episodes of one model's
# unsteered arm teacher-forced at every layer, token means plus projection sums for the named directions and random
# nulls. An array task takes every n-th seed of SEEDS (n = array size); a rerun skips the seeds already written.
# Usage (from the repo root; export in the shell first, sbatch --export splits values at commas):
#   export MODEL=<hf id> DATA=<data dir> TAG=<main|pilot> RUN=<unsteered arm dir> SEEDS="0 1 2 ..." OUT=<dir> \
#          DIRECTIONS="gi_v2=<pt> gf_all=<pt>:<key> ..." CHAT_KWARGS='{"reasoning_effort":"low"}' [MAX_LENGTH=40960] [ATTN=<kernel>]
#   sbatch --array=0-3 --export=ALL [-p cs2] experiments/deceiver-axis/slurm/axis_acts.sbatch
# Placement: gpt-oss needs a Hopper GPU for its FA3 kernel (-p cs2); Gemma 4 31B and Qwen3.8-27B turns of 30-70k
# tokens want a B200 (-p cs3); Muse fits an A100.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOO_CHAT_KWARGS="${CHAT_KWARGS:?set CHAT_KWARGS to the chat-template kwargs of the run as JSON}"
MODEL="${MODEL:?set MODEL}"; DATA="${DATA:?set DATA}"; TAG="${TAG:?set TAG}"; RUN="${RUN:?set RUN}"; OUT="${OUT:?set OUT}"
SEEDS="${SEEDS:?set SEEDS}"; DIRECTIONS="${DIRECTIONS:?set DIRECTIONS}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEEDS="$(echo "$SEEDS" | tr ' ' '\n' | awk -v n="$SLURM_ARRAY_TASK_COUNT" -v i="$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))" '(NR - 1) % n == i' | tr '\n' ' ')"
fi
read -r -a SEED_LIST <<< "$SEEDS"
DIRECTION_ARGS=()
for SPEC in $DIRECTIONS; do DIRECTION_ARGS+=(--direction "$SPEC"); done
EXTRA=()
if [ -n "${ATTN:-}" ]; then EXTRA+=(--attn-implementation "$ATTN"); fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# cs A100 nodes: a 1-GPU task can see a sibling's GPU; pin to the least-used one by UUID (see main_run.sbatch).
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$SLURM_SUBMIT_DIR/.venv-313}"
source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "start $(date): $MODEL seeds ${SEED_LIST[*]} -> $OUT"
python scripts/axis_acts_extract.py --model "$MODEL" --data "$DATA" --tag "$TAG" --run "$RUN" --arm unsteered --seeds "${SEED_LIST[@]}" \
    "${DIRECTION_ARGS[@]}" --null-directions "${NULLS:-32}" --max-length "${MAX_LENGTH:-40960}" "${EXTRA[@]}" --out "$OUT"
echo "=== axis-acts done $(date): $OUT seeds ${SEED_LIST[*]} ==="

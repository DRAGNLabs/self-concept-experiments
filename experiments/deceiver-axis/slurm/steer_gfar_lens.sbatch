#!/bin/bash --login
#SBATCH --job-name=steer-gfar-lens
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs,cs2,cs3
#SBATCH --qos=cs
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# gi_v2-steered vs unsteered per-token gfar projections on finished unsteered episodes (steer_gfar_lens_extract.py).
# An array task takes every n-th seed of SEEDS (n = array size); a rerun skips the seeds already written. With
# NULL_DIRECTIONS ("name=path ...") an array task instead takes every n-th null direction for all SEEDS and writes to
# $OUT/chunk<task>.
# Usage (from the repo root; export in the shell first, sbatch --export splits values at commas):
#   export MODEL=<hf id> DATA=<data dir> TAG=<main|pilot> RUN=<unsteered arm dir> SEEDS="0 1 2 ..." OUT=<dir> \
#          GI_V2=<persona_contrast.pt> GFAR=<directions_headline.pt>:<key> STEER_LAYER=<int> ALPHA=<float> LAYER_NORMS=<pt> \
#          CHAT_KWARGS='{"reasoning_effort":"low"}' [MAX_LENGTH=40960] [N=<episodes per seed>] [ARMS="steered steered_gfar_component"]
#          [NULL_DIRECTIONS="name=path ..." TOKEN_STRIDE=<k>]
#   sbatch --array=0-3 --export=ALL [-p cs2] experiments/deceiver-axis/slurm/steer_gfar_lens.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOO_CHAT_KWARGS="${CHAT_KWARGS:?set CHAT_KWARGS to the chat-template kwargs of the run as JSON}"
SEEDS="${SEEDS:?set SEEDS}"
NULL_LIST=()
if [ -n "${NULL_DIRECTIONS:-}" ]; then
    TASK_INDEX="$((${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0}))"
    for SPEC in $(echo "$NULL_DIRECTIONS" | tr ' ' '\n' | awk -v n="${SLURM_ARRAY_TASK_COUNT:-1}" -v i="$TASK_INDEX" '(NR - 1) % n == i'); do
        NULL_LIST+=(--null-direction "$SPEC")
    done
    OUT="${OUT:?}/chunk$TASK_INDEX"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SEEDS="$(echo "$SEEDS" | tr ' ' '\n' | awk -v n="$SLURM_ARRAY_TASK_COUNT" -v i="$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))" '(NR - 1) % n == i' | tr '\n' ' ')"
fi
read -r -a SEED_LIST <<< "$SEEDS"
EXTRA=()
if [ -n "${N:-}" ]; then EXTRA+=(--n "$N"); fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# cs A100 nodes: a 1-GPU task can see a sibling's GPU; pin to the least-used one by UUID.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"

echo "start $(date): $MODEL seeds ${SEED_LIST[*]} -> $OUT"
python scripts/steer_gfar_lens_extract.py --model "${MODEL:?}" --data "${DATA:?}" --tag "${TAG:?}" --run "${RUN:?}" --seeds "${SEED_LIST[@]}" \
    --gi-v2 "${GI_V2:?}" --gfar "${GFAR:?}" --steer-layer "${STEER_LAYER:?}" --alpha "${ALPHA:?}" --layer-norms "${LAYER_NORMS:?}" \
    --arms ${ARMS:-steered} "${NULL_LIST[@]}" --token-stride "${TOKEN_STRIDE:-1}" --max-length "${MAX_LENGTH:-40960}" "${EXTRA[@]}" --out "${OUT:?}"
echo "=== steer-gfar-lens done $(date): $OUT seeds ${SEED_LIST[*]} ==="

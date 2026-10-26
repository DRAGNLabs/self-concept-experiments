#!/bin/bash --login
#SBATCH --job-name=schemer-pilot-gemma31b
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cs3
#SBATCH --gpus-per-node=b200:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root

set -euo pipefail

# Steering dose pilot on google/gemma-4-31B-it over the arm-search problems with a STRONG or WEAK reward-hack verdict
# (impossible_original, evilgenie and evilgenie_lcbpro rows filtered into DATA). One Slurm array task per setting:
# SETTINGS is a TSV, name <TAB> extra args for steered_rh_vllm.py (the steering args and --sample-seed; empty extra
# args = unsteered) [<TAB> scenario files of this shard, space-separated; default $SCENARIOS]. A shard needs its own
# name: the harness writes records to <tag>_<scenario of the examples>.jsonl, so shards sharing a name would append to
# one file. Recipe: neutral feedback, critical_no_carve, 5 stated attempts, Gemma sampling (top-p 0.95,
# top-k 64), thinking on, 32 768 new tokens. Output: $ROOT/<name>/pilot_<scenario>.jsonl (+ reasoning sidecars and
# summaries). Rerunning a task resumes (the harness skips finished example ids). Summarize with scripts/pilot_summary.py.
# Usage: sbatch --array=0-$((N-1)) --export=ALL,SETTINGS=<tsv>,ROOT=<dir>,DATA=<dir> experiments/schemer-axis/slurm/dose_pilot_gemma31b.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"enable_thinking":true}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1  # cached graphs are keyed on the model, not on the steering patch
nvidia-smi

MODEL="${MODEL:-google/gemma-4-31B-it}"
SETTINGS="${SETTINGS:?set SETTINGS to the settings TSV}"
ROOT="${ROOT:?set ROOT to the pilot output dir}"
DATA="${DATA:?set DATA to the directory of filtered problem files}"
SCENARIOS="${SCENARIOS:-impossible_original evilgenie evilgenie_lcbpro}"
RECIPE=(--data "$DATA" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 32768 --max-model-len 40960
        --temperature 1.0 --top-p 0.95 --top-k 64
        --engine-kwargs '{"limit_mm_per_prompt":{"image":0,"video":0,"audio":0}}')

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTINGS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2 -s)
SHARD_SCENARIOS=$(printf '%s' "$LINE" | cut -f3 -s)
read -r -a EXTRA <<< "$EXTRA_STR"
read -r -a SCEN <<< "${SHARD_SCENARIOS:-$SCENARIOS}"
[ -n "$NAME" ] || { echo "no setting at index $SLURM_ARRAY_TASK_ID in $SETTINGS"; exit 1; }
echo "########## setting $SLURM_ARRAY_TASK_ID: $NAME | ${EXTRA[*]:-unsteered} ##########"

source ../../activate.sh

SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios "${SCEN[@]}" "${RECIPE[@]}" \
    --out "$ROOT/$NAME" --tag pilot "${EXTRA[@]}"
echo "setting $NAME took ${SECONDS}s"
echo "=== schemer-pilot-gemma31b done: $ROOT/$NAME ==="

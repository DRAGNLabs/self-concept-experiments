#!/bin/bash --login
#SBATCH --job-name=schemer-pilot
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=standby
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Phase 3 dose pilot (PLAN.md): one Slurm array task per steering setting, 45 hard problems x 1 seed each, on the
# compiled vLLM engine with the steering patched into the block class. SETTINGS is a TSV: name <TAB> extra args for
# steered_rh_vllm.py (empty extra args = unsteered). Output: $ROOT/<name>/pilot_impossible_original.jsonl (+ reasoning
# sidecars and summary). Rerunning a task resumes (the harness skips finished example ids).
# Usage: sbatch --array=0-$((N-1)) --export=ALL,SETTINGS=<tsv>,ROOT=<dir> experiments/schemer-axis/slurm/dose_pilot.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1  # cached graphs are keyed on the model, not on the steering patch
nvidia-smi

MODEL="openai/gpt-oss-120b"
MIRROR="/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments"
SETTINGS="${SETTINGS:?set SETTINGS to the settings TSV}"
ROOT="${ROOT:?set ROOT to the pilot output dir}"
SEED="${SEED:-0}"
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed "$SEED")

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SETTINGS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2 -s)
read -r -a EXTRA <<< "$EXTRA_STR"
[ -n "$NAME" ] || { echo "no setting at index $SLURM_ARRAY_TASK_ID in $SETTINGS"; exit 1; }
echo "########## setting $SLURM_ARRAY_TASK_ID: $NAME | ${EXTRA[*]:-unsteered} ##########"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 "${RECIPE[@]}" \
    --out "$ROOT/$NAME" --tag pilot "${EXTRA[@]}"
echo "setting $NAME took ${SECONDS}s"
echo "=== schemer-pilot done: $ROOT/$NAME ==="

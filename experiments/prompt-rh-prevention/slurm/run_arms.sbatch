#!/bin/bash --login
#SBATCH --job-name=prompt-rh
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/prompt-rh-prevention  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Prompt arms on the recipe (PLAN.md), adapted from deceiver-axis/slurm/main_run.sbatch. One Slurm array task per
# (arm, seed): task index i -> arm line i / NSEEDS of the ARMS TSV, seed SEED_BASE + i % NSEEDS (default 67, so
# seeds 67-78 with NSEEDS=12; the study's seeds, decided 2026-10-30).
# ARMS TSV: name <TAB> prompt file (relative to this experiment dir, e.g. data/prompts/honesty.txt). The file's text
# goes in front of each problem's first user message, then a blank line, then the recipe's task message unchanged
# (selfconcept.codebench.run --prompt-prefix-file). An empty second column = no prefix (the baseline arm).
# Problems: the 45 hard ImpossibleBench problems in rh_hillclimb/data (ib_solvhard_s0-2), as in the recipe and
# deceiver-axis. 16384 new tokens per turn.
# Output: $ROOT/<arm>/seed<s>/ (ROOT defaults to ~/nobackup/autodelete/prompt-rh-prevention-results/<TAG>):
# <TAG>_impossible_original.jsonl (all three files' records), a reasoning sidecar, a summary, and run_info/ (git
# commit and diff, arms TSV, the arm's prompt, problem checksums, Slurm job info) so the run can be checked later.
# Rerunning a task resumes (the harness skips example ids already recorded; run_info/ is rewritten).
# Usage (from the repo root):
#   sbatch --array=0-$((N_ARMS*NSEEDS-1)) --export=ALL,ARMS=data/arms.tsv,TAG=<run name>[,NSEEDS=12][,ROOT=<dir>] \
#       experiments/prompt-rh-prevention/slurm/run_arms.sbatch
# A smoke run on a few problems: add EXTRA_ARGS="--n 3" to --export (the first n of each file).

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null  # the cluster's OpenSSL FIPS mode aborts otherwise
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'  # the recipe's gpt-oss reasoning effort
export VLLM_ENABLE_V1_MULTIPROCESSING=0
nvidia-smi
# On the cs A100 nodes a --gpus-per-node=1 task can see two GPUs, one of them a sibling task's (deceiver-axis job
# 14121029). Pin this task to the visible GPU with the least memory in use, by UUID.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"
PROBLEMS="${PROBLEMS:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data}"
SCENARIOS="${SCENARIOS:-ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2}"
ARMS="${ARMS:?set ARMS to the arms TSV}"
NSEEDS="${NSEEDS:-12}"
SEED_BASE="${SEED_BASE:-67}"
TAG="${TAG:-main}"
ROOT="${ROOT:-$HOME/nobackup/autodelete/prompt-rh-prevention-results/$TAG}"
read -r -a EXTRA_RUN_ARGS <<< "${EXTRA_ARGS:-}"
read -r -a SCENARIO_LIST <<< "$SCENARIOS"
ARM_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID % NSEEDS))
# The recipe (experiments/codebench/reward_hack_recipe.md); the same task message in every arm.
RECIPE=(--data "$PROBLEMS" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --temperature 1.0 --sample-seed "$SEED" --max-new-tokens 16384
        --max-model-len 20480)  # as deceiver-axis's unsteered main run, which is this study's baseline arm

LINE=$(sed -n "$((ARM_INDEX + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
PROMPT_FILE=$(printf '%s' "$LINE" | cut -f2 -s)
[ -n "$NAME" ] || { echo "no arm at index $ARM_INDEX in $ARMS"; exit 1; }
PREFIX_ARGS=()
if [ -n "$PROMPT_FILE" ]; then
    [ -s "$PROMPT_FILE" ] || { echo "arm $NAME: prompt file $PROMPT_FILE is missing or empty"; exit 1; }
    PREFIX_ARGS=(--prompt-prefix-file "$PROMPT_FILE")
fi
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED | prefix: ${PROMPT_FILE:-none} ##########"

OUT="$ROOT/$NAME/seed$SEED"
PROBLEM_FILES=()
for scenario in "${SCENARIO_LIST[@]}"; do PROBLEM_FILES+=("$PROBLEMS/$scenario.jsonl"); done
source slurm/run_info.sh
RUN_INFO_EXTRA="seed: $SEED  tag: $TAG  model: $MODEL  problems: $PROBLEMS  scenarios: $SCENARIOS  extra args: ${EXTRA_ARGS:-}" \
    write_run_info "$OUT" "$NAME" "$PROMPT_FILE" "${PROBLEM_FILES[@]}"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv"
source ../../activate.sh

SECONDS=0
python -m selfconcept.codebench.run --backend vllm-harmony --model "$MODEL" \
    --scenarios "${SCENARIO_LIST[@]}" "${RECIPE[@]}" "${PREFIX_ARGS[@]}" \
    --out "$OUT" --tag "$TAG" "${EXTRA_RUN_ARGS[@]}"
echo "arm $NAME seed $SEED took ${SECONDS}s"

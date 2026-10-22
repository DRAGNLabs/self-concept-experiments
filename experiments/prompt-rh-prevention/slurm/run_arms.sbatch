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

# Prompt arms on the recipe (PLAN.md), adapted from schemer-axis/slurm/main_run.sbatch. One Slurm array task per
# (arm, seed): task index i -> arm line i / NSEEDS of the ARMS TSV (name <TAB> extra args for selfconcept.codebench.run,
# e.g. --impossible-prompt strict), seed i % NSEEDS.
# Output: $ROOT/<arm>/seed<s>/<TAG>_impossible_original.jsonl plus the reasoning sidecar. Rerunning a task resumes
# (the harness skips example ids already recorded).
# Usage (from the repo root):
#   sbatch --array=0-$((N_ARMS*NSEEDS-1)) --export=ALL,ARMS=data/arms.tsv,ROOT=<dir>[,NSEEDS=12][,TAG=main] \
#       experiments/prompt-rh-prevention/slurm/run_arms.sbatch
# A smoke run on a few problems: add EXTRA_ARGS="--n 3" to --export.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null  # the cluster's OpenSSL FIPS mode aborts otherwise
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'  # the recipe's gpt-oss reasoning effort
export VLLM_ENABLE_V1_MULTIPROCESSING=0
nvidia-smi
# On the cs A100 nodes a --gpus-per-node=1 task can see two GPUs, one of them a sibling task's (schemer-axis job
# 14121029). Pin this task to the visible GPU with the least memory in use, by UUID.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"  # Koby's shared data
ARMS="${ARMS:?set ARMS to the arms TSV}"
ROOT="${ROOT:?set ROOT to the output dir}"
NSEEDS="${NSEEDS:-12}"
TAG="${TAG:-main}"
read -r -a EXTRA_RUN_ARGS <<< "${EXTRA_ARGS:-}"
ARM_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NSEEDS))
# The recipe (experiments/codebench/reward_hack_recipe.md). Arm args come after it; argparse's last value wins.
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed "$SEED")

LINE=$(sed -n "$((ARM_INDEX + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
ARM_ARGS_STR=$(printf '%s' "$LINE" | cut -f2 -s)
read -r -a ARM_ARGS <<< "$ARM_ARGS_STR"
[ -n "$NAME" ] || { echo "no arm at index $ARM_INDEX in $ARMS"; exit 1; }
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED | ${ARM_ARGS[*]:-recipe only} ##########"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv"
source ../../activate.sh

SECONDS=0
python -m selfconcept.codebench.run --backend vllm-harmony --model "$MODEL" \
    --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 "${RECIPE[@]}" \
    --out "$ROOT/$NAME/seed$SEED" --tag "$TAG" "${ARM_ARGS[@]}" "${EXTRA_RUN_ARGS[@]}"
echo "arm $NAME seed $SEED took ${SECONDS}s"

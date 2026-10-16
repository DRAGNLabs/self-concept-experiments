#!/bin/bash --login
#SBATCH --job-name=schemer-main
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

# Phase 3 main run (PLAN.md): the five arms on the same 45 hard problems x NSEEDS seeds, one Slurm array task per
# (arm, seed): task index i -> arm line i / NSEEDS of the ARMS TSV (name <TAB> extra args for steered_rh_vllm.py,
# empty extra args = unsteered), seed i % NSEEDS. Output: $ROOT/<arm>/seed<s>/main_impossible_original.jsonl plus
# the reasoning sidecar; rerunning a task resumes (the harness skips finished example ids).
# Usage: sbatch --array=0-$((N_ARMS*NSEEDS-1)) --export=ALL,ARMS=<tsv>,ROOT=<dir>[,NSEEDS=12] experiments/schemer-axis/slurm/main_run.sbatch
# On an 80 GB GPU (cs2 H100, paid qos cs, no preemption) override the partition on the command line and raise the vLLM
# memory fraction so the KV cache fits beside the ~61 GiB of weights:
#   sbatch --partition=cs2 --qos=cs --gpus-per-node=h100:1 --array=... --export=ALL,ARMS=...,ROOT=...,VLLM_GPU_MEM=0.95 ...

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
DEFAULT_CHAT_KWARGS='{"reasoning_effort":"low"}'  # gpt-oss low effort; Gemma 4 runs export CHAT_KWARGS='{"enable_thinking":true}'
export SOO_CHAT_KWARGS="${CHAT_KWARGS:-$DEFAULT_CHAT_KWARGS}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1  # cached graphs are keyed on the model, not on the steering patch
nvidia-smi
# On the cs A100 nodes a --gpus-per-node=1 task can see two GPUs, one of them a sibling task's (job 14121029: 7 of 12
# tasks died at startup on a GPU with 77 GB already in use). Pin this task to the visible GPU with the least memory
# in use, by UUID so the choice does not depend on index remapping.
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="${MODEL:-openai/gpt-oss-120b}"  # Gemma Phase 3: MODEL=google/gemma-4-12B-it with CHAT_KWARGS as above
MIRROR="${MIRROR:-/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments}"  # Koby's shared data (rh_hillclimb/data)
ARMS="${ARMS:?set ARMS to the arms TSV}"
ROOT="${ROOT:?set ROOT to the main-run output dir}"
NSEEDS="${NSEEDS:-12}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-}"  # optional --gpu-memory-utilization override (steered_rh_vllm.py default 0.9)
ARM_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NSEEDS))
RECIPE=(--data "$MIRROR/rh_hillclimb/data" --impossible-prompt critical_no_carve --feedback neutral
        --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 --sample-seed "$SEED")

LINE=$(sed -n "$((ARM_INDEX + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2 -s)
read -r -a EXTRA <<< "$EXTRA_STR"
[ -n "$NAME" ] || { echo "no arm at index $ARM_INDEX in $ARMS"; exit 1; }
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED | ${EXTRA[*]:-unsteered} ##########"

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_solvhard_s0 ib_solvhard_s1 ib_solvhard_s2 "${RECIPE[@]}" \
    --out "$ROOT/$NAME/seed$SEED" --tag main ${VLLM_GPU_MEM:+--gpu-memory-utilization "$VLLM_GPU_MEM"} "${EXTRA[@]}"
echo "arm $NAME seed $SEED took ${SECONDS}s"
echo "=== schemer-main done: $ROOT/$NAME/seed$SEED ==="

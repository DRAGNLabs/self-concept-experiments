#!/bin/bash --login
# Superseded by stagec_rh_steer.sbatch. This file still points at the old 68-pair axis.
#SBATCH --job-name=rh-axis-gptoss
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Steer gpt-oss-120b with the reward-hack axis, in the schemer main-run harness.
# The axis was fit on 13 LiveCodeBench problems. Twelve of those are in the 45-problem
# schemer set, so this run uses the other 33. Seeds 0-11 match the schemer main run.
# Layer 18 is the only layer the axis was extracted at. Alpha 0.2 matches shift_plus
# and shift_minus. The schemer episodes are already on disk and are not rerun.
#
# Norms, once:  sbatch --export=ALL,NORMS_ONLY=1 experiments/reward-hack-axis/slurm/gptoss_rh_axis_steer.sbatch
# Steering:     sbatch --array=0-23 --dependency=afterok:<norms job> experiments/reward-hack-axis/slurm/gptoss_rh_axis_steer.sbatch

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL="openai/gpt-oss-120b"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.95}"  # gpt-oss-120b MXFP4 on an 80 GB H100, as in the schemer main run
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis/gptoss_main}"
DATA="$ROOT/data"
DIRECTION="$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis/hack_axis_layers.pt"
NORMS="$ROOT/norms/layer_norms.pt"
ARMS="$ROOT/arms.tsv"
NSEEDS=12

export SELFCONCEPT_VENV="$SLURM_SUBMIT_DIR/.venv-313"
source ../../activate.sh

if [ "${NORMS_ONLY:-0}" = 1 ]; then
    echo "########## layer norms on the 33 eval prompts ##########"
    python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios ib_rhsteer \
        --data "$DATA" --impossible-prompt critical_no_carve --feedback neutral \
        --max-attempts 5 --state-attempt-budget --gpu-memory-utilization "$VLLM_GPU_MEM" \
        --measure-norms --out "$ROOT/norms"
    echo "=== norms done: $NORMS ==="
    exit 0
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "steering expects an array task" >&2
    exit 1
fi
ARM_INDEX=$((SLURM_ARRAY_TASK_ID / NSEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % NSEEDS))
LINE=$(sed -n "$((ARM_INDEX + 1))p" "$ARMS")
NAME=$(printf '%s' "$LINE" | cut -f1)
EXTRA_STR=$(printf '%s' "$LINE" | cut -f2-)
read -r -a EXTRA <<< "$EXTRA_STR"
[ -n "$NAME" ] || { echo "no arm at index $ARM_INDEX" >&2; exit 1; }
echo "########## task $SLURM_ARRAY_TASK_ID: arm $NAME seed $SEED ##########"

SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" \
    --scenarios ib_rhsteer --data "$DATA" \
    --impossible-prompt critical_no_carve --feedback neutral \
    --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 \
    --sample-seed "$SEED" \
    --direction "$DIRECTION" --layer-norms "$NORMS" \
    --gpu-memory-utilization "$VLLM_GPU_MEM" \
    --out "$ROOT/$NAME/seed$SEED" --tag main "${EXTRA[@]}"
echo "arm $NAME seed $SEED took ${SECONDS}s"
echo "=== rh-axis-gptoss done: $ROOT/$NAME/seed$SEED ==="

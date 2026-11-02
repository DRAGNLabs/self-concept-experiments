#!/bin/bash --login
#SBATCH --job-name=rh-stagec-steer
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --requeue
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --chdir=experiments/schemer-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Steer with the Stage C reward-hack axis in the schemer harness.
# Problems used to fit the axis are left out. Seeds 0-11 match the schemer main run.
# GPT-OSS: layer 18 (the only layer the axis exists at), alpha ±0.2, the schemer main dose.
#   The schemer headline vector was steered at layer 17.
# Muse: layer 26, alpha +0.08 and -0.16, the schemer main doses.
# Usage, from the repo root, options before the script:
#   sbatch experiments/reward-hack-axis/slurm/stagec_rh_steer.sbatch gptoss norms
#   sbatch --array=0-23 --dependency=afterok:<norms> experiments/reward-hack-axis/slurm/stagec_rh_steer.sbatch gptoss steer
#   sbatch experiments/reward-hack-axis/slurm/stagec_rh_steer.sbatch muse norms
#   sbatch --array=0-23 --dependency=afterok:<norms> experiments/reward-hack-axis/slurm/stagec_rh_steer.sbatch muse steer

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENSSL_CONF=/dev/null
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1
nvidia-smi
if [ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)"
    echo "several GPUs visible; pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

WHICH="${1:?gptoss or muse}"
MODE="${2:?norms or steer}"
BASE="$HOME/nobackup/autodelete/self-concept-experiments/reward-hack-axis"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.95}"
NSEEDS=12

case "$WHICH" in
    gptoss)
        MODEL="openai/gpt-oss-120b"
        export SOO_CHAT_KWARGS='{"reasoning_effort":"low"}'
        ROOT="$BASE/stagec_gptoss/steer"
        DIRECTION="$BASE/stagec_gptoss/hack_axis_layers.pt"
        EXTRA_SAMPLE=()
        ;;
    muse)
        MODEL="meta-models/Muse-Glimmer-30B"
        export SOO_CHAT_KWARGS='{"reasoning_strength":"low"}'
        ROOT="$BASE/stagec_muse/steer"
        DIRECTION="$BASE/stagec_muse/hack_axis_layers.pt"
        EXTRA_SAMPLE=(--top-p 0.95 --top-k 64)
        ;;
    *)
        echo "unknown model $WHICH" >&2
        exit 1
        ;;
esac

DATA="$ROOT/data"
NORMS="$ROOT/norms/layer_norms.pt"
ARMS="$ROOT/arms.tsv"
mapfile -t SCENARIOS < "$DATA/scenarios.txt"
[ "${#SCENARIOS[@]}" -gt 0 ] || { echo "no scenarios in $DATA/scenarios.txt" >&2; exit 1; }

# This checkout's Python 3.13 env with vLLM is .venv. .venv-313 is not present.
unset SELFCONCEPT_VENV
source ../../activate.sh

if [ "$MODE" = "norms" ]; then
    echo "########## layer norms for $WHICH ##########"
    python scripts/steered_rh_vllm.py --model "$MODEL" --scenarios "${SCENARIOS[@]}" \
        --data "$DATA" --impossible-prompt critical_no_carve --feedback neutral \
        --max-attempts 5 --state-attempt-budget --gpu-memory-utilization "$VLLM_GPU_MEM" \
        "${EXTRA_SAMPLE[@]}" \
        --measure-norms --out "$ROOT/norms"
    echo "=== norms done: $NORMS ==="
    exit 0
fi

if [ "$MODE" != "steer" ]; then
    echo "mode must be norms or steer" >&2
    exit 1
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
echo "########## task $SLURM_ARRAY_TASK_ID: $WHICH arm $NAME seed $SEED ##########"

SECONDS=0
python scripts/steered_rh_vllm.py --model "$MODEL" \
    --scenarios "${SCENARIOS[@]}" --data "$DATA" \
    --impossible-prompt critical_no_carve --feedback neutral \
    --max-attempts 5 --state-attempt-budget --max-new-tokens 16384 --temperature 1.0 \
    --sample-seed "$SEED" \
    --direction "$DIRECTION" --layer-norms "$NORMS" \
    --gpu-memory-utilization "$VLLM_GPU_MEM" \
    "${EXTRA_SAMPLE[@]}" \
    --out "$ROOT/$NAME/seed$SEED" --tag main "${EXTRA[@]}"
echo "arm $NAME seed $SEED took ${SECONDS}s"
echo "=== rh-stagec-steer done: $ROOT/$NAME/seed$SEED ==="

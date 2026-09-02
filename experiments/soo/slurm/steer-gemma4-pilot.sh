#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma4-pilot
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

mamba activate ../../.env

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# GPU isolation isn't enforced on these nodes and Slurm's CUDA_VISIBLE_DEVICES
# can point at a GPU already occupied by other/orphaned processes (seen on
# dw-2-4) — or at a GPU another job holds exclusively. Keep the assigned GPU
# if it looks free; otherwise re-pin to the emptiest visible one.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
pick=$CUDA_VISIBLE_DEVICES
used=$([ -n "$pick" ] && nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$pick" | tr -d ' ')
if [ -z "$used" ] || [ "$used" -gt 1000 ]; then
    pick=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "Assigned GPU busy or unset (${used:-?} MiB used); re-pinned to emptiest"
fi
export CUDA_VISIBLE_DEVICES=$pick
echo "Using GPU $CUDA_VISIBLE_DEVICES"

set -e
# Gemma 4 pilot. No LoRA-validated band exists for this model, so sweep
# depth: L12/L18/L24/L30 of 60 brackets gemma-2's working depth (L14/46 ~ 30%).
# Alphas 8 and 32 are where gemma-2 first moved and where it hit 48% honest.
# Baselines run in both suffix conventions (gemma-2 needed room_only; unknown
# whether gemma-4 does) — the grid uses room_only pending that comparison.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --n 50 --out "$OUT" --tag "$tag" "$@"
}

run base_room_only --suffix room_only --scenarios $SCENARIOS
run base_i_would --suffix i_would --scenarios main
for L in 12 18 24 30; do
    for A in 8 32; do
        run "steer_L${L}_a${A}" --suffix room_only --scenarios main \
            --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done

echo "=== gemma4 pilot complete ==="

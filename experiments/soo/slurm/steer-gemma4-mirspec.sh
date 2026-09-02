#!/bin/bash --login
#SBATCH --job-name=soo-steer-g4-mirspec
#SBATCH --time=04:00:00
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
# Close the mirrored-specificity gap at gemma-4-31B's a16 working cell
# (real mir 44% vs mir base 0%, but no random-mir control yet) and map
# the mirrored dose between the direction-specific (a16) and
# direction-agnostic (a>=24) regimes.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
OUT=results/steering_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" --data data/eval_mirrored --scenarios main "$@"
}

run steer_L30_rand_a16_mir \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 16 --steer-random-seed 0
for A in 20 24; do
    run "steer_L30_a${A}_mir" \
        --steer-vectors $VEC --steer-layer 30 --steer-alpha $A
done
run steer_L30_rand_a24_mir \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 24 --steer-random-seed 0

echo "=== gemma4 mirrored specificity complete ==="

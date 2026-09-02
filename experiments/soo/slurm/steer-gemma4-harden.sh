#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma4-harden
#SBATCH --time=12:00:00
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
# Harden the gemma-4 pilot result: L30 a32 flipped main 0 -> 100% honest
# (verified against each example's honest room; single-word answers) while
# a8 and L12/18/24 were null. Here: mirrored orientation, random-vector
# control, dose points between the null a8 and total a32, neighboring
# layers to map the band, the other scenarios at the working cell, and
# an n=250 confirmation.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --out "$OUT" --tag "$tag" "$@"
}

# Mirrored baseline + working cell, all scenarios
run base_mir --n 50 --data data/eval_mirrored --scenarios $SCENARIOS
run steer_L30_a32_mir --n 50 --data data/eval_mirrored --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 32

# Working cell on the other scenarios, original orientation
run steer_L30_a32 --n 50 --scenarios perspectives treasure_hunt \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 32

# Random matched-norm control at the working cell
run steer_L30_rand_a32 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 32 --steer-random-seed 0

# Dose points between null (a8) and total (a32)
for A in 12 16 24; do
    run "steer_L30_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 30 --steer-alpha $A
done

# Band mapping: neighbors of L30 at the working strength
for L in 27 33 36; do
    run "steer_L${L}_a32" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer $L --steer-alpha 32
done

# n=250 confirmation of the headline cell
run steer_L30_a32_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 32

echo "=== gemma4 hardening complete ==="

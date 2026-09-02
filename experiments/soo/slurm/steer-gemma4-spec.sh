#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma4-spec
#SBATCH --time=08:00:00
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
# Direction-specificity disambiguation for gemma-4-31B L30: the hardening
# round's random matched-norm control at a32 flipped main to 50/50 honest,
# identical to the real vector (norms match: 8.0817). Map whether the real
# direction is privileged at all: random dose curve vs the real one
# (real: a12 3/50 -> a16 49/50), a second random seed, the reversed real
# vector, and random at the layers where the real vector was null.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
OUT=results/steering_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" --scenarios main "$@"
}

# Random threshold curve (seed 0) against the real a12->a16 transition
for A in 8 12 16 24; do
    run "steer_L30_rand_a${A}" \
        --steer-vectors $VEC --steer-layer 30 --steer-alpha $A --steer-random-seed 0
done

# Seed robustness
for A in 16 32; do
    run "steer_L30_rand2_a${A}" \
        --steer-vectors $VEC --steer-layer 30 --steer-alpha $A --steer-random-seed 1
done

# Sign check: reversed real vector (toward "self")
for A in 16 32; do
    run "steer_L30_a-${A}" \
        --steer-vectors $VEC --steer-layer 30 --steer-alpha -$A
done

# Layer specificity of the perturbation effect: random where real was null
for L in 27 33; do
    run "steer_L${L}_rand_a32" \
        --steer-vectors $VEC --steer-layer $L --steer-alpha 32 --steer-random-seed 0
done

echo "=== gemma4 specificity complete ==="

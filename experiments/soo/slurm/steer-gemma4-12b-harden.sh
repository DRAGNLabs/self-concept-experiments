#!/bin/bash --login
#SBATCH --job-name=soo-steer-g4-12b-harden
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
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
# gemma-4-12B hardening + specificity in one round (12B is cheap). Pilot:
# L19 a32 = 50/50, L14 a32 = 48/50, a8 null, L24/L29 null — but random
# matched-norm at L24 a32 flips 50/50 where the real vector does nothing,
# so the 31B's direction-agnostic perturbation regime exists here too and
# specificity must be shown at the working layers. Dose transitions, randoms
# at L19/L14 (two seeds), reversed vector, mirrored, scenarios, and paired
# real-vs-random n=250 at the working cell.
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma4_12b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --out "$OUT" --tag "$tag" "$@"
}

# Dose transitions at the two working layers
for A in 12 16 24; do
    run "steer_L19_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 19 --steer-alpha $A
done
for A in 16 24; do
    run "steer_L14_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 14 --steer-alpha $A
done

# Random matched-norm at the working layers, two seeds
for A in 16 32; do
    run "steer_L19_rand_a${A}" --n 50 --scenarios main \
        --steer-vectors $VEC --steer-layer 19 --steer-alpha $A --steer-random-seed 0
done
run steer_L19_rand2_a32 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 32 --steer-random-seed 1
run steer_L14_rand_a32 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 14 --steer-alpha 32 --steer-random-seed 0

# Reversed vector
run steer_L19_a-32 --n 50 --scenarios main \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha -32

# Mirrored + scenarios at the working cell
run base_mir --n 50 --data data/eval_mirrored --scenarios $SCENARIOS
run steer_L19_a32_mir --n 50 --data data/eval_mirrored --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 32
run steer_L19_a32_scen --n 50 --scenarios perspectives treasure_hunt \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 32

# Paired n=250: real vs random at the working cell
run steer_L19_a32_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 32
run steer_L19_rand_a32_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 19 --steer-alpha 32 --steer-random-seed 0

echo "=== gemma4-12b hardening complete ==="

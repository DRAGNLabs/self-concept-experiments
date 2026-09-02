#!/bin/bash --login
#SBATCH --job-name=soo-steer-g4-12b-edge
#SBATCH --time=03:00:00
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
# Last 12B disambiguation: is the real direction privileged at its own
# threshold (a=12, where real = 50/50 but a8 = 0/50), the way the 31B's
# was at a16? Real vs two random seeds vs reversed, all at a12, plus a10
# for the real vector. If randoms match real here too, the 12B flip is
# perturbation-honesty at every measurable strength.
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
OUT=results/steering_eval/gemma4_12b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" --scenarios main "$@"
}

run steer_L19_a10 --steer-vectors $VEC --steer-layer 19 --steer-alpha 10
run steer_L19_rand_a12 --steer-vectors $VEC --steer-layer 19 --steer-alpha 12 --steer-random-seed 0
run steer_L19_rand2_a12 --steer-vectors $VEC --steer-layer 19 --steer-alpha 12 --steer-random-seed 1
run steer_L19_a-12 --steer-vectors $VEC --steer-layer 19 --steer-alpha -12

echo "=== gemma4-12b edge complete ==="

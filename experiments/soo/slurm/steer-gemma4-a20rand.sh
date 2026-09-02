#!/bin/bash --login
#SBATCH --job-name=soo-steer-g4-a20rand
#SBATCH --time=02:00:00
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
# Pin the 31B operating point: real a20 gives 94% mirrored; random-mir is
# 0% at a16 and 58% at a24. Measure random at a20 in both orientations to
# quote the direction-vs-random differential at the recommended strength.
MODEL=google/gemma-4-31B-it
VEC=results/steering/gemma4_31b.pt
OUT=results/steering_eval/gemma4_31b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --n 50 --out "$OUT" --tag "$tag" --scenarios main "$@"
}

run steer_L30_rand_a20 \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 20 --steer-random-seed 0
run steer_L30_rand_a20_mir --data data/eval_mirrored \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 20 --steer-random-seed 0
run steer_L30_a20 \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 20

echo "=== gemma4 a20 controls complete ==="

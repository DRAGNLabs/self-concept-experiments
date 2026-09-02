#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma4-a16
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
# Re-anchor the gemma-4-31B working cell at a16, the direction-specific
# regime (real 98% vs rand 18%/4% honest; a>=24 is confounded by the
# direction-agnostic perturbation flip). Headline checks re-run at a16:
# mirrored, other scenarios, n=250 real, and n=250 of the random control
# to pin the differential at scale.
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

run steer_L30_a16_mir --n 50 --data data/eval_mirrored --scenarios $SCENARIOS \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 16
run steer_L30_a16_scen --n 50 --scenarios perspectives treasure_hunt \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 16
run steer_L30_a16_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 16
run steer_L30_rand_a16_n250 --n 250 --scenarios main \
    --steer-vectors $VEC --steer-layer 30 --steer-alpha 16 --steer-random-seed 0

echo "=== gemma4 a16 re-anchor complete ==="

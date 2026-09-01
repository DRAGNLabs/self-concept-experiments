#!/bin/bash --login
#SBATCH --job-name=soo-steer-gemma
#SBATCH --time=06:00:00
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
# Steering pilot on Gemma: L14 is the validated LoRA honesty band (the
# cleanest in the study) — if the extracted direction is inert here too, the
# Mistral "direction is not load-bearing" verdict generalizes. L20 = paper's
# layer. Gemma eval protocol uses --suffix room_only (see gemma-layer-sweep).
MODEL=google/gemma-2-27b-it
VEC=results/steering/gemma27b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/gemma27b

run() {  # run <tag> [steering args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" --suffix room_only \
        --scenarios $SCENARIOS --n 50 --out "$OUT" --tag "$tag" "$@"
}

run base
for A in 1 2 4 8 16; do
    run "steer_L14_a${A}" --steer-vectors $VEC --steer-layer 14 --steer-alpha $A
done
run steer_L14_proj_a1 --steer-vectors $VEC --steer-layer 14 --steer-alpha 1 --steer-mode project
run steer_L14_rand_a8 --steer-vectors $VEC --steer-layer 14 --steer-alpha 8 --steer-random-seed 0
for A in 2 8; do
    run "steer_L20_a${A}" --steer-vectors $VEC --steer-layer 20 --steer-alpha $A
done

echo "=== gemma steering pilot complete ==="

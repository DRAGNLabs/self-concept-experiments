#!/bin/bash --login
#SBATCH --job-name=soo-steer-olmo
#SBATCH --time=04:00:00
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
# Steering pilot on OLMo: L16 (lasttok LoRA's partial main effect) and L19
# (full-mode LoRA's treasure-hunt effect) — does either effect reappear via
# the extracted direction, and does token convention (last vs mean vector)
# matter the way soo_mode did for LoRA?
MODEL=allenai/OLMo-2-1124-7B-Instruct
VEC=results/steering/olmo7b.pt
SCENARIOS="main perspectives treasure_hunt"
OUT=results/steering_eval/olmo7b

run() {  # run <tag> [steering args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --scenarios $SCENARIOS --n 50 --out "$OUT" --tag "$tag" "$@"
}

run base
for L in 16 19; do
    for A in 1 2 4 8 16; do
        run "steer_L${L}_a${A}" --steer-vectors $VEC --steer-layer $L --steer-alpha $A
    done
done
run steer_L16_proj_a1 --steer-vectors $VEC --steer-layer 16 --steer-alpha 1 --steer-mode project
run steer_L19_mean_a4 --steer-vectors $VEC --steer-layer 19 --steer-alpha 4 --steer-token-mode mean
run steer_L16_rand_a8 --steer-vectors $VEC --steer-layer 16 --steer-alpha 8 --steer-random-seed 0

echo "=== olmo steering pilot complete ==="

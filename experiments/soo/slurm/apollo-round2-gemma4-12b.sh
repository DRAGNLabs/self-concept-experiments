#!/bin/bash --login
#SBATCH --job-name=soo-apollo2-g4-12b
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/soo  # relative to the submission dir -- run sbatch from the repo root

source ../../.venv/bin/activate

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
# Apollo OOD round 2, 12B: round 1 found LoRA seed0 transfers (dec
# 52->31, hon 10->32) while the vector is inert. Certify: LoRA seeds
# 1-2 (does the transfer replicate?), rand seed 1 (thicken the
# steering null's control distribution).
MODEL=google/gemma-4-12B-it
VEC=results/steering/gemma4_12b.pt
OUT=results/apollo_eval/gemma4_12b

run() {  # run <tag> [eval args...]
    local tag=$1; shift
    echo "=== $tag ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --data data/eval_apollo --scenarios roleplaying --suffix none \
        --max-new-tokens 256 --out "$OUT" --tag "$tag" "$@"
}

run ap_lora_s1 --adapter results/checkpoints/gemma4-12b-L24/seed1
run ap_lora_s2 --adapter results/checkpoints/gemma4-12b-L24/seed2
run ap_rand_s1 --steer-vectors $VEC --steer-layer 19 --steer-alpha 12 --steer-random-seed 1

echo "=== apollo round2 gemma4-12b complete ==="

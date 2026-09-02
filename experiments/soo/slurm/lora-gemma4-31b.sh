#!/bin/bash --login
#SBATCH --job-name=soo-lora-g4-31b
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
# LoRA-vs-age within the Gemma family: gemma-2-27b paper recipe unchanged,
# layer sweep at depth-matched positions of the 60-layer stack — L18 (30%,
# Gemma-2's validated L14 depth), L26 (43%, the paper's original depth),
# L30 (50%, this model's steering layer). One seed per layer for the sweep.
MODEL=google/gemma-4-31B-it
SCENS="main treasure_hunt perspectives"

echo "=== baseline eval ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios $SCENS --n 50 --suffix room_only --tag baseline_lora_g4_31b

for L in 18 26 30; do
    echo "=== gemma-4-31B LoRA L${L} seed 0 ==="
    python -m selfconcept.soo.train --config configs/gemma4-31b.yaml --layer ${L} --seeds 0
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma4-31b-L${L}/seed0" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_g4_31b_L${L}_seed0"
done

echo "=== gemma-4-31B LoRA sweep complete ==="

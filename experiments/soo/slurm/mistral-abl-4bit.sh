#!/bin/bash --login
#SBATCH --job-name=soo-mistral-4bit
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
MODEL=mistralai/Mistral-7B-Instruct-v0.2
SCENS="main treasure_hunt perspectives"

echo "=== ablation B: 4-bit QLoRA (paper-exact), seeds 0 1 ==="
python -m selfconcept.soo.train --config configs/mistral-7b-4bit.yaml --seeds 0 1
echo "--- 4-bit baseline eval (no adapter) for reference ---"
python -m selfconcept.soo.evaluate --model "$MODEL" --quant-4bit \
    --scenarios $SCENS --n 50 --tag baseline_mistral7b_4bit
for seed in 0 1; do
    python -m selfconcept.soo.evaluate --model "$MODEL" --quant-4bit \
        --adapter "results/checkpoints/mistral-7b-4bit/seed${seed}" \
        --scenarios $SCENS --n 50 --tag "soo_mistral7b_4bit_seed${seed}"
done

echo "=== ablations complete ==="

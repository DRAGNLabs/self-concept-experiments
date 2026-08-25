#!/bin/bash --login
#SBATCH --job-name=soo-seed-robust
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
SCENS="main treasure_hunt perspectives"

# Seed-robustness check: Mistral last-token reproduced the paper on seed 1 but
# broke on seed 0; OLMo-7B (full mode) worked on all of seeds 0-4. One fresh
# seed each to firm up "Mistral is seed-fragile, OLMo is robust".

echo "=== Mistral lasttok seed 2 ==="
python -m selfconcept.soo.train --config configs/mistral-7b-lasttok.yaml --seeds 2
python -m selfconcept.soo.evaluate --model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter results/checkpoints/mistral-7b-lasttok/seed2 \
    --scenarios $SCENS --n 50 --tag soo_mistral7b_lasttok_seed2

echo "=== OLMo-2-7B seed 5 ==="
python -m selfconcept.soo.train --config configs/olmo2-7b.yaml --seeds 5
python -m selfconcept.soo.evaluate --model allenai/OLMo-2-1124-7B-Instruct \
    --adapter results/checkpoints/olmo2-7b/seed5 \
    --scenarios $SCENS --n 50 --tag soo_olmo7b_seed5

echo "=== seed robustness complete ==="

#!/bin/bash --login
#SBATCH --job-name=soo-layer-sweep
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/jansen05/self-concept-experiments

mamba activate ./.env

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

# Layer sweep guided by scripts/layer_gap.py (results/layer_gap_*.json):
#   L2  = largest relative self/other gap overall (rel_last 1.66, early outlier)
#   L16 = largest relative gap in the mid-stack (1.20)
#   L24 = large on both metrics (rel 1.06, abs 5.6e-4)
#   L30 = absolute gap (1.4e-3) comparable to OLMo-7B's L19 (3.5e-3)
# Paper's choice was L19 (rel 0.97, abs 6.9e-4). lasttok lr1e-4 recipe.
for L in 2 16 24 30; do
    for seed in 0 1; do
        echo "=== Mistral lasttok L${L} seed ${seed} ==="
        python -m soo.train --config configs/mistral-7b-lasttok.yaml \
            --layer ${L} --seeds ${seed}
        python -m soo.evaluate --model mistralai/Mistral-7B-Instruct-v0.2 \
            --adapter "results/checkpoints/mistral-7b-lasttok-L${L}/seed${seed}" \
            --scenarios $SCENS --n 50 --tag "soo_mistral7b_lasttok_L${L}_seed${seed}"
    done
done

echo "=== layer sweep complete ==="

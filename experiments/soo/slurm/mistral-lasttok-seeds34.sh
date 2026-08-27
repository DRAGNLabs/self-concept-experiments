#!/bin/bash --login
#SBATCH --job-name=soo-lasttok-s34
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

# Mistral last-token seeds 3-4: seeds 0/1/2 landed in three different basins
# (evasive / paper-like / partial-with-ToM-damage). Two more seeds to pin down
# what fraction of seeds actually reproduce the paper.
for seed in 3 4; do
    echo "=== Mistral lasttok seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/mistral-7b-lasttok.yaml --seeds ${seed}
    python -m selfconcept.soo.evaluate --model mistralai/Mistral-7B-Instruct-v0.2 \
        --adapter "results/checkpoints/mistral-7b-lasttok/seed${seed}" \
        --scenarios $SCENS --n 50 --tag "soo_mistral7b_lasttok_seed${seed}"
done

echo "=== lasttok seeds 3-4 complete ==="

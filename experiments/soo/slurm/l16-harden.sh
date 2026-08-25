#!/bin/bash --login
#SBATCH --job-name=soo-l16-harden
#SBATCH --time=03:30:00
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
MODEL=mistralai/Mistral-7B-Instruct-v0.2
SCENS="main treasure_hunt perspectives"

# Harden the L16 result: it worked on seeds 0-1 (layer sweep); add seeds 2-4
# and evaluate everything at the paper's n=250 (prior evals were n=50).

echo "=== train L16 seeds 2 3 4 ==="
python -m selfconcept.soo.train --config configs/mistral-7b-lasttok.yaml --layer 16 --seeds 2 3 4

echo "=== baseline n=250 ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios $SCENS --n 250 --tag baseline_mistral7b_n250

for seed in 0 1 2 3 4; do
    echo "=== eval L16 seed ${seed} n=250 ==="
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/mistral-7b-lasttok-L16/seed${seed}" \
        --scenarios $SCENS --n 250 --tag "soo_mistral7b_lasttok_L16_seed${seed}_n250"
done

echo "=== L16 hardening complete ==="

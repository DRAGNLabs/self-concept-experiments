#!/bin/bash --login
#SBATCH --job-name=soo-lowlr
#SBATCH --time=04:00:00
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

# LR-scale test: Mistral's q/v weights are 4-5x smaller than OLMo's, so the
# paper's lr=1e-4 perturbs Mistral 2-3x more relative to base weights. If
# that's why Mistral is seed-fragile, lowering lr to 2e-5 / 1e-5 should give
# deception reduction with Perspectives intact and less variance across seeds.
for lr in lr2e5 lr1e5; do
    for seed in 0 1 2; do
        echo "=== Mistral lasttok ${lr} seed ${seed} ==="
        python -m selfconcept.soo.train --config "configs/mistral-7b-lasttok-${lr}.yaml" --seeds ${seed}
        python -m selfconcept.soo.evaluate --model mistralai/Mistral-7B-Instruct-v0.2 \
            --adapter "results/checkpoints/mistral-7b-lasttok-${lr}/seed${seed}" \
            --scenarios $SCENS --n 50 --tag "soo_mistral7b_lasttok_${lr}_seed${seed}"
    done
done

echo "=== low-lr sweep complete ==="

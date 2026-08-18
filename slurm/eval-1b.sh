#!/bin/bash --login
#SBATCH --job-name=soo-eval-1b
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/jansen05/self-concept-experiments

mamba activate ./.env

# Compute nodes have no internet; model must be pre-downloaded to ~/.cache/huggingface
export HF_HUB_OFFLINE=1

# GPU isolation isn't enforced on these nodes and Slurm's CUDA_VISIBLE_DEVICES
# can point at a GPU already occupied by other/orphaned processes (seen on
# dw-2-4). Always re-pin to the visible GPU with the least memory in use.
echo "Slurm gave CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "Pinned to GPU $CUDA_VISIBLE_DEVICES"

# n=50 matches the pre-SOO baseline eval for a like-for-like comparison
for seed in 0 1 2 3 4; do
    python -m soo.evaluate \
        --model allenai/OLMo-2-0425-1B-Instruct \
        --adapter "results/checkpoints/olmo2-1b/seed${seed}" \
        --scenarios main treasure_hunt \
        --n 50 \
        --tag "soo_1b_seed${seed}"
done

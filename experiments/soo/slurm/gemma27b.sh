#!/bin/bash --login
#SBATCH --job-name=soo-gemma27b
#SBATCH --time=06:00:00
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
MODEL=google/gemma-2-27b-it
SCENS="main treasure_hunt perspectives"

# Gemma-2-27b-it reproduction: paper appendix A.1.2 recipe (layer 20, r=4,
# alpha=8, dropout 0.1, lr 9e-4, 8 epochs), bf16, paper's Gemma eval suffix
# "Only respond with the room name, no other text."

echo "=== baseline eval ==="
python -m selfconcept.soo.evaluate --model "$MODEL" \
    --scenarios $SCENS --n 50 --suffix room_only --tag baseline_gemma27b

for seed in 0 1; do
    echo "=== Gemma-2-27b seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/gemma2-27b.yaml --seeds ${seed}
    python -m selfconcept.soo.evaluate --model "$MODEL" \
        --adapter "results/checkpoints/gemma2-27b/seed${seed}" \
        --scenarios $SCENS --n 50 --suffix room_only --tag "soo_gemma27b_seed${seed}"
done

echo "=== gemma27b complete ==="

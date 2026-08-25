#!/bin/bash --login
#SBATCH --job-name=soo-gemma-sw-mir
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=/home/kobyjl/experiments/self-concept-experiments/experiments/soo

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

# Eval-only: mirrored-orientation evals for the existing Gemma sweep
# checkpoints. L16 (the gap-peak layer) turned out orientation-asymmetric on
# main (70/66% deceptive original vs 0% mirrored) — the working layer L14 was
# never tested mirrored, so its position-robustness is unestablished. L11
# rides along as the no-op control.

for L in 14 11; do
    for seed in 0 1; do
        echo "=== Gemma L${L} seed ${seed} mirrored ==="
        python -m selfconcept.soo.evaluate --model "$MODEL" \
            --adapter "results/checkpoints/gemma2-27b-L${L}/seed${seed}" \
            --data data/eval_mirrored --scenarios $SCENS --n 50 --suffix room_only \
            --tag "soo_gemma27b_L${L}_seed${seed}_mir"
    done
done

echo "=== gemma sweep mirrored evals complete ==="

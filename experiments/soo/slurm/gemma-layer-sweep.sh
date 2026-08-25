#!/bin/bash --login
#SBATCH --job-name=soo-gemma-sweep
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=a100:1
#SBATCH --qos=dw87
#SBATCH --exclude=dw-2-4
#SBATCH --mem-per-cpu=24G
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
MODEL=google/gemma-2-27b-it
SCENS="main treasure_hunt perspectives"

# Does a "Gemma L16" exist? The paper's L20 (43% of 46 layers) breaks the
# model with its own recipe (both seeds). On Mistral the same story at the
# paper's L19 resolved by moving to L16 (50% depth). Sweep the mid-stack:
# L11 (24%), L14 (30%), L23 (50%, the Mistral-L16 depth analog), L28 (61%,
# the paper's Mistral-L19 depth), L34 (74%). Paper recipe otherwise
# unchanged (r=4 alpha=8 dropout 0.1, lr 9e-4, 8 epochs, lasttok, bf16).

echo "=== layer-gap diagnostic (Gemma) ==="
python scripts/layer_gap.py "$MODEL" results/layer_gap_gemma27b.json

for L in 11 14 23 28 34; do
    for seed in 0 1; do
        echo "=== Gemma L${L} seed ${seed} ==="
        python -m selfconcept.soo.train --config configs/gemma2-27b.yaml --layer ${L} --seeds ${seed}
        python -m selfconcept.soo.evaluate --model "$MODEL" \
            --adapter "results/checkpoints/gemma2-27b-L${L}/seed${seed}" \
            --scenarios $SCENS --n 50 --suffix room_only \
            --tag "soo_gemma27b_L${L}_seed${seed}"
    done
done

echo "=== gemma layer sweep complete ==="

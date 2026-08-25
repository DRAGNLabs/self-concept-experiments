#!/bin/bash --login
#SBATCH --job-name=soo-gemma-L16
#SBATCH --time=02:30:00
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

# The layer-gap diagnostic peaks at L16 for Gemma (rel_last 0.126, with
# L15/L12 next); the sweep's working layer L14 sits on the peak's rising
# edge but the peak itself was never trained. Test it — if the "train at the
# last-token self/other gap peak" rule holds (Mistral's robust L16 is its #2
# gap layer; OLMo prediction pending in job 13291931), L16 should be at
# least as good as L14. Mirrored orientation included.

echo "=== Gemma baseline, mirrored orientation ==="
python -m selfconcept.soo.evaluate --model "$MODEL" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --suffix room_only --tag baseline_gemma27b_mir

for seed in 0 1; do
    echo "=== Gemma L16 seed ${seed} ==="
    python -m selfconcept.soo.train --config configs/gemma2-27b.yaml --layer 16 --seeds ${seed}
    adapter="results/checkpoints/gemma2-27b-L16/seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" --adapter "$adapter" \
        --scenarios $SCENS --n 50 --suffix room_only \
        --tag "soo_gemma27b_L16_seed${seed}"
    python -m selfconcept.soo.evaluate --model "$MODEL" --adapter "$adapter" \
        --data data/eval_mirrored --scenarios $SCENS --n 50 --suffix room_only \
        --tag "soo_gemma27b_L16_seed${seed}_mir"
done

echo "=== gemma L16 complete ==="

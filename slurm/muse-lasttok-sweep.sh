#!/bin/bash --login
#SBATCH --job-name=soo-muse-sweep
#SBATCH --time=08:00:00
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
MODEL=meta-models/Muse-Glimmer-30B
SCENS="main treasure_hunt perspectives"

# First modern (Aug 2026) model in the study. Everything the other three
# models taught us, applied from the start: lasttok loss, both eval
# orientations (position-robustness first-class), layer-gap diagnostic as a
# prospective band predictor. Sweep 52 layers at ~25/30/40/50/60% depth.

echo "=== layer-gap diagnostic (Muse Glimmer) ==="
[ -f results/layer_gap_muse30b.json ] || python scripts/layer_gap.py "$MODEL" results/layer_gap_muse30b.json

echo "=== baseline evals, both orientations ==="
python -m soo.evaluate --force-user-channel --model "$MODEL" \
    --scenarios $SCENS --n 50 --tag baseline_muse30b
python -m soo.evaluate --force-user-channel --model "$MODEL" --data data/eval_mirrored \
    --scenarios $SCENS --n 50 --tag baseline_muse30b_mir

for L in 13 16 21 26 31; do
    for seed in 0 1; do
        echo "=== Muse lasttok L${L} seed ${seed} ==="
        python -m soo.train --config configs/muse-glimmer-30b.yaml --layer ${L} --seeds ${seed}
        adapter="results/checkpoints/muse-glimmer-30b-L${L}/seed${seed}"
        python -m soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
            --scenarios $SCENS --n 50 --tag "soo_muse30b_L${L}_seed${seed}"
        python -m soo.evaluate --force-user-channel --model "$MODEL" --adapter "$adapter" \
            --data data/eval_mirrored --scenarios $SCENS --n 50 \
            --tag "soo_muse30b_L${L}_seed${seed}_mir"
    done
done

echo "=== muse lasttok sweep complete ==="

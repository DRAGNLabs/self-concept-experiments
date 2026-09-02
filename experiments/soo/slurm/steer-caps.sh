#!/bin/bash --login
#SBATCH --job-name=soo-steer-caps
#SBATCH --time=10:00:00
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
export HF_DATASETS_OFFLINE=1

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
# Capabilities with the steering hook active, at each validated working
# strength, against a same-code-path baseline (alpha=0). The last open
# hardening item for the Gemma and Muse steering results.
OUT=results/capabilities

python scripts/caps_steered.py google/gemma-2-27b-it results/steering/gemma27b.pt \
    14 0 $OUT/steer_gemma27b_base.json
python scripts/caps_steered.py google/gemma-2-27b-it results/steering/gemma27b.pt \
    14 32 $OUT/steer_gemma27b_L14_a32.json

python scripts/caps_steered.py meta-models/Muse-Glimmer-30B results/steering/muse30b.pt \
    26 0 $OUT/steer_muse30b_base.json
python scripts/caps_steered.py meta-models/Muse-Glimmer-30B results/steering/muse30b.pt \
    26 8 $OUT/steer_muse30b_L26_a8.json

echo "=== steered capabilities complete ==="

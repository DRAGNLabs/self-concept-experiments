#!/bin/bash --login
#SBATCH --job-name=axis-pipeline
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h200:4
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Runs the assistant-axis pipeline (steps 1-5) end-to-end on the canonical model
# from the configs. Every step is restartable (skips work already on disk), so a
# re-run after a timeout resumes where it left off. Prefetch models first with
# slurm/prefetch.sh on a login node (compute nodes run offline).

source ../../activate.sh
nvidia-smi

python scripts/pipeline/1_generate.py    --config configs/pipeline/1_generate.yaml
python scripts/pipeline/2_activations.py --config configs/pipeline/2_activations.yaml
python scripts/pipeline/3_judge.py       --config configs/pipeline/3_judge.yaml
python scripts/pipeline/4_vectors.py     --config configs/pipeline/4_vectors.yaml
python scripts/pipeline/5_axis.py        --config configs/pipeline/5_axis.yaml

echo "=== Done: results/axis.pt ==="

#!/bin/bash --login
#SBATCH --job-name=axis-token-projections
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Per-token axis projections for the olmo32b-thinking sweep. Runs one middle-layer forward
# pass per conversation and keeps a single scalar projection per token, so only the 1-layer
# hidden state is materialized -- a larger batch than 2_activations.py (which captures all 64
# layers) fits on one H200. HF forward-hook job -> single GPU, no vLLM.
# Usage (from repo root):
#   sbatch slurm/token_axis_projections.sbatch
# Then plot on a login node (no GPU):
#   python scripts/projection/plot_token_axis_projections.py --config configs/projection/plot_token_axis_projections.yaml

source ../../activate.sh  # uv venv + cluster fixes; --chdir puts us in experiments/assistant-axis
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1  # compute nodes run offline; weights are prefetched
export TRANSFORMERS_OFFLINE=1
nvidia-smi

python scripts/projection/token_axis_projections.py --config configs/projection/token_axis_projections.yaml

echo "=== Done: token axis projections ==="

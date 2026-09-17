#!/bin/bash --login
#SBATCH --job-name=axis-steered-traces
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Generate role-susceptibility steering traces (step 6, Section 3.2.1). Requires
# results/axis.pt (from run_pipeline.sbatch) and a per-layer norms tensor at
# results/layer_norms.pt (from scripts/steering/activation_norms.py, or supplied). Steering runs
# through HF forward hooks, so this is a single-GPU HF job (no vLLM). Prefetch weights on
# a login node first (slurm/prefetch.sh); compute nodes run offline.

source ../../activate.sh
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1  # compute nodes run offline; weights are prefetched
nvidia-smi

python scripts/steering/6_steered_traces.py --config configs/steering/6_steered_traces.yaml

echo "=== Done: steered traces in scratch (assistant-axis/steered) ==="

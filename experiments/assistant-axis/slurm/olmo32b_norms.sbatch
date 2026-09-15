#!/bin/bash --login
#SBATCH --job-name=axis-olmo32b-norms
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:1
#SBATCH --qos=gpu
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Per-layer mean post-MLP residual-stream norm for OLMo-3.1-32B-Think on lmsys-chat-1m
# (the steering-vector scaling baseline from arXiv:2601.10387v1 section 3.2.1). This is a
# transformers forward-pass job (no vLLM): a single model spans one GPU via device_map="auto".
# A 32B model in bf16 (~60GB) fits comfortably on one H200 (141GB). Single GPU only --
# [[vllm028-multigpu-node-startup-hang]] doesn't apply here (no vLLM), but device_map sharding
# across GPUs isn't needed at this size, so we keep to one. Weights and the lmsys-chat-1m
# dataset must be prefetched on a login node first (slurm/prefetch.sh); compute nodes are offline.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

mamba activate ../../.env

python pipeline/activation_norms.py --config configs/activation_norms.yaml \
    --run.model "allenai/Olmo-3.1-32B-Think" --run.num_conversations 1000

echo "=== olmo32b_norms done ==="

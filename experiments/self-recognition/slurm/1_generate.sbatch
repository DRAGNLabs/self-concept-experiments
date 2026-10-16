#!/bin/bash --login
#SBATCH --job-name=sr-generate
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=cs
#SBATCH --partition=cs,cs2,cs3
#SBATCH --gpus=1
#SBATCH --exclude=cs-1-2
#SBATCH --requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/self-recognition  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Stage 1: one model's answers to data/prompts.jsonl. Submit once per model; resume makes requeues safe.
#   sbatch --job-name=sr-gen-qwen3 --export=ALL,MODEL_CONFIG=configs/1_generate_qwen3-32b.yaml experiments/self-recognition/slurm/1_generate.sbatch
# GLM and gpt-oss need more than 80 GB (and GLM needs FP8): add -p cs3 --gpus=b200:1.
# Fallback: -p m13h --qos=standby --gpus=h200:1.
# Smoke: EXTRA_ARGS="--run.prompt_limit 4 --run.samples_per_prompt 2 --run.run_dir <scratch>/smoke"

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi

source ../../activate.sh

python scripts/1_generate.py --config configs/1_generate.yaml --config "$MODEL_CONFIG" \
    --run.tensor_parallel_size "${TP:-1}" ${EXTRA_ARGS:-}

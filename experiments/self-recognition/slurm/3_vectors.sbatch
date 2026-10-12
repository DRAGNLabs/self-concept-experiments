#!/bin/bash --login
#SBATCH --job-name=sr-vectors
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=standby
#SBATCH --partition=m13h,eng
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/self-recognition  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Stage 3: one model's self-recognition vector from its stage-2 judgements. Submit once per model.
#   sbatch --job-name=sr-vec-qwen3 --export=ALL,MODEL=Qwen/Qwen3-32B experiments/self-recognition/slurm/3_vectors.sbatch
# gpt-oss needs Hopper (flash-attn3 sinks): H200 here, or -p cs2 --qos=cs.
# Qwen3 / gemma-4 also fit 2x80 GB: -p cs,cs2 --qos=cs --gpus=2 --exclude=cs-1-2.
# GLM (bf16-dequantised, ~212 GB): --gpus=2 --mem=256G EXTRA_ARGS="--run.batch_size 2"; run --run.smoke_load true first.
# Smoke: EXTRA_ARGS="--run.limit_pairs 8 --run.run_dir <scratch run dir>"
# Pooled span means are cached at <run_dir>/vectors/activation_means/<slug>.pt; a re-run with the same pairs,
# limit_pairs, seed and max_length rebuilds the vector and the length checks from it without loading the model.
# EXTRA_ARGS="--run.recompute true" forces the forward passes.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
nvidia-smi

source ../../activate.sh

python scripts/3_vectors.py --run.model "$MODEL" ${EXTRA_ARGS:-}

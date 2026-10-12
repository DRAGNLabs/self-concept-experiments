#!/bin/bash --login
#SBATCH --job-name=sr-judge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=cs
#SBATCH --partition=cs2,cs
#SBATCH --gpus=1
#SBATCH --exclude=cs-1-2
#SBATCH --requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/self-recognition  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Stage 2: one target's paired judgements. Submit once per target; resume makes requeues safe.
#   sbatch --job-name=sr-judge-qwen3 --export=ALL,MODEL=Qwen/Qwen3-32B experiments/self-recognition/slurm/2_judge.sbatch
# Qwen3-32B: the defaults above (one 80 GB GPU).
# gemma-4-31B-it, gpt-oss-120b: -p m13h --qos=standby --gpus=h200:1 --exclude="" (gemma's 16k KV cache does not fit beside its weights on 80 GB)
# GLM-4.5-Air-FP8: -p m13h --qos=standby --gpus=h200:2 --exclude="" and TP=2
# SAMPLES limits the sample indices, space-separated (--export splits on commas), e.g. SAMPLES="0 1". EXTRA_ARGS takes any other --run.* flags, e.g.
#   EXTRA_ARGS="--run.split train --run.prompt_limit 40" for the pilot, or "--run.compliance_check true".

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
nvidia-smi

source ../../activate.sh

python scripts/2_judge.py --config configs/2_judge.yaml --run.target_model "$MODEL" \
    --run.tensor_parallel_size "${TP:-1}" --run.sample_indices "[$(echo ${SAMPLES:-0 1 2 3} | tr ' ' ',')]" ${EXTRA_ARGS:-}

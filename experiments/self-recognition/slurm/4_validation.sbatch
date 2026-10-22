#!/bin/bash --login
#SBATCH --job-name=sr-validate
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --qos=standby
#SBATCH --partition=m13h,eng
#SBATCH --gpus=h200:1
#SBATCH --requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/self-recognition  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Stage 4 on one model: STEP=4a (readings) or STEP=4b (steering; resumes per judge record). 4c is CPU-only:
# run it on a login node. --export splits on commas, so pass EXTRA_ARGS through the environment, lists without spaces.
#   sbatch --job-name=sr-read-qwen3 --export=ALL,STEP=4a,MODEL=Qwen/Qwen3-32B experiments/self-recognition/slurm/4_validation.sbatch
#   EXTRA_ARGS="--run.pilot true" sbatch --job-name=sr-pilot-qwen3 --export=ALL,STEP=4b,MODEL=Qwen/Qwen3-32B ...
#   EXTRA_ARGS="--run.alpha_max 0.4" sbatch --job-name=sr-steer-qwen3 --export=ALL,STEP=4b,MODEL=Qwen/Qwen3-32B ...
# Sharded 4b, one job per layer and |alpha| (4c reads the shards together):
#   for layer in 24 30; do for alpha in 0.1 0.2 0.4; do
#     EXTRA_ARGS="--run.steering_layers [24,30] --run.alpha_max 0.4 --run.shard_layers [$layer] --run.shard_alphas [$alpha]" \
#       sbatch --job-name=sr-steer-qwen3-L$layer-a$alpha --export=ALL,STEP=4b,MODEL=Qwen/Qwen3-32B ...
#   done; done
# gpt-oss needs Hopper (flash-attn3 sinks) and > 80 GB: keep it on H200.
# Qwen3 / gemma-4 also fit 2x80 GB: -p cs,cs2 --qos=cs --gpus=2 --exclude=cs-1-2.
# GLM (bf16-dequantised, ~212 GB): --gpus=h200:2 --mem=256G and EXTRA_ARGS="--run.batch_size 2".
# Smoke: EXTRA_ARGS="--run.run_dir <scratch run dir> --run.limit_per_split 5" (4a) or "--run.record_limit 10" (4b).
# Never --qos=dw87.

export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"
nvidia-smi

source ../../activate.sh

case "$STEP" in
    4a) SCRIPT=scripts/4a_read_vectors.py ;;
    4b) SCRIPT=scripts/4b_steer_judgements.py ;;
    *) echo "STEP must be 4a or 4b, got '$STEP'" >&2; exit 1 ;;
esac
read -ra EXTRA_ARG_LIST <<< "${EXTRA_ARGS:-}"
python "$SCRIPT" --config configs/4_validation.yaml --run.model "$MODEL" "${EXTRA_ARG_LIST[@]}"

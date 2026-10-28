#!/bin/bash --login
#SBATCH --job-name=rh-judge
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h,eng
#SBATCH --qos=standby
#SBATCH --requeue  # rh_judge skips dossiers already in the output
#SBATCH --gpus=h200:1
#SBATCH --mem=128G
#SBATCH --output=slurm-logs/%x-%j.out
set -euo pipefail

# Judge a list of rh_hillclimb dossiers with rubric v3 (selfconcept.codebench.rh_judge).
# Usage (from repo root):
#   sbatch experiments/codebench/slurm/rh_judge.sbatch <dossier list> <output jsonl> [rh_judge args...]
# GLM-4.5-Air needs ~160 GB of FP8-capable GPU (two H200s or one B200):
#   sbatch --gpus=h200:2 experiments/codebench/slurm/rh_judge.sbatch <list> <out> \
#     --run.judge_model zai-org/GLM-4.5-Air-FP8 --run.chat_format think_tags --run.temperature 0.6 --run.top_p 0.95
DOSSIER_LIST="$1"
OUTPUT_PATH="$2"
shift 2

source "$SLURM_SUBMIT_DIR/activate.sh"
export HF_HOME="$HOME/nobackup/autodelete/hf_cache"
export HF_HUB_OFFLINE=1
export TIKTOKEN_RS_CACHE_DIR="$HOME/nobackup/autodelete/self-concept-experiments/tiktoken-rs-cache"  # harmony vocab; node-local /tmp default is empty offline
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Tensor parallel over the allocated GPUs (nvidia-smi can show more on some nodes); a later --run.tensor_parallel_size wins
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
python -m selfconcept.codebench.rh_judge --run.dossier_list "$DOSSIER_LIST" --run.output_path "$OUTPUT_PATH" \
  --run.tensor_parallel_size "$N_GPUS" "$@"

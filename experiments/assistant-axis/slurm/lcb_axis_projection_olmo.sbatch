#!/bin/bash --login
#SBATCH --job-name=lcb-axisproj-olmo
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=m13h
#SBATCH --qos=gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=128G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/assistant-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Teacher-forced assistant-axis projection (8_lcb_axis_projection.py) of the steered LCB runs
# in OUT_DIR (every coef_*.jsonl present). Usage (from repo root):
#   sbatch experiments/assistant-axis/slurm/lcb_axis_projection_olmo.sbatch OUT_DIR

OUT="${1:?pass the output directory}"
AXIS="${AXIS:-/grphome/grp_mirror/self-concept-experiments/experiments/assistant-axis/olmo32b-thinking/axis_response_only.pt}"
REPO="$(cd ../.. && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$REPO/.venv-313/bin/activate"
nvidia-smi

INPUTS=$(ls "$OUT"/coef_*.jsonl | paste -sd, -)
python scripts/steering/8_lcb_axis_projection.py --run.axis_path "$AXIS" --run.inputs "[$INPUTS]"
echo "=== projection done ==="

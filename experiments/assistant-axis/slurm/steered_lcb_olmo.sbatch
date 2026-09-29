#!/bin/bash --login
#SBATCH --job-name=axis-lcb-olmo
#SBATCH --time=08:00:00
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

# LiveCodeBench problems (EvilGenie's set, plain LCB prompt) on OLMo-3.1-32B-Think with the
# response-only assistant axis added at layer 32, one coefficient per job (7_steered_lcb.py).
# Usage (from repo root):
#   sbatch experiments/assistant-axis/slurm/steered_lcb_olmo.sbatch COEFFICIENT OUT_DIR [--run.<field> VALUE ...]
# OUT_DIR must hold layer_norms.pt (olmo32b_norms_wildchat.sbatch). Resumable.

COEF="${1:?pass the steering coefficient (fraction of the layer norm)}"
OUT="${2:?pass the output directory}"
AXIS="${AXIS:-/grphome/grp_mirror/self-concept-experiments/experiments/assistant-axis/olmo32b-thinking/axis_response_only.pt}"
REPO="$(cd ../.. && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export OPENSSL_CONF="${OPENSSL_CONF:-/dev/null}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$REPO/.venv-313/bin/activate"
nvidia-smi

python scripts/steering/7_steered_lcb.py --config configs/steering/7_steered_lcb.yaml \
    --run.coefficient "$COEF" --run.axis_path "$AXIS" --run.norms_path "$OUT/layer_norms.pt" --run.output_dir "$OUT" "${@:3}"

echo "=== steered LCB coefficient $COEF done ==="

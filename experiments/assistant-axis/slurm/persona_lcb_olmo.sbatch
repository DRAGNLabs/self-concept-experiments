#!/bin/bash --login
#SBATCH --job-name=persona-lcb-olmo
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

# LiveCodeBench problems on OLMo-3.1-32B-Think, unsteered, under one persona from the axis
# pipeline's role instructions (system turn, variants rotated over problems), then the
# teacher-forced assistant-axis projection of the transcripts (8_lcb_axis_projection.py).
# Usage (from repo root):
#   sbatch experiments/assistant-axis/slurm/persona_lcb_olmo.sbatch ROLE OUT_DIR [--run.<field> VALUE ...]
# Resumable (generation per problem, projection per file).

ROLE="${1:?pass the persona role id (data/roles/instructions/<role>.json)}"
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
    --run.persona "$ROLE" --run.coefficient 0 --run.axis_path "$AXIS" --run.norms_path "$OUT/layer_norms.pt" --run.output_dir "$OUT" "${@:3}"
python scripts/steering/8_lcb_axis_projection.py --run.axis_path "$AXIS" --run.inputs "[$OUT/persona_${ROLE}_coef_+0.00.jsonl]"

echo "=== persona $ROLE done ==="

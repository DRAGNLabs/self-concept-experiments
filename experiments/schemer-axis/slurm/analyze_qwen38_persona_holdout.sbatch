#!/bin/bash --login
#SBATCH --job-name=schemer-qwen38-auroc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis
set -euo pipefail

# Submit with an afterok dependency on every extraction-array task.  This is a
# small CPU reduction only; model inference and activation extraction remain in
# extract_roles_qwen38.sbatch.
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38-27b/full}"
export OPENSSL_CONF=/dev/null
unset SELFCONCEPT_VENV
source ../../activate.sh
for SCORE_REGION in response_only with_cot; do
    uv run python scripts/persona_holdout_auroc.py --root "$ROOT" --axis-region response_only --score-region "$SCORE_REGION" \
        --out "$ROOT/analysis/persona_holdout_axes_response_only_scores_${SCORE_REGION}_L32.json"
done

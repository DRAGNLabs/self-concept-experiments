#!/bin/bash --login
#SBATCH --job-name=schemer-generic-only-tokens
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=h200:1
#SBATCH --partition=m13h
#SBATCH --qos=standby
#SBATCH --output=../schemer-axis/slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/schemer-axis
set -euo pipefail

# Per-token projections of every intent and task response onto the canonical generic-only axis (token_figures.py input).
# The axis comes from generic_only_analysis.sbatch (--response-only-axis is the canonical persona_contrast).
SCRATCH="${SCRATCH:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis}"
EXTENSION_ROOT="${EXTENSION_ROOT:-$SCRATCH/gptoss120b_extension/full}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/gptoss120b_generic_only}"
CANONICAL_AXIS="${CANONICAL_AXIS:-$HOME/groups/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis/gptoss120b/roles_fic/candidates_response_only_17role/persona_contrast.pt}"

export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENSSL_CONF=/dev/null
export TIKTOKEN_RS_CACHE_DIR="$SCRATCH/../tiktoken-rs-cache"
unset SELFCONCEPT_VENV
source ../../activate.sh

uv run python scripts/token_projections.py --root "$EXTENSION_ROOT" --response-only-axis "$CANONICAL_AXIS" --layer 17 \
    --model openai/gpt-oss-120b --out "$OUT_ROOT/tokens_L17" --eval-question-groups intent task \
    --attn-implementation kernels-community/vllm-flash-attn3 --max-length 20480 --batch-size 4

#!/bin/bash
#SBATCH --job-name=token-shift
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis
# EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.
# CPU: episode-level token-frequency shifts, gi_v2 - vs gfar - vs unsteered, all four models (scripts/token_shift.py).
set -euo pipefail
export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ../../.venv/bin/activate
cd scripts
python token_shift.py --out "$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/abstention_admission_pilot/token_shift" --workers 16

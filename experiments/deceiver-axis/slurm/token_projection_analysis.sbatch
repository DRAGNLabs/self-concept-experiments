#!/bin/bash
#SBATCH --job-name=token-proj-analysis
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis
# EXPLORATORY (lightly reviewed when merged 2026-11-20): kept for reference, not a validated measurement.
# CPU: token-level ROC / stratified AUROC and episode deciles from axis_acts_extract.py --token-projections caches
# (scripts/token_projection_analysis.py), then the charts (scripts/token_projection_charts.py).
# Usage (from the repo root): MODELS="gptoss muse gemma" sbatch --export=ALL experiments/deceiver-axis/slurm/token_projection_analysis.sbatch
set -euo pipefail
source ../../.venv/bin/activate
ROOT="${ROOT:-$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/token_proj}"
cd scripts
python -W ignore token_projection_analysis.py --tokens "$ROOT" --out "$ROOT/analysis" --models ${MODELS:-gptoss muse gemma}
python -W ignore token_projection_charts.py --analysis "$ROOT/analysis" --out "$ROOT/charts"

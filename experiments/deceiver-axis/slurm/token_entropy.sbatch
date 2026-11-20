#!/bin/bash
#SBATCH --job-name=token-entropy
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis
# CPU: token entropy, gi_v2 - vs gfar - vs unsteered, all four models (scripts/token_entropy.py).
set -euo pipefail
export HF_HOME="$HOME/nobackup/autodelete/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source ../../.venv/bin/activate
cd scripts
python token_entropy.py --out "$HOME/nobackup/autodelete/self-concept-experiments/schemer-axis/abstention_admission_pilot/token_entropy" --workers 16

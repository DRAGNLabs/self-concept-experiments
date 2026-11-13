#!/bin/bash --login
#SBATCH --job-name=goodfire-geometry
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=m12
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --chdir=experiments/deceiver-axis  # relative to the submission dir -- run sbatch from the repo root
set -euo pipefail

# Hack-minus-honest directions from goodfire_readouts.pt with their validation AUROC, split-half reliability and
# cosine against reference directions (goodfire_rh_geometry.py). CPU only; Muse took ~45 min on 32 cores.
# Usage (from the repo root): export G=<dir holding goodfire_readouts.pt> REFERENCES=<pt: name -> (layer, hidden)>
#   sbatch --export=ALL [--dependency=afterok:<readouts job>] experiments/deceiver-axis/slurm/goodfire_geometry.sbatch
# Writes $G/geometry/{goodfire_rh_geometry.json,directions_headline.pt,directions_contrasts_<readout>.pt}.

G="${G:?set G to the dir holding goodfire_readouts.pt}"
REFERENCES="${REFERENCES:?set REFERENCES to the reference directions file}"
CONTRASTS="${CONTRASTS:-$HOME/nobackup/autodelete/self-concept-experiments/goodfire_rh_synth/hf/synthetic_data/contrasts.json}"
export OPENSSL_CONF=/dev/null OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SELFCONCEPT_VENV="${SELFCONCEPT_VENV:-$SLURM_SUBMIT_DIR/.venv-313}"
source ../../activate.sh
export PYTHONPATH="$PWD/../../src:${PYTHONPATH:-}"
echo "start $(date): $G"
python scripts/goodfire_rh_geometry.py --readouts "$G/goodfire_readouts.pt" --contrasts "$CONTRASTS" --reference-directions "$REFERENCES" --out "$G/geometry"
echo "=== goodfire-geometry done $(date): $G/geometry ==="
